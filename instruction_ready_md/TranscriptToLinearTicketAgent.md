# Guide: Converting Meeting Transcripts to Linear Tickets with Mistral AI

## Overview
This guide demonstrates how to build an automated pipeline that converts customer call transcripts into actionable development tickets on Linear. The process uses Mistral AI's LLM for intelligent document processing and a GraphQL client to interact with Linear's API.

The pipeline consists of two main stages:
1.  **PRD Generation:** Transforms a raw transcript into a structured Product Requirements Document (PRD).
2.  **Ticket Creation:** Parses the PRD into discrete features and creates corresponding tickets in your Linear project.

## Prerequisites & Setup

Before you begin, ensure you have the following:

1.  **API Keys:**
    *   A **Mistral AI API Key** from the [Mistral AI console](https://console.mistral.ai/api-keys/).
    *   A **Linear API Key** (OAuth Token) from your Linear settings (`Settings` → `API`).
2.  **Linear Team ID:** The unique identifier for your Linear team or project.
3.  **Python Environment:** A Python environment where you can install the required packages.

### Step 1: Install Required Libraries
Run the following command to install the necessary Python packages.

```bash
pip install mistralai==1.5.1 gql==3.5.0 pydantic==2.10.6 pypdf==5.3.0
```

### Step 2: Import Modules
Create a new Python script or notebook and import the required modules.

```python
from mistralai import Mistral
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pypdf import PdfReader
import json
```

### Step 3: Download a Sample Transcript
For demonstration, we'll use a synthetic transcript about a product called "LeChat". Download it using the command below.

```bash
wget 'https://raw.githubusercontent.com/mistralai/cookbook/main/mistral/agents/non_framework/transcript_linearticket_agent/lechat_product_call_trascript.pdf' -O './lechat_product_call_trascript.pdf'
```

## Configuration

Now, let's set up the configuration for both Mistral AI and Linear. Replace the placeholder values with your actual API keys and team ID.

```python
@dataclass
class Config:
    """Configuration settings for the application."""
    LINEAR_API_KEY: str # OAuth token for Linear API authentication
    LINEAR_TEAM_ID: str # Unique identifier for your Linear team/project
    LINEAR_GRAPHQL_URL: str # Linear's GraphQL API endpoint
    MISTRAL_API_KEY: str # API Key for accessing Mistral LLMs
    MISTRAL_MODEL: str # Specific Mistral model to use

config = Config(
    LINEAR_API_KEY = "YOUR_LINEAR_API_KEY_HERE",
    LINEAR_TEAM_ID = "YOUR_LINEAR_TEAM_ID_HERE",
    LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql",
    MISTRAL_API_KEY = "YOUR_MISTRAL_API_KEY_HERE",
    MISTRAL_MODEL = "mistral-large-latest", # Corrected model name
)
```

## Define Data Models

We'll use Pydantic to define a structured model for the features extracted from the PRD. This ensures clean data validation.

```python
class FeaturesList(BaseModel):
    """Pydantic model for structured feature data."""
    Features: List[str]
    DescriptionOfFeatures: List[str]
```

## Step 4: Build the PRD Generation Agent

The `PRDAgent` is responsible for the iterative process of creating and refining a PRD from a transcript.

```python
class PRDAgent:
    """Agent responsible for generating and refining PRD from transcripts."""

    def __init__(self, transcript: str, mistral_client: Mistral, model: str = "mistral-large-latest"):
        self.transcript: str = transcript
        self.prd: Optional[str] = None
        self.feedback: Optional[str] = None
        self.client: Mistral = mistral_client
        self.model: str = model

    def generate_initial_prd(self) -> str:
        """Generate initial PRD from transcript."""
        prompt = f"""
        Based on the following call transcript, create an initial Product Requirements Document (PRD) with some or all of these sections:
        1. Title
        2. Purpose
        3. Scope
        4. Features and Requirements
        5. User Personas
        6. Technical Requirements
        7. Constraints
        8. Success Metrics
        9. Timeline and Milestones

        Transcript:
        {self.transcript}

        Align everything only with the information provided in the transcript. If any section is not present in the transcript, you can skip it in the PRD.

        PRD:
        """
        response = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        self.prd = response.choices[0].message.content
        return self.prd

    def get_feedback(self) -> str:
        """Get feedback on current PRD by comparing it to the original transcript."""
        prompt = f"""
            Review the following Product Requirements Document (PRD) based on the original call transcript. Provide feedback on:
            - Missing information in PRD that are present in the transcript.
            - Inconsistencies in the PRD that are not aligned with the transcript.

            Transcript:
            {self.transcript}

            Current PRD:
            {self.prd}

            Align the feedback only with the information provided in the transcript. We are not looking for additional information based on your knowledge.

            If no feedback is required, respond with "None." and don't provide any further feedback. Your task is only to review the alignment between the PRD and the transcript and provide feedback based on that. Don't refine the PRD at this stage.

            Feedback:
            """
        response = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        self.feedback = response.choices[0].message.content
        return self.feedback

    def refine_prd(self) -> str:
        """Refine PRD based on feedback."""
        prompt = f"""
        Refine the PRD based on the provided feedback and aligning it with the transcript:

        Current PRD:
        {self.prd}

        Feedback:
        {self.feedback}

        Transcript:
        {self.transcript}
        """
        response = self.client.chat.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        self.prd = response.choices[0].message.content
        return self.prd

    def run(self, max_iterations: int = 3) -> str:
        """
        Run the PRD generation and refinement process.
        Iterates until feedback is 'None' or the max iterations are reached.
        """
        print("Generating initial PRD...")
        self.generate_initial_prd()
        print(f"Initial PRD:\n{self.prd}")

        for iteration in range(max_iterations):
            print(f"\nIteration {iteration}: Requesting feedback...")
            feedback = self.get_feedback()
            print(f"Feedback:\n{feedback}")

            if "none" in feedback.strip().lower():
                print("\nNo further feedback. Finalizing PRD...")
                break

            print("\nRefining PRD...")
            self.refine_prd()
            print(f"Refined PRD:\n{self.prd}")

        return self.prd
```

## Step 5: Build the Ticket Creation Agent

The `TicketCreationAgent` handles communication with Linear's GraphQL API to create tickets from the parsed PRD features.

```python
class TicketCreationAgent:
    """Agent responsible for creating Linear tickets from PRD."""

    def __init__(self, api_key: str, team_id: str, mistral_client: Mistral, graphql_url: str):
        self.client = Client(
            transport=RequestsHTTPTransport(
                url=graphql_url,
                headers={'Authorization': api_key},
                verify=True,
                retries=3
            ),
            fetch_schema_from_transport=True
        )
        self.team_id = team_id
        self.mistral_client = mistral_client

    def parse_prd(self, prd_text: str) -> Dict[str, List[str]]:
        """Parse PRD text into a structured list of features and descriptions using Mistral's structured output."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant helping to create Features list and their descriptions from a Product Requirements Document (PRD)."
                    "The description should contain a brief explanation of the feature that includes Technical requirements (if any), Constraints (if any), Success metrics (if any), User personas (if any), and Timeline and Milestones (if any)."
                )
            },
            {
                "role": "user",
                "content": f"PRD:\n\n{prd_text}"
            }
        ]

        chat_response = self.mistral_client.chat.parse(
            model="mistral-large-latest",
            messages=messages,
            response_format=FeaturesList,
            max_tokens=2048,
            temperature=0.1
        )
        # The .parse method returns a Pydantic model, we convert it to a dict
        return chat_response.choices[0].message.parsed.dict()

    def create_ticket(self, title: str, description: str) -> Dict[str, Any]:
        """Create a single Linear ticket via GraphQL mutation."""
        mutation = gql("""
        mutation CreateIssue($title: String!, $description: String!, $teamId: String!) {
            issueCreate(
                input: {
                    title: $title,
                    description: $description,
                    teamId: $teamId
                }
            ) {
                success
                issue {
                    id
                    url
                }
            }
        }
        """)

        variables = {
            "title": title,
            "description": description,
            "teamId": self.team_id
        }

        result = self.client.execute(mutation, variable_values=variables)
        print(f"Created ticket: {result['issueCreate']['issue']['url']}")
        return result

    def create_tickets_from_prd(self, parsed_items: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Create Linear tickets from parsed PRD items."""
        results = []
        for title, description in zip(
            parsed_items['Features'],
            parsed_items['DescriptionOfFeatures']
        ):
            result = self.create_ticket(title, description)
            results.append(result)
        return results
```

## Step 6: Create a Workflow Orchestrator

The `WorkflowOrchestrator` ties everything together, managing the sequence from transcript to tickets.

```python
class WorkflowOrchestrator:
    """Orchestrates the entire workflow from transcript to Linear tickets."""

    def __init__(self, config: Config, transcript: str):
        mistral_client = Mistral(api_key=config.MISTRAL_API_KEY)
        self.prd_agent = PRDAgent(
            transcript=transcript,
            mistral_client=mistral_client
        )
        self.linear_agent = TicketCreationAgent(
            api_key=config.LINEAR_API_KEY,
            team_id=config.LINEAR_TEAM_ID,
            mistral_client=mistral_client,
            graphql_url=config.LINEAR_GRAPHQL_URL
        )

    def run(self) -> Dict[str, Any]:
        """Run the complete workflow."""
        print("Generating and finalizing PRD...")
        prd = self.prd_agent.run()

        print("\nParsing PRD into actionable items...")
        parsed_items = self.linear_agent.parse_prd(prd)

        print("\nCreating Linear tickets...")
        ticket_results = self.linear_agent.create_tickets_from_prd(parsed_items)

        return {
            "prd": prd,
            "parsed_items": parsed_items,
            "ticket_results": ticket_results
        }
```

## Step 7: Parse the Transcript PDF

To extract text from a PDF transcript, we'll use Mistral's OCR model. This function handles the upload and processing.

```python
def parse_transcript(config: Config, file_path: str) -> str:
    """Parse a transcript PDF file and extract text from all pages using Mistral OCR."""
    mistral_client = Mistral(api_key=config.MISTRAL_API_KEY)

    uploaded_pdf = mistral_client.files.upload(
        file={
            "file_name": file_path,
            "content": open(file_path, "rb"),
        },
        purpose="ocr"
    )

    signed_url = mistral_client.files.get_signed_url(file_id=uploaded_pdf.id)

    ocr_response = mistral_client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        }
    )

    text = "\n".join([x.markdown for x in (ocr_response.pages)])
    return text
```

## Step 8: Execute the Pipeline

Now, let's run the entire pipeline with our sample transcript.

```python
# 1. Extract text from the PDF transcript
transcript = parse_transcript(config, "./lechat_product_call_trascript.pdf")

# 2. Initialize and run the orchestrator
orchestrator = WorkflowOrchestrator(config, transcript)
results = orchestrator.run()
```

When you run this code, you will see output similar to the following, showing the iterative PRD refinement and ticket creation process:

```
Generating and finalizing PRD...
Generating initial PRD...
Initial PRD:
[Initial PRD content...]

Iteration 0: Requesting feedback...
Feedback:
[Feedback content...]

Refining PRD...
Refined PRD:
[Refined PRD content...]
...
Parsing PRD into actionable items...

Creating Linear tickets...
Created ticket: https://linear.app/your-team/issue/ABC-123/feature-one
Created ticket: https://linear.app/your-team/issue/ABC-124/feature-two
```

## Step 9: Inspect the Results

After the pipeline runs, you can inspect the generated artifacts.

```python
# View the final PRD
print("Final PRD:")
print(results["prd"])

# View the parsed features and descriptions
print("\n--- Parsed Features ---")
for feature, desc in zip(
    results["parsed_items"]["Features"],
    results["parsed_items"]["DescriptionOfFeatures"]
):
    print(f"\nFeature: {feature}")
    print(f"Description: {desc}")

# View the Linear ticket creation results
print("\n--- Ticket Creation Results ---")
for result in results["ticket_results"]:
    print(result)
```

## Next Steps & Extensions

You can extend this pipeline to better suit your workflow:

1.  **Add Ticket Prioritization:** Modify the `create_ticket` mutation to include priority fields from Linear.
2.  **Include Custom Fields:** If your Linear team uses custom fields (e.g., "Effort Points"), add them to the ticket creation variables.
3.  **Integrate Other Tools:** Adapt the `TicketCreationAgent` to work with other project management tools like Jira by changing the GraphQL client to their respective API client.
4.  **Add Error Handling:** Implement more robust error handling and logging for production use.
5.  **Batch Processing:** Modify the pipeline to process multiple transcripts or PRDs in sequence.

This guide provides a foundational automated workflow to transform meeting discussions into tracked development work, saving time and reducing information loss.