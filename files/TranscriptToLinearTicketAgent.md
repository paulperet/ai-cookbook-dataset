# Call Transcript-to-PRD-to-Ticket Agent: Converting Meeting Transcripts to Linear Tickets using Mistral AI LLMs

## Problem Statement
In modern software development, a significant challenge is efficiently converting customer calls and meetings into actionable development tickets. This process typically involves:
- Manual note-taking during calls
- Converting notes into Product Requirements Documents (PRDs)
- Breaking down PRDs into actionable tickets
- Creating and managing tickets in project management tools (ex:- Linear)

This manual process is:
- Time-consuming
- Prone to information loss
- Subject to inconsistencies
- Difficult to scale

## Our Solution
We've created an automated pipeline that leverages Mistral's LLM and OCR models to streamline this process:

### Stage 1: PRD Generation
- Takes raw call transcripts as input (parsed using Mistral OCR)
- Uses Mistral AI LLM to generate structured PRD
- Implements iterative refinement for accuracy
- Ensures alignment with original discussion (from transcript)

### Stage 2: Feature & Technical requirements Extraction
- Analyzes PRD to identify distinct features
- Extracts technical requirements
- Captures constraints and success metrics
- Maintains traceability to original content(call/ transcript)

## Mistral LLM Integration

The solution uses several Mistral AI LLM capabilities:

1. **Chat Completion API**
   - Used for PRD generation
   - Handles iterative refinement
   - Processes feedback and improvements

2. **Structured Output**
   - Formats PRD content
   - Extracts feature lists
   - Generates ticket descriptions

3. **Context Management**
   - Maintains consistency across iterations
   - Preserves original transcript context
   - Ensures accurate information flow

This notebook walks through the implementation of this pipeline, demonstrating how to automate the journey from call transcripts to PRD creation to actionable development tickets on Linear.

### Installation


```python
!pip install mistralai==1.5.1 # MistralAI
!pip install gql==3.5.0 # GraphQL
!pip install pydantic==2.10.6 # Data validation
!pip install pypdf==5.3.0  # PDF processing
```

[Collecting mistralai==1.5.1, ..., Successfully installed pypdf-5.3.0]

### Imports


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

### Download Call Transcript

For this demonstration we will use a product call regarding LeChat.

*Note*: The trascript is synthetically generated just for the demonstration purposes.


```python
!wget 'https://raw.githubusercontent.com/mistralai/cookbook/main/mistral/agents/non_framework/transcript_linearticket_agent/lechat_product_call_trascript.pdf' -O './lechat_product_call_trascript.pdf'
```

### Configuration and Setup

Our pipeline integrates Mistral AI LLM for PRD generation and Linear for ticket management. Let's set up the required configurations:

## API Setup
1. **Linear Configuration**
   - Get API key from Linear (Settings → API)
   - Get your Team ID
   - GraphQL endpoint: https://api.linear.app/graphql

2. **Mistral AI Configuration**
   - Get API key from Mistral AI
   - We use "mistral-large-latest" model


```python
@dataclass
class Config:
    """Configuration settings for the application."""
    LINEAR_API_KEY: str # OAuth token for Linear API authentication
    LINEAR_TEAM_ID: str # Unique identifier for your Linear team/project
    LINEAR_GRAPHQL_URL: str # Linear's GraphQL API endpoint (usually "https://api.linear.app/graphql")
    MISTRAL_API_KEY: str # API Key for accessing Mistral LLMs
    MISTRAL_MODEL: str # Specific Mistral model to use (e.g., "mistral-large-latest")

config = Config(
    LINEAR_API_KEY = "YOUR API KEY ON LINEAR",
    LINEAR_TEAM_ID = "YOUR TEAM ID ON LINEAR",
    LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql",
    MISTRAL_API_KEY = "YOUR MISTRAL API KEY", # Get your API key from https://console.mistral.ai/api-keys/
    MISTRAL_MODEL = "ministral-large-latest",
)
```

### Data Models

We also define our data structures for Features and descriptions that we create on Linear based on PRD.


```python
class FeaturesList(BaseModel):
    """Pydantic model for structured feature data."""
    Features: List[str]
    DescriptionOfFeatures: List[str]
```

### PRD Generation Agent

The PRD Generation Agent (`PRDAgent`) is responsible for converting call transcripts into accurate PRDs through an iterative process:

1. First creates initial PRD (`generate_initial_prd`)
2. Then gets feedback (`get_feedback`)
3. Refines based on feedback (`refine_prd`)
4. Repeats until quality is satisfactory (max 3 times) (`run`)





```python
class PRDAgent:
    """Agent responsible for generating and refining PRD from transcripts."""

    def __init__(self, transcript: str, mistral_client: Mistral, model: str = "mistral-large-latest"):
        """
        Initialize PRD agent.

        Args:
            transcript (str): Call transcript text
            mistral_client (Mistral): Initialized Mistral client
            model (str): Model name to use
        """
        self.transcript: str = transcript
        self.prd: Optional[str] = None
        self.feedback: Optional[str] = None
        self.client: Mistral = mistral_client
        self.model: str = model

    def generate_initial_prd(self) -> str:
        """
        Generate initial PRD from transcript.

        Returns:
            str: Generated PRD text
        """
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
        """
        Get feedback on current PRD.

        Returns:
            str: Feedback text
        """
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
        """
        Refine PRD based on feedback.

        Returns:
            str: Refined PRD text
        """
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

        Args:
            max_iterations (int): Maximum number of refinement iterations

        Returns:
            str: Final PRD text
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

### Ticket Creation Agent

The Ticket Creation Agent converts PRDs into actionable tickets on Linear through three main steps:

1. Parses PRD into structured features and descriptions (`parse_prd`)
2. Converts each feature into a ticket format (`create_ticket`)
3. Creates tickets in Linear via GraphQL API (`create_tickets_from_prd`)


```python
class TicketCreationAgent:
    """Agent responsible for creating Linear tickets from PRD."""

    def __init__(self, api_key: str, team_id: str, mistral_client: Mistral, graphql_url: str):
        """
        Initialize Linear ticket agent.

        Args:
            api_key (str): Linear API key
            team_id (str): Linear team ID
            mistral_client (Mistral): Initialized Mistral client
            graphql_url (str): Linear GraphQL API URL
        """
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
        """
        Parse PRD into structured feature data.

        Args:
            prd_text (str): PRD text to parse

        Returns:
            Dict[str, List[str]]: Structured feature data
        """
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

        return json.loads(chat_response.choices[0].message.content)

    def create_ticket(self, title: str, description: str) -> Dict[str, Any]:
        """
        Create a single Linear ticket.

        Args:
            title (str): Ticket title
            description (str): Ticket description

        Returns:
            Dict[str, Any]: Creation result from Linear API
        """
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
        """
        Create Linear tickets from parsed PRD items.

        Args:
            parsed_items (Dict[str, List[str]]): Parsed feature data

        Returns:
            List[Dict[str, Any]]: List of ticket creation results
        """
        results = []
        for title, description in zip(
            parsed_items['Features'],
            parsed_items['DescriptionOfFeatures']
        ):
            result = self.create_ticket(title, description)
            results.append(result)
        return results

```

### Workflow Orchestrator

The Workflow Orchestrator:
- Coordinates the entire process
- Manages communication between agents
- Handles the overall workflow


```python
class WorkflowOrchestrator:
    """Orchestrates the entire workflow from transcript to Linear tickets."""

    def __init__(self, config: Config, transcript: str):
        """
        Initialize workflow orchestrator.

        Args:
            config (Config): Application configuration
            transcript (str): Call transcript text
        """
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
        """
        Run the complete workflow.

        Returns:
            Dict[str, Any]: Workflow results including PRD and ticket data
        """
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

### Parse The Call Transcript

We will use Mistral OCR model to parse the downloaded call transcript file.


```python
def parse_transcript(config: Config, file_path: str) -> str:
  """Parse a transcriot PDF file and extract text from all pages using Mistral OCR."""

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


```python
transcript = parse_transcript(config, "./lechat_product_call_trascript.pdf")
```

### Running the Pipeline

Let's test the pipeline with a sample transcript that discusses about LeChat product call.


```python
orchestrator = WorkflowOrchestrator(config, transcript)
results = orchestrator.run()
```

    Generating and finalizing PRD...
    Generating initial PRD...
    Initial PRD:
    [Initial PRD content, ..., Final PRD content]
    
    Iteration 0: Requesting feedback...
    Feedback:
    [Feedback content, ..., Feedback content]
    
    Refining PRD...
    Refined PRD:
    [Refined PRD content, ..., Final PRD content]
    
    Iteration 1: Requesting feedback...
    Feedback:
    [Feedback content, ..., Feedback content]
    
    Refining PRD...
    Refined PRD:
    [Refined PRD content, ..., Final PRD content]
    
    Iteration 2: Requesting feedback...
    Feedback:
    [Feedback content, ..., Feedback content]
    
    Refining PRD...
    Refined PRD:
    [Refined PRD content, ..., Final PRD content]
    
    Parsing PRD into actionable items...
    
    Creating Linear tickets...
    [Created ticket: https://linear.app/mistralai/issue/MIS-97/infrastructure-optimization, ..., Created ticket: https://linear.app/mistralai/issue/MIS-119/personalized-learning-experiences]

### Understanding the Output
The pipeline produces:
1. A structured PRD
2. List of features and descriptions
3. Linear tickets with URLs

#### PRD


```python
print(results["prd"])
```

    [PRD content]

#### Features


```python
for feature, desc in zip(
    results["parsed_items"]["Features"],
    results["parsed_items"]["DescriptionOfFeatures"]
):
    print(f"\nFeature: {feature}")
    print(f"Description: {desc}")
```

    [Features and Descriptions]

#### Linear Tickets Created


```python
for result in results["ticket_results"]:
    print(result)
```

    [Ticket creation results]

### Next Steps
You can extend this pipeline by:
1. Adding priority levels to tickets.
2. Including custom fields in Linear tickets.
3. Incorporate similar pipeline with Jira.