# HubSpot Dynamic Multi-Agent System with Magistral Reasoning

This guide demonstrates how to build an intelligent, multi-agent system that integrates the Magistral reasoning model with HubSpot's CRM. You will create a system that can understand complex business queries, plan multi-step workflows, and execute sophisticated CRM operations automatically.

## Prerequisites

Before you begin, ensure you have:
- A HubSpot account with API access.
- A Mistral AI API key.

## Setup

### 1. Install Required Libraries

Install the necessary Python packages.

```bash
pip install hubspot-api-client==12.0.0 mistralai==1.9.3
```

### 2. Import Dependencies

Import the required modules.

```python
import requests
import json
from mistralai import Mistral, ThinkChunk, TextChunk
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import re
```

### 3. Configure API Keys

Set your HubSpot and Mistral API keys. Replace the placeholder values with your actual keys.

```python
HUBSPOT_API_KEY = "<YOUR_HUBSPOT_API_KEY>"
MISTRAL_API_KEY = "<YOUR_MISTRAL_API_KEY>"
```

### 4. Initialize the Mistral Client

Create a client to interact with the Mistral AI API.

```python
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
```

## Building the System Components

### Step 1: Create the HubSpot Connector

This class handles all interactions with the HubSpot API, including fetching data, updating records, and discovering available properties.

```python
class HubSpotConnector:
    """Handles all HubSpot API operations"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.hubapi.com/crm/v3/objects"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def get_properties(self) -> Dict:
        """Load all HubSpot properties for deals, contacts, and companies"""
        properties = {}

        for obj_type in ['deals', 'contacts', 'companies']:
            url = f"https://api.hubapi.com/crm/v3/properties/{obj_type}"
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                props = response.json()['results']
                prop_list = []
                for prop in sorted(props, key=lambda x: x['name']):
                    prop_str = f"'{prop['name']}' - {prop['label']}"
                    if 'options' in prop and prop['options']:
                        valid_values = [opt['value'] for opt in prop['options']]
                        prop_str += f" | Valid values: {valid_values}"
                    prop_list.append(prop_str)

                properties[obj_type] = prop_list

        return properties

    def get_data(self, object_type: str) -> List[Dict]:
        """Fetch data from HubSpot for a given object type"""
        url = f"{self.base_url}/{object_type}"
        params = {"limit": 100}
        all_data = []

        while url:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code != 200:
                raise Exception(f"HubSpot API error: {response.text}")

            data = response.json()
            all_data.extend(data.get("results", []))
            url = data.get("paging", {}).get("next", {}).get("link")
            params = {}

        return all_data

    def batch_update(self, updates: Dict) -> None:
        """Perform batch updates to HubSpot records"""
        for object_type, update_list in updates.items():
            if not update_list:
                continue

            url = f"{self.base_url}/{object_type}/batch/update"
            headers = {**self.headers, "Content-Type": "application/json"}

            # Process in batches of 100
            for i in range(0, len(update_list), 100):
                batch = update_list[i:i+100]
                payload = {"inputs": batch}

                response = requests.post(url, headers=headers, json=payload)
                if response.status_code not in [200, 202]:
                    raise Exception(f"HubSpot update error: {response.text}")
```

### Step 2: Define the LLM Helper Functions

Create two helper functions to interact with the Mistral models: one for complex reasoning with Magistral and another for fast execution with Mistral Small.

```python
def magistral_reasoning(prompt: str) -> Dict[str, str]:
    """Use the Magistral reasoning model for complex query analysis and planning"""
    response = mistral_client.chat.complete(
        model="magistral-medium-latest",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content

    reasoning = ""
    conclusion = ""

    for r in content:
      if isinstance(r, ThinkChunk):
          reasoning = r.thinking[0].text
      elif isinstance(r, TextChunk):
          conclusion = r.text

    return {
        "reasoning": reasoning,
        "conclusion": conclusion
    }

def mistral_small_execution(prompt: str) -> str:
    """Use the Mistral Small model for content generation and task execution"""
    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### Step 3: Implement the Lead Agent

The Lead Agent uses the Magistral model to analyze the user's query and create a detailed execution plan, determining which sub-agents are needed.

```python
class LeadAgent:
    """Lead Agent powered by Magistral reasoning model for query analysis and planning"""

    def __init__(self, hubspot_properties):
        self.hubspot_properties = hubspot_properties
        self.name = "LeadAgent"

    def analyze_query(self, query: str) -> Dict:
        """Analyze query using Magistral reasoning and create execution plan"""
        analysis_prompt = f"""
        Analyze this HubSpot query and create a detailed execution plan based on different hubspot properties provided by following the shared rules:

        HUBSPOT_PROPERTIES: {self.hubspot_properties}

        QUERY: {query}

        RULES:
        1. What is the user asking for?
        2. Is this a read-only query or does it require HubSpot updates?
        3. What sub-agents are needed to accomplish this?
        4. What HubSpot data is required?
        5. What's the execution sequence?
        6. What should be the final output format?
        7. Query can also be combination of read-only and write-back.
        8. Query is read-only if it requires data read from HubSpot.
        9. Query is write-back if it requires an update to existing values or writing/ assigning new values.
        10. In the final conclusion just give only one JSON string nothing else. I don't need any explanation.

        Provide a JSON execution plan with:
        {{
            "sub_agents": [
                {{
                    "name": "agent_name",
                    "task": "specific task description",
                    "task_type": "read_only" or "write_back",
                    "input_data": ["deals", "contacts", "companies"],
                    "output_format": "expected output"
                }}
            ]
        }}
        """

        # Use existing magistral_reasoning function
        analysis = magistral_reasoning(analysis_prompt)

        try:
            # Extract JSON execution plan from conclusion
            json_match = re.search(r'\{.*\}', analysis["conclusion"], re.DOTALL)
            if json_match:
                execution_plan = json.loads(json_match.group(0))
            else:
                raise ValueError("No JSON found in analysis")
        except Exception as e:
            execution_plan = {
                "sub_agents": [{
                    "name": "general_analyzer",
                    "task": query,
                    "task_type": "read_only",
                    "input_data": ["deals", "contacts", "companies"],
                    "output_format": "summary"
                }]
            }

        return {
            "reasoning": analysis["reasoning"],
            "execution_plan": execution_plan,
            "conclusion": analysis["conclusion"]
        }
```

### Step 4: Implement the Dynamic Sub-Agent

Sub-agents are created dynamically based on the execution plan. They execute specific tasks, which can be read-only analysis or write-back operations to HubSpot.

```python
class SubAgent:
    """Dynamic Sub-Agent created on-the-fly for specific tasks"""

    def __init__(self, name: str, task: str, task_type: str, input_data: List[str],
                 output_format: str):
        self.name = name
        self.task = task
        self.task_type = task_type
        self.input_data = input_data
        self.output_format = output_format

    def execute(self, data: Dict, properties_context: str, hubspot_updater=None) -> Dict:
        """Execute the assigned task"""
        if self.task_type == 'read_only':
            agent_prompt = f"""
            You are a {self.name} agent.

            TASK: {self.task}

            AVAILABLE HUBSPOT PROPERTIES:
            {properties_context}

            DATA AVAILABLE:
            {json.dumps(data, indent=2)}

            OUTPUT FORMAT: {self.output_format}

            Provide your analysis only based on the available data.
            """
        else:  # write_back
            agent_prompt = f"""
            You are a {self.name} agent.

            TASK: {self.task}

            AVAILABLE HUBSPOT PROPERTIES:
            {properties_context}

            DATA AVAILABLE:
            {json.dumps(data, indent=2)}

            CRITICAL: Use exact HubSpot property names from the list above in your JSON output.

            OUTPUT FORMAT: JSON format with the properties to be written to HubSpot

            Provide updates using exact HubSpot property names.
            """

        # Use existing mistral_small_execution function
        result = mistral_small_execution(agent_prompt)

        # Handle write-back operations
        if self.task_type == 'write_back' and hubspot_updater:
            try:
                json_match = re.search(r'\{.*\}', result, re.DOTALL)
                if json_match:
                    updates = json.loads(json_match.group(0))
                    hubspot_updater.batch_update(updates)
            except Exception as e:
                return {"status": "error", "error": str(e), "raw_result": result}

        return {"status": "success", "result": result}
```

### Step 5: Implement the Synthesis Agent

The Synthesis Agent combines the results from all sub-agents into a single, coherent, and user-friendly response.

```python
class SynthesisAgent:
    """Final agent to synthesize all results into user-friendly response"""

    def __init__(self):
        self.name = "SynthesisAgent"

    def synthesize(self, query: str, sub_agent_results: List[Dict], execution_plan: Dict) -> str:
        """Combine all sub-agent results into final answer"""
        # Prepare context from all sub-agent results
        results_context = ""
        for result in sub_agent_results:
            results_context += f"\n{result['agent'].upper()} ({result['task_type']}):\n"
            if result['result']['status'] == 'success':
                results_context += f"{result['result']['result']}\n"
            else:
                results_context += f"Error: {result['result'].get('error', 'Unknown error')}\n"
            results_context += "---\n"

        synthesis_prompt = f"""
        You are a final synthesizer agent. Create a comprehensive, user-friendly response based on all sub-agent results.

        ORIGINAL QUERY: {query}

        SUB-AGENT RESULTS:
        {results_context}

        TASK: Synthesize all the above results into a clear, actionable response for the user.

        Guidelines:
        1. Start with a direct answer to the user's query
        2. Include key insights and findings
        3. If updates were made, summarize what was changed
        4. Provide actionable next steps if relevant
        5. Keep it concise but comprehensive
        6. Use a professional but friendly tone

        Provide the final synthesized response:
        """

        # Use existing mistral_small_execution function
        final_answer = mistral_small_execution(synthesis_prompt)
        return final_answer
```

### Step 6: Create the Agent Orchestrator

The AgentOrchestrator is the main controller. It initializes all components, manages the workflow, and processes user queries through the multi-agent system.

```python
class AgentOrchestrator:
    """Main orchestrator that coordinates all agents"""

    def __init__(self, hubspot_api_key: str, mistral_api_key: str):
        # Initialize global mistral client for existing functions
        global mistral_client
        mistral_client = Mistral(api_key=mistral_api_key)

        # Initialize HubSpot connector
        self.hubspot_connector = HubSpotConnector(hubspot_api_key)

        # Load HubSpot data and properties
        self.hubspot_properties = self.hubspot_connector.get_properties()
        self.hubspot_data = {
            "deals": self.hubspot_connector.get_data("deals"),
            "contacts": self.hubspot_connector.get_data("contacts"),
            "companies": self.hubspot_connector.get_data("companies")
        }

        # Initialize agents
        self.lead_agent = LeadAgent(self.hubspot_properties)
        self.synthesis_agent = SynthesisAgent()
        self.active_sub_agents = []

    def process_query(self, query: str) -> Dict:
        """Main method to process user queries through multi-agent workflow"""
        # Step 1: Analyze the query with the Lead Agent
        analysis_result = self.lead_agent.analyze_query(query)
        execution_plan = analysis_result["execution_plan"]

        sub_agent_results = []

        # Step 2: Execute each sub-agent defined in the plan
        for agent_spec in execution_plan["sub_agents"]:
            # Create the sub-agent dynamically
            sub_agent = SubAgent(
                name=agent_spec["name"],
                task=agent_spec["task"],
                task_type=agent_spec["task_type"],
                input_data=agent_spec["input_data"],
                output_format=agent_spec["output_format"]
            )

            # Prepare the data this agent needs
            agent_data = {}
            for data_type in agent_spec["input_data"]:
                agent_data[data_type] = self.hubspot_data.get(data_type, [])

            # Prepare the properties context
            properties_context = ""
            for obj_type in agent_spec["input_data"]:
                properties_context += f"\n{obj_type.upper()} PROPERTIES:\n"
                properties_context += "\n".join(self.hubspot_properties.get(obj_type, []))
                properties_context += "\n"

            # Execute the sub-agent's task
            result = sub_agent.execute(
                data=agent_data,
                properties_context=properties_context,
                hubspot_updater=self.hubspot_connector if agent_spec["task_type"] == "write_back" else None
            )

            sub_agent_results.append({
                "agent": agent_spec["name"],
                "task_type": agent_spec["task_type"],
                "result": result
            })

        # Step 3: Synthesize all results
        final_answer = self.synthesis_agent.synthesize(query, sub_agent_results, execution_plan)

        return {
            "query": query,
            "analysis": analysis_result,
            "sub_agent_results": sub_agent_results,
            "final_answer": final_answer
        }
```

## Running the System

### Step 7: Initialize and Execute a Query

Now you can instantiate the orchestrator and process a business query.

```python
# Initialize the orchestrator with your API keys
orchestrator = AgentOrchestrator(
    hubspot_api_key=HUBSPOT_API_KEY,
    mistral_api_key=MISTRAL_API_KEY
)

# Process a sample query
query = "Assign priorities to all deals based on deal value"
result = orchestrator.process_query(query)

# Print the final, synthesized answer
print("Final Answer:")
print(result["final_answer"])
```

## Conclusion

You have successfully built a dynamic multi-agent system that integrates advanced AI reasoning with HubSpot CRM. This system can:
1. **Understand** complex business queries using the Magistral model.
2. **Plan** multi-step workflows by dynamically creating specialized sub-agents.
3. **Execute** both data analysis and CRM update operations.
4. **Synthesize** results into actionable, user-friendly insights.

This architecture provides a powerful foundation for automating complex CRM tasks and generating strategic business intelligence directly from natural language queries.