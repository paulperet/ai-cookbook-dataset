# HubSpot Dynamic Multi-Agent System with Magistral Reasoning

This cookbook demonstrates the power of Magistral reasoning model combined with HubSpot CRM integration to create an intelligent, multi-agent system that can understand complex business queries and execute sophisticated CRM operations automatically.
\
The system transforms natural language business questions into actionable insights and automated CRM updates, showcasing how advanced AI reasoning can streamline sales operations and strategic decision-making.

## Problem Statement

### Traditional CRM Challenges
Modern sales and marketing teams face several critical challenges when working with CRM systems like HubSpot:

**Manual Data Analysis:** Teams spend hours manually analyzing deals, contacts, and companies to extract insights
**Complex Query Processing:** Business stakeholders struggle to get answers to multi-faceted questions that require data from multiple CRM objects
**Strategic Planning:** Market analysis and expansion planning requires combining CRM data with business intelligence in ways that aren't natively supported

### Sample Query

"Assign priorities to all deals based on deal value"

These queries require:

- Understanding business context
- Analyzing multiple data sources
- Applying business logic
- Generating actionable recommendations
- Sometimes updating CRM records automatically

## Solution Architecture

**Core Innovation:** Magistral Reasoning + HubSpot Integration + Multi-Agent Orchestration

Our solution combines **Mistral's Magistral reasoning model** with **HubSpot's comprehensive CRM API** through a sophisticated multi-agent system that can:

- **Understand** complex business queries using Magistral's advanced reasoning capabilities
- **Plan** multi-step execution strategies with dynamically created specialized agents
- **Execute** both data analysis and CRM updates through coordinated agent workflows
- **Synthesize** results into actionable business insights with strategic recommendations

### AgentOrchestrator
**Master coordinator** that manages the entire multi-agent workflow and HubSpot integration. Orchestrates the complete flow from query analysis through sub-agent execution to final synthesis, while managing agent lifecycle and data connectivity.

### LeadAgent
Powered by **Magistral reasoning model** with `<think>` pattern processing, the Lead Agent performs sophisticated query analysis to understand business intent, determine data requirements, and create detailed execution plans specifying which sub-agents to create dynamically.

### Dynamic Sub-Agents
Sub-agents are **created on-the-fly** based on specific query requirements - not pre-defined templates. Each agent is dynamically generated with specialized roles (e.g., priority_calculator, market_analyzer, deals_updater), specific tasks, and targeted data access patterns using **Mistral Small** for fast execution.

### HubSpot API Connector
Dedicated connector providing comprehensive access to CRM data and operations:
- **Property Discovery:** Automatically maps all available HubSpot fields and valid values
- **Data Fetching:** Retrieves deals, contacts, and companies with full property sets
- **Batch Updates:** Efficiently updates multiple records in batches of 100

### SynthesisAgent
**Final orchestrator** that combines all sub-agent results into coherent, actionable business insights using **Mistral Small**. Transforms technical agent outputs into user-friendly responses with strategic recommendations and next steps.

#### Installation

We need `hubspot-api-client` and `mistralai` packages for the demonstration.

```python
!pip install hubspot-api-client=="12.0.0" mistralai=="1.9.3"
```

[Collecting hubspot-api-client, ..., Successfully installed eval-type-backport-0.2.2 hubspot-api-client-12.0.0 mistralai-1.9.3]

#### Imports

```python
import requests
import json
from mistralai import Mistral, ThinkChunk, TextChunk
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import re

```

#### Setup API Keys

```python
HUBSPOT_API_KEY = "<YOUR HUBSPOT API KEY>"  # Replace with your HubSpot API key
MISTRAL_API_KEY = "<YOUR MISTRAL API KEY>"  # Get it from https://console.mistral.ai/api-keys
```

#### Setup MistralAI Client

```python
mistral_client = Mistral(api_key=MISTRAL_API_KEY)
```

#### HubSpot API connector

- `get_data`: Fetches CRM data from HubSpot API, Retrieves deals, contacts, and companies data for the analysis.

- `batch_update`: Performs batch updates to HubSpot records, the updates and writes them back to HubSpot in efficient batches of 100 records.

- `get_properties`: Automatically fetches and formats all HubSpot deal, contact, and company properties, including valid values and dropdown options, so agents can update data reliably without errors.

```python
class HubSpotConnector:
    """Handles all HubSpot API operations"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.hubapi.com/crm/v3/objects"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def get_properties(self) -> Dict:
        """Load all HubSpot properties"""
        print("📡 HubSpotConnector: Loading properties...")
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

        print(f"✅ HubSpotConnector: Loaded properties for {len(properties)} object types")
        return properties

    def get_data(self, object_type: str) -> List[Dict]:
        """Fetch data from HubSpot"""
        print(f"📡 HubSpotConnector: Fetching {object_type} data...")

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

        print(f"✅ HubSpotConnector: Loaded {len(all_data)} {object_type}")
        return all_data

    def batch_update(self, updates: Dict) -> None:
        """Perform batch updates to HubSpot"""
        for object_type, update_list in updates.items():
            if not update_list:
                continue

            print(f"📡 HubSpotConnector: Updating {len(update_list)} {object_type}...")

            url = f"{self.base_url}/{object_type}/batch/update"
            headers = {**self.headers, "Content-Type": "application/json"}

            # Process in batches of 100
            for i in range(0, len(update_list), 100):
                batch = update_list[i:i+100]
                payload = {"inputs": batch}

                response = requests.post(url, headers=headers, json=payload)
                if response.status_code not in [200, 202]:
                    raise Exception(f"HubSpot update error: {response.text}")

            print(f"✅ HubSpotConnector: {object_type} updates completed")
```

#### Magistral (reasoning) and Mistral small LLM functions

- `magistral_reasoning`: Uses Magistral reasoning model for complex query analysis and execution planning with thinking process.

- `mistral_small_execution`: Uses Mistral Small model for sub-agent task execution.

```python
def magistral_reasoning(prompt: str) -> Dict[str, str]:
    """Use reasoning model for query analysis and planning"""
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
    """Use Mistral Small for content generation"""
    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

#### LeadAgent

**Powered by Magistral reasoning model** for sophisticated query analysis and execution planning

- `analyze_query`: Uses Magistral's `<think>` pattern to understand business intent, determine data requirements, and create detailed execution plans with dynamic sub-agent specifications
- Determines whether queries require read-only analysis or write-back operations to HubSpot

```python
class LeadAgent:
    """Lead Agent powered by Magistral reasoning model for query analysis and planning"""

    def __init__(self, hubspot_properties):
        self.hubspot_properties = hubspot_properties
        self.name = "LeadAgent"

    def analyze_query(self, query: str) -> Dict:
        """Analyze query using Magistral reasoning and create execution plan"""
        print(f"🧠 {self.name}: Analyzing query with Magistral reasoning...")

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
            print(f"⚠️ {self.name}: JSON parsing failed, using fallback plan")
            execution_plan = {
                "sub_agents": [{
                    "name": "general_analyzer",
                    "task": query,
                    "task_type": "read_only",
                    "input_data": ["deals", "contacts", "companies"],
                    "output_format": "summary"
                }]
            }

        print(f"✅ {self.name}: Plan created - {len(execution_plan['sub_agents'])} sub-agents needed")

        return {
            "reasoning": analysis["reasoning"],
            "execution_plan": execution_plan,
            "conclusion": analysis["conclusion"]
        }
```

#### SubAgent

**Dynamic agents created on-the-fly** based on query complexity and requirements

- `execute`: Uses Mistral Small for fast task execution including data analysis, business logic application, and CRM updates
- Specialized roles generated automatically (e.g., priority_calculator, market_analyzer, deals_updater)
- Handles both read-only operations and write-back operations with proper HubSpot property validation

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
        print(f"🤖 {self.name} ({self.task_type}): Executing task...")

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
                    print(f"✅ {self.name}: Successfully updated HubSpot records")
            except Exception as e:
                print(f"❌ {self.name}: Update failed - {str(e)}")
                return {"status": "error", "error": str(e), "raw_result": result}

        print(f"✅ {self.name}: Task completed successfully")
        return {"status": "success", "result": result}
```

#### SynthesisAgent

**Final orchestrator** that combines all sub-agent results into coherent business insights

- `synthesize`: Uses Mistral Small to create user-friendly responses with actionable recommendations and next steps
- Transforms technical agent outputs into executive-ready summaries and strategic guidance

```python
class SynthesisAgent:
    """Final agent to synthesize all results into user-friendly response"""

    def __init__(self):
        self.name = "SynthesisAgent"

    def synthesize(self, query: str, sub_agent_results: List[Dict], execution_plan: Dict) -> str:
        """Combine all sub-agent results into final answer"""
        print(f"🔄 {self.name}: Synthesizing results from {len(sub_agent_results)} agents...")

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
        print(f"✅ {self.name}: Final answer synthesized")

        return final_answer
```

#### AgentOrchestrator

**Master coordinator** that manages the entire multi-agent workflow and HubSpot integration

- `process_query`: Orchestrates the complete flow from query analysis through sub-agent execution to final synthesis
- Manages agent lifecycle, data flow between agents, and HubSpot connectivity
- Provides rich logging and monitoring of the multi-agent process

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

        print(f"🚀 AgentOrchestrator: System initialized with {sum(len(data) for data in self.hubspot_data.values())} HubSpot records")

    def process_query(self, query: str) -> Dict:
        """Main method to process user queries through multi-agent workflow"""
        print(f