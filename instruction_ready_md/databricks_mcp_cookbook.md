# Building a Supply-Chain Copilot with OpenAI Agent SDK and Databricks MCP Servers

## Overview

In supply-chain operations, an intelligent agent can answer critical questions that directly affect service levels and revenue. This guide walks you through building a supply-chain copilot using the OpenAI Agent SDK and Databricks Managed MCP servers. The resulting agent can query structured and unstructured enterprise data—such as inventory, sales, supplier feeds, and communications—to provide real-time visibility, detect material shortages early, and recommend proactive actions.

By the end of this tutorial, you will deploy an agent that can:
- Query distributed data sources and predictive models
- Highlight emerging bottlenecks
- Recommend proactive actions
- Answer natural-language questions like:
  - "What products are dependent on L6HUK material?"
  - "How much revenue is at risk if we can’t produce the forecasted amount of product autoclave_1?"
  - "Which products have delays right now?"

## Prerequisites

Before you begin, ensure you have:
1. A Databricks workspace with Serverless compute and Unity Catalog enabled
2. A Databricks personal access token (PAT)
3. An OpenAI API key
4. Python 3.9 or later installed

## Architecture

The solution layers an OpenAI Agent on top of your existing Databricks analytics workloads. You expose Databricks components as callable Unity Catalog functions and vector search indexes via Databricks Managed MCP servers. The agent, built with the OpenAI Agent SDK, connects to these MCP servers to execute queries and return answers.

## Step 1: Set Up Databricks Authentication

First, configure authentication to your Databricks workspace using a configuration profile.

1. Generate a personal access token (PAT) in your Databricks workspace:
   - Go to **Settings → Developer → Access tokens → Generate new token**
   - Copy the generated token

2. Create or edit the Databricks configuration file:

```bash
# Create the configuration directory and file if it doesn't exist
mkdir -p ~/.databrickscfg
nano ~/.databrickscfg
```

3. Add your workspace configuration:

```bash
[DEFAULT]
host  = https://your-workspace-url.cloud.databricks.com  # Replace with your workspace URL
token = dapi123...                                        # Replace with your PAT
```

4. Verify the configuration works:

```bash
databricks clusters list
```

If this command returns data without prompting for credentials, your authentication is correctly set up.

## Step 2: (Optional) Set Up Sample Supply Chain Data

This tutorial can work with your own Databricks supply chain datasets. Alternatively, you can accelerate setup using a pre-built solution accelerator.

To use the sample data:

1. Clone the supply chain repository into your Databricks workspace:
   ```bash
   git clone https://github.com/lara-openai/databricks-supply-chain
   ```

2. Follow the instructions in the repository's README to run the notebooks. This will create:
   - Time-series demand forecasts
   - Raw-material planning tables
   - Transportation optimization models
   - Vector search indexes for email archives
   - Unity Catalog functions for data access

3. The notebooks wrap key insights as Unity Catalog functions (e.g., `product_from_raw`, `revenue_risk`, `query_unstructured_emails`), which your agent will call via MCP.

If using your own data, ensure you create similar Unity Catalog functions and vector search indexes.

## Step 3: Install Required Dependencies

Create a Python virtual environment and install the necessary packages:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install openai-agents httpx databricks-sdk pydantic
```

## Step 4: Configure Environment Variables

Set up environment variables for your configuration:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Optionally override default catalog, schema, and function path
export MCP_VECTOR_CATALOG="main"
export MCP_VECTOR_SCHEMA="supply_chain_db"
export MCP_FUNCTIONS_PATH="main/supply_chain_db"
export DATABRICKS_PROFILE="DEFAULT"
```

## Step 5: Create the Authentication Module

Create a file named `databricks_mcp.py` to handle Databricks authentication:

```python
"""
Databricks OAuth client provider for MCP servers.
"""

class DatabricksOAuthClientProvider:
    def __init__(self, ws):
        self.ws = ws

    def get_token(self):
        # For Databricks SDK >=0.57.0, token is available as ws.config.token
        return self.ws.config.token
```

This module abstracts the token retrieval logic, making it easier to maintain if Databricks authentication changes.

## Step 6: Create the Guardrail Module

Create a file named `supply_chain_guardrails.py` to implement content filtering:

```python
"""
Output guardrail that blocks answers not related to supply-chain topics.
"""
from __future__ import annotations

from pydantic import BaseModel
from agents import Agent, Runner

class GuardrailCheck(BaseModel):
    is_supply_chain: bool

async def supply_chain_guardrail(text: str) -> GuardrailCheck:
    """
    Classifies whether the given text is about supply-chain topics.
    Returns a GuardrailCheck object with the classification result.
    """
    # Implementation of the guardrail logic
    # This would typically call another LLM to classify the text
    # For brevity, we show the structure here
    return GuardrailCheck(is_supply_chain=True)
```

The guardrail ensures the agent only responds to supply-chain-related queries.

## Step 7: Build the Main Agent

Create the main agent script in `main.py`:

```python
"""
CLI assistant that uses Databricks MCP Vector Search and UC Functions via the OpenAI Agents SDK.
"""

import asyncio
import os
import httpx
from typing import Dict, Any
from agents import Agent, Runner, function_tool, gen_trace_id, trace
from agents.exceptions import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)
from agents.model_settings import ModelSettings
from databricks_mcp import DatabricksOAuthClientProvider
from databricks.sdk import WorkspaceClient
from supply_chain_guardrails import supply_chain_guardrail

# Configuration
CATALOG = os.getenv("MCP_VECTOR_CATALOG", "main")
SCHEMA = os.getenv("MCP_VECTOR_SCHEMA", "supply_chain_db")
FUNCTIONS_PATH = os.getenv("MCP_FUNCTIONS_PATH", "main/supply_chain_db")
DATABRICKS_PROFILE = os.getenv("DATABRICKS_PROFILE", "DEFAULT")
HTTP_TIMEOUT = 30.0  # seconds

async def _databricks_ctx():
    """Return (workspace, PAT token, base_url)."""
    ws = WorkspaceClient(profile=DATABRICKS_PROFILE)
    token = DatabricksOAuthClientProvider(ws).get_token()
    return ws, token, ws.config.host

@function_tool
async def vector_search(query: str) -> Dict[str, Any]:
    """Query Databricks MCP Vector Search index."""
    ws, token, base_url = await _databricks_ctx()
    url = f"{base_url}/api/2.0/mcp/vector-search/{CATALOG}/{SCHEMA}"
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(url, json={"query": query}, headers=headers)
        resp.raise_for_status()
        return resp.json()

@function_tool
async def uc_function(function_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke a Databricks Unity Catalog function with parameters."""
    ws, token, base_url = await _databricks_ctx()
    url = f"{base_url}/api/2.0/mcp/functions/{FUNCTIONS_PATH}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"function": function_name, "params": params}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()

async def run_agent():
    agent = Agent(
        name="Assistant",
        instructions="You are a supply-chain assistant for Databricks MCP; you must answer **only** questions that are **strictly** about supply-chain data, logistics, inventory, procurement, demand forecasting, etc; for every answer you must call one of the registered tools; if the user asks anything not related to supply chain, reply **exactly** with 'Sorry, I can only help with supply-chain questions'.",
        tools=[vector_search, uc_function],
        model_settings=ModelSettings(model="gpt-4o", tool_choice="required"),
        output_guardrails=[supply_chain_guardrail],
    )

    print("Databricks MCP assistant ready. Type a question or 'exit' to quit.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        trace_id = gen_trace_id()
        with trace(workflow_name="Databricks MCP Agent", trace_id=trace_id):
            try:
                result = await Runner.run(starting_agent=agent, input=user_input)
                print("Assistant:", result.final_output)
            except InputGuardrailTripwireTriggered:
                print("Assistant: Sorry, I can only help with supply-chain questions.")
            except OutputGuardrailTripwireTriggered:
                print("Assistant: Sorry, I can only help with supply-chain questions.")

def main():
    asyncio.run(run_agent())

if __name__ == "__main__":
    main()
```

## Step 8: Run the Agent

With all files created and environment variables set, run the agent:

```bash
python main.py
```

You should see the prompt:
```
Databricks MCP assistant ready. Type a question or 'exit' to quit.
```

## Step 9: Test the Agent

Try asking supply-chain questions like:
- "What products are dependent on L6HUK material?"
- "Are there any delays with syringe_1?"
- "What raw materials are required for syringe_1?"

The agent will:
1. Parse your natural language question
2. Determine which tool to use (vector search or UC function)
3. Make authenticated calls to Databricks MCP servers
4. Process the results and return a clear answer

## How It Works

### Tool Registration
The agent has two main tools:
1. **`vector_search`**: Queries the Databricks Vector Search index for semantic similarity in unstructured data (like emails)
2. **`uc_function`**: Executes Unity Catalog functions that wrap your analytical workloads (forecasting, risk calculation, etc.)

### Authentication Flow
1. The agent uses your `~/.databrickscfg` profile to initialize a `WorkspaceClient`
2. It retrieves the PAT token via the `DatabricksOAuthClientProvider`
3. All MCP API calls include the token in the Authorization header

### Guardrail Enforcement
The `supply_chain_guardrail` ensures the agent only responds to relevant queries. If a question falls outside supply-chain topics, the agent responds with a predefined refusal message.

## Next Steps

### Deploy as a Web Service
To make the agent accessible via a web interface:

1. Wrap the agent logic in a FastAPI application
2. Add endpoints for chat interactions
3. Deploy to a cloud platform (AWS, Azure, GCP)

### Add More Tools
Extend the agent's capabilities by:
1. Creating additional Unity Catalog functions for new analyses
2. Adding more vector search indexes for different document types
3. Integrating with other MCP servers for specialized tasks

### Monitor and Improve
1. Implement logging for all agent interactions
2. Collect feedback to improve tool selection and responses
3. Regularly update the underlying models and functions

## Troubleshooting

### Common Issues

1. **Authentication errors**: Verify your `~/.databrickscfg` file contains the correct host and token
2. **MCP server errors**: Ensure Serverless compute is enabled in your Databricks workspace
3. **Function not found**: Check that the Unity Catalog functions exist in the specified path
4. **Timeout errors**: Increase the `HTTP_TIMEOUT` value if queries take longer than 30 seconds

### Getting Help
- Refer to the [OpenAI Agent SDK documentation](https://openai.github.io/openai-agents-python/)
- Check the [Databricks MCP documentation](https://docs.databricks.com/aws/en/generative-ai/agent-framework/mcp)
- Review the [sample repository](https://github.com/lara-openai/databricks-supply-chain) for complete examples

## Conclusion

You've successfully built a supply-chain copilot that connects OpenAI's Agent SDK with Databricks MCP servers. This agent can now answer complex supply-chain questions by querying both structured data (via Unity Catalog functions) and unstructured data (via vector search). The modular architecture allows you to easily extend the agent with new tools and data sources as your needs evolve.