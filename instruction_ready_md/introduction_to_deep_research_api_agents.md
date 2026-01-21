# Deep Research Agents Cookbook

This guide demonstrates how to build advanced, agentic research workflows using the OpenAI Deep Research API and the OpenAI [Agents SDK](https://openai.github.io/openai-agents-python/). It builds upon the [fundamentals cookbook](https://cookbook.openai.com/examples/deep_research_api/introduction_to_deep_research_api). If you are new to the Deep Research API, we recommend reviewing that content first.

You will learn to orchestrate single and multi-agent pipelines, enrich user queries to improve output quality, stream research progress, integrate web search, and incorporate internal knowledge via MCP (Model Context Protocol).

**Use Case:** Deep Research Agents are ideal for tasks requiring planning, synthesis, tool use, or multi-step reasoning. They are not suited for simple fact lookups or short Q&A, where a standard `openai.responses` API would be more efficient.

## Prerequisites

*   An OpenAI API key set as the `OPENAI_API_KEY` environment variable.
*   The OpenAI Python SDK and Agents SDK.

## Setup and Configuration

### 1. Install Dependencies

Begin by installing the required Python packages.

```bash
pip install --upgrade "openai>=1.88" "openai-agents>=0.0.19"
```

### 2. Import Libraries and Configure Client

Import the necessary modules and configure the OpenAI client. The following setup includes an environment variable to disable tracing, which is useful for organizations operating under a Zero Data Retention (ZDR) policy. If ZDR is not a requirement, you may omit this to benefit from integrated tracing and debugging tools.

```python
import os
from agents import Agent, Runner, WebSearchTool, RunConfig, set_default_openai_client, HostedMCPTool
from typing import List, Dict, Optional
from pydantic import BaseModel
from openai import AsyncOpenAI

# Configure the AsyncOpenAI client with a long timeout for research tasks
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=600.0)
set_default_openai_client(client)

# Optional: Disable tracing for Zero Data Retention compliance
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
```

## Tutorial 1: Basic Single-Agent Research

This section introduces a basic research agent that uses the `o4-mini-deep-research-alpha` model. This model is faster than the full `o3-deep-research` model while maintaining strong research capabilities. The agent has native access to web search and streams its progress.

**Learning Objective:** Run a single-agent research task and stream its intermediate steps.

### Step 1: Define the Research Agent

Create an agent configured for deep empirical research with web search capabilities.

```python
research_agent = Agent(
    name="Research Agent",
    model="o4-mini-deep-research-2025-06-26",
    tools=[WebSearchTool()],
    instructions="You perform deep empirical research based on the user's question."
)
```

### Step 2: Create a Streaming Research Function

Write an asynchronous function that runs the agent and prints streaming events, such as agent switches and web search queries, for transparency.

```python
async def basic_research(query):
    print(f"Researching: {query}")
    result_stream = Runner.run_streamed(
        research_agent,
        query
    )

    async for ev in result_stream.stream_events():
        if ev.type == "agent_updated_stream_event":
            print(f"\n--- switched to agent: {ev.new_agent.name} ---")
            print(f"\n--- RESEARCHING ---")
        elif (
            ev.type == "raw_response_event"
            and hasattr(ev.data, "item")
            and hasattr(ev.data.item, "action")
        ):
            action = ev.data.item.action or {}
            if action.get("type") == "search":
                print(f"[Web search] query={action.get('query')!r}")

    # Streaming is complete; final_output is now available
    return result_stream.final_output
```

### Step 3: Execute the Research

Run the function with a research query and print the final result.

```python
result = await basic_research("Research the economic impact of semaglutide on global healthcare systems.")
print(result)
```

## Tutorial 2: Multi-Agent Research Pipeline

To improve research quality, we can implement a multi-agent architecture. This pipeline enriches the user's initial query with more context and structure before the final research step, leading to more comprehensive and relevant outputs.

The pipeline consists of four specialized agents:

1.  **Triage Agent:** Inspects the query and decides if clarification is needed.
2.  **Clarifier Agent:** Asks follow-up questions to gather missing context.
3.  **Instruction Builder Agent:** Transforms the enriched query into a detailed research brief.
4.  **Research Agent:** Performs the actual deep research using web search and an internal knowledge MCP server.

### Step 1: Define Agent Prompts

First, define the system prompts that guide the Clarifier and Instruction Builder agents.

```python
CLARIFYING_AGENT_PROMPT = """
If the user hasn't specifically asked for research (unlikely), ask them what research they would like you to do.

GUIDELINES:
1. **Be concise while gathering all necessary information** Ask 2–3 clarifying questions to gather more context for research.
2. **Maintain a Friendly and Non-Condescending Tone**
3. **Adhere to Safety Guidelines**
"""

RESEARCH_INSTRUCTION_AGENT_PROMPT = """
Based on the following guidelines, take the user's query and rewrite it into detailed research instructions. OUTPUT ONLY THE RESEARCH INSTRUCTIONS, NOTHING ELSE. Transfer to the research agent.

GUIDELINES:
1. **Maximize Specificity and Detail**
2. **Fill in Unstated But Necessary Dimensions as Open-Ended**
3. **Avoid Unwarranted Assumptions**
4. **Use the First Person**
5. **Tables:** Explicitly request tables when they would help organize information (e.g., product comparisons, project plans).
6. **Headers and Formatting:** Specify the expected output format (e.g., "Format as a report with appropriate headers").
7. **Language:** Instruct the model to respond in the user's language unless otherwise specified.
8. **Sources:** Prioritize primary sources (official websites, original papers) and internal knowledge.
"""
```

### Step 2: Define Structured Output for Clarifications

Use Pydantic to define a structured output model for the Clarifier Agent's questions.

```python
class Clarifications(BaseModel):
    questions: List[str]
```

### Step 3: Create the Agent Team

Instantiate the four agents, connecting them via `handoffs`. The Research Agent is equipped with both `WebSearchTool` and a `HostedMCPTool` for accessing internal files.

> **Note:** Replace `<url>` in the `HostedMCPTool` configuration with the URL of your MCP server. For guidance on building an MCP server for Deep Research, refer to [this resource](https://cookbook.openai.com/examples/deep_research_api/how_to_build_a_deep_research_mcp_server/readme).

```python
# Research Agent (uses the full o3-deep-research model)
research_agent = Agent(
    name="Research Agent",
    model="o3-deep-research-2025-06-26",
    instructions="Perform deep empirical research based on the user's instructions.",
    tools=[
        WebSearchTool(),
        HostedMCPTool(
            tool_config={
                "type": "mcp",
                "server_label": "file_search",
                "server_url": "https://<url>/sse", # Replace with your MCP server URL
                "require_approval": "never",
            }
        )
    ]
)

# Instruction Builder Agent
instruction_agent = Agent(
    name="Research Instruction Agent",
    model="gpt-4o-mini",
    instructions=RESEARCH_INSTRUCTION_AGENT_PROMPT,
    handoffs=[research_agent], # Hands off to the Research Agent
)

# Clarifier Agent
clarifying_agent = Agent(
    name="Clarifying Questions Agent",
    model="gpt-4o-mini",
    instructions=CLARIFYING_AGENT_PROMPT,
    output_type=Clarifications, # Uses the structured output model
    handoffs=[instruction_agent],
)

# Triage Agent
triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "Decide whether clarifications are required.\n"
        "• If yes → call transfer_to_clarifying_questions_agent\n"
        "• If no  → call transfer_to_research_instruction_agent\n"
        "Return exactly ONE function-call."
    ),
    handoffs=[clarifying_agent, instruction_agent], # Can hand off to either agent
)
```

### Step 4: Implement an Auto-Clarify Runner

Create a helper function that runs the pipeline. If the Clarifier Agent asks questions, this function can automatically provide mock answers to simulate a user response, allowing for fully automated runs.

```python
async def basic_research(
    query: str,
    mock_answers: Optional[Dict[str, str]] = None,
    verbose: bool = False,
):
    stream = Runner.run_streamed(
        triage_agent,
        query,
        run_config=RunConfig(tracing_disabled=True),
    )

    async for ev in stream.stream_events():
        # If the Clarifier Agent outputs questions, send mock answers
        if isinstance(getattr(ev, "item", None), Clarifications):
            reply = []
            for q in ev.item.questions:
                ans = (mock_answers or {}).get(q, "No preference.")
                reply.append(f"**{q}**\n{ans}")
            stream.send_user_message("\n\n".join(reply))
            continue
        if verbose:
            print(ev)

    return stream
```

### Step 5: Run the Multi-Agent Pipeline

Execute the pipeline with a research query. You can provide an empty `mock_answers` dictionary for a fully automated run, or populate it with specific answers.

```python
result = await basic_research(
    "Research the economic impact of semaglutide on global healthcare systems.",
    mock_answers={},   # Provide canned answers here if desired
)
```

## Tutorial 3: Analyzing the Agent Workflow

The Agents SDK provides detailed traces. The following helper function parses the interaction stream and prints a human-readable summary of the agent steps, handoffs, and tool calls.

```python
import json

def parse_agent_interaction_flow(stream):
    print("=== Agent Interaction Flow ===")
    count = 1

    for item in stream.new_items:
        agent_name = getattr(item.agent, "name", "Unknown Agent") if hasattr(item, "agent") else "Unknown Agent"

        if item.type == "handoff_call_item":
            func_name = getattr(item.raw_item, "name", "Unknown Function")
            print(f"{count}. [{agent_name}] → Handoff Call: {func_name}")
            count += 1

        elif item.type == "handoff_output_item":
            print(f"{count}. [{agent_name}] → Handoff Output")
            count += 1

        elif item.type == "tool_call_item":
            tool_name = getattr(item.raw_item, "name", "")
            if tool_name:
                args = getattr(item.raw_item, "arguments", None)
                args_str = ""
                if args:
                    try:
                        parsed_args = json.loads(args)
                        if parsed_args:
                            args_str = json.dumps(parsed_args)
                    except Exception:
                        if args.strip() and args.strip() != "{}":
                            args_str = args.strip()
                args_display = f" with args {args_str}" if args_str else ""
                print(f"{count}. [{agent_name}] → Tool Call: {tool_name}{args_display}")
                count += 1

        elif item.type == "message_output_item":
            print(f"{count}. [{agent_name}] → Message Output")
            count += 1

        elif item.type == "reasoning_item":
            print(f"{count}. [{agent_name}] → Reasoning step")
            count += 1

        else:
            # Optionally print other event types
            # print(f"{count}. [{agent_name}] → {item.type}")
            # count += 1
            pass

# Usage
parse_agent_interaction_flow(result)
```

## Tutorial 4: Extracting Citations

The final research output contains URL citations. This function extracts and prints them, showing the source title, URL, and the context in which it was cited.

```python
def print_final_output_citations(stream, preceding_chars=50):
    # Iterate over new_items in reverse to find the last message_output_item(s)
    for item in reversed(stream.new_items):
        if item.type == "message_output_item":
            for content in getattr(item.raw_item, 'content', []):
                if not hasattr(content, 'annotations') or not hasattr(content, 'text'):
                    continue
                text = content.text
                for ann in content.annotations:
                    if getattr(ann, 'type', None) == 'url_citation':
                        title = getattr(ann, 'title', '<no title>')
                        url = getattr(ann, 'url', '<no url>')
                        start = getattr(ann, 'start_index', None)
                        end = getattr(ann, 'end_index', None)

                        if start is not None and end is not None and isinstance(text, str):
                            pre_start = max(0, start - preceding_chars)
                            preceding_text = text[pre_start:start].replace('\n', ' ').strip()
                            excerpt = text[start:end].replace('\n', ' ').strip()
                            print("# --------")
                            print("# CITATION SAMPLE:")
                            print(f"#   Title:       {title}")
                            print(f"#   URL:         {url}")
                            print(f"#   Location:    chars {start}–{end}")
                            print(f"#   Preceding:   '{preceding_text}'")
                            print(f"#   Excerpt:     '{excerpt}'\n")
                        else:
                            # fallback if no indices available
                            print(f"- {title}: {url}")
            break

# Usage
print_final_output_citations(result)
```

### Step 5: View the Final Research Report

Finally, print the complete research artifact generated by the pipeline.

```python
print(result.final_output)
```

## Conclusion

This cookbook provides a foundation for building scalable, production-ready research workflows using OpenAI Deep Research Agents. You have learned to:

1.  Configure and run a basic single-agent research task with streaming.
2.  Architect a multi-agent pipeline (Triage, Clarifier, Instruction Builder, Research) to significantly enhance output quality.
3.  Integrate both web search and internal knowledge (via MCP) into the research process.
4.  Analyze the agent interaction flow for debugging and transparency.
5.  Extract and review citations from the final research report.

These modular, agentic patterns enable you to tackle complex, multi-step research tasks that require planning, synthesis, and tool use across various domains.

**Happy researching!**