# Building a One-Liner Research Agent

Research tasks consume hours of expert time: market analysts manually gathering competitive intelligence, legal teams tracking regulatory changes, engineers investigating bug reports across documentation. The core challenge isn't finding information but knowing what to search for next based on what you just discovered.

The Claude Agent SDK makes it possible to build agents that autonomously explore external systems without a predefined workflow. Unlike traditional workflow automations that follow fixed steps, research agents adapt their strategy based on what they find—following promising leads, synthesizing conflicting sources, and knowing when they have enough information to answer the question.

## By the end of this guide, you'll be able to:

- Build a research agent that autonomously searches and synthesizes information with a few lines of code

This foundation applies to any task where the information needed isn't available upfront: competitive analysis, technical troubleshooting, investment research, or literature reviews.

## Why Research Agents?

Research is an ideal agentic use case for two reasons:

1.  **Information isn't self-contained.** The input question alone doesn't contain the answer. The agent must interact with external systems (search engines, databases, APIs) to gather what it needs.
2.  **The path emerges during exploration.** You can't predetermine the workflow. Whether an agent should search for company financials or regulatory filings depends on what it discovers about the business model. The optimal strategy reveals itself through investigation.

In its simplest form, a research agent searches the web and synthesizes findings. Below, we'll build exactly that with the Claude Agent SDK's built-in web search tool in just a few lines of code.

**Note:** You can also view the full list of [Claude Code's built-in tools](https://docs.claude.com/en/docs/claude-code/settings#tools-available-to-claude).

## Prerequisites

Before following this guide, ensure you have:

**Required Knowledge**

*   Python fundamentals - comfortable with async/await, functions, and basic data structures
*   Basic understanding of agentic patterns - we recommend reading [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) first if you're new to agents

**Required Tools**

*   Python 3.11 or higher
*   Anthropic API key [(get one here)](https://console.anthropic.com)

**Recommended:**

*   Familiarity with the Claude Agent SDK concepts
*   Understanding of tool use patterns in LLMs

## Setup

First, install the required dependencies:

```bash
pip install -U claude-agent-sdk python-dotenv
```

Create a `.env` file in your project directory and add your Anthropic API key:

```bash
ANTHROPIC_API_KEY=your_key_here
```

Now, load your environment variables and configure the client in your Python script:

```python
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-5"
```

## Step 1: Build Your First Research Agent

Let's start with the simplest possible implementation: a research agent that can search the web and synthesize findings. With the Claude Agent SDK, this takes just a few lines of code.

The key is the `query()` function, which creates a stateless agent interaction. We'll provide Claude with a single tool, `WebSearch`, and let it autonomously decide when and how to use it based on our research question.

Create a new Python file and add the following code:

```python
from claude_agent_sdk import ClaudeAgentOptions, query

messages = []
async for msg in query(
    prompt="Research the latest trends in AI agents and give me a brief summary and relevant citation links.",
    options=ClaudeAgentOptions(model=MODEL, allowed_tools=["WebSearch"]),
):
    print(msg)
    messages.append(msg)
```

**What's happening here:**

1.  **`query()`** creates a single-turn, stateless agent interaction. Each call is independent—it has no conversation memory or context from previous queries. This is perfect for one-off research tasks.
2.  **`allowed_tools=["WebSearch"]`** gives Claude permission to search the web autonomously without asking for approval. This is critical for the agent's independent operation.
3.  The agent autonomously decides when to search, what queries to run, and how to synthesize the results into a final answer.

## Step 2: Visualize the Agent's Activity (Optional)

To better understand what the agent is doing, you can use visualization utilities. First, install the helper module if needed (these are often part of example repositories). Then, update your code:

```python
from utils.agent_visualizer import (
    display_agent_response,
    print_activity,
)
from claude_agent_sdk import ClaudeAgentOptions, query

messages = []
async for msg in query(
    prompt="Research the latest trends in AI agents and give me a brief summary and relevant citation links.",
    options=ClaudeAgentOptions(model=MODEL, allowed_tools=["WebSearch"]),
):
    print_activity(msg)  # Shows the agent's actions in real-time
    messages.append(msg)

# Render the final response as a styled HTML card
display_agent_response(messages)
```

When you run this code, `print_activity` will output a real-time log of the agent's actions, such as `[🤖 Using: WebSearch(), ..., 🤖 Thinking...]`. The `display_agent_response` function will then present the agent's final, synthesized answer in a clean, formatted card.

**Example Output Summary:**

The agent will provide a structured summary. Here is a condensed example of what you might receive:

```
## Latest Trends in AI Agents (2025) - Summary

### 🚀 Market Growth & Adoption
The AI agent market is experiencing explosive growth, nearly doubling from $3.7 billion (2023) to $7.38 billion (2025), with projections reaching $103.6 billion by 2032.

### 🔑 Key Trends
1.  **Rise of Multi-Agent Systems:** The "orchestra approach" where multiple specialized agents collaborate using frameworks like CrewAI and AutoGen.
2.  **From Assistants to Autonomous Decision-Makers:** Agents are evolving to complete multi-step tasks without constant human input.
3.  **Model Context Protocol (MCP):** Anthropic's open standard for connecting LLMs to external systems.

### 💼 Impact
-   66% of adopters report measurable productivity gains.
-   Early movers are cutting operational costs by up to 40%.

### Sources:
-   The State of AI in 2025 - McKinsey
-   PwC's AI Agent Survey
-   Gartner Hype Cycle Identifies Top AI Innovations in 2025
-   ... (and several more with links)
```

## Step 3: Inspect the Full Conversation Timeline

To see a detailed breakdown of every step the agent took—including each individual web search it performed—you can visualize the entire conversation:

```python
from utils.agent_visualizer import visualize_conversation

# Assuming `messages` was populated from the previous query
visualize_conversation(messages)
```

This will generate a timeline view showing:
*   **System Initialization**
*   **Tool Calls:** Each specific `WebSearch` query the agent executed (e.g., "AI agents trends 2025 latest developments", "autonomous AI agents enterprise adoption 2025").
*   **Assistant Response:** The final synthesized answer.

This visualization confirms that the agent performed multiple, adaptive searches to gather comprehensive information before formulating its response.

## Understanding the Architecture

### How Tool Permissions Work

The `allowed_tools` parameter is key to autonomous operation:
*   **Allowed tools:** Claude can use these freely (in our case, just `WebSearch`).
*   **Other tools:** Available but would require explicit user approval before use.
*   **Read-only tools:** Tools like `Read` are always allowed by default.
*   **Disallowed tools:** You can explicitly remove tools from Claude's context using the `disallowed_tools` parameter.

### When to Use Stateless Queries (`query()`)

The `query()` function is stateless. Use it for:
*   One-off research questions where context doesn't matter.
*   Parallel processing of independent research tasks.
*   Scenarios where you want a fresh context for each query.

**When not to use it:**
*   Multi-turn investigations that build on previous findings.
*   Iterative refinement of research based on initial results.
*   Complex analysis requiring sustained context across multiple exchanges. (For these, you would use a stateful `Agent` session).

## Conclusion

You've successfully built a fully autonomous research agent in just a few lines of code. This agent can:
1.  Accept a broad research question.
2.  Autonomously decide what to search for.
3.  Execute multiple web searches.
4.  Synthesize the findings into a coherent, well-cited summary.

This pattern forms the foundation for virtually any research task. You can extend it by adding more tools (like database connectors or specialized APIs) or by moving to a stateful `Agent` for more complex, multi-turn research workflows.