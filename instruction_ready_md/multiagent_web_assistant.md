# Build a Multi-Agent Web Browsing System

_Authored by: [Aymeric Roucher](https://huggingface.co/m-ric)_

> **Prerequisite:** This is an advanced tutorial. We recommend first understanding the basics from [the foundational agents cookbook](agents).

In this guide, you will build a **multi-agent web browser**—a hierarchical system where multiple AI agents collaborate to solve problems using web search and content retrieval.

## System Architecture

Our system uses a simple hierarchy with a manager agent overseeing a specialized web search agent:

```
              +----------------+
              | Manager agent  |
              +----------------+
                       |
        _______________|______________
       |                              |
  Code interpreter   +--------------------------------+
       tool          |         Managed agent          |
                     |      +------------------+      |
                     |      | Web Search agent |      |
                     |      +------------------+      |
                     |         |            |         |
                     |  Web Search tool     |         |
                     |             Visit webpage tool |
                     +--------------------------------+
```

## Setup and Installation

First, install the required dependencies.

```bash
pip install markdownify duckduckgo-search smolagents --upgrade -q
```

Next, authenticate with the Hugging Face Hub to use the Inference API.

```python
from huggingface_hub import notebook_login

notebook_login()
```

## Configure the Language Model

We'll power our agents using the **Qwen/Qwen2.5-72B-Instruct** model via the Hugging Face Inference API. This API provides easy access to a wide range of open-source models.

```python
model_id = "Qwen/Qwen2.5-72B-Instruct"
```

> **Note:** The Inference API hosts models based on various criteria, and deployed models may be updated or replaced without prior notice. Learn more [here](https://huggingface.co/docs/api-inference/supported-models).

## Create the Web Browsing Tools

Our web agent needs two core capabilities: searching the web and reading webpage content.

### 1. Build a Webpage Visitor Tool

While `smolagents` includes a built-in `VisitWebpageTool`, we'll create our own version to understand the implementation. This tool fetches a webpage and converts its HTML content to clean Markdown.

```python
import re
import requests
from markdownify import markdownify as md
from requests.exceptions import RequestException
from smolagents import tool


@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        markdown_content = md(response.text).strip()
        # Clean up excessive line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        return markdown_content
    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"
```

Test the tool to ensure it works correctly.

```python
print(visit_webpage("https://en.wikipedia.org/wiki/Hugging_Face")[:500])
```

## Assemble the Multi-Agent System

Now we'll create the agent hierarchy.

### 1. Create the Web Search Agent

This agent handles the actual web browsing. Since web browsing is a sequential task, we'll use a `ToolCallingAgent` (which uses JSON tool calling). We'll also increase the maximum iterations to allow for thorough page exploration.

```python
from smolagents import ToolCallingAgent, InferenceClientModel, DuckDuckGoSearchTool

model = InferenceClientModel(model_id)

web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), visit_webpage],
    model=model,
    max_iterations=10,
)
```

### 2. Wrap the Web Agent for Management

The `ManagedAgent` wrapper makes our web agent callable by the manager agent, turning it into a "tool" that the manager can use.

```python
from smolagents import ManagedAgent

managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="search_agent",
    description="Runs web searches for you. Give it your query as an argument.",
)
```

### 3. Create the Manager Agent

The manager is responsible for planning and high-level reasoning. A `CodeAgent` is ideal for this role as it can write and execute Python code. We'll also authorize datetime imports so it can perform date calculations.

```python
from smolagents import CodeAgent

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[managed_web_agent],
    additional_authorized_imports=["time", "datetime"],
)
```

## Run the System

Let's test our multi-agent system with a question that requires web search and date calculation.

```python
manager_agent.run("How many years ago was Stripe founded?")
```

The system will execute as follows:

1. **Manager Agent Plans:** The manager determines it needs to find Stripe's founding year and calculate the time elapsed.
2. **Manager Delegates:** It calls the `search_agent` (our managed web agent) with the query "When was Stripe founded?"
3. **Web Agent Searches:** The web agent performs a DuckDuckGo search, visits relevant pages, and extracts the founding year.
4. **Web Agent Reports Back:** It returns a detailed answer to the manager.
5. **Manager Calculates:** Using the founding year and current date, the manager calculates how many years ago Stripe was founded.
6. **Final Answer:** The manager provides the final result.

## Summary

You've successfully built a hierarchical multi-agent system for web browsing. The key components are:

- **Web Search Agent:** Specialized in searching and reading web content
- **ManagedAgent Wrapper:** Makes the web agent callable as a tool
- **Manager Agent:** Orchestrates the workflow and performs complex reasoning

This architecture demonstrates how to decompose complex tasks (web research with calculation) across specialized agents, enabling more sophisticated AI applications.

**Next Steps:** Experiment with different questions, add more specialized agents to the hierarchy, or integrate additional tools for even more capable systems.