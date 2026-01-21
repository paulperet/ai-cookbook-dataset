# Tool Search with Embeddings: Scaling Claude to Thousands of Tools

Building Claude applications with dozens of specialized tools quickly hits a wall: providing all tool definitions upfront consumes your context window, increases latency and costs, and makes it harder for Claude to find the right tool. Beyond ~100 tools, this approach becomes impractical.

Semantic tool search solves this by treating tools as discoverable resources. Instead of front-loading hundreds of definitions, you give Claude a single `tool_search` tool that returns relevant capabilities on demand, cutting context usage by 90%+ while enabling applications that scale to thousands of tools.

**By the end of this cookbook, you'll be able to:**
- Implement client-side tool search to scale Claude applications from dozens to thousands of tools
- Use semantic embeddings to dynamically discover relevant tools based on task context
- Apply this pattern to domain-specific tool libraries (APIs, databases, internal systems)

This pattern is used in production by teams managing large tool ecosystems where context efficiency is critical. While we'll demonstrate with a small set of tools for clarity, the same approach scales seamlessly to libraries with hundreds or thousands of tools.

## Prerequisites

Before following this guide, ensure you have:

**Required Knowledge**
- Python fundamentals - comfortable with functions, dictionaries, and basic data structures
- Basic understanding of Claude tool use - we recommend reading the [Tool Use Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) first

**Required Tools**
- Python 3.11 or higher
- Anthropic API key ([get one here](https://docs.anthropic.com/claude/reference/getting-started-with-the-api))

## Setup

First, install the required dependencies:

```bash
pip install anthropic sentence-transformers numpy python-dotenv
```

Ensure your `.env` file contains:
```
ANTHROPIC_API_KEY=your_key_here
```

Load your environment variables and configure the client:

```python
import json
import random
from datetime import datetime, timedelta
from typing import Any

import anthropic
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Define model constant for easy updates
MODEL = "claude-sonnet-4-5-20250929"

# Initialize Claude client (API key loaded from environment)
claude_client = anthropic.Anthropic()

# Load the SentenceTransformer model
# all-MiniLM-L6-v2 is a lightweight model with 384 dimensional embeddings
# It will be downloaded from HuggingFace on first use
print("Loading SentenceTransformer model...")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("✓ Clients initialized successfully")
```

## Step 1: Define Your Tool Library

Before we can implement semantic search, we need tools to search through. We'll create a library of 8 tools across two categories: Weather and Finance.

In production applications, you might manage hundreds or thousands of tools across your internal APIs, database operations, or third-party integrations. The semantic search approach scales to these larger libraries without modification - we're using a small set here purely for demonstration clarity.

```python
# Define our tool library with 2 domains
TOOL_LIBRARY = [
    # Weather Tools
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature",
                },
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_forecast",
        "description": "Get the weather forecast for multiple days ahead",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state",
                },
                "days": {
                    "type": "number",
                    "description": "Number of days to forecast (1-10)",
                },
            },
            "required": ["location", "days"],
        },
    },
    {
        "name": "get_timezone",
        "description": "Get the current timezone and time for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or timezone identifier",
                }
            },
            "required": ["location"],
        },
    },
    {
        "name": "get_air_quality",
        "description": "Get current air quality index and pollutant levels for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or coordinates",
                }
            },
            "required": ["location"],
        },
    },
    # Finance Tools
    {
        "name": "get_stock_price",
        "description": "Get the current stock price and market data for a given ticker symbol",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., AAPL, GOOGL)",
                },
                "include_history": {
                    "type": "boolean",
                    "description": "Include historical data",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another using current exchange rates",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Amount to convert",
                },
                "from_currency": {
                    "type": "string",
                    "description": "Source currency code (e.g., USD)",
                },
                "to_currency": {
                    "type": "string",
                    "description": "Target currency code (e.g., EUR)",
                },
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "calculate_compound_interest",
        "description": "Calculate compound interest for investments over time",
        "input_schema": {
            "type": "object",
            "properties": {
                "principal": {
                    "type": "number",
                    "description": "Initial investment amount",
                },
                "rate": {
                    "type": "number",
                    "description": "Annual interest rate (as percentage)",
                },
                "years": {"type": "number", "description": "Number of years"},
                "frequency": {
                    "type": "string",
                    "enum": ["daily", "monthly", "quarterly", "annually"],
                    "description": "Compounding frequency",
                },
            },
            "required": ["principal", "rate", "years"],
        },
    },
    {
        "name": "get_market_news",
        "description": "Get recent financial news and market updates for a specific company or sector",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Company name, ticker symbol, or sector",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of news articles to return",
                },
            },
            "required": ["query"],
        },
    },
]

print(f"✓ Defined {len(TOOL_LIBRARY)} tools in the library")
```

## Step 2: Create Tool Embeddings

Semantic search works by comparing the *meaning* of text, rather than just searching for keywords. To enable this, we need to convert each tool definition into an **embedding vector** that captures its semantic meaning.

Since our tool definitions are structured JSON objects with names, descriptions, and parameters, we first convert each tool into a human-readable text representation, then generate embedding vectors using SentenceTransformer's `all-MiniLM-L6-v2` model.

We picked this model because it is:
- **Lightweight and fast** (only 384 dimensions vs 768+ for larger models)
- **Runs locally** without requiring API calls
- **Sufficient for tool search** (you can experiment with larger models for better accuracy)

Let's start by creating a function that converts tool definitions into searchable text:

```python
def tool_to_text(tool: dict[str, Any]) -> str:
    """
    Convert a tool definition into a text representation for embedding.
    Combines the tool name, description, and parameter information.
    """
    text_parts = [
        f"Tool: {tool['name']}",
        f"Description: {tool['description']}",
    ]

    # Add parameter information
    if "input_schema" in tool and "properties" in tool["input_schema"]:
        params = tool["input_schema"]["properties"]
        param_descriptions = []
        for param_name, param_info in params.items():
            param_desc = param_info.get("description", "")
            param_type = param_info.get("type", "")
            param_descriptions.append(f"{param_name} ({param_type}): {param_desc}")

        if param_descriptions:
            text_parts.append("Parameters: " + ", ".join(param_descriptions))

    return "\n".join(text_parts)


# Test with one tool
sample_text = tool_to_text(TOOL_LIBRARY[0])
print("Sample tool text representation:")
print(sample_text)
```

Now let's create embeddings for all our tools:

```python
# Create embeddings for all tools
print("Creating embeddings for all tools...")

tool_texts = [tool_to_text(tool) for tool in TOOL_LIBRARY]

# Embed all tools at once using SentenceTransformer
# The model returns normalized embeddings by default
tool_embeddings = embedding_model.encode(tool_texts, convert_to_numpy=True)

print(f"✓ Created embeddings with shape: {tool_embeddings.shape}")
print(f"  - {tool_embeddings.shape[0]} tools")
print(f"  - {tool_embeddings.shape[1]} dimensions per embedding")
```

## Step 3: Implement Tool Search

With our tools embedded as vectors, we can now implement semantic search. If two pieces of text have similar meanings, their embedding vectors will be close together in vector space. We measure this "closeness" using **cosine similarity**.

The search process:
1. **Embed the query**: Convert Claude's natural language search request into the same vector space as our tools
2. **Calculate similarity**: Compute cosine similarity between the query vector and each tool vector
3. **Rank and return**: Sort tools by similarity score and return the top N matches

With semantic search, Claude can search using natural language like "I need to check the weather" or "calculate investment returns" rather than exact tool names.

Let's implement the search function and test it with a sample query:

```python
def search_tools(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Search for tools using semantic similarity.

    Args:
        query: Natural language description of what tool is needed
        top_k: Number of top tools to return

    Returns:
        List of tool definitions most relevant to the query
    """
    # Embed the query using SentenceTransformer
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)

    # Calculate cosine similarity using dot product
    # SentenceTransformer returns normalized embeddings, so dot product = cosine similarity
    similarities = np.dot(tool_embeddings, query_embedding)

    # Get top k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # Return the corresponding tools with their scores
    results = []
    for idx in top_indices:
        results.append({"tool": TOOL_LIBRARY[idx], "similarity_score": float(similarities[idx])})

    return results


# Test the search function
test_query = "I need to check the weather"
test_results = search_tools(test_query, top_k=3)

print(f"Search query: '{test_query}'\n")
print("Top 3 matching tools:")
for i, result in enumerate(test_results, 1):
    tool_name = result["tool"]["name"]
    score = result["similarity_score"]
    print(f"{i}. {tool_name} (similarity: {score:.3f})")
```

## Step 4: Define the tool_search Tool

Now we'll implement the **meta-tool** that allows Claude to discover other tools on demand. When Claude needs a capability it doesn't have, it searches for it using this `tool_search` tool, receives the tool definitions in the result, and can use those newly discovered tools immediately.

This is the only tool we provide to Claude initially:

```python
# The tool_search tool definition
TOOL_SEARCH_DEFINITION = {
    "name": "tool_search",
    "description": "Search for available tools that can help with a task. Returns tool definitions for matching tools. Use this when you need a tool but don't have it available yet.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language description of what kind of tool you need (e.g., 'weather information', 'currency conversion', 'stock prices')",
            },
            "top_k": {
                "type": "number",
                "description": "Number of tools to return (default: 5)",
            },
        },
        "required": ["query"],
    },
}

print("✓ Tool search definition created")
```

Now let's implement the handler that processes `tool_search` calls from Claude and returns discovered tools:

```python
def handle_tool_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Handle a tool_search invocation and return tool references.

    Returns a list of tool_reference content blocks for discovered tools.
    """
    # Search for relevant tools
    results = search_tools(query, top_k=top_k)

    # Create tool_reference objects instead of full definitions
    tool_references = [
        {"type": "tool_reference", "tool_name": result["tool"]["name"]} for result in results
    ]

    print(f"\n🔍 Tool search: '{query}'")
    print(f"   Found {len(tool_references)} tools:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['tool']['name']} (similarity: {result['similarity_score']:.3f})")

    return tool_references


# Test the handler
test_result = handle_tool_search("stock market data", top_k=3)
print(f"\nReturned {len(test_result)} tool references:")
for ref in test_result:
    print(f"  {ref}")
```

## Step 5: Mock Tool Execution

For this demonstration, we'll create mock responses for tool executions. In a real application, these would call actual APIs or services:

```python
def mock_tool_execution(tool_name: str, tool_input: dict[str, Any]) -> str:
    """
    Generate realistic mock responses for tool executions.

    Args:
        tool_name: Name of the tool being executed
        tool_input: Input parameters for the tool

    Returns:
        Mock response string appropriate for the tool
    """
    # Weather tools
    if tool_name == "get_weather":
        location = tool_input.get("location", "Unknown")
        unit = tool_input.get("unit", "fahrenheit")
        temp = random.randint(15, 30) if unit == "celsius" else random.randint(60, 85)
        conditions = random.choice(["sunny", "partly cloudy", "cloudy", "rainy"])
        return json.dumps(
            {
                "location": location,
                "temperature": temp,
                "unit": unit,
                "conditions": conditions,
                "humidity": random.randint(40, 80),
                "wind_speed": random.randint(5, 20),
            }
        )

    elif tool_name == "get_forecast":
        location = tool_input.get("location", "Unknown")
        days = int(tool_input.get("days", 5))
        forecast = []
        for i in range(days):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            forecast.append(
                {
                    "date": date,
                    "high": random.randint(65, 85),
                    "low": random.randint(45, 65),
                    "conditions": random.choice(["sunny", "partly cloudy", "rainy"]),
                }
            )
        return json.dumps({"location": location, "forecast": forecast})

    elif tool_name == "get_timezone":
        location = tool_input.get("location", "Unknown")
        return json.dumps(
            {
                "location": location,
                "timezone": random.choice(["EST", "PST", "CST", "GMT"]),
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    elif tool_name == "get_air_quality":
        location = tool_input.get("location", "Unknown")
        return json.dumps(
            {
                "location": location,
                "aqi": random.randint(0, 300),
                "pm2_5": random.uniform(0, 100),
                "pm10": random.uniform(0, 150),
                "status": random.choice(["Good", "Moderate", "Unhealthy"]),
            }
        )

    # Finance tools
    elif tool_name == "get_stock_price":
        ticker = tool_input.get("ticker", "UNKNOWN")
        return json.dumps(
            {
                "ticker": ticker,
                "price": round(random.uniform(100, 500), 2),
                "change": round(random.uniform(-10, 10), 2),
                "volume": random.randint(1000000, 50000000),
                "market_cap": random.randint(1000000000, 5000000000),
            }
        )

    elif tool_name == "convert_currency":
        amount = tool_input.get("amount", 100)
        from_curr = tool_input.get("from_currency", "USD")
        to_curr = tool_input.get("to_currency", "EUR")
        rate = round(random.uniform(0.8, 