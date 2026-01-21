# Tool Search: Alternate Approaches

This guide demonstrates two alternate approaches to tool discovery with Claude, focusing on techniques that keep requests small and preserve prompt caching, even with access to thousands of tools.

## Prerequisites

**Required Knowledge**
*   Python fundamentals (functions, dictionaries, basic data structures)
*   Basic understanding of Claude tool use (recommended: [Tool Use Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use))

**Required Tools**
*   Python 3.11 or higher
*   Anthropic API key ([get one here](https://docs.anthropic.com/claude/reference/getting-started-with-the-api))

## Setup

First, install the required packages and initialize the Anthropic client.

```bash
pip install anthropic python-dotenv
```

```python
import anthropic
import json
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-5-20250929"
client = anthropic.Anthropic()
```

## Step 1: Define Your Tool Library

Create a dictionary containing all available tools. In a production system, this could be loaded from a database or configuration file.

```python
TOOL_LIBRARY = {
    "get_weather": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
    "get_stock_price": {
        "name": "get_stock_price",
        "description": "Get current stock price for a ticker symbol",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker (e.g., AAPL)"},
            },
            "required": ["ticker"],
        },
    },
    "convert_currency": {
        "name": "convert_currency",
        "description": "Convert amount between currencies",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string"},
                "to_currency": {"type": "string"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    "calculate_tip": {
        "name": "calculate_tip",
        "description": "Calculate tip amount for a bill",
        "input_schema": {
            "type": "object",
            "properties": {
                "bill_amount": {"type": "number"},
                "tip_percent": {"type": "number", "default": 20},
            },
            "required": ["bill_amount"],
        },
    },
    "send_email": {
        "name": "send_email",
        "description": "Send an email to a recipient",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
}
```

## Step 2: Create the `describe_tool` Discovery Tool

Instead of using semantic search, you can provide Claude with a simple `describe_tool` tool. Claude calls this tool with a specific tool name to load that tool's full definition into its context.

The system prompt lists all available tool names, so Claude knows what's available without needing embeddings.

```python
DESCRIBE_TOOL = {
    "name": "describe_tool",
    "description": "Load a tool's full definition into context. Call this before using any tool for the first time.",
    "input_schema": {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Name of the tool to load",
            },
        },
        "required": ["tool_name"],
    },
}

# Build the system prompt with the tool catalog
tool_names = list(TOOL_LIBRARY.keys())
SYSTEM_PROMPT = f"""You are a helpful assistant with access to various tools.

Available tools: {', '.join(tool_names)}

Before using any tool, you must first call describe_tool with the tool name to load it."""
```

## Step 3: Implement Mock Tool Execution

For demonstration, create a simple function that returns mock responses.

```python
def execute_tool(name: str, inputs: dict) -> str:
    """Mock tool execution."""
    if name == "get_weather":
        return json.dumps({"city": inputs["city"], "temp": "72°F", "conditions": "Sunny"})
    elif name == "get_stock_price":
        return json.dumps({"ticker": inputs["ticker"], "price": 185.50, "change": "+1.2%"})
    elif name == "convert_currency":
        rate = 0.92 if inputs["to_currency"] == "EUR" else 1.0
        converted = inputs["amount"] * rate
        return json.dumps({"converted": round(converted, 2), "to": inputs["to_currency"]})
    elif name == "calculate_tip":
        tip = inputs["bill_amount"] * (inputs.get("tip_percent", 20) / 100)
        return json.dumps({"tip": round(tip, 2), "total": round(inputs["bill_amount"] + tip, 2)})
    elif name == "send_email":
        return json.dumps({"status": "sent", "to": inputs["to"]})
    return json.dumps({"error": f"Unknown tool: {name}"})
```

## Step 4: Build the Conversation Loop with Dynamic Loading

This is the core pattern. The conversation loop starts with only the `describe_tool` in the tools list. When Claude requests a tool via `describe_tool`, you add that tool to the active list with `defer_loading=True` and return a `tool_reference`.

```python
def run_conversation(user_message: str, max_turns: int = 10):
    """Run a conversation with dynamic tool loading."""
    print(f"\n{'='*60}")
    print(f"USER: {user_message}")
    print(f"{'='*60}\n")

    messages = [{"role": "user", "content": user_message}]
    
    # Start with ONLY describe_tool
    active_tools = [DESCRIBE_TOOL]
    loaded_tools = set()  # Track which tools we've added

    for turn in range(max_turns):
        print(f"--- Turn {turn + 1} (tools in request: {len(active_tools)}) ---")

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=active_tools,
            messages=messages,
            extra_headers={"anthropic-beta": "advanced-tool-use-2025-11-20"},
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    print(f"\nASSISTANT: {block.text}")
            break

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "text" and block.text:
                print(f"ASSISTANT: {block.text}")
            
            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

                if tool_name == "describe_tool":
                    requested_tool = tool_input["tool_name"]
                    print(f"🔍 describe_tool({requested_tool})")

                    if requested_tool in TOOL_LIBRARY:
                        # Add tool to active_tools with defer_loading=True
                        # This is critical for prompt caching!
                        if requested_tool not in loaded_tools:
                            tool_def = {**TOOL_LIBRARY[requested_tool], "defer_loading": True}
                            active_tools.append(tool_def)
                            loaded_tools.add(requested_tool)
                            print(f"   ✓ Added {requested_tool} to tools (defer_loading=True)")

                        # Return tool_reference so Claude can use it
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": [{"type": "tool_reference", "tool_name": requested_tool}],
                        })
                    else:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Tool '{requested_tool}' not found.",
                        })
                else:
                    # Execute the discovered tool
                    print(f"🔧 {tool_name}({json.dumps(tool_input)})")
                    result = execute_tool(tool_name, tool_input)
                    print(f"   → {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    print(f"\n{'='*60}\n")
```

## Step 5: Run Examples

### Example 1: Simple Weather Query

Run a conversation where Claude needs to discover and use a single tool.

```python
run_conversation("What's the weather in Tokyo?")
```

**Expected Flow:**
1.  Claude sees `get_weather` in the system prompt's tool list.
2.  Claude calls `describe_tool("get_weather")` to load it.
3.  Your code adds `get_weather` to `active_tools` with `defer_loading=True` and returns a `tool_reference`.
4.  Claude now uses the `get_weather` tool.

### Example 2: Multi-Tool Query

Run a conversation that requires multiple tools.

```python
run_conversation("Convert $100 to EUR, then calculate a 20% tip on a $85 dinner bill.")
```

**Expected Flow:**
Claude will sequentially call `describe_tool` for `convert_currency` and `calculate_tip`, load each one, and then use them.

## Key Concept: Why `defer_loading=True` Matters

Normally, when you add a tool to the `tools` list, its definition is loaded at the very beginning of Claude's context window. Adding new tools invalidates most of the prompt cache because the start of the context has changed.

**With `defer_loading=True`:**
*   The tool definition is **not** included at the beginning of Claude's context window.
*   It is loaded into context only when Claude encounters the `tool_reference` in the conversation.
*   This preserves the cache for your system prompt and initial tools, even as new tools are discovered.

**The Essential Pattern:**
```python
# Initial request - only describe_tool, system prompt is cached
tools = [DESCRIBE_TOOL]

# After Claude calls describe_tool("get_weather")
# Add with defer_loading to preserve cache
tools.append({**TOOL_LIBRARY["get_weather"], "defer_loading": True})

# Return tool_reference so Claude knows it's available
tool_result = [{"type": "tool_reference", "tool_name": "get_weather"}]
```

This technique is essential for applications with hundreds or thousands of tools, as it allows you to:
*   Keep the initial request size small.
*   Preserve prompt caching across tool discoveries.
*   Load only the tools Claude actually needs.

## Conclusion

The core insight is that **tools do not need to be in the `tools` list until Claude needs them**. By combining `defer_loading=True` with `tool_reference`, you can scale to thousands of tools while maintaining small request sizes and efficient prompt caching. Your application must provide a discovery mechanism, like the `describe_tool` approach shown here.

Other discovery patterns you can implement include:
*   **`list_tools`**: Returns tool names matching a category or keyword.
*   **Hierarchical discovery**: Browse tool categories before loading specific tools.
*   **Hybrid approaches**: Combine listing with semantic search for large catalogs.

The fundamental pattern remains:
1.  Return a `tool_reference` when a tool is discovered.
2.  Add the tool with `defer_loading=True` to preserve caching.
3.  Claude can then use the tool immediately.

For an embeddings-based search approach, see the companion guide: **Tool Search with Embeddings**.