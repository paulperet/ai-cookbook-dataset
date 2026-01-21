# Guide: Using the Wolfram Alpha LLM API as a Tool with Claude

This guide demonstrates how to integrate the Wolfram Alpha LLM API as a tool for Claude. By the end, Claude will be able to send queries to Wolfram Alpha and use the computed responses to answer complex user questions.

## Prerequisites

Before you begin, ensure you have:
1. An Anthropic API key for Claude.
2. A Wolfram Alpha App ID. You can sign up and create one for free at the [Wolfram Alpha Developer Portal](https://developer.wolframalpha.com/access).

## Step 1: Set Up Your Environment

First, install the required Python libraries and configure your API credentials.

```bash
pip install anthropic requests
```

Now, create a new Python script and import the necessary modules. Set your Wolfram Alpha App ID and choose your Claude model.

```python
import json
import urllib.parse
import requests
from anthropic import Anthropic

# Initialize the Anthropic client
client = Anthropic()

# Replace 'YOUR_APP_ID' with your actual Wolfram Alpha AppID
WOLFRAM_APP_ID = "YOUR_APP_ID"
MODEL_NAME = "claude-3-haiku-20240307"  # You can use another Claude model if preferred
```

## Step 2: Define the Wolfram Alpha Query Function

You'll create a function that sends a query to the Wolfram Alpha LLM API and returns the response.

```python
def wolfram_alpha_query(query):
    """
    Sends a query to the Wolfram Alpha LLM API and returns the computed result.
    
    Args:
        query (str): The natural language query to compute.
    
    Returns:
        str: The API response text or an error message.
    """
    # URL-encode the query
    encoded_query = urllib.parse.quote(query)

    # Construct the API URL
    url = (
        f"https://www.wolframalpha.com/api/v1/llm-api"
        f"?input={encoded_query}&appid={WOLFRAM_APP_ID}"
    )
    
    # Make the HTTP request
    response = requests.get(url, timeout=30)

    if response.status_code == 200:
        return response.text
    else:
        return f"Error: {response.status_code}: {response.text}"
```

This function handles the HTTP request and basic error checking. A successful request returns the computed answer as plain text.

## Step 3: Define the Tool for Claude

Claude uses a structured tool definition to understand when and how to call external APIs. Define the `wolfram_alpha` tool with a clear description and input schema.

```python
tools = [
    {
        "name": "wolfram_alpha",
        "description": "Query the Wolfram Alpha knowledge base. Use for mathematical calculations, scientific data, and factual questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "The natural language query to send to Wolfram Alpha.",
                }
            },
            "required": ["search_query"],
        },
    }
]
```

**Important:** The `input_schema` must match the parameter name your function expects. Here, the function `wolfram_alpha_query` takes a `query` argument, but the tool schema defines a `search_query` property. You'll need to handle this mapping in the next step.

## Step 4: Create the Interaction Logic

Now, build the main function that orchestrates the conversation between the user, Claude, and the Wolfram Alpha tool.

### 4.1 Define a Tool Call Handler

Create a function that routes tool calls from Claude to the correct function.

```python
def process_tool_call(tool_name, tool_input):
    """Executes the specified tool and returns its result."""
    if tool_name == "wolfram_alpha":
        # Map the schema's 'search_query' to the function's 'query' parameter
        return wolfram_alpha_query(tool_input["search_query"])
    else:
        return f"Error: Unknown tool '{tool_name}'"
```

### 4.2 Build the Chat Function

This is the core function that manages the multi-turn conversation with Claude.

```python
def chat_with_claude(user_message):
    """
    Sends a user message to Claude, handles tool use if needed, and returns the final answer.
    """
    print(f"\n{'=' * 50}")
    print(f"User Message: {user_message}")
    print(f"{'=' * 50}")

    # Construct the initial prompt for Claude
    prompt = f"""Here is a question: {user_message}. Please use the Wolfram Alpha tool to answer it. Do not reflect on the quality of the returned search results in your response."""

    # Send the first message to Claude
    message = client.beta.tools.messages.create(
        model=MODEL_NAME,
        max_tokens=4096,
        tools=tools,
        messages=[{"role": "user", "content": prompt}],
    )

    print("\nInitial Claude Response:")
    print(f"Stop Reason: {message.stop_reason}")

    # Check if Claude wants to use a tool
    if message.stop_reason == "tool_use":
        # Extract the tool use request from Claude's response
        tool_use = next(block for block in message.content if block.type == "tool_use")
        tool_name = tool_use.name
        tool_input = tool_use.input

        print(f"\nClaude requested to use tool: {tool_name}")
        print("Tool Input:")
        print(json.dumps(tool_input, indent=2))

        # Execute the tool
        tool_result = process_tool_call(tool_name, tool_input)

        print("\nTool Result from Wolfram Alpha:")
        print(tool_result)

        # Send the tool result back to Claude for a final answer
        response = client.beta.tools.messages.create(
            model=MODEL_NAME,
            max_tokens=2000,
            tools=tools,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": message.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": str(tool_result),
                        }
                    ],
                },
            ],
        )
    else:
        # If no tool was needed, use the initial response
        response = message

    # Extract the final text response from Claude
    final_response = None
    for block in response.content:
        if hasattr(block, "text"):
            final_response = block.text
            break

    print(f"\nFinal Answer: {final_response}")
    print(f"{'=' * 50}\n")

    return final_response
```

This function follows a standard tool-use pattern:
1.  Claude receives the user's question.
2.  If it decides a tool is needed, it returns a `tool_use` request.
3.  Your code executes the tool and gets the result.
4.  The result is sent back to Claude, which then synthesizes the final answer.

## Step 5: Test the Integration

Let's test the integration with a few example questions.

```python
if __name__ == "__main__":
    # Example 1: Factual world knowledge
    answer1 = chat_with_claude("What are the 5 largest countries in the world by population?")
    
    # Example 2: Mathematical calculation
    answer2 = chat_with_claude("Calculate the square root of 1764.")
    
    # Example 3: Scientific data
    answer3 = chat_with_claude("What is the distance between Earth and Mars?")
```

When you run this script, you should see output similar to the following for each question:

```
==================================================
User Message: Calculate the square root of 1764.
==================================================

Initial Claude Response:
Stop Reason: tool_use

Claude requested to use tool: wolfram_alpha
Tool Input:
{
  "search_query": "square root of 1764"
}

Tool Result from Wolfram Alpha:
The square root of 1764 is 42.

Final Answer: The square root of 1764 is 42.
==================================================
```

## Summary

You have successfully created an AI agent that combines Claude's reasoning with Wolfram Alpha's computational knowledge. The key steps were:
1.  Setting up API credentials.
2.  Creating a function to call the Wolfram Alpha API.
3.  Defining a tool schema so Claude knows when to use it.
4.  Building a conversation loop that handles tool execution and response synthesis.

This pattern can be extended to integrate other APIs and tools, enabling Claude to tackle an even wider range of tasks.