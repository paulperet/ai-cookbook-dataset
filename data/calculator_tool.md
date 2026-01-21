# Cookbook: Using a Calculator Tool with Claude

This guide demonstrates how to extend Claude's capabilities by providing it with a calculator tool. You will define a simple arithmetic tool and implement the logic for Claude to call it, enabling it to solve mathematical problems accurately.

## Prerequisites

Ensure you have the Anthropic Python library installed.

```bash
pip install anthropic
```

## Step 1: Import Libraries and Initialize the Client

Begin by importing the necessary library and setting up the Anthropic client.

```python
from anthropic import Anthropic

# Initialize the client
client = Anthropic()
MODEL_NAME = "claude-3-opus-20240229"  # Or your preferred Claude model
```

## Step 2: Define the Calculator Function and Tool Schema

You will create a function that evaluates mathematical expressions. **Important:** The example uses Python's `eval()` for simplicity in a controlled environment. In a production application, you should use a safer alternative, such as `ast.literal_eval` or a dedicated math expression parser.

```python
import re

def calculate(expression):
    """
    Evaluates a basic mathematical expression.
    Args:
        expression (str): A string containing a mathematical expression (e.g., "2 + 3 * 4").
    Returns:
        str: The result as a string, or an error message.
    """
    # Sanitize the input: remove any characters that are not digits, operators, or parentheses.
    expression = re.sub(r"[^0-9+\-*/().]", "", expression)

    try:
        # WARNING: eval() is used here for demonstration only.
        # It can execute arbitrary code and is a security risk.
        result = eval(expression)  # noqa: S307
        return str(result)
    except (SyntaxError, ZeroDivisionError, NameError, TypeError, OverflowError):
        return "Error: Invalid expression"
```

Next, define the tool schema that Claude will use. This tells the model the tool's name, purpose, and expected input format.

```python
tools = [
    {
        "name": "calculator",
        "description": "A simple calculator that performs basic arithmetic operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '2 + 3 * 4').",
                }
            },
            "required": ["expression"],
        },
    }
]
```

## Step 3: Create the Interaction Logic

Now, build the main function that handles the conversation with Claude, manages tool calls, and returns the final answer.

```python
def process_tool_call(tool_name, tool_input):
    """Routes the tool call to the appropriate function."""
    if tool_name == "calculator":
        return calculate(tool_input["expression"])
    # Add other tools here as needed
    return "Error: Tool not found."

def chat_with_claude(user_message):
    """
    Sends a message to Claude, processes any tool use, and returns the final response.
    """
    print(f"\n{'=' * 50}\nUser Message: {user_message}\n{'=' * 50}")

    # 1. Send the initial user message to Claude
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4096,
        messages=[{"role": "user", "content": user_message}],
        tools=tools,
    )

    print("\nInitial Claude Response:")
    print(f"Stop Reason: {message.stop_reason}")
    # Extract and print the initial text content, if any
    initial_text = next((block.text for block in message.content if block.type == "text"), None)
    if initial_text:
        print(f"Content: {initial_text}")

    # 2. Check if Claude wants to use a tool
    if message.stop_reason == "tool_use":
        # Find the tool_use block in the response
        tool_use = next(block for block in message.content if block.type == "tool_use")
        tool_name = tool_use.name
        tool_input = tool_use.input

        print(f"\nTool Used: {tool_name}")
        print(f"Tool Input: {tool_input}")

        # 3. Execute the tool locally
        tool_result = process_tool_call(tool_name, tool_input)
        print(f"Tool Result: {tool_result}")

        # 4. Send the tool result back to Claude for a final answer
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": message.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": tool_result,
                        }
                    ],
                },
            ],
            tools=tools,
        )
    else:
        # If no tool was used, the first response is the final one.
        response = message

    # 5. Extract and return the final text response from Claude
    final_response = next(
        (block.text for block in response.content if block.type == "text"),
        None,
    )
    print(f"\nFinal Response: {final_response}")
    return final_response
```

## Step 4: Test the Calculator Tool

Let's verify the implementation by asking Claude to solve a few arithmetic problems.

```python
# Example 1: Large multiplication
result1 = chat_with_claude("What is the result of 1,984,135 * 9,343,116?")
# Output: Final Response: 1,984,135 multiplied by 9,343,116 equals 18,532,894,860,660.

# Example 2: Complex expression
result2 = chat_with_claude("Calculate (12851 - 593) * 301 + 76")
# Output: Final Response: (12851 - 593) * 301 + 76 = 3,688,334.

# Example 3: Division
result3 = chat_with_claude("What is 15910385 divided by 193053?")
# Output: Final Response: So 15910385 divided by 193053 equals approximately 82.4146.
```

## Summary

You have successfully created a tool-augmented Claude agent. The process follows these steps:
1.  **Define the Tool:** Create a Python function and its corresponding schema.
2.  **Initial Query:** Send a user message to Claude with the available tools.
3.  **Tool Execution:** Claude requests tool use; you run the function locally.
4.  **Final Answer:** You provide the tool's result back to Claude, which synthesizes the final response.

This pattern is the foundation for building more complex agents with multiple tools, such as database queries, API calls, or custom business logic. Remember to replace the `eval()` function with a secure expression evaluator for any production deployment.