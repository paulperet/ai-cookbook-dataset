# Managing Function Calls with Reasoning Models: A Practical Guide

OpenAI's reasoning models, such as `o3` and `o4-mini`, are trained to follow logical chains of thought, making them exceptionally well-suited for complex, multi-step tasks like coding, scientific reasoning, and agentic workflows. While using these models via the API is straightforward, there are important nuances to understand, especially when integrating custom function calls.

This guide walks you through the process of building a robust agent that can reason through a problem, decide when to use tools, and handle multiple sequential function calls.

## Prerequisites

Ensure you have the OpenAI Python library installed and your API key configured.

```bash
pip install openai
```

## 1. Initial Setup and Basic API Call

First, import the necessary libraries and set up your client with default parameters for a reasoning model.

```python
import json
from openai import OpenAI
from uuid import uuid4
from typing import Callable

client = OpenAI()

MODEL_DEFAULTS = {
    "model": "o4-mini",  # 200,000 token context window
    "reasoning": {"effort": "low", "summary": "auto"},  # Automatically summarize reasoning
}
```

Now, let's make a simple call to the model using the Responses API. This API conveniently manages conversation state.

```python
# First question
response = client.responses.create(
    input="Which of the last four Olympic host cities has the highest average temperature?",
    **MODEL_DEFAULTS
)
print(response.output_text)

# Follow-up question using the previous response ID for context
response = client.responses.create(
    input="what about the lowest?",
    previous_response_id=response.id,
    **MODEL_DEFAULTS
)
print(response.output_text)
```

The model handles the multi-step reasoning internally. You can inspect the hidden reasoning tokens and usage details.

```python
# Find and print the reasoning summary from the response
for rx in response.output:
    if rx.type == 'reasoning':
        print(rx.summary[0].text)
        break

# Check token usage
print(response.usage.to_dict())
```

**Key Insight:** Reasoning models consume tokens for both the visible output and the internal reasoning process (`reasoning_tokens`). This means they use the context window faster than traditional chat models.

## 2. Integrating Custom Functions

Let's enhance our agent with custom tools. We'll create a simple function that returns a fake UUID for a given city and define the corresponding tool schema for the model.

```python
def get_city_uuid(city: str) -> str:
    """Return a fake internal ID (UUID) for a given city."""
    uuid = str(uuid4())
    return f"{city} ID: {uuid}"

# Define the tool schema for the model
tools = [
    {
        "type": "function",
        "name": "get_city_uuid",
        "description": "Retrieve the internal ID for a city from the internal database. Only invoke this function if the user needs to know the internal ID for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The name of the city to get information about"}
            },
            "required": ["city"]
        }
    }
]

# Map tool names to the actual Python functions
tool_mapping = {
    "get_city_uuid": get_city_uuid
}

# Add tools to our default parameters
MODEL_DEFAULTS["tools"] = tools
```

Now, ask a question that requires the model to use this tool.

```python
response = client.responses.create(
    input="What's the internal ID for the lowest-temperature city?",
    previous_response_id=response.id,  # Continue the previous conversation
    **MODEL_DEFAULTS
)

# The model may not return text immediately if it decides a tool call is needed
print(response.output_text)  # This might be empty
```

Inspect the response output. You'll likely see a `function_call` item, indicating the model has paused its reasoning to request tool execution.

```python
print(response.output)
```

## 3. Handling Function Call Responses

When the model requests a tool call, you must execute the corresponding function and send the result back so the model can continue reasoning. Function outputs must be sent as a special message type.

```python
# Extract function calls from the response
new_conversation_items = []
function_calls = [rx for rx in response.output if rx.type == 'function_call']

for function_call in function_calls:
    target_tool = tool_mapping.get(function_call.name)
    if not target_tool:
        raise ValueError(f"No tool found for function call: {function_call.name}")

    # Parse arguments and execute the tool
    arguments = json.loads(function_call.arguments)
    tool_output = target_tool(**arguments)

    # Structure the response for the API
    new_conversation_items.append({
        "type": "function_call_output",
        "call_id": function_call.call_id,  # Link output to the specific call
        "output": tool_output
    })

# Send the tool results back to the model
response = client.responses.create(
    input=new_conversation_items,
    previous_response_id=response.id,
    **MODEL_DEFAULTS
)
print(response.output_text)
```

## 4. Adding a Web Search Tool

Let's add a second, more complex tool. We'll create a custom web search function that uses a different model (`gpt-4o-mini`) with web search capabilities.

```python
def web_search(query: str) -> str:
    """Search the web for information and return a summary."""
    result = client.responses.create(
        model="gpt-4o-mini",
        input=f"Search the web for '{query}' and reply with only the result.",
        tools=[{"type": "web_search_preview"}],
    )
    return result.output_text

# Add the new tool to our schema and mapping
tools.append({
    "type": "function",
    "name": "web_search",
    "description": "Search the web for information and return back a summary of the results",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The query to search the web for."}
        },
        "required": ["query"]
    }
})
tool_mapping["web_search"] = web_search
```

## 5. Building a Robust Execution Loop

Reasoning models may require a sequence of tool calls, where some steps depend on the results of previous ones. We need a general pattern to handle this. First, let's create a helper function to execute all tools from a model response.

```python
def invoke_functions_from_response(response, tool_mapping: dict[str, Callable]) -> list[dict]:
    """
    Execute all function calls found in a model response.
    Returns a list of messages to send back to the model.
    """
    intermediate_messages = []
    for response_item in response.output:
        if response_item.type == 'function_call':
            target_tool = tool_mapping.get(response_item.name)
            if target_tool:
                try:
                    arguments = json.loads(response_item.arguments)
                    print(f"Invoking tool: {response_item.name}({arguments})")
                    tool_output = target_tool(**arguments)
                except Exception as e:
                    msg = f"Error executing function call: {response_item.name}: {e}"
                    tool_output = msg
                    print(msg)
            else:
                msg = f"ERROR - No tool registered for function call: {response_item.name}"
                tool_output = msg
                print(msg)

            intermediate_messages.append({
                "type": "function_call_output",
                "call_id": response_item.call_id,
                "output": tool_output
            })
        elif response_item.type == 'reasoning':
            # Optionally log reasoning steps
            print(f'Reasoning step: {response_item.summary}')
    return intermediate_messages
```

Now, we can create a loop that continues the conversation until the model finishes its reasoning and produces a final message.

```python
initial_question = (
    "What are the internal IDs for the cities that have hosted the Olympics in the last 20 years, "
    "and which of those cities have recent news stories (in 2025) about the Olympics? "
    "Use your internal tools to look up the IDs and the web search tool to find the news stories."
)

# Get the initial model response
response = client.responses.create(
    input=initial_question,
    **MODEL_DEFAULTS,
)

# Loop until reasoning is complete
while True:
    # Execute any required tools
    function_responses = invoke_functions_from_response(response, tool_mapping)

    if len(function_responses) == 0:
        # No more tool calls needed; reasoning is complete
        print(response.output_text)
        break
    else:
        # Send tool results back and get the next response
        print("More reasoning required, continuing...")
        response = client.responses.create(
            input=function_responses,
            previous_response_id=response.id,
            **MODEL_DEFAULTS
        )
```

This loop will handle complex, multi-step reasoning where the model interleaves thinking and tool use.

## 6. Manual Conversation Orchestration

For production use cases—where you might need to manage context window size, store messages externally, or allow conversation navigation—you can take full control by managing the conversation history yourself. This requires you to preserve all reasoning steps and function call responses in the history you send to the API.

Here's a simplified example that tracks token usage:

```python
# Initialize conversation history
conversation_history = []
total_tokens_used = 0

# User's first message
user_message_1 = (
    "Of those cities that have hosted the summer Olympic games in the last 20 years - "
    "do any of them have IDs beginning with a number and a temperate climate? "
    "Use your available tools to look up the IDs for each city and make sure to search the web to find out about the climate."
)
conversation_history.append({"role": "user", "content": user_message_1})

# Get the model's first response
response = client.responses.create(
    input=conversation_history,
    **MODEL_DEFAULTS
)
total_tokens_used += response.usage.total_tokens

# Add the model's full response (including reasoning and function calls) to history
# This is crucial for the model to maintain its chain of thought
conversation_history.extend(response.output)

# Now you would enter a loop similar to the one above, but you would:
# 1. Check for function calls in `response.output`
# 2. Execute them and append the `function_call_output` messages to `conversation_history`
# 3. Send the updated `conversation_history` for the next turn
# 4. Continue until a final message is produced

print(f"Total tokens used so far: {total_tokens_used}")
```

**Critical Note:** When managing history manually, you **must** include every `reasoning` and `function_call` item from previous responses in the input you send back. Omitting these will break the model's chain of thought and cause errors.

## Summary

You've now built a sophisticated agent using OpenAI's reasoning models that can:
1. Perform complex, multi-step reasoning.
2. Decide when to use custom tools.
3. Handle sequential function calls where later steps may depend on earlier results.
4. Operate in both automated and manually-orchestrated conversation modes.

Remember the key considerations:
- **Token Usage:** Reasoning models consume tokens for internal reasoning, impacting your context window budget.
- **State Management:** The Responses API can manage conversation state, but for advanced control you can manage history yourself—just be sure to include all reasoning and function call artifacts.
- **Tool Execution:** Always map tool calls to their implementations and structure the outputs correctly as `function_call_output` messages.

This pattern forms the foundation for building powerful, reliable AI agents capable of tackling intricate real-world tasks.