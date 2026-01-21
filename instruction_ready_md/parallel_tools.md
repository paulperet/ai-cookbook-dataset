# Guide: Enabling Parallel Tool Calls with Claude 3.7 Sonnet Using a Batch Tool

Claude 3.7 Sonnet may sometimes avoid making parallel tool calls in a single response, even when parallel tool use is enabled. This guide demonstrates a workaround: introducing a "batch tool" that acts as a meta-tool, allowing Claude to wrap multiple tool invocations into one simultaneous call.

## Prerequisites

Ensure you have the Anthropic Python client installed.

```bash
pip install anthropic
```

## Step 1: Initial Setup

First, import the necessary library and define the model name.

```python
from anthropic import Anthropic

client = Anthropic()
MODEL_NAME = "claude-sonnet-4-5"
```

## Step 2: Define Basic Tools

Create two simple tools: one to get the weather and another to get the time for a given location.

```python
def get_weather(location):
    # Pretend to get the weather, and just return a fixed value.
    return f"The weather in {location} is 72 degrees and sunny."


def get_time(location):
    # Pretend to get the time, and just return a fixed value.
    return f"The time in {location} is 12:32 PM."


weather_tool = {
    "name": "get_weather",
    "description": "Gets the weather for in a given location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
        "required": ["location"],
    },
}

time_tool = {
    "name": "get_time",
    "description": "Gets the time in a given location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
        "required": ["location"],
    },
}


def process_tool_call(tool_name, tool_input):
    if tool_name == "get_weather":
        return get_weather(tool_input["location"])
    elif tool_name == "get_time":
        return get_time(tool_input["location"])
    else:
        raise ValueError(f"Unexpected tool name: {tool_name}")
```

## Step 3: Create a Helper Function for Queries

Define a function to send a query to Claude and print the response, including any tool calls.

```python
def make_query_and_print_result(messages, tools=None):
    response = client.messages.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=1000,
        tool_choice={"type": "auto"},
        tools=tools or [weather_tool, time_tool],
    )

    for block in response.content:
        match block.type:
            case "text":
                print(block.text)
            case "tool_use":
                print(f"Tool: {block.name}({block.input})")
            case _:
                raise ValueError(f"Unexpected block type: {block.type}")

    return response
```

## Step 4: Observe the Default Sequential Behavior

Now, let's see how Claude handles a query that requires both tools without the batch tool.

```python
MESSAGES = [{"role": "user", "content": "What's the weather and time in San Francisco?"}]

response = make_query_and_print_result(MESSAGES)
```

You'll notice that Claude only calls one tool (e.g., `get_weather`) in its first response, even though the query asks for both weather and time.

To get the second piece of information, you must provide the result of the first tool call and let Claude continue.

```python
last_tool_call = response.content[1]

MESSAGES.append({"role": "assistant", "content": response.content})
MESSAGES.append(
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": last_tool_call.id,
                "content": process_tool_call(response.content[1].name, response.content[1].input),
            }
        ],
    }
)

response = make_query_and_print_result(MESSAGES)
```

Now Claude makes the second tool call (e.g., `get_time`). This sequential approach works but introduces unnecessary latency due to multiple back-and-forth turns.

## Step 5: Implement the Batch Tool Workaround

To encourage parallel tool calls, define a `batch_tool` that can wrap multiple tool invocations.

```python
import json

batch_tool = {
    "name": "batch_tool",
    "description": "Invoke multiple other tool calls simultaneously",
    "input_schema": {
        "type": "object",
        "properties": {
            "invocations": {
                "type": "array",
                "description": "The tool calls to invoke",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the tool to invoke",
                        },
                        "arguments": {
                            "type": "string",
                            "description": "The arguments to the tool",
                        },
                    },
                    "required": ["name", "arguments"],
                },
            }
        },
        "required": ["invocations"],
    },
}


def process_tool_with_maybe_batch(tool_name, tool_input):
    if tool_name == "batch_tool":
        results = []
        for invocation in tool_input["invocations"]:
            results.append(
                process_tool_call(invocation["name"], json.loads(invocation["arguments"]))
            )
        return "\n".join(results)
    else:
        return process_tool_call(tool_name, tool_input)
```

**Note:** The original schema had a typo (`"types"` instead of `"type"`). The corrected version uses `"type"` for the `name` and `arguments` properties, as shown above.

## Step 6: Test with the Batch Tool

Now, provide Claude with all three tools (weather, time, and batch) and repeat the query.

```python
MESSAGES = [{"role": "user", "content": "What's the weather and time in San Francisco?"}]

response = make_query_and_print_result(MESSAGES, tools=[weather_tool, time_tool, batch_tool])
```

This time, Claude should use the `batch_tool` to request both the weather and time in a single tool call, demonstrating parallel invocation.

Finally, process the batch tool result and let Claude generate the final answer.

```python
last_tool_call = response.content[1]

MESSAGES.append({"role": "assistant", "content": response.content})
MESSAGES.append(
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": last_tool_call.id,
                "content": process_tool_with_maybe_batch(
                    response.content[1].name, response.content[1].input
                ),
            }
        ],
    }
)

response = make_query_and_print_result(MESSAGES)
```

Claude will now receive both results at once and can synthesize a complete answer in a single step, reducing overall latency.

## Summary

By introducing a batch tool, you provide Claude 3.7 Sonnet with a mechanism to bundle multiple tool calls into one. This workaround can improve efficiency when your application requires data from several tools simultaneously, minimizing the conversational turns needed to complete a task.