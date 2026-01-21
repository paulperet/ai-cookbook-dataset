# Optimizing Function Calling with o3/o4-mini Models: A Developer's Guide

## Introduction

The o3 and o4-mini models represent a significant advancement in OpenAI's reasoning capabilities. These models are specifically trained to use tools natively within their chain of thought, enabling more sophisticated function calling behavior than previous generations. This guide provides best practices for maximizing function calling performance with these powerful reasoning models.

## Prerequisites

Before implementing these best practices, ensure you have the OpenAI Python client installed:

```bash
pip install openai
```

## Core Concepts

### Developer Prompts vs. System Prompts

In o-series models, any system message you provide is automatically converted to a developer message internally. For practical purposes, you can treat the developer prompt as analogous to the traditional system prompt.

### Function Descriptions

A function description is the explanatory text in the `description` field of each function object within the `tools` parameter. This text tells the model when and how to use the function.

```python
tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for provided coordinates in celsius.",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number"},
            "longitude": {"type": "number"}
        },
        "required": ["latitude", "longitude"],
        "additionalProperties": False
    },
    "strict": True
}]
```

## Best Practices for Developer Prompts

### 1. Set Clear Context with Role Prompting

Establish the agent's role, tone, and available actions at the beginning of your developer prompt:

```python
developer_prompt = """
You are an AI retail agent.

As a retail agent, you can help users cancel or modify pending orders, return or exchange delivered orders, modify their default user address, or provide information about their own profile, orders, and related products.
"""
```

### 2. Define Function Call Ordering

Explicitly outline the sequence of function calls for complex tasks to prevent ordering mistakes:

```python
developer_prompt = """
To process a refund for a delivered order, follow these steps:
1. Confirm the order was delivered. Use: `order_status_check`
2. Check the refund eligibility policy. Use: `refund_policy_check`
3. Create the refund request. Use: `refund_create`
4. Notify the user of refund status. Use: `user_notify`
"""
```

### 3. Establish Tool Usage Boundaries

Clarify when to use tools and when not to:

```python
developer_prompt = """
Be proactive in using tools to accomplish the user's goal. If a task cannot be completed with a single step, keep going and use multiple tools as needed until the task is completed.

- Use tools when:
  - The user wants to cancel or modify an order.
  - The user wants to return or exchange a delivered product.
  - The user wants to update their address or contact details.
  - The user asks for current or personalized order or profile info.

- Do not use tools when:
  - The user asks a general question like "What's your return policy?"
  - The user asks something outside your retail role (e.g., "Write a poem").

If a task is not possible due to real constraints, explain why clearly and do not call tools blindly.
"""
```

## Optimizing Function Descriptions

### 1. Include Usage Criteria

Add specific criteria for when a function should be invoked:

```python
function_description = """
Creates a new file with the specified name and contents in a target directory. This function should be used when persistent storage is needed and the file does not already exist.
- Only call this function if the target directory exists. Check first using the `directory_check` tool.
- Do not use for temporary or one-off content—prefer direct responses for those cases.
- Do not overwrite existing files. Always ensure the file name is unique.
"""
```

### 2. Use Few-Shot Examples for Complex Arguments

Provide examples for functions with complex argument requirements:

```python
function_description = """
Use this tool to run fast, exact regex searches over text files using the `ripgrep` engine.

- Always escape special regex characters: ( ) [ ] { } + * ? ^ $ | . \\
- Use `\\` to escape any of these characters when they appear in your search string.
- Do NOT perform fuzzy or semantic matches.
- Return only a valid regex pattern string.

Examples:
Literal            -> Regex Pattern         
function(          -> function\\(           
value[index]       -> value\\[index\\]      
file.txt           -> file\\.txt            
user|admin         -> user\\|admin          
path\to\file       -> path\\\\to\\\\file     
"""
```

### 3. Place Key Rules First

Put the most important instructions at the beginning of your function descriptions. Our testing shows this approach can improve tool calling accuracy by up to 6%.

## Preventing Common Issues

### Guarding Against Hallucinations

Add explicit instructions to minimize hallucinations:

```python
developer_prompt = """
Do NOT promise to call a function later. If a function call is required, emit it now; otherwise respond normally.
Validate arguments against the format before sending the call; if you are unsure, ask for clarification instead of guessing.
"""
```

### Enable Strict Mode

Always set `strict: true` in your function definitions to ensure reliable schema adherence:

```python
tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for provided coordinates in celsius.",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number"},
            "longitude": {"type": "number"}
        },
        "required": ["latitude", "longitude"],
        "additionalProperties": False
    },
    "strict": True  # Always enable strict mode
}]
```

### Addressing Lazy Behavior

If you encounter lazy responses, try these strategies:

1. **Start new conversations** for unrelated topics to maintain focus
2. **Summarize long histories** instead of passing all previous tool calls
3. **Provide explicit detail requests** when you need comprehensive answers

## Using the Responses API for Maximum Performance

The Responses API allows you to persist reasoning items between tool calls, which significantly improves performance for o3/o4-mini models.

### Step 1: Set Up Your Environment

```python
from openai import OpenAI
import requests
import json

client = OpenAI()
```

### Step 2: Define Your Tools

```python
def get_weather(latitude, longitude):
    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = response.json()
    return data['current']['temperature_2m']

tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for provided coordinates in celsius.",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number"},
            "longitude": {"type": "number"}
        },
        "required": ["latitude", "longitude"],
        "additionalProperties": False
    },
    "strict": True
}]
```

### Step 3: Make the Initial Request

```python
context = [{"role": "user", "content": "What's the weather like in Paris today?"}]

response = client.responses.create(
    model="o3",
    input=context,
    tools=tools,
    store=False,
    include=["reasoning.encrypted_content"]  # Encrypted chain of thought is passed back
)

context += response.output  # Add the response to the context
tool_call = response.output[1]
args = json.loads(tool_call.arguments)
```

### Step 4: Execute the Function and Continue

```python
result = get_weather(args["latitude"], args["longitude"])

context.append({                               
    "type": "function_call_output",
    "call_id": tool_call.call_id,
    "output": str(result)
})

response_2 = client.responses.create(
    model="o3",
    input=context,
    tools=tools,
    store=False,
    include=["reasoning.encrypted_content"]
)

print(response_2.output_text)
```

## Working with Hosted Tools

When mixing hosted tools with custom functions, provide clear decision boundaries:

```python
developer_prompt = """
You are a helpful research assistant with access to the following tools:
- python tool: for any computation involving math, statistics, or code execution
- calculator: for basic arithmetic or unit conversions when speed is preferred

Always use the python tool for anything involving logic, scripts, or multistep math. Use the calculator tool only for simple 1-step math problems.
"""
```

## Important Considerations

### Avoid Chain of Thought Prompting

Since o3/o4-mini are reasoning models, they don't need explicit prompts to plan between tool calls. Asking them to reason more may actually hurt performance.

### Function Quantity Guidelines

While there's no hard limit on the number of functions, practical guidance suggests keeping your toolset focused and well-organized for optimal performance.

## Conclusion

By following these best practices—setting clear context, optimizing function descriptions, using the Responses API, and establishing proper tool boundaries—you can maximize the function calling capabilities of o3 and o4-mini models. Remember that these models are designed to think through complex tool usage scenarios, so provide them with clear guidance and let their native reasoning capabilities shine.