# Guide: Maximizing Performance with Reasoning Models and the Responses API

## Overview

This guide demonstrates how to use OpenAI's Responses API to unlock higher intelligence, lower costs, and more efficient token usage with reasoning models like `o3` and `o4-mini`. You'll learn how reasoning works, how to properly handle function calls, leverage caching, use encrypted reasoning items for stateless workflows, and access reasoning summaries.

## Prerequisites

Ensure you have the OpenAI Python library installed and your API key configured.

```bash
pip install openai requests
```

```python
import os
import json
import requests
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

## 1. Understanding Reasoning Models

Reasoning models like `o3` and `o4-mini` break down problems step-by-step, producing an internal chain of thought. For safety, these reasoning tokens are not directly exposed but are summarized. In a multi-turn conversation, reasoning tokens from previous turns are typically discarded unless you explicitly pass them back using the Responses API.

Let's examine a basic response object to understand its structure.

```python
response = client.responses.create(
    model="o4-mini",
    input="tell me a joke",
)

print(json.dumps(response.model_dump(), indent=2))
```

The response includes an `output` array containing items of type `reasoning` and `message`. The reasoning item has an ID (e.g., `rs_...`) representing the internal reasoning tokens. The `usage` field shows token counts, distinguishing between reasoning tokens and final output tokens.

## 2. Enhancing Function Calls with Reasoning Items

When your workflow involves tool/function calls, you must include reasoning items from previous steps to maximize model intelligence. Here's how to do it.

### Step 1: Define a Tool and Initial Context

First, define a simple weather function and set up the initial conversation context.

```python
def get_weather(latitude, longitude):
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
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

context = [{"role": "user", "content": "What's the weather like in Paris today?"}]
```

### Step 2: Make the First API Call

Request a response from the model, which will reason about the query and decide if a tool call is needed.

```python
response = client.responses.create(
    model="o4-mini",
    input=context,
    tools=tools,
)

print(response.output)
```

The output will contain a `reasoning` item and a `function_call` item. The model has determined it needs to call `get_weather`.

### Step 3: Execute the Tool and Prepare the Next Turn

Append the entire response output (including the reasoning item) to the context, execute the function, and add the result.

```python
# Add the full response (including reasoning) to the context
context += response.output

# Extract and execute the tool call
tool_call = response.output[1]
args = json.loads(tool_call.arguments)
result = get_weather(args["latitude"], args["longitude"])

# Append the function result to the context
context.append({
    "type": "function_call_output",
    "call_id": tool_call.call_id,
    "output": str(result)
})
```

### Step 4: Send the Updated Context for Completion

Make a second API call with the enriched context. The model now has access to its previous reasoning.

```python
response_2 = client.responses.create(
    model="o4-mini",
    input=context,
    tools=tools,
)

print(response_2.output_text)
```

By including the reasoning item, you help the model maintain continuity, which can improve performance. Internal benchmarks show about a **3% improvement** on tasks like SWE-bench.

## 3. Leveraging Caching for Cost and Latency

The Responses API improves cache utilization compared to the Completions API. Caching only affects prompts longer than 1024 tokens. Higher cache utilization reduces costs (cached input tokens for `o4-mini` are 75% cheaper) and improves latency.

In a multi-turn conversation, reasoning items from previous turns are ignored unless you include them. While the API discards irrelevant reasoning items, including them is harmless and ensures you don't miss potential cache hits.

## 4. Using Encrypted Reasoning Items for Stateless Workflows

Organizations with Zero Data Retention (ZDR) requirements cannot use the stateful `previous_response_id`. For these cases, use encrypted reasoning items to keep workflows stateless while preserving reasoning benefits.

### Step 1: Request Encrypted Content

Set `store=False` and include `"reasoning.encrypted_content"` in the `include` parameter.

```python
context = [{"role": "user", "content": "What's the weather like in Paris today?"}]

response = client.responses.create(
    model="o3",
    input=context,
    tools=tools,
    store=False,
    include=["reasoning.encrypted_content"]
)

print(response.output[0])
```

The reasoning item now contains an `encrypted_content` field. This encrypted state is persisted client-side; OpenAI retains no data.

### Step 2: Use the Encrypted Item in the Next Turn

Proceed as before, appending the encrypted reasoning item to the context.

```python
context += response.output

tool_call = response.output[1]
args = json.loads(tool_call.arguments)

# Mock function result for demonstration
result = 20

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

This approach allows you to benefit from reasoning items without storing state on OpenAI's servers.

## 5. Accessing Reasoning Summaries

While raw reasoning tokens are not exposed, you can request a summary of the model's chain of thought.

```python
response = client.responses.create(
    model="o3",
    input="What are the main differences between photosynthesis and cellular respiration?",
    reasoning={"summary": "auto"},
)

# Extract the summary text
first_reasoning_item = response.output[0]
first_summary_text = first_reasoning_item.summary[0].text if first_reasoning_item.summary else None
print("First reasoning summary text:\n", first_summary_text)
```

The summary provides a high-level view of the model's internal reasoning process.

## Summary

You now know how to:
1. **Structure conversations** with reasoning models using the Responses API.
2. **Include reasoning items** during function calls to boost performance.
3. **Benefit from improved caching** for lower costs and latency.
4. **Use encrypted reasoning items** for stateless, ZDR-compliant workflows.
5. **Access reasoning summaries** to gain insight into the model's thought process.

By fully leveraging the Responses API, you ensure your applications achieve maximum intelligence, efficiency, and cost-effectiveness with OpenAI's latest reasoning models.