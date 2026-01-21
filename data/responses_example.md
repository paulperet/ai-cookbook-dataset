# Getting Started with the OpenAI Responses API

## Introduction

The OpenAI Responses API is a modern, stateful interface designed for building advanced AI applications. It simplifies multi-turn conversations, provides seamless access to hosted tools like web search, and offers fine-grained control over context. Unlike previous APIs, it's built from the ground up for asynchronous, complex reasoning tasks.

This guide walks you through the core features with practical, step-by-step examples.

## Prerequisites

Ensure you have the OpenAI Python library installed and your API key configured.

```bash
pip install openai
```

Set up your client by importing the library and initializing it with your API key.

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

## Step 1: Making Your First API Call

The basic structure of the Responses API is straightforward. You send an input and receive a response object.

```python
response = client.responses.create(
    model="gpt-4o-mini",
    input="tell me a joke",
)

print(response.output[0].content[0].text)
```

**Output:**
```
Why did the scarecrow win an award?

Because he was outstanding in his field!
```

## Step 2: Understanding Stateful Conversations

A key feature is statefulness. The API manages conversation history for you. You can retrieve any previous response, which includes the full context.

```python
fetched_response = client.responses.retrieve(response_id=response.id)
print(fetched_response.output[0].content[0].text)
```

**Output:**
```
Why did the scarecrow win an award?

Because he was outstanding in his field!
```

## Step 3: Continuing and Forking Conversations

You can continue a conversation by referencing the previous response's ID.

```python
response_two = client.responses.create(
    model="gpt-4o-mini",
    input="tell me another",
    previous_response_id=response.id
)

print(response_two.output[0].content[0].text)
```

**Output:**
```
Why don't skeletons fight each other?

They don't have the guts!
```

You can also *fork* a conversation from any point, creating a new branch. Here, we fork from the first response and ask for a different joke and an analysis.

```python
response_two_forked = client.responses.create(
    model="gpt-4o-mini",
    input="I didn't like that joke, tell me another and tell me the difference between the two jokes",
    previous_response_id=response.id # Forking from the first response
)

output_text = response_two_forked.output[0].content[0].text
print(output_text)
```

**Output:**
```
Sure! Here’s another joke:

Why don’t scientists trust atoms?

Because they make up everything!

**Difference:** The first joke plays on a pun involving "outstanding" in a literal sense versus being exceptional, while the second joke relies on a play on words about atoms "making up" matter versus fabricating stories. Each joke uses wordplay, but they target different concepts (farming vs. science).
```

## Step 4: Using Hosted Tools

The Responses API provides easy access to hosted tools like `web_search`. You simply specify the tools in your request, and the API handles the execution.

### Example: Fetching Live News

Let's ask for the latest AI news and have the model use web search to get current information.

```python
response = client.responses.create(
    model="gpt-4o",
    input="What's the latest news about AI?",
    tools=[{"type": "web_search"}]
)
```

The response object contains the search results and the model's synthesized answer. You can inspect the structure.

```python
import json
print(json.dumps(response.output, default=lambda o: o.__dict__, indent=2))
```

The output will be a detailed JSON object containing the search call status and the assistant's message with citations. The assistant's text will summarize recent AI developments with links to sources.

## Step 5: Building a Multimodal, Tool-Augmented Interaction

The API natively supports text, images, and audio. You can combine modalities and tools in a single call.

In this example, we provide an image of a cat and ask the model to generate related keywords, perform a web search, and summarize the findings.

```python
response_multimodal = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text":
                 "Come up with keywords related to the image, and search on the web using the search tool for any news related to the keywords, summarize the findings and cite the sources."},
                {"type": "input_image", "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/2880px-Cat_August_2010-4.jpg"}
            ]
        }
    ],
    tools=[{"type": "web_search"}]
)
```

You can examine the full response object to see the step-by-step reasoning, the web search call, and the final summarized output with citations.

```python
import json
print(json.dumps(response_multimodal.__dict__, default=lambda o: o.__dict__, indent=4))
```

The response will show the assistant first proposing keywords (e.g., "Cat", "Feline"), then executing a web search, and finally delivering a summary of recent cat-related news stories with proper source citations.

## Summary

You've now learned the fundamentals of the OpenAI Responses API:
1.  **Making simple calls** to generate text.
2.  **Leveraging statefulness** for conversation history and forking.
3.  **Integrating hosted tools** like `web_search` to augment responses with live data.
4.  **Creating multimodal interactions** that combine images and tools in a single, powerful request.

This API provides a unified, flexible foundation for building sophisticated AI-driven applications.