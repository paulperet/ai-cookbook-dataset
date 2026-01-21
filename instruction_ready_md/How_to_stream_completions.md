# Streaming OpenAI Completions: A Practical Guide

By default, the OpenAI API generates the entire completion before sending it back in a single response. For long completions, this can mean waiting several seconds. Streaming allows you to receive and process the response incrementally as it's being generated, providing a faster perceived response time.

## Key Considerations

**Production Note:** Using `stream=True` in production applications can make content moderation more challenging, as partial completions may be harder to evaluate. Consider this when implementing streaming for approved use cases.

## Prerequisites

First, install the OpenAI Python package and set up your environment:

```bash
pip install openai
```

```python
import time
from openai import OpenAI
import os

# Initialize the client with your API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
```

## Step 1: Understanding Standard Chat Completions

Let's start with a standard, non-streaming completion to establish a baseline. We'll ask the model to count to 100 and measure the response time.

```python
# Record the time before the request
start_time = time.time()

# Send a standard ChatCompletion request
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'user', 'content': 'Count to 100, with a comma between each number and no newlines. E.g., 1, 2, 3, ...'}
    ],
    temperature=0,
)

# Calculate the response time
response_time = time.time() - start_time

print(f"Full response received {response_time:.2f} seconds after request")
```

The response arrives as a complete object. You can extract the assistant's reply:

```python
# Extract the full message object
reply = response.choices[0].message
print(f"Extracted reply: \n{reply}")

# Extract just the content
reply_content = response.choices[0].message.content
print(f"Extracted content: \n{reply_content}")
```

## Step 2: Implementing Streaming Completions

Now let's implement streaming by setting `stream=True`. This returns an event stream where each chunk contains incremental updates to the response.

```python
# Send a streaming ChatCompletion request
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'user', 'content': "What's 1+1? Answer in one word."}
    ],
    temperature=0,
    stream=True  # Enable streaming
)

# Iterate through the stream of chunks
for chunk in response:
    print(chunk)
    print(chunk.choices[0].delta.content)
    print("****************")
```

**Key Difference:** Streaming responses use a `delta` field instead of a `message` field. The `delta` contains incremental updates like:
- Role tokens: `{"role": "assistant"}`
- Content tokens: `{"content": "Two"}`
- Empty objects: `{}` (when the stream ends)

## Step 3: Measuring Streaming Performance

Let's compare the timing by streaming the same "count to 100" request:

```python
# Record the start time
start_time = time.time()

# Send a streaming request
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'user', 'content': 'Count to 100, with a comma between each number and no newlines. E.g., 1, 2, 3, ...'}
    ],
    temperature=0,
    stream=True
)

# Variables to collect the stream
collected_chunks = []
collected_messages = []

# Process each chunk as it arrives
for chunk in response:
    chunk_time = time.time() - start_time
    collected_chunks.append(chunk)
    chunk_message = chunk.choices[0].delta.content
    collected_messages.append(chunk_message)
    print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")

# Clean and assemble the full response
print(f"Full response received {chunk_time:.2f} seconds after request")
collected_messages = [m for m in collected_messages if m is not None]
full_reply_content = ''.join(collected_messages)
print(f"Full conversation received: {full_reply_content}")
```

**Performance Insight:** While both requests may take similar total time to complete (4-5 seconds), streaming delivers the first token much faster (typically 0.1 seconds), with subsequent tokens arriving every 0.01-0.02 seconds. This creates a much more responsive user experience.

## Step 4: Retrieving Token Usage for Streamed Responses

To get token usage statistics with streaming, add `stream_options={"include_usage": True}`. This adds a final chunk containing usage data.

```python
# Request with token usage tracking
response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {'role': 'user', 'content': "What's 1+1? Answer in one word."}
    ],
    temperature=0,
    stream=True,
    stream_options={"include_usage": True},  # Enable usage tracking
)

# Process the stream
for chunk in response:
    print(f"choices: {chunk.choices}\nusage: {chunk.usage}")
    print("****************")
```

**Important Notes:**
- All chunks except the last have `usage: None`
- The final chunk contains the complete usage statistics in its `usage` field
- The final chunk's `choices` field is always an empty array `[]`

## Summary

Streaming completions provide significant UX improvements for longer responses by delivering content incrementally. Remember to:
1. Set `stream=True` in your API call
2. Process the `delta` field instead of `message` in each chunk
3. Consider using `stream_options={"include_usage": True}` for token tracking
4. Be mindful of content moderation challenges in production environments

This approach is particularly valuable for applications where perceived responsiveness is critical, such as chatbots, writing assistants, or any interactive AI interface.