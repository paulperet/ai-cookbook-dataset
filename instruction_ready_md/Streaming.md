# Streaming Responses with the Gemini API

This guide demonstrates how to use the streaming capabilities of the Google Gemini Python SDK. By default, the SDK returns a complete response after the model finishes generation. With streaming, you can receive and process the response in chunks as they are generated, enabling real-time interactions and more responsive applications.

## Prerequisites

Before you begin, ensure you have the following:

1.  **Python Environment:** A working Python environment.
2.  **API Key:** A Google AI API key. Store it securely in an environment variable named `GOOGLE_API_KEY`.
3.  **SDK:** The `google-genai` Python SDK installed.

### Setup

First, install the required Python package.

```bash
pip install -U "google-genai"
```

Next, import the necessary modules and configure the client with your API key.

```python
from google import genai
import os

# Retrieve your API key from the environment variable
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)

# Define the model you want to use
MODEL_ID = "gemini-3-flash-preview"  # You can change this to any supported model
```

## Step 1: Stream a Response Synchronously

To receive a streamed response, use the `client.models.generate_content_stream` method. This returns an iterator that yields response chunks.

```python
for chunk in client.models.generate_content_stream(
  model=MODEL_ID,
  contents='Tell me a story in 300 words.'
):
    # Print each text chunk as it arrives
    if chunk.text:
        print(chunk.text, end='', flush=True)
```

**What's happening?** Instead of waiting for the entire story to be generated, the code prints each piece of text immediately as the model produces it. The `end=''` and `flush=True` arguments ensure the output appears smoothly without extra newlines.

## Step 2: Stream a Response Asynchronously

For asynchronous applications, use the `client.aio.models.generate_content_stream` method. This is ideal for web servers or applications that need to handle multiple requests concurrently.

```python
import asyncio

async def stream_story():
    async for chunk in await client.aio.models.generate_content_stream(
        model=MODEL_ID,
        contents="Write a cute story about cats."
    ):
        if chunk.text:
            print(chunk.text, end='', flush=True)

# Run the async function
await stream_story()
```

**Key Difference:** This uses `async for` to iterate over the stream within an asynchronous context. The `await` keyword is required before the method call to initiate the stream.

## Step 3: Demonstrate Concurrent Asynchronous Execution

A major advantage of async streaming is that it doesn't block your entire program. You can run other tasks concurrently while the model generates its response.

The following example creates two asynchronous tasks: one streams a story, and another prints a simple message at regular intervals.

```python
import asyncio

async def get_response():
    """Task 1: Stream a long story from the model."""
    async for chunk in await client.aio.models.generate_content_stream(
        model=MODEL_ID,
        contents='Tell me a story in 500 words.'
    ):
        if chunk.text:
            print(chunk.text, end='', flush=True)

async def something_else():
    """Task 2: Simulate another concurrent process."""
    for i in range(5):
        print("\n========== Other task is not blocked! ==========")
        await asyncio.sleep(1)  # Simulate work with a sleep

async def async_demo():
    # Create tasks for concurrent execution
    task1 = asyncio.create_task(get_response())
    task2 = asyncio.create_task(something_else())
    
    # Wait for both tasks to complete
    await asyncio.gather(task1, task2)

# Execute the demo
await async_demo()
```

**Expected Behavior:** You will see the text from the story streaming while the "Other task is not blocked!" messages print at one-second intervals. This demonstrates that the long-running LLM call does not prevent other asynchronous operations from proceeding.

## Summary

You have successfully learned how to:
1.  **Stream responses synchronously** for immediate, chunk-by-chunk output in a standard script.
2.  **Stream responses asynchronously** to integrate with async/await patterns in modern Python applications.
3.  **Leverage concurrency** by running other tasks simultaneously during an async stream, improving application responsiveness.

Streaming is essential for creating interactive AI experiences, such as chatbots or real-time content generators, where low latency and a seamless user experience are critical.