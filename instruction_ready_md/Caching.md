# Gemini API Context Caching Guide

This guide demonstrates how to use context caching with the Gemini API to efficiently query a large document multiple times. You will work with the Apollo 11 mission transcript, caching it once and then asking several questions without resending the entire file.

## Prerequisites

Ensure you have the required library installed and your API key configured.

### 1. Install the SDK

```bash
pip install -q -U "google-genai>=1.0.0"
```

### 2. Configure Your API Key

Set your `GOOGLE_API_KEY` as an environment variable or within your script.

```python
import os
from google import genai

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 1: Download and Upload the Document

First, obtain the Apollo 11 transcript and upload it via the File API.

```python
# Download the transcript file
import requests

url = "https://storage.googleapis.com/generativeai-downloads/data/a11.txt"
response = requests.get(url)
with open("a11.txt", "wb") as f:
    f.write(response.content)

# Inspect the first few lines
with open("a11.txt", "r") as f:
    print(f.read(500))
```

Now, upload the file to the Gemini API.

```python
document = client.files.upload(file="a11.txt")
```

## Step 2: Create a Cached Content Object

Context caching allows you to store a prompt (including files and system instructions) server-side, reducing token costs for subsequent queries. Caches are model-specific.

```python
MODEL_ID = "gemini-2.5-flash"  # Choose your model

apollo_cache = client.caches.create(
    model=MODEL_ID,
    config={
        'contents': [document],
        'system_instruction': 'You are an expert at analyzing transcripts.',
    },
)

print(f"Cached {apollo_cache.usage_metadata.total_token_count} tokens.")
```

## Step 3: Manage Cache Expiry

By default, caches expire after one hour. You can extend their lifetime.

```python
from google.genai import types

# Update the cache to expire in 2 hours (7200 seconds)
client.caches.update(
    name=apollo_cache.name,
    config=types.UpdateCachedContentConfig(ttl="7200s")
)

# Fetch the updated cache object
apollo_cache = client.caches.get(name=apollo_cache.name)
print(f"New expiry: {apollo_cache.expire_time}")
```

## Step 4: Generate Content Using the Cache

With the cache created, you can now query the model. The cached document and system instruction are automatically included.

```python
from IPython.display import Markdown

response = client.models.generate_content(
    model=MODEL_ID,
    contents='Find a lighthearted moment from this transcript',
    config=types.GenerateContentConfig(
        cached_content=apollo_cache.name,
    )
)

print(response.text)
```

### Understanding Token Usage

Inspect the `usage_metadata` to see how tokens are allocated between the cache, your new prompt, the model's reasoning, and the final output.

```python
meta = response.usage_metadata
print(f"Cached tokens: {meta.cached_content_token_count}")
print(f"Prompt tokens (total): {meta.prompt_token_count}")
print(f"  -> New prompt tokens: {meta.prompt_token_count - meta.cached_content_token_count}")
print(f"Thought tokens: {meta.thoughts_token_count}")
print(f"Output tokens: {meta.candidates_token_count}")
print(f"Total tokens: {meta.total_token_count}")
```

## Step 5: Conduct a Multi-Turn Chat

You can also use the cache within a chat session, reusing the context across multiple messages.

```python
# Start a chat session with the cache
chat = client.chats.create(
  model=MODEL_ID,
  config={"cached_content": apollo_cache.name}
)

# First question
response = chat.send_message(message="Give me a quote from the most important part of the transcript.")
print(response.text)

# Follow-up question (context is maintained)
response = chat.send_message(
    message="What was recounted after that?",
    config={"cached_content": apollo_cache.name}
)
print(response.text)
```

Check the token usage after the second message to see the efficiency gains.

```python
meta = response.usage_metadata
print(f"New prompt tokens for follow-up: {meta.prompt_token_count - meta.cached_content_token_count}")
```

## Step 6: Clean Up the Cache

Caches incur a small storage cost. Delete them when they are no longer needed.

```python
print(f"Deleting cache: {apollo_cache.name}")
client.caches.delete(name=apollo_cache.name)
```

## Summary

Context caching is a powerful feature for applications that repeatedly query the same large documents. By caching the initial prompt (including files and system instructions), you significantly reduce the token count and cost of subsequent requests, while maintaining full conversational context.

### Next Steps
- Explore the full [Caching API documentation](https://ai.google.dev/gemini-api/docs/caching).
- Learn about the [File API](https://ai.google.dev/gemini-api/docs/file_api) for handling various media types.
- Review [pricing details](https://ai.google.dev/pricing) to understand cost savings from cached tokens.