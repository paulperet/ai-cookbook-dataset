# Mistral AI API Quickstart Guide

This guide will walk you through the essential steps to get started with the Mistral AI API, covering both chat completions and text embeddings.

## Prerequisites

Before you begin, ensure you have:
1. An active Mistral AI account on [La Plateforme](https://console.mistral.ai/)
2. API access enabled (requires payment activation on your account)
3. Your API key ready

## Setup

First, install the official Mistral AI Python client:

```bash
pip install mistralai
```

## Step 1: Initialize the Client

Import the library and create a client instance using your API key:

```python
from mistralai import Mistral

# Replace with your actual API key
api_key = "YOUR_API_KEY_HERE"
client = Mistral(api_key=api_key)
```

## Step 2: Chat Completions

The chat endpoint allows you to interact with Mistral's language models conversationally. Let's start with a simple query.

### 2.1 Define Your Model and Query

Choose a model and prepare your message. Here we'll use `mistral-large-latest`:

```python
model = "mistral-large-latest"

chat_response = client.chat.complete(
    model=model,
    messages=[
        {"role": "user", "content": "What is the best French cheese?"}
    ]
)
```

### 2.2 Extract and Display the Response

The response contains structured data. Extract the assistant's message content:

```python
print(chat_response.choices[0].message.content)
```

You should see a detailed response about French cheeses, demonstrating the model's knowledge and conversational capabilities.

## Step 3: Text Embeddings

Embeddings convert text into numerical vectors that capture semantic meaning. These are useful for search, clustering, and other ML tasks.

### 3.1 Generate Embeddings

Use the embeddings endpoint with the `mistral-embed` model:

```python
model = "mistral-embed"

embeddings_response = client.embeddings.create(
    model=model,
    inputs=[
        "Embed this sentence.",
        "As well as this one."
    ]
)
```

### 3.2 Understand the Response

The response contains an `EmbeddingsResponse` object. Let's examine its structure:

```python
print(f"Response ID: {embeddings_response.id}")
print(f"Object type: {embeddings_response.object}")
print(f"Number of embeddings: {len(embeddings_response.data)}")
print(f"First embedding vector length: {len(embeddings_response.data[0].embedding)}")
print(f"First few values of first embedding: {embeddings_response.data[0].embedding[:5]}")
```

Each input text is converted into a high-dimensional vector (embedding) that you can use for similarity comparisons or as input to other machine learning models.

## Next Steps

Now that you've successfully made your first API calls, consider exploring:

1. **Conversational History**: Build multi-turn conversations by maintaining message history
2. **Streaming Responses**: Use streaming for real-time token-by-token generation
3. **Embedding Applications**: Implement semantic search or document clustering using embeddings
4. **Model Selection**: Experiment with different models (`mistral-tiny`, `mistral-small`, etc.) for your specific use case

Remember to keep your API key secure and never commit it to version control. Consider using environment variables for production applications:

```python
import os
api_key = os.environ.get("MISTRAL_API_KEY")
```

Happy building with Mistral AI!