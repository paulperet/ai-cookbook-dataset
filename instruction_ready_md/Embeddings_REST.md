# Gemini API: Embedding Quickstart with REST

This guide provides step-by-step examples for generating text embeddings using the Gemini API's REST endpoint with `curl`. You'll learn how to embed single prompts, batch multiple requests, and use advanced parameters to tailor the embeddings for your specific use case.

## Prerequisites

To follow this tutorial, you need:
1.  A **Gemini API Key**. You can obtain one from [Google AI Studio](https://makersuite.google.com/app/apikey).
2.  The ability to run `curl` commands from your terminal or command line.

## Step 1: Set Your API Key

Before making any API calls, you must set your API key as an environment variable. Replace `YOUR_API_KEY_HERE` with your actual key.

```bash
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

## Step 2: Generate a Single Embedding

Let's start by embedding a simple text string. The core endpoint for this is `embedContent`. We'll use the `gemini-embedding-001` model.

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=$GEMINI_API_KEY" \
-H 'Content-Type: application/json' \
-d '{
  "model": "models/gemini-embedding-001",
  "content": {
    "parts":[{
      "text": "Hello world"
    }]
  }
}'
```

**What's happening?**
*   The `curl` command sends a `POST` request to the Gemini API endpoint.
*   The `-H` flag sets the request header to specify we are sending JSON data.
*   The `-d` flag contains the JSON payload with our request parameters: the model name and the content to embed.

**Expected Output:**
The API responds with a JSON object containing an `embedding` field. Its `values` key holds a long array of floating-point numbers—this is your text embedding vector.

```json
{
  "embedding": {
    "values": [
      -0.02342152,
      0.01676572,
      0.009261323,
      ... // Many more values
    ]
  }
}
```

## Step 3: Batch Embed Multiple Prompts

For efficiency, you can embed multiple text snippets in a single API call using the `batchEmbedContents` endpoint. This is ideal for processing datasets.

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key=$GEMINI_API_KEY" \
-H 'Content-Type: application/json' \
-d '{
  "requests": [
    {
      "model": "models/gemini-embedding-001",
      "content": {
        "parts":[{
          "text": "What is the meaning of life?"
        }]
      }
    },
    {
      "model": "models/gemini-embedding-001",
      "content": {
        "parts":[{
          "text": "How much wood would a woodchuck chuck?"
        }]
      }
    },
    {
      "model": "models/gemini-embedding-001",
      "content": {
        "parts":[{
          "text": "How does the brain work?"
        }]
      }
    }
  ]
}'
```

**Understanding the Response:**
The API returns a JSON object with an `embeddings` array. Each object in this array corresponds to one of the requests you sent, containing the `values` for that specific text.

## Step 4: Control Embedding Size with `output_dimensionality`

The `gemini-embedding-001` model allows you to reduce the size of the output vector using the `output_dimensionality` parameter. This is useful for applications where storage or computational efficiency is critical, as it truncates the full embedding to the specified length.

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=$GEMINI_API_KEY" \
-H 'Content-Type: application/json' \
-d '{
  "model": "models/gemini-embedding-001",
  "output_dimensionality": 256,
  "content": {
    "parts":[{
      "text": "Hello world"
    }]
  }
}'
```

In this example, the returned embedding vector will contain only the first 256 values from the full embedding.

## Step 5: Specify a `task_type` for Your Use Case

You can provide a hint to the model about how you plan to use the embeddings. This optional `task_type` parameter can help the model generate embeddings better suited for specific downstream tasks.

The available task types are:
*   `RETRIEVAL_QUERY`: The text is a search query.
*   `RETRIEVAL_DOCUMENT`: The text is a document in a search corpus.
*   `SEMANTIC_SIMILARITY`: The text will be used for Semantic Textual Similarity (STS).
*   `CLASSIFICATION`: The text will be classified.
*   `CLUSTERING`: The embeddings will be used for clustering.

Here's an example for embedding a document intended for a retrieval system:

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=$GEMINI_API_KEY" \
-H 'Content-Type: application/json' \
-d '{
  "model": "models/gemini-embedding-001",
  "content": {
    "parts":[{
      "text": "Hello world"
    }]
  },
  "task_type": "RETRIEVAL_DOCUMENT"
}'
```

## Next Steps

You've successfully learned the basics of generating embeddings with the Gemini API via REST. To dive deeper:

*   Explore the official [Gemini API Embeddings Guide](https://ai.google.dev/gemini-api/docs/embeddings) for detailed model information and best practices.
*   Check out more advanced examples and workflows in the [Google Gemini Cookbook](https://github.com/google-gemini/cookbook).