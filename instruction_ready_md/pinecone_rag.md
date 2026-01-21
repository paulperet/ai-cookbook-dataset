# RAG with Mistral AI and Pinecone: A Step-by-Step Guide

This guide walks you through building a Retrieval-Augmented Generation (RAG) system using Mistral AI for embeddings and generation, and Pinecone as a vector database. You will index a dataset of AI research papers and create a pipeline that retrieves relevant context to answer user queries.

## Prerequisites & Setup

Before starting, ensure you have API keys for both [Mistral AI](https://console.mistral.ai/api-keys/) and [Pinecone](https://app.pinecone.io). The estimated cost to run this tutorial is less than $1 for embeddings.

First, install the required Python libraries.

```bash
pip install -qU datasets mistralai pinecone
```

## Step 1: Load and Prepare the Dataset

We'll use a dataset of semantically chunked ArXiv papers focused on LLMs and Generative AI.

```python
from datasets import load_dataset

# Load the first 10,000 chunks for demonstration
data = load_dataset(
    "jamescalam/ai-arxiv2-semantic-chunks",
    split="train[:10000]"
)
print(f"Dataset loaded. Number of records: {len(data)}")
```

Each record contains a chunk of text (1-2 paragraphs). Let's format the data to keep only the essential fields: a unique `id`, the `text` content for embedding, and associated `metadata`.

```python
# Map the dataset to our required format
data = data.map(lambda x: {
    "id": x["id"],
    "metadata": {
        "title": x["title"],
        "content": x["content"],
    }
})

# Remove the original columns we no longer need
columns_to_drop = ["title", "content", "prechunk_id", "postchunk_id", "arxiv_id", "references"]
data = data.remove_columns(columns_to_drop)

print(data)
```

## Step 2: Initialize the Mistral AI Client

We'll use Mistral AI's `mistral-embed` model to create vector embeddings. Initialize the client with your API key.

```python
import os
from mistralai import Mistral
import getpass

# Securely fetch your Mistral API key
mistral_api_key = os.getenv("MISTRAL_API_KEY") or getpass.getpass("Enter your Mistral API key: ")

# Initialize the client
mistral = Mistral(api_key=mistral_api_key)
embed_model = "mistral-embed"
```

Test the embedding model to confirm it works and to get the dimensionality of the vectors, which is required for setting up the vector index.

```python
# Create a test embedding
embeds = mistral.embeddings.create(
    model=embed_model,
    inputs=["this is a test"]
)

# Get the vector dimension
dims = len(embeds.data[0].embedding)
print(f"Embedding dimension: {dims}")
```

## Step 3: Set Up the Pinecone Vector Index

Initialize the Pinecone client and create a serverless index to store our vectors.

```python
from pinecone import Pinecone, ServerlessSpec
import time

# Securely fetch your Pinecone API key
api_key = os.getenv("PINECONE_API_KEY") or getpass.getpass("Enter your Pinecone API key: ")
pc = Pinecone(api_key=api_key)

# Define the serverless specification (cloud and region)
spec = ServerlessSpec(cloud="aws", region="us-east-1")

# Define your index name
index_name = "mistral-rag"

# Check if the index already exists
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    # Create the index with the correct dimension and metric
    pc.create_index(
        index_name,
        dimension=dims,          # 1024 for mistral-embed
        metric='dotproduct',     # Compatible with mistral-embed
        spec=spec
    )
    # Wait for the index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Connect to the index
index = pc.Index(index_name)
time.sleep(1)  # Brief pause

# View index statistics
print(index.describe_index_stats())
```

## Step 4: Create a Robust Embedding Function

The Mistral API has a token limit per batch. This function handles batching and automatically reduces the batch size if an error occurs.

```python
def embed(metadata: list[dict]):
    """
    Embeds a list of metadata dictionaries.
    Each dict should have 'title' and 'content' keys.
    """
    batch_size = len(metadata)
    while batch_size >= 1:
        try:
            embeds = []
            for j in range(0, len(metadata), batch_size):
                j_end = min(len(metadata), j + batch_size)
                # Combine title and content for embedding
                input_texts = [
                    x["title"] + "\n" + x["content"] for x in metadata[j:j_end]
                ]
                embed_response = mistral.embeddings.create(
                    inputs=input_texts,
                    model=embed_model
                )
                embeds.extend([x.embedding for x in embed_response.data])
            return embeds
        except Exception as e:
            # Halve the batch size and retry
            batch_size = int(batch_size / 2)
            print(f"Hit an exception: {e}, attempting batch_size={batch_size}")
    raise Exception("Failed to embed metadata after multiple attempts.")
```

## Step 5: Populate the Vector Index

Now, we'll embed our dataset in batches and upsert the vectors into Pinecone.

**Note:** Embedding the full 10,000 chunks has a small associated cost.

```python
from tqdm.auto import tqdm

batch_size = 32  # Number of records to process per batch

for i in tqdm(range(0, len(data), batch_size)):
    # Define the batch range
    i_end = min(len(data), i + batch_size)
    batch = data[i:i_end]

    # Generate embeddings for the batch
    embeds = embed(batch["metadata"])
    assert len(embeds) == (i_end - i)

    # Prepare data for upsert: (id, vector, metadata)
    to_upsert = list(zip(batch["id"], embeds, batch["metadata"]))

    # Upsert to Pinecone
    index.upsert(vectors=to_upsert)

print("Indexing complete.")
```

## Step 6: Build the Retrieval Function

Create a function that takes a user query, embeds it, and retrieves the most relevant document chunks from Pinecone.

```python
def get_docs(query: str, top_k: int) -> list[str]:
    """Retrieves the top_k most relevant document chunks for a query."""
    # Encode the query
    xq = mistral.embeddings.create(
        inputs=[query],
        model=embed_model
    ).data[0].embedding

    # Query the Pinecone index
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)

    # Extract the text content from the results
    docs = [x["metadata"]['content'] for x in res["matches"]]
    return docs
```

Let's test the retrieval.

```python
query = "can you tell me about mistral LLM?"
docs = get_docs(query, top_k=5)

print("Retrieved document chunks:")
print("\n---\n".join(docs))
```

## Step 7: Generate Answers with Context

With the relevant context retrieved, we can now use the powerful `mistral-large-latest` model to generate a coherent answer.

```python
def generate(query: str, docs: list[str]):
    """Generates an answer using the provided context (docs) and user query."""
    system_message = (
        "You are a helpful assistant that answers questions about AI using the "
        "context provided below.\n\n"
        "CONTEXT:\n"
        "\n---\n".join(docs)
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]

    # Call the Mistral chat completion API
    chat_response = mistral.chat.complete(
        model="mistral-large-latest",
        messages=messages
    )
    return chat_response.choices[0].message.content
```

Now, generate the final answer.

```python
answer = generate(query=query, docs=docs)
print("Generated Answer:")
print(answer)
```

## Step 8: Clean Up Resources

To avoid incurring unnecessary costs, delete the Pinecone index once you are finished.

```python
pc.delete_index(index_name)
print(f"Index '{index_name}' deleted.")
```

## Summary

You have successfully built a complete RAG pipeline:
1.  **Data Preparation:** Loaded and formatted a dataset of AI research papers.
2.  **Embedding:** Used Mistral AI's `mistral-embed` model to create vector representations.
3.  **Vector Database:** Stored and indexed vectors in a Pinecone serverless index.
4.  **Retrieval:** Implemented a function to find the most relevant context for a user query.
5.  **Generation:** Used `mistral-large-latest` to synthesize an accurate answer based on the retrieved context.

This system can be extended with more sophisticated query routing, hybrid search, or multi-stage retrieval for production applications.