# Semantic Search with Pinecone and OpenAI: A Step-by-Step Guide

This guide will walk you through building a semantic search pipeline using OpenAI's Embedding API and Pinecone's vector database. You'll learn how to generate text embeddings, store them in a scalable index, and perform fast, meaning-based searches—even when queries don't contain exact keywords.

## Prerequisites

Before you begin, ensure you have:
- An [OpenAI API key](https://platform.openai.com)
- A [Pinecone API key](https://app.pinecone.io)

## Step 1: Environment Setup

Install the required Python libraries:

```bash
pip install -qU \
    pinecone-client==3.0.2 \
    openai==1.10.0 \
    datasets==2.16.1
```

## Step 2: Initialize OpenAI and Pinecone Clients

First, import the necessary modules and set up your API clients.

```python
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import time

# Initialize OpenAI client
client = OpenAI(api_key="your-openai-api-key")  # Replace with your key

# Initialize Pinecone client
pc = Pinecone(api_key="your-pinecone-api-key")  # Replace with your key
```

## Step 3: Create and Inspect Embeddings

Let's test the embedding generation with OpenAI's `text-embedding-3-small` model.

```python
MODEL = "text-embedding-3-small"

# Generate embeddings for sample text
res = client.embeddings.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ],
    model=MODEL
)

# Extract embeddings to a list
embeds = [record.embedding for record in res.data]

print(f"Number of embeddings: {len(embeds)}")
print(f"Dimension of each vector: {len(embeds[0])}")
```

You should see output confirming two embeddings were created, each with 1536 dimensions.

## Step 4: Create a Pinecone Index

Now, create a Pinecone index to store your vectors. We'll use a serverless specification for simplicity.

```python
index_name = 'semantic-search-openai'
spec = ServerlessSpec(cloud="aws", region="us-west-2")

# Check if index exists, create if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=len(embeds[0]),  # Matches embedding dimension
        metric='dotproduct',
        spec=spec
    )
    # Wait for index initialization
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Connect to the index
index = pc.Index(index_name)
time.sleep(1)

# View index statistics
print(index.describe_index_stats())
```

The output should show an empty index with 1536 dimensions.

## Step 5: Load and Prepare Dataset

We'll use the TREC dataset from Hugging Face, loading the first 1,000 questions.

```python
from datasets import load_dataset

# Load dataset
trec = load_dataset('trec', split='train[:1000]')
print(f"Dataset loaded: {len(trec)} rows")
print(f"Sample row: {trec[0]}")
```

## Step 6: Populate the Index with Embeddings

Now, we'll process the dataset in batches, generate embeddings for each question, and upload them to Pinecone.

```python
from tqdm.auto import tqdm

batch_size = 32

for i in tqdm(range(0, len(trec['text']), batch_size)):
    # Define batch boundaries
    i_end = min(i + batch_size, len(trec['text']))
    
    # Get batch of text and create IDs
    lines_batch = trec['text'][i:i_end]
    ids_batch = [str(n) for n in range(i, i_end)]
    
    # Generate embeddings for the batch
    res = client.embeddings.create(input=lines_batch, model=MODEL)
    embeds = [record.embedding for record in res.data]
    
    # Prepare metadata
    meta = [{'text': line} for line in lines_batch]
    
    # Create upsert data structure
    to_upsert = zip(ids_batch, embeds, meta)
    
    # Upload to Pinecone
    index.upsert(vectors=list(to_upsert))
```

This process will show a progress bar as all 1,000 questions are embedded and indexed.

## Step 7: Perform Semantic Searches

With the index populated, you can now query it using natural language.

### Basic Query

```python
query = "What caused the 1929 Great Depression?"

# Generate query embedding
xq = client.embeddings.create(input=query, model=MODEL).data[0].embedding

# Query Pinecone
res = index.query(vector=[xq], top_k=5, include_metadata=True)

# Display results
for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")
```

You should see semantically similar questions about the Great Depression, ranked by relevance.

### Testing Semantic Understanding

Let's test the system's ability to understand meaning beyond keywords by using different phrasing.

```python
# Query with synonym "recession"
query = "What was the cause of the major recession in the early 20th century?"
xq = client.embeddings.create(input=query, model=MODEL).data[0].embedding
res = index.query(vector=[xq], top_k=5, include_metadata=True)

print("\nQuery with 'recession':")
for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")

# Query with descriptive phrase
query = "Why was there a long-term economic downturn in the early 20th century?"
xq = client.embeddings.create(input=query, model=MODEL).data[0].embedding
res = index.query(vector=[xq], top_k=5, include_metadata=True)

print("\nQuery with descriptive phrase:")
for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")
```

Notice how the system correctly identifies questions about the Great Depression even when using different terminology.

## Step 8: Clean Up Resources

When you're finished, delete the index to free up resources.

```python
pc.delete_index(index_name)
print(f"Index '{index_name}' deleted.")
```

## Conclusion

You've successfully built a semantic search pipeline that can:
1. Generate text embeddings using OpenAI's API
2. Store and index embeddings in Pinecone
3. Perform fast, meaning-based searches that understand synonyms and related concepts

This foundation can be extended to build applications like question-answering systems, recommendation engines, or document retrieval tools.