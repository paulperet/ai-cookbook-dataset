# Guide: Using Redis Vectors with JSON and OpenAI

This guide demonstrates how to store and search vector embeddings alongside JSON data in Redis. You'll learn to perform semantic and hybrid searches using OpenAI's embedding models and Redis Stack's vector search capabilities.

## Prerequisites

Before starting, ensure you have:
- A Redis instance with Redis Search and Redis JSON modules enabled
- The `redis-py` Python client library
- An OpenAI API key

## Setup

### 1. Install Required Packages

Begin by installing the necessary Python packages:

```bash
pip install redis openai python-dotenv
```

### 2. Configure OpenAI API Key

Create a `.env` file in your project directory and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_key_here
```

### 3. Import Libraries and Initialize OpenAI

```python
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
```

## Step 1: Generate Text Embeddings

First, you'll create a helper function to generate vector embeddings from text using OpenAI's API. Then, you'll apply this function to three sample news articles.

```python
def get_vector(text, model="text-embedding-3-small"):
    """Generate embedding vector for input text."""
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

# Sample news articles
text_1 = """Japan narrowly escapes recession..."""  # Full text from original
text_2 = """Dibaba breaks 5,000m world record..."""  # Full text from original
text_3 = """Google's toolbar sparks concern..."""  # Full text from original

# Create documents with content and vector embeddings
doc_1 = {"content": text_1, "vector": get_vector(text_1)}
doc_2 = {"content": text_2, "vector": get_vector(text_2)}
doc_3 = {"content": text_3, "vector": get_vector(text_3)}
```

## Step 2: Start Redis Stack

If you're using Docker, start a Redis Stack container with the necessary modules:

```bash
docker run -d -p 6379:6379 redis/redis-stack:latest
```

## Step 3: Connect to Redis

Establish a connection to your Redis instance:

```python
from redis import from_url

REDIS_URL = 'redis://localhost:6379'
client = from_url(REDIS_URL)
client.ping()  # Should return True if connected
```

## Step 4: Create a Search Index

Create a Redis Search index that can handle both vector similarity search and full-text search on JSON documents.

```python
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# Define the index schema
schema = [
    VectorField(
        '$.vector', 
        "FLAT", 
        {
            "TYPE": 'FLOAT32', 
            "DIM": len(doc_1['vector']), 
            "DISTANCE_METRIC": "COSINE"
        }, 
        as_name='vector'
    ),
    TextField('$.content', as_name='content')
]

# Define index configuration
idx_def = IndexDefinition(index_type=IndexType.JSON, prefix=['doc:'])

# Drop existing index if it exists, then create new one
try: 
    client.ft('idx').dropindex()
except:
    pass

client.ft('idx').create_index(schema, definition=idx_def)
```

## Step 5: Store Documents as JSON

Store your documents in Redis as JSON objects with the vector embeddings included:

```python
client.json().set('doc:1', '$', doc_1)
client.json().set('doc:2', '$', doc_2)
client.json().set('doc:3', '$', doc_3)
```

## Step 6: Perform Semantic Search

Now you can perform vector similarity search. Let's search for articles similar to a new sports-related article:

```python
from redis.commands.search.query import Query
import numpy as np

# New query article about athletics
text_4 = """Radcliffe yet to answer GB call..."""  # Full text from original

# Generate embedding for the query
vec = np.array(get_vector(text_4), dtype=np.float32).tobytes()

# Build the KNN query
q = Query('*=>[KNN 3 @vector $query_vec AS vector_score]')\
    .sort_by('vector_score')\
    .return_fields('vector_score', 'content')\
    .dialect(2)

params = {"query_vec": vec}

# Execute the search
results = client.ft('idx').search(q, query_params=params)

# Display results
for doc in results.docs:
    print(f"Distance: {round(float(doc['vector_score']), 3)}")
    print(f"Content: {doc['content'][:200]}...\n")
```

The search will return the three most similar articles, with the sports-related article (about Dibaba's world record) appearing first due to its semantic similarity.

## Step 7: Perform Hybrid Search

Combine full-text filtering with vector similarity search. Let's find articles about "recession" that are semantically similar to an article about Ethiopia's crop production:

```python
# New query article about Ethiopia's economy
text_5 = """Ethiopia's crop production up 24%..."""  # Full text from original

# Generate embedding for the query
vec = np.array(get_vector(text_5), dtype=np.float32).tobytes()

# Build hybrid query: full-text filter + vector similarity
q = Query('@content:recession => [KNN 3 @vector $query_vec AS vector_score]')\
    .sort_by('vector_score')\
    .return_fields('vector_score', 'content')\
    .dialect(2)

params = {"query_vec": vec}

# Execute the search
results = client.ft('idx').search(q, query_params=params)

# Display results
for doc in results.docs:
    print(f"Distance: {round(float(doc['vector_score']), 3)}")
    print(f"Content: {doc['content'][:200]}...\n")
```

This search first filters for articles containing the word "recession" in their content, then performs vector similarity search on those filtered results. Only the Japan recession article will match, as it's the only one containing "recession" while also being semantically related to economic topics.

## Summary

You've successfully implemented a Redis-based vector search system that:
1. Generates embeddings using OpenAI's API
2. Stores documents as JSON with vector fields
3. Creates a search index for both vector and text search
4. Performs semantic search using vector similarity
5. Combines full-text filtering with vector search for hybrid queries

This approach enables powerful semantic search capabilities while maintaining the flexibility of JSON document storage. You can extend this pattern to build sophisticated recommendation systems, document retrieval applications, or any system requiring semantic understanding of text data.