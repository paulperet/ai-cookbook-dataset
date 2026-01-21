# Building a Vector Search Application with Redis

This guide walks you through implementing a semantic search system using Redis as a vector database. You'll learn how to index and search embedded data, enabling production-ready applications like chatbots and recommendation engines.

## Prerequisites

Ensure you have the following installed:
- Python 3.7+
- Docker and Docker Compose
- An OpenAI API key (for generating embeddings)

## 1. Setup Environment

First, install the required Python packages.

```bash
pip install redis pandas numpy wget openai
```

Now, import the necessary libraries and configure your environment.

```python
import openai
import pandas as pd
import numpy as np
import os
import wget
import warnings
from ast import literal_eval

# Redis client
import redis
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.commands.search.field import TextField, VectorField

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set your embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
```

## 2. Load and Prepare the Dataset

You'll use a pre-embedded Wikipedia articles dataset. This dataset contains article titles, content, and their corresponding vector embeddings.

Download and extract the dataset:

```python
# Download the dataset (approx. 700 MB)
embeddings_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'
wget.download(embeddings_url)

# Extract the files
import zipfile
with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
    zip_ref.extractall("../data")
```

Load the dataset into a pandas DataFrame:

```python
# Load the CSV file
article_df = pd.read_csv('../data/vector_database_wikipedia_articles_embedded.csv')

# Convert stringified vectors back to Python lists
article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
article_df['content_vector'] = article_df.content_vector.apply(literal_eval)

# Ensure the ID column is a string
article_df['vector_id'] = article_df['vector_id'].apply(str)

# Inspect the data structure
print(article_df.info(show_counts=True))
```

## 3. Deploy Redis with RediSearch

You'll use Redis Stack, which includes the RediSearch module with vector search capabilities. The easiest way to run it is via Docker.

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  redis:
    image: redis/redis-stack:latest
    ports:
      - "6379:6379"
      - "8001:8001"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

Start the Redis Stack container:

```bash
docker-compose up -d
```

This command starts Redis with RediSearch enabled and also launches RedisInsight, a management GUI available at `http://localhost:8001`.

## 4. Connect to Redis

Initialize a connection to your Redis instance.

```python
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = ""  # Default for passwordless setup

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD
)

# Test the connection
print("Connected to Redis:", redis_client.ping())
```

## 5. Create a Search Index

Define the schema for your vector search index. This index will store both textual metadata and vector embeddings.

```python
# Configuration constants
VECTOR_DIM = len(article_df['title_vector'][0])  # Dimension of embeddings
VECTOR_NUMBER = len(article_df)                  # Number of vectors to index
INDEX_NAME = "embeddings-index"                  # Index identifier
PREFIX = "doc"                                   # Key prefix for documents
DISTANCE_METRIC = "COSINE"                       # Similarity measure

# Define fields for the index schema
title = TextField(name="title")
url = TextField(name="url")
text = TextField(name="text")
title_embedding = VectorField("title_vector", "FLAT", {
    "TYPE": "FLOAT32",
    "DIM": VECTOR_DIM,
    "DISTANCE_METRIC": DISTANCE_METRIC,
    "INITIAL_CAP": VECTOR_NUMBER,
})
text_embedding = VectorField("content_vector", "FLAT", {
    "TYPE": "FLOAT32",
    "DIM": VECTOR_DIM,
    "DISTANCE_METRIC": DISTANCE_METRIC,
    "INITIAL_CAP": VECTOR_NUMBER,
})

fields = [title, url, text, title_embedding, text_embedding]

# Create the index if it doesn't exist
try:
    redis_client.ft(INDEX_NAME).info()
    print("Index already exists")
except:
    redis_client.ft(INDEX_NAME).create_index(
        fields=fields,
        definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
    )
    print(f"Created index: {INDEX_NAME}")
```

## 6. Index Documents

Now, load the Wikipedia articles into Redis. The function below converts vector lists to byte arrays, which is the required format for Redis vector fields.

```python
def index_documents(client: redis.Redis, prefix: str, documents: pd.DataFrame):
    """
    Index documents into Redis as hash objects.
    """
    records = documents.to_dict("records")
    for doc in records:
        key = f"{prefix}:{str(doc['id'])}"

        # Convert vectors to byte arrays
        title_embedding = np.array(doc["title_vector"], dtype=np.float32).tobytes()
        content_embedding = np.array(doc["content_vector"], dtype=np.float32).tobytes()

        # Replace list fields with byte arrays
        doc["title_vector"] = title_embedding
        doc["content_vector"] = content_embedding

        # Store as a Redis hash
        client.hset(key, mapping=doc)

# Execute indexing
index_documents(redis_client, PREFIX, article_df)
print(f"Loaded {redis_client.info()['db0']['keys']} documents into Redis.")
```

## 7. Perform Vector Search Queries

Create a search function that converts a natural language query into an embedding, then performs a nearest-neighbor search in Redis.

```python
# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or set directly: "sk-..."

def search_redis(
    redis_client: redis.Redis,
    user_query: str,
    index_name: str = "embeddings-index",
    vector_field: str = "title_vector",
    return_fields: list = ["title", "url", "text", "vector_score"],
    hybrid_fields: str = "*",
    k: int = 20,
) -> list:
    """
    Perform a vector similarity search in Redis.
    """
    # Generate embedding for the query
    embedded_query = openai.Embedding.create(
        input=user_query,
        model=EMBEDDING_MODEL,
    )["data"][0]['embedding']

    # Build the RediSearch query
    base_query = f'{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]'
    query = (
        Query(base_query)
        .return_fields(*return_fields)
        .sort_by("vector_score")
        .paging(0, k)
        .dialect(2)
    )
    params_dict = {"vector": np.array(embedded_query).astype(dtype=np.float32).tobytes()}

    # Execute search
    results = redis_client.ft(index_name).search(query, params_dict)

    # Display results
    for i, article in enumerate(results.docs):
        score = 1 - float(article.vector_score)  # Convert distance to similarity score
        print(f"{i}. {article.title} (Score: {round(score, 3)})")

    return results.docs
```

Test the search function with a sample query:

```python
results = search_redis(redis_client, 'modern art in Europe', k=5)
```

You can also search by content embeddings:

```python
results = search_redis(redis_client,
                       'Famous battles in Scottish history',
                       vector_field='content_vector',
                       k=5)
```

## 8. Implement Hybrid Search

Combine vector similarity with traditional field filters (like full-text search) for more precise results.

First, create a helper to build hybrid query filters:

```python
def create_hybrid_field(field_name: str, value: str) -> str:
    """
    Create a RediSearch filter for a specific field and value.
    """
    return f'@{field_name}:"{value}"'
```

Now, run a hybrid query that finds articles about "Famous battles in Scottish history" but only includes those with "Scottish" in the title:

```python
results = search_redis(redis_client,
                       "Famous battles in Scottish history",
                       vector_field="title_vector",
                       k=5,
                       hybrid_fields=create_hybrid_field("title", "Scottish"))
```

Another example: find articles about "Art" but filter to those mentioning "Leonardo da Vinci" in the text:

```python
results = search_redis(redis_client,
                       "Art",
                       vector_field="title_vector",
                       k=5,
                       hybrid_fields=create_hybrid_field("text", "Leonardo da Vinci"))

# Extract a specific sentence mentioning Leonardo da Vinci
if results:
    mention = [sentence for sentence in results[0].text.split("\n")
               if "Leonardo da Vinci" in sentence][0]
    print("Mention found:", mention)
```

## Next Steps

You've successfully built a vector search pipeline with Redis. To extend this application:

- Experiment with different distance metrics (`L2`, `IP`).
- Implement pagination for large result sets.
- Use RedisJSON for more flexible document storage.
- Explore the [Redis vector search documentation](https://redis.io/docs/stack/search/reference/vectors/) for advanced features.

Remember to clean up your Docker container when finished:

```bash
docker-compose down
```