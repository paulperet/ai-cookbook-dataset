# Building Hybrid Search with Redis and OpenAI Embeddings

This guide demonstrates how to implement a hybrid search system using Redis as a vector database. You will combine vector similarity search (VSS) with traditional Redis Query and Search filtering (lexical, tag, numeric) to perform powerful, multi-faceted queries on an e-commerce dataset.

## Prerequisites

Before you begin, ensure you have the following:

1.  **A running Redis instance with RediSearch:** The easiest method is to use the Redis Stack Docker container.
2.  **An OpenAI API Key:** Required for generating text embeddings.

## Step 1: Environment Setup

### 1.1 Start Redis

Launch a Redis Stack container using Docker Compose. This provides Redis with the RediSearch module and the RedisInsight management GUI.

```bash
docker-compose up -d
```

Once running, you can access RedisInsight at `http://localhost:8001`.

### 1.2 Install Required Python Libraries

Install the necessary packages using pip.

```bash
pip install redis pandas openai
```

### 1.3 Configure Your OpenAI API Key

Set your OpenAI API key as an environment variable. Replace `<YOUR_OPENAI_API_KEY>` with your actual key.

```python
import os
import openai

# Set your API key
os.environ["OPENAI_API_KEY"] = '<YOUR_OPENAI_API_KEY>'

# Verify the key is loaded
if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print("OPENAI_API_KEY is ready")
else:
    print("OPENAI_API_KEY environment variable not found")
```

## Step 2: Load and Prepare the Dataset

You will use a sample e-commerce dataset containing product information. First, load and clean the data.

```python
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("../../data/styles_2k.csv", on_bad_lines='skip')

# Clean the data: remove null rows and fix data types
df.dropna(inplace=True)
df["year"] = df["year"].astype(int)

# Display dataset info and a preview
print("Dataset Info:")
df.info()
print("\nFirst 5 rows:")
print(df.head())
```

Next, create a consolidated text field for each product. This field will be used to generate semantic embeddings.

```python
# Create a combined text description for embedding generation
df["product_text"] = df.apply(
    lambda row: f"name {row['productDisplayName']} category {row['masterCategory']} subcategory {row['subCategory']} color {row['baseColour']} gender {row['gender']}".lower(),
    axis=1
)

# Rename the 'id' column for clarity
df.rename({"id": "product_id"}, axis=1, inplace=True)

# Verify the new field
print("Example product text for embedding:")
print(df["product_text"][0])
```

## Step 3: Connect to Redis

Establish a connection to your running Redis instance.

```python
import redis
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.commands.search.field import TagField, NumericField, TextField, VectorField

# Connection parameters
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = ""  # Default for passwordless Redis

# Create the Redis client
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD
)

# Test the connection
print("Redis connection successful:", redis_client.ping())
```

## Step 4: Create a Search Index

To perform searches, you must first define and create a RediSearch index on your Redis data.

### 4.1 Define Index Constants and Schema

Set constants for your index and define its schema, specifying the fields and their types.

```python
# Index configuration
INDEX_NAME = "product_embeddings"
PREFIX = "doc"
DISTANCE_METRIC = "L2"  # Options: COSINE, IP, L2
NUMBER_OF_VECTORS = len(df)

# Define the schema for the search index
name = TextField(name="productDisplayName")
category = TagField(name="masterCategory")
articleType = TagField(name="articleType")
gender = TagField(name="gender")
season = TagField(name="season")
year = NumericField(name="year")

# Define the vector field for embeddings
text_embedding = VectorField("product_vector", "FLAT", {
    "TYPE": "FLOAT32",
    "DIM": 1536,  # Dimension of OpenAI's text-embedding-3-small model
    "DISTANCE_METRIC": DISTANCE_METRIC,
    "INITIAL_CAP": NUMBER_OF_VECTORS,
})

# Combine all fields
fields = [name, category, articleType, gender, season, year, text_embedding]
```

### 4.2 Create the Index

Create the index in Redis if it doesn't already exist.

```python
# Check for an existing index, create if not found
try:
    redis_client.ft(INDEX_NAME).info()
    print("Index already exists")
except:
    redis_client.ft(INDEX_NAME).create_index(
        fields=fields,
        definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
    )
    print(f"Index '{INDEX_NAME}' created successfully.")
```

## Step 5: Generate Embeddings and Load Data

Now, generate vector embeddings for your product descriptions using OpenAI's API and load the data into the Redis index.

### 5.1 Define Helper Functions

Create functions to batch-process embeddings and index documents efficiently.

```python
from utils.embeddings_utils import get_embeddings

EMBEDDING_MODEL = "text-embedding-3-small"

def embeddings_batch_request(documents: pd.DataFrame):
    """Generate embeddings for a batch of product texts."""
    records = documents.to_dict("records")
    print(f"Records to process: {len(records)}")
    product_vectors = []
    docs = []
    batchsize = 1000

    for idx, doc in enumerate(records, start=1):
        docs.append(doc["product_text"])
        if idx % batchsize == 0:
            product_vectors += get_embeddings(docs, EMBEDDING_MODEL)
            docs.clear()
            print(f"Vectors processed: {len(product_vectors)}", end='\r')
    # Process any remaining docs
    product_vectors += get_embeddings(docs, EMBEDDING_MODEL)
    print(f"Total vectors processed: {len(product_vectors)}")
    return product_vectors

def index_documents(client: redis.Redis, prefix: str, documents: pd.DataFrame):
    """Generate embeddings and load documents into the Redis index."""
    product_vectors = embeddings_batch_request(documents)
    records = documents.to_dict("records")
    batchsize = 500

    # Use a pipeline for efficient batch operations
    pipe = client.pipeline()
    for idx, doc in enumerate(records, start=1):
        key = f"{prefix}:{str(doc['product_id'])}"

        # Convert the embedding to a byte vector
        text_embedding = np.array((product_vectors[idx-1]), dtype=np.float32).tobytes()
        doc["product_vector"] = text_embedding

        # Store the document as a Redis Hash
        pipe.hset(key, mapping=doc)

        # Execute the pipeline in batches
        if idx % batchsize == 0:
            pipe.execute()
    pipe.execute()
    print(f"Indexing complete.")
```

### 5.2 Execute Data Loading

Run the indexing function to populate your Redis database.

```python
index_documents(redis_client, PREFIX, df)
print(f"Loaded {redis_client.info()['db0']['keys']} documents into Redis index: {INDEX_NAME}")
```

## Step 6: Perform Vector Similarity Searches

With the data indexed, you can now perform semantic searches. The following function queries the vector database using a natural language prompt.

```python
def search_redis(
    redis_client: redis.Redis,
    user_query: str,
    index_name: str = "product_embeddings",
    vector_field: str = "product_vector",
    return_fields: list = ["productDisplayName", "masterCategory", "gender", "season", "year", "vector_score"],
    hybrid_fields: str = "*",
    k: int = 20,
    print_results: bool = True,
) -> list:
    """
    Perform a hybrid vector search.
    Args:
        hybrid_fields: A RediSearch filter query string. Use '*' for pure vector search.
    """

    # Generate an embedding for the user's query
    embedded_query = openai.Embedding.create(
        input=user_query,
        model="text-embedding-3-small",
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

    # Execute the search
    results = redis_client.ft(index_name).search(query, params_dict)

    # Print formatted results
    if print_results:
        print(f"\nQuery: '{user_query}'")
        print(f"Filter: {hybrid_fields}")
        for i, product in enumerate(results.docs):
            score = 1 - float(product.vector_score)  # Convert distance to similarity score
            print(f"  {i+1}. {product.productDisplayName} (Score: {round(score, 3)})")
    return results.docs
```

### 6.1 Run a Simple Vector Search

Test the function with a basic semantic query.

```python
print("=== Simple Vector Search ===")
results = search_redis(redis_client, 'man blue jeans', k=10)
```

## Step 7: Execute Hybrid Queries

The real power of this system is combining vector similarity with traditional database filters. The `hybrid_fields` parameter allows you to apply RediSearch filters.

### 7.1 Vector + Text Search

Combine semantic search with a phrase match in the product title.

```python
print("\n=== Hybrid Search: Vector + Text Phrase ===")
results = search_redis(
    redis_client,
    "man blue jeans",
    hybrid_fields='@productDisplayName:"blue jeans"',
    k=10
)
```

### 7.2 Vector + Tag Filter

Find semantically similar items but restrict results to a specific category using a tag filter.

```python
print("\n=== Hybrid Search: Vector + Tag Filter ===")
results = search_redis(
    redis_client,
    "watch",
    hybrid_fields='@masterCategory:{Accessories}',
    k=10
)
```

### 7.3 Vector + Numeric Range Filter

Search for items similar to "sandals" but only from a specific year range.

```python
print("\n=== Hybrid Search: Vector + Numeric Range ===")
results = search_redis(
    redis_client,
    "sandals",
    hybrid_fields='@year:[2011 2012]',
    k=10
)
```

### 7.4 Complex Hybrid Query

Combine multiple filter types in a single query: numeric range, tag selection, and text matching.

```python
print("\n=== Hybrid Search: Complex Multi-Field Filter ===")
results = search_redis(
    redis_client,
    "brown belt",
    hybrid_fields='(@year:[2012 2012] @articleType:{Shirts | Belts} @productDisplayName:"Wrangler")',
    k=10
)
```

## Summary

You have successfully built a hybrid search system using Redis and OpenAI. This tutorial covered:

1.  Setting up a Redis Stack instance.
2.  Loading and preparing an e-commerce dataset.
3.  Creating a RediSearch index with vector and traditional fields.
4.  Generating embeddings and populating the database.
5.  Performing pure vector similarity searches.
6.  Executing powerful hybrid queries that combine semantic understanding with precise filtering on text, tags, and numeric ranges.

This architecture is ideal for applications like e-commerce search, content discovery, or any domain requiring nuanced, multi-faceted retrieval.