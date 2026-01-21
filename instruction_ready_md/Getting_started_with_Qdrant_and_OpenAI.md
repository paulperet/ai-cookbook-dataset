# Guide: Using Qdrant as a Vector Database for OpenAI Embeddings

This guide walks you through using **Qdrant** as a high-performance vector database to store and search OpenAI embeddings. You'll learn how to set up a local Qdrant instance, load precomputed embeddings, index them, and perform semantic searches.

## What is Qdrant?
[Qdrant](https://qdrant.tech) is an open-source vector database written in Rust. It stores vector embeddings along with metadata (payloads) and provides efficient nearest-neighbor search with built-in filtering capabilities. Qdrant can be deployed locally, on Kubernetes, or via [Qdrant Cloud](https://cloud.qdrant.io/).

## Prerequisites

Before you begin, ensure you have:

1. **Docker** installed and running.
2. An **OpenAI API key** (available from [OpenAI's platform](https://platform.openai.com/settings/organization/api-keys)).
3. Python 3.7+.

## Step 1: Set Up Your Environment

### 1.1 Start the Qdrant Server
Launch a local Qdrant instance using Docker Compose. Create a `docker-compose.yaml` file with the following content:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

Then, start the service:

```bash
docker compose up -d
```

Verify the server is running:

```bash
curl http://localhost:6333
```

You should see a JSON response confirming the service is operational.

### 1.2 Install Required Python Packages
Install the necessary libraries:

```bash
pip install openai qdrant-client pandas wget
```

### 1.3 Configure Your OpenAI API Key
Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

To verify the key is set correctly in your Python environment:

```python
import os

if os.getenv("OPENAI_API_KEY") is not None:
    print("OPENAI_API_KEY is ready")
else:
    print("OPENAI_API_KEY environment variable not found")
```

## Step 2: Connect to Qdrant

Initialize the Qdrant client to interact with your local server:

```python
import qdrant_client

client = qdrant_client.QdrantClient(
    host="localhost",
    prefer_grpc=True,  # gRPC is faster for large data transfers
)
```

Test the connection by listing existing collections (initially, there will be none):

```python
client.get_collections()
```

## Step 3: Load the Dataset

We'll use a precomputed dataset of Wikipedia article embeddings to save time and API credits.

### 3.1 Download the Embeddings File
Download the compressed dataset (approximately 700 MB):

```python
import wget

embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"
wget.download(embeddings_url)
```

### 3.2 Extract the Data
Extract the contents of the ZIP file:

```python
import zipfile

with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
    zip_ref.extractall("../data")
```

### 3.3 Load the CSV into a DataFrame
Load the CSV file and convert the stringified vectors back into Python lists:

```python
import pandas as pd
from ast import literal_eval

article_df = pd.read_csv('../data/vector_database_wikipedia_articles_embedded.csv')
article_df["title_vector"] = article_df.title_vector.apply(literal_eval)
article_df["content_vector"] = article_df.content_vector.apply(literal_eval)

article_df.head()
```

## Step 4: Index Data in Qdrant

Qdrant organizes data into **collections**. Each point in a collection has one or more vectors and optional metadata (payload).

### 4.1 Create a Collection
Create a collection named `"Articles"` with two vector fields: `title` and `content`. Both will use cosine distance for similarity measurement.

```python
from qdrant_client.http import models as rest

vector_size = len(article_df["content_vector"][0])  # Dynamically get vector dimension

client.create_collection(
    collection_name="Articles",
    vectors_config={
        "title": rest.VectorParams(
            distance=rest.Distance.COSINE,
            size=vector_size,
        ),
        "content": rest.VectorParams(
            distance=rest.Distance.COSINE,
            size=vector_size,
        ),
    }
)
```

### 4.2 Insert the Embeddings
Insert all articles into the collection. Each point includes both title and content vectors, along with the full article metadata as a payload.

```python
client.upsert(
    collection_name="Articles",
    points=[
        rest.PointStruct(
            id=k,
            vector={
                "title": v["title_vector"],
                "content": v["content_vector"],
            },
            payload=v.to_dict(),
        )
        for k, v in article_df.iterrows()
    ],
)
```

### 4.3 Verify the Insertion
Check that all points have been stored:

```python
client.count(collection_name="Articles")
```

## Step 5: Search the Vector Database

Now you can query the collection using natural language. The query will be embedded using the same OpenAI model (`text-embedding-ada-002`) and compared against the stored vectors.

### 5.1 Define a Search Function
Create a helper function to handle embedding generation and Qdrant search:

```python
from openai import OpenAI

openai_client = OpenAI()

def query_qdrant(query, collection_name, vector_name="title", top_k=20):
    # Convert the query text into an embedding vector
    embedded_query = openai_client.embeddings.create(
        input=query,
        model="text-embedding-ada-002",
    ).data[0].embedding

    # Search Qdrant for the nearest vectors
    query_results = client.search(
        collection_name=collection_name,
        query_vector=(vector_name, embedded_query),
        limit=top_k,
    )

    return query_results
```

### 5.2 Perform a Title-Based Search
Search for articles related to "modern art in Europe" using the title vectors:

```python
query_results = query_qdrant("modern art in Europe", "Articles")
for i, article in enumerate(query_results):
    print(f"{i + 1}. {article.payload['title']} (Score: {round(article.score, 3)})")
```

### 5.3 Perform a Content-Based Search
Search for "Famous battles in Scottish history" using the more detailed content vectors:

```python
query_results = query_qdrant("Famous battles in Scottish history", "Articles", "content")
for i, article in enumerate(query_results):
    print(f"{i + 1}. {article.payload['title']} (Score: {round(article.score, 3)})")
```

## Conclusion

You've successfully set up Qdrant as a vector database for OpenAI embeddings. You can now:

- Store large sets of precomputed embeddings efficiently.
- Perform fast, accurate semantic searches using either title or content vectors.
- Scale your setup by deploying Qdrant on Kubernetes or using Qdrant Cloud for production workloads.

To explore further, refer to the [Qdrant documentation](https://qdrant.tech/documentation/) and experiment with filtering, payload indexing, and hybrid search features.