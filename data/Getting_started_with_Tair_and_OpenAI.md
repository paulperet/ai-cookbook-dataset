# Guide: Using Tair as a Vector Database for OpenAI Embeddings

This guide provides a step-by-step tutorial for using Tair, a cloud-native in-memory database, as a vector store for OpenAI embeddings. You will learn how to store precomputed embeddings, perform similarity searches, and build a retrieval system.

## What is Tair?

[Tair](https://www.alibabacloud.com/help/en/tair/latest/what-is-tair) is a cloud-native in-memory database service developed by Alibaba Cloud. It is compatible with open-source Redis and offers various data models and enterprise capabilities for real-time applications.

**TairVector** is a built-in data structure for high-performance vector storage and retrieval. It supports indexing algorithms like HNSW and Flat Search, along with distance metrics such as Euclidean distance and inner product. Key advantages include:

- In-memory storage with real-time index updates for low latency.
- Optimized data structures for efficient memory usage.
- A simple, out-of-the-box architecture without complex dependencies.

### Deployment Options
- Use the [Tair Cloud Vector Database](https://www.alibabacloud.com/product/tair) for a managed deployment.

## Prerequisites

Before starting, ensure you have:
1. A Tair cloud server instance.
2. An [OpenAI API key](https://beta.openai.com/account/api-keys).

## Setup

Install the required Python libraries:

```bash
pip install openai redis tair pandas wget
```

## Step 1: Configure Your OpenAI API Key

Your OpenAI API key is used to generate embeddings for text queries. Securely input it using `getpass`:

```python
import getpass
import openai

openai.api_key = getpass.getpass("Input your OpenAI API key: ")
```

## Step 2: Connect to Your Tair Instance

First, obtain your Tair connection URL. The format is: `redis://[[username]:[password]]@host:port/db`.

```python
TAIR_URL = getpass.getpass("Input your Tair URL: ")
```

Now, establish a connection using the Tair client:

```python
from tair import Tair as TairClient

url = TAIR_URL
client = TairClient.from_url(url)

# Test the connection
client.ping()
```

## Step 3: Download and Extract Sample Data

We'll use a precomputed dataset of Wikipedia article embeddings. Download and extract it:

```python
import wget
import zipfile
import os

# Download the embeddings file (~700 MB)
embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"
wget.download(embeddings_url)

# Extract the ZIP file
current_directory = os.getcwd()
zip_file_path = os.path.join(current_directory, "vector_database_wikipedia_articles_embedded.zip")
output_directory = os.path.join(current_directory, "../../data")

with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(output_directory)

# Verify the CSV file exists
file_name = "vector_database_wikipedia_articles_embedded.csv"
data_directory = os.path.join(current_directory, "../../data")
file_path = os.path.join(data_directory, file_name)

if os.path.exists(file_path):
    print(f"The file {file_name} exists in the data directory.")
else:
    print(f"The file {file_name} does not exist in the data directory.")
```

## Step 4: Create Vector Indexes in Tair

Tair stores vectors in indexes. Each index corresponds to a key containing a vector and associated metadata. We'll create two indexes: one for article titles and one for content.

```python
# Define index parameters
index = "openai_test"
embedding_dim = 1536  # Dimension of OpenAI embeddings
distance_type = "L2"  # Euclidean distance
index_type = "HNSW"   # Hierarchical Navigable Small World algorithm
data_type = "FLOAT32"

# Create indexes for title and content vectors
index_names = [index + "_title_vector", index + "_content_vector"]

for index_name in index_names:
    index_connection = client.tvs_get_index(index_name)
    if index_connection is not None:
        print(f"Index '{index_name}' already exists.")
    else:
        client.tvs_create_index(
            name=index_name,
            dim=embedding_dim,
            distance_type=distance_type,
            index_type=index_type,
            data_type=data_type
        )
        print(f"Index '{index_name}' created.")
```

## Step 5: Load Data into Tair

Load the precomputed embeddings from the CSV file and insert them into the Tair indexes.

```python
import pandas as pd
from ast import literal_eval

# Load the CSV file
csv_file_path = '../../data/vector_database_wikipedia_articles_embedded.csv'
article_df = pd.read_csv(csv_file_path)

# Convert string vectors back to lists
article_df['title_vector'] = article_df.title_vector.apply(literal_eval).values
article_df['content_vector'] = article_df.content_vector.apply(literal_eval).values

# Insert data into Tair indexes
for i in range(len(article_df)):
    # Add to title_vector index
    client.tvs_hset(
        index=index_names[0],
        key=article_df.id[i].item(),
        vector=article_df.title_vector[i],
        is_binary=False,
        **{"url": article_df.url[i], "title": article_df.title[i], "text": article_df.text[i]}
    )
    # Add to content_vector index
    client.tvs_hset(
        index=index_names[1],
        key=article_df.id[i].item(),
        vector=article_df.content_vector[i],
        is_binary=False,
        **{"url": article_df.url[i], "title": article_df.title[i], "text": article_df.text[i]}
    )

print("Data loading complete.")
```

Verify the data count in each index:

```python
for index_name in index_names:
    stats = client.tvs_get_index(index_name)
    count = int(stats["current_record_count"]) - int(stats["delete_record_count"])
    print(f"Record count in '{index_name}': {count}")
```

## Step 6: Perform Similarity Searches

Now, you can query the Tair indexes for the nearest neighbors to a given query. The function below generates an embedding for the query and searches the specified index.

```python
import openai
import numpy as np

def query_tair(client, query, vector_name="title_vector", top_k=5):
    """
    Query a Tair vector index for similar items.

    Args:
        client: Tair client instance.
        query: Text query string.
        vector_name: Type of vector to search ("title_vector" or "content_vector").
        top_k: Number of results to return.

    Returns:
        List of search results (key, distance).
    """
    # Generate embedding for the query
    embedded_query = openai.Embedding.create(
        input=query,
        model="text-embedding-3-small",
    )["data"][0]['embedding']
    embedded_query = np.array(embedded_query)

    # Perform k-NN search
    query_result = client.tvs_knnsearch(
        index=index + "_" + vector_name,
        k=top_k,
        vector=embedded_query
    )
    return query_result
```

### Example 1: Search by Title Vector

Search for articles related to "modern art in Europe" using title vectors:

```python
query_result = query_tair(client=client, query="modern art in Europe", vector_name="title_vector")

print("Top results for 'modern art in Europe':")
for i, (key, distance) in enumerate(query_result):
    title = client.tvs_hmget(index + "_content_vector", key.decode('utf-8'), "title")
    print(f"{i + 1}. {title[0].decode('utf-8')} (Distance: {round(distance, 3)})")
```

### Example 2: Search by Content Vector

Search for articles about "Famous battles in Scottish history" using content vectors:

```python
query_result = query_tair(client=client, query="Famous battles in Scottish history", vector_name="content_vector")

print("Top results for 'Famous battles in Scottish history':")
for i, (key, distance) in enumerate(query_result):
    title = client.tvs_hmget(index + "_content_vector", key.decode('utf-8'), "title")
    print(f"{i + 1}. {title[0].decode('utf-8')} (Distance: {round(distance, 3)})")
```

## Summary

You have successfully set up Tair as a vector database for OpenAI embeddings. The process involved:

1.  Connecting to a Tair instance.
2.  Creating vector indexes for title and content embeddings.
3.  Loading a dataset of precomputed embeddings.
4.  Performing similarity searches using both title and content vectors.

This setup can be extended to build applications like semantic search engines, recommendation systems, or Retrieval-Augmented Generation (RAG) pipelines.