# Guide: Using MyScale as a Vector Database for OpenAI Embeddings

This guide provides a step-by-step tutorial for using MyScale as a high-performance vector database with OpenAI embeddings. You will learn how to store precomputed embeddings, generate new ones from text queries, and perform efficient vector similarity searches.

## What is MyScale?

[MyScale](https://myscale.com) is a managed database built on ClickHouse that combines vector search with full SQL analytics. It enables fast, joint queries on both structured data and vector embeddings, making it ideal for AI applications like retrieval-augmented generation (RAG) and semantic search.

## Prerequisites

Before you begin, ensure you have:

1.  A deployed MyScale cluster. Follow the [MyScale Quickstart](https://docs.myscale.com/en/quickstart/) to create one.
2.  An [OpenAI API key](https://platform.openai.com/account/api-keys) for generating embeddings.

## Step 1: Install Required Libraries

Install the necessary Python packages using pip.

```bash
pip install openai clickhouse-connect wget pandas tqdm
```

## Step 2: Set Up Your OpenAI API Key

Import the OpenAI library and configure it with your API key to authenticate requests.

```python
import openai

# Replace with your actual OpenAI API key
openai.api_key = "OPENAI_API_KEY"

# Optional: Verify authentication by listing available engines
openai.Engine.list()
```

## Step 3: Connect to Your MyScale Cluster

Retrieve your cluster connection details (host, username, password) from the [MyScale Console](https://console.myscale.com). Use the `clickhouse_connect` library to establish a connection.

```python
import clickhouse_connect

# Initialize the client with your cluster credentials
client = clickhouse_connect.get_client(
    host='YOUR_CLUSTER_HOST',
    port=8443,
    username='YOUR_USERNAME',
    password='YOUR_CLUSTER_PASSWORD'
)
```

## Step 4: Load the Sample Dataset

We'll use a sample dataset of Wikipedia article embeddings provided by OpenAI. First, download the compressed file.

```python
import wget

embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"

# The file is ~700 MB; download may take a moment
wget.download(embeddings_url)
```

Next, extract the downloaded ZIP file.

```python
import zipfile

with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
    zip_ref.extractall("../data")
```

Now, load the extracted CSV file into a Pandas DataFrame. The embeddings are stored as string representations of lists, so we need to convert them back to Python lists.

```python
import pandas as pd
from ast import literal_eval

# Read data from the CSV file
article_df = pd.read_csv('../data/vector_database_wikipedia_articles_embedded.csv')

# Select relevant columns
article_df = article_df[['id', 'url', 'title', 'text', 'content_vector']]

# Convert the string representation of vectors back into lists
article_df["content_vector"] = article_df.content_vector.apply(literal_eval)

# Inspect the first few rows
article_df.head()
```

## Step 5: Create a Table and Index the Data

We will create a table in MyScale to store our data. This table includes a vector column and a vector index to enable fast similarity search.

First, determine the dimensionality of the embeddings (OpenAI's `text-embedding-3-small` produces 1536-dimensional vectors).

```python
# Get the length of the first embedding vector
embedding_len = len(article_df['content_vector'][0])  # Should be 1536
print(f"Embedding dimension: {embedding_len}")
```

Now, execute an SQL command to create the table. The `VECTOR INDEX` clause creates an HNSW index using cosine distance for similarity search.

```python
# Create the articles table with a vector index
client.command(f"""
CREATE TABLE IF NOT EXISTS default.articles
(
    id UInt64,
    url String,
    title String,
    text String,
    content_vector Array(Float32),
    CONSTRAINT cons_vector_len CHECK length(content_vector) = {embedding_len},
    VECTOR INDEX article_content_index content_vector TYPE HNSWFLAT('metric_type=Cosine')
)
ENGINE = MergeTree ORDER BY id
""")
```

With the table created, we can insert the data from our DataFrame. We'll do this in batches for efficiency.

```python
from tqdm.auto import tqdm

batch_size = 100
total_records = len(article_df)

# Prepare data and column names for insertion
data = article_df.to_records(index=False).tolist()
column_names = article_df.columns.tolist()

# Insert data in batches with a progress bar
for i in tqdm(range(0, total_records, batch_size)):
    i_end = min(i + batch_size, total_records)
    client.insert("default.articles", data[i:i_end], column_names=column_names)
```

## Step 6: Verify Data Insertion and Index Status

Before searching, confirm that the data was inserted successfully and that the vector index has been built. Index building happens asynchronously.

```python
# Check the number of records in the table
record_count = client.command('SELECT count(*) FROM default.articles')
print(f"Articles count: {record_count}")

# Check the build status of the vector index
get_index_status = "SELECT status FROM system.vector_indices WHERE name='article_content_index'"
index_status = client.command(get_index_status)
print(f"Index build status: {index_status}")

# Ensure the status is 'Built' before proceeding
```

## Step 7: Perform a Vector Search

Now, let's perform a semantic search. We'll convert a text query into an embedding using the OpenAI API, then use MyScale to find the most similar articles.

First, generate an embedding for your search query.

```python
query = "Famous battles in Scottish history"

# Create an embedding vector from the user query
embed_response = openai.Embedding.create(
    input=query,
    model="text-embedding-3-small",
)
query_embedding = embed_response["data"][0]["embedding"]
```

Next, execute a search query in MyScale. We use the `distance` function with cosine metric (as defined in our index) to find the nearest neighbors.

```python
top_k = 10

# Query the database for the top K most similar articles
results = client.query(f"""
SELECT id, url, title, distance(content_vector, {query_embedding}) as dist
FROM default.articles
ORDER BY dist
LIMIT {top_k}
""")

# Display the results
for i, r in enumerate(results.named_results()):
    print(f"{i+1}. {r['title']} (Distance: {r['dist']:.4f})")
```

This will output a ranked list of Wikipedia article titles most semantically related to "Famous battles in Scottish history," along with their similarity scores.

## Summary

You have successfully set up MyScale as a vector database for OpenAI embeddings. The process involved:
1.  Connecting to a MyScale cluster.
2.  Loading and storing a dataset of precomputed embeddings.
3.  Creating a vector index for fast similarity search.
4.  Generating an embedding for a text query and using it to perform a nearest-neighbor search.

This foundation enables you to build powerful applications like semantic search engines, recommendation systems, or RAG pipelines by combining MyScale's vector search capabilities with OpenAI's powerful embedding models.