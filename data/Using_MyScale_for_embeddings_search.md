# Building a Semantic Search Engine with MyScale

This guide walks you through creating a semantic search engine using OpenAI embeddings and the MyScale vector database. You'll learn how to index a dataset of Wikipedia articles and perform similarity searches to find relevant content based on natural language queries.

## Prerequisites

Ensure you have the following installed:

```bash
pip install clickhouse-connect wget pandas numpy openai tqdm
```

You'll also need:
- An OpenAI API key (set as an environment variable `OPENAI_API_KEY`)
- A MyScale cluster (get your host, username, and password from the [MyScale Console](https://console.myscale.com))

## 1. Setup and Imports

First, import the necessary libraries and configure your environment.

```python
import openai
import pandas as pd
import numpy as np
import os
import wget
from ast import literal_eval
import clickhouse_connect
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set your embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
```

## 2. Load and Prepare the Dataset

We'll use a pre-embedded Wikipedia articles dataset. This dataset contains article content along with their vector embeddings.

```python
# Download the dataset
embeddings_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'
wget.download(embeddings_url)

# Extract the downloaded file
import zipfile
with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
    zip_ref.extractall("../data")

# Load the dataset into a DataFrame
article_df = pd.read_csv('../data/vector_database_wikipedia_articles_embedded.csv')

# Convert string vectors back to Python lists
article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
article_df['content_vector'] = article_df.content_vector.apply(literal_eval)

# Ensure vector_id is a string
article_df['vector_id'] = article_df['vector_id'].apply(str)

# Display dataset information
print(f"Dataset loaded with {len(article_df)} articles")
print(article_df.info(show_counts=True))
```

## 3. Connect to MyScale

Establish a connection to your MyScale cluster using the credentials from your MyScale Console.

```python
# Replace with your actual cluster credentials
client = clickhouse_connect.get_client(
    host='YOUR_CLUSTER_HOST',
    port=8443,
    username='YOUR_USERNAME',
    password='YOUR_CLUSTER_PASSWORD'
)

print("Successfully connected to MyScale")
```

## 4. Create and Populate the Vector Index

Now, create a table in MyScale to store your articles with a vector index for efficient similarity search.

```python
# Determine the embedding dimension
embedding_len = len(article_df['content_vector'][0])
print(f"Embedding dimension: {embedding_len}")

# Create the articles table with vector index
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

print("Table 'articles' created successfully")

# Prepare data for insertion
article_df = article_df[['id', 'url', 'title', 'text', 'content_vector']]
data = article_df.to_records(index=False).tolist()
column_names = article_df.columns.tolist()

# Insert data in batches for better performance
from tqdm.auto import tqdm

batch_size = 100
total_records = len(article_df)

for i in tqdm(range(0, total_records, batch_size)):
    i_end = min(i + batch_size, total_records)
    client.insert("default.articles", data[i:i_end], column_names=column_names)

print("Data insertion complete")
```

## 5. Verify the Index Build

Before searching, ensure the vector index has been built successfully.

```python
# Check the number of inserted records
article_count = client.command('SELECT count(*) FROM default.articles')
print(f"Articles indexed: {article_count}")

# Check vector index status
get_index_status = "SELECT status FROM system.vector_indices WHERE name='article_content_index'"
index_status = client.command(get_index_status)
print(f"Vector index status: {index_status}")

# Wait for index to be ready if needed
if index_status != 'Built':
    print("Waiting for index to build...")
    # Add a small delay or check again
```

## 6. Perform Semantic Search

Now you can query your indexed data using natural language. The search converts your query to an embedding and finds the most similar articles.

```python
def search_articles(query: str, top_k: int = 10):
    """Search for articles similar to the query."""
    
    # Generate embedding for the query
    embed = openai.Embedding.create(
        input=query,
        model=EMBEDDING_MODEL,
    )["data"][0]["embedding"]
    
    # Execute vector similarity search
    results = client.query(f"""
    SELECT id, url, title, distance(content_vector, {embed}) as dist
    FROM default.articles
    ORDER BY dist
    LIMIT {top_k}
    """)
    
    # Display results
    print(f"\nTop {top_k} results for: '{query}'\n")
    for i, r in enumerate(results.named_results()):
        print(f"{i+1}. {r['title']} (distance: {r['dist']:.4f})")
    
    return results.named_results()

# Example search
search_results = search_articles("Famous battles in Scottish history")
```

## 7. Additional Search Examples

Try different queries to explore the semantic search capabilities:

```python
# Search for technology-related articles
tech_results = search_articles("Advances in artificial intelligence", top_k=5)

# Search for historical figures
history_results = search_articles("Renaissance artists and their works", top_k=8)

# Search for scientific concepts
science_results = search_articles("Quantum computing principles", top_k=6)
```

## Next Steps

You've successfully built a semantic search engine with MyScale! Here are some ways to extend this project:

1. **Add Filtering**: Combine vector search with SQL WHERE clauses to filter by date, category, or other metadata
2. **Build a Chatbot**: Use these search results as context for a RAG (Retrieval-Augmented Generation) system
3. **Optimize Performance**: Experiment with different index types and parameters for your specific use case
4. **Scale Up**: Load larger datasets and monitor query performance as your data grows

For more advanced features and SQL examples, refer to the [MyScale documentation](https://docs.myscale.com/).