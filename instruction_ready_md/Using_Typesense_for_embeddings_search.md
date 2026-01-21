# Building a Semantic Search Engine with Typesense and OpenAI Embeddings

This guide walks you through creating a semantic search engine using Typesense, an open-source vector database, and OpenAI's text embeddings. You'll learn how to index a dataset of Wikipedia articles and perform similarity searches based on meaning rather than just keywords.

## Prerequisites

Before starting, ensure you have:
- An OpenAI API key
- Docker installed and running (for local Typesense setup)
- Python 3.7 or higher

## Setup

First, install the required Python packages:

```bash
pip install typesense wget pandas numpy openai
```

Now, import the necessary libraries and configure your environment:

```python
import openai
import pandas as pd
import numpy as np
import os
import wget
from ast import literal_eval
import typesense
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set your embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
```

## Step 1: Load and Prepare the Dataset

We'll use a pre-embedded Wikipedia articles dataset for this tutorial.

### Download the Dataset

```python
# Download the embeddings dataset
embeddings_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'
wget.download(embeddings_url)
```

### Extract and Load the Data

```python
import zipfile

# Extract the downloaded zip file
with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
    zip_ref.extractall("../data")

# Load the CSV file into a DataFrame
article_df = pd.read_csv('../data/vector_database_wikipedia_articles_embedded.csv')

# Convert string vectors back to lists
article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
article_df['content_vector'] = article_df.content_vector.apply(literal_eval)

# Ensure vector_id is a string
article_df['vector_id'] = article_df['vector_id'].apply(str)

# Verify the data structure
print(f"Dataset loaded with {len(article_df)} articles")
print(f"Columns: {article_df.columns.tolist()}")
```

## Step 2: Set Up Typesense

Typesense can be run locally via Docker or used through Typesense Cloud. For this tutorial, we'll use a local Docker instance.

### Start Typesense with Docker

Create a `docker-compose.yml` file with the following content:

```yaml
version: '3.4'
services:
  typesense:
    image: typesense/typesense:0.24.1
    restart: on-failure
    ports:
      - "8108:8108"
    volumes:
      - ./typesense-data:/data
    command: '--data-dir /data --api-key=xyz --enable-cors'
```

Then start the Typesense service:

```bash
docker-compose up -d
```

### Initialize the Python Client

```python
# Configure the Typesense client
typesense_client = typesense.Client({
    "nodes": [{
        "host": "localhost",  # Use your Typesense Cloud hostname if using cloud
        "port": "8108",       # Use 443 for Typesense Cloud
        "protocol": "http"    # Use https for Typesense Cloud
    }],
    "api_key": "xyz",  # Default API key from docker-compose.yml
    "connection_timeout_seconds": 60
})
```

## Step 3: Create a Collection and Index Data

In Typesense, a collection is similar to a table in traditional databases. We'll create one to store our article embeddings.

### Define the Collection Schema

```python
# Delete existing collection if it exists
try:
    typesense_client.collections['wikipedia_articles'].delete()
    print("Deleted existing collection")
except Exception as e:
    print("No existing collection found")

# Create a new collection schema
schema = {
    "name": "wikipedia_articles",
    "fields": [
        {
            "name": "content_vector",
            "type": "float[]",
            "num_dim": len(article_df['content_vector'][0])
        },
        {
            "name": "title_vector",
            "type": "float[]",
            "num_dim": len(article_df['title_vector'][0])
        }
    ]
}

# Create the collection
create_response = typesense_client.collections.create(schema)
print(f"Created collection: {create_response['name']}")
```

### Import Documents in Batches

```python
print("Indexing vectors in Typesense...")

document_counter = 0
documents_batch = []

# Process each article and add it to the collection
for _, row in article_df.iterrows():
    document = {
        "title_vector": row["title_vector"],
        "content_vector": row["content_vector"],
        "title": row["title"],
        "content": row["text"],
    }
    documents_batch.append(document)
    document_counter += 1

    # Import in batches of 100 for better performance
    if document_counter % 100 == 0 or document_counter == len(article_df):
        response = typesense_client.collections['wikipedia_articles'].documents.import_(documents_batch)
        documents_batch = []
        print(f"Processed {document_counter} / {len(article_df)} articles")

print(f"Successfully imported {len(article_df)} articles into Typesense")

# Verify the import
collection = typesense_client.collections['wikipedia_articles'].retrieve()
print(f'Collection "{collection["name"]}" has {collection["num_documents"]} documents')
```

## Step 4: Implement Semantic Search

Now that our data is indexed, we can create a search function that finds articles semantically similar to a user's query.

### Create the Search Function

```python
def query_typesense(query, field='title', top_k=20):
    """
    Search for documents similar to the query using vector similarity.
    
    Args:
        query (str): The search query
        field (str): Which vector field to search ('title' or 'content')
        top_k (int): Number of results to return
    
    Returns:
        dict: Search results from Typesense
    """
    # Set your OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Generate embedding for the query
    embedded_query = openai.Embedding.create(
        input=query,
        model=EMBEDDING_MODEL,
    )['data'][0]['embedding']
    
    # Convert the embedding list to a comma-separated string
    vector_string = ','.join(str(v) for v in embedded_query)
    
    # Perform the vector search
    typesense_results = typesense_client.multi_search.perform({
        "searches": [{
            "q": "*",
            "collection": "wikipedia_articles",
            "vector_query": f"{field}_vector:([{vector_string}], k:{top_k})"
        }]
    }, {})
    
    return typesense_results
```

### Test the Search Function

Let's run some example searches to see our semantic search in action:

```python
# Search by title similarity
print("Searching for articles with titles similar to 'modern art in Europe':")
query_results = query_typesense('modern art in Europe', 'title')

for i, hit in enumerate(query_results['results'][0]['hits']):
    document = hit["document"]
    vector_distance = hit["vector_distance"]
    print(f'{i + 1}. {document["title"]} (Distance: {vector_distance:.4f})')
```

```python
# Search by content similarity
print("\nSearching for articles with content similar to 'Famous battles in Scottish history':")
query_results = query_typesense('Famous battles in Scottish history', 'content')

for i, hit in enumerate(query_results['results'][0]['hits']):
    document = hit["document"]
    vector_distance = hit["vector_distance"]
    print(f'{i + 1}. {document["title"]} (Distance: {vector_distance:.4f})')
```

## Step 5: Advanced Usage and Customization

### Combining Vector Search with Filters

Typesense allows you to combine vector similarity with traditional filtering. Here's an example:

```python
def filtered_vector_search(query, filter_condition, field='title', top_k=10):
    """
    Perform a vector search with additional filtering.
    
    Args:
        query (str): The search query
        filter_condition (str): Typesense filter expression
        field (str): Which vector field to search
        top_k (int): Number of results to return
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Generate embedding for the query
    embedded_query = openai.Embedding.create(
        input=query,
        model=EMBEDDING_MODEL,
    )['data'][0]['embedding']
    
    vector_string = ','.join(str(v) for v in embedded_query)
    
    # Perform filtered vector search
    typesense_results = typesense_client.multi_search.perform({
        "searches": [{
            "q": "*",
            "collection": "wikipedia_articles",
            "vector_query": f"{field}_vector:([{vector_string}], k:{top_k})",
            "filter_by": filter_condition
        }]
    }, {})
    
    return typesense_results
```

### Search with Multiple Vector Fields

You can search across multiple vector fields simultaneously:

```python
def multi_field_search(query, top_k=10):
    """
    Search across both title and content vectors.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Generate embedding for the query
    embedded_query = openai.Embedding.create(
        input=query,
        model=EMBEDDING_MODEL,
    )['data'][0]['embedding']
    
    vector_string = ','.join(str(v) for v in embedded_query)
    
    # Search across both fields
    typesense_results = typesense_client.multi_search.perform({
        "searches": [
            {
                "q": "*",
                "collection": "wikipedia_articles",
                "vector_query": f"title_vector:([{vector_string}], k:{top_k})"
            },
            {
                "q": "*",
                "collection": "wikipedia_articles",
                "vector_query": f"content_vector:([{vector_string}], k:{top_k})"
            }
        ]
    }, {})
    
    return typesense_results
```

## Conclusion

You've successfully built a semantic search engine using Typesense and OpenAI embeddings. Key takeaways:

1. **Vector databases** like Typesense enable efficient similarity search on embedding vectors
2. **Semantic search** finds documents based on meaning rather than just keyword matching
3. **Batch processing** is essential when indexing large datasets
4. **Combined search** allows you to mix vector similarity with traditional filtering

This foundation can be extended to build more complex applications like:
- Question-answering systems
- Recommendation engines
- Content discovery platforms
- Chatbots with contextual memory

To clean up your local Typesense instance when you're done:

```bash
docker-compose down
```

For production deployments, consider using [Typesense Cloud](https://cloud.typesense.org) for managed hosting with automatic scaling and backups.