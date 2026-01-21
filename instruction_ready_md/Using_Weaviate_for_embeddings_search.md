# Building a Semantic Search Engine with Weaviate

This guide walks you through creating a semantic search engine using Weaviate, a powerful vector database. You'll learn how to index pre-computed embeddings and perform similarity searches, enabling use cases like intelligent document retrieval, recommendation systems, and question-answering applications.

## Prerequisites

Before starting, ensure you have:
- An OpenAI API key set as an environment variable (`OPENAI_API_KEY`)
- Docker installed and running (for local Weaviate deployment)
- Basic Python knowledge

## Setup

First, install the required Python packages and import necessary libraries.

```bash
pip install weaviate-client wget pandas numpy openai
```

```python
import openai
import pandas as pd
import numpy as np
import os
import wget
import warnings
from ast import literal_eval

# Weaviate client library
import weaviate

# Set embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# Suppress warnings for cleaner output
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
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

# Load the CSV file
article_df = pd.read_csv('../data/vector_database_wikipedia_articles_embedded.csv')

# Preview the data
print(article_df.head())
```

### Prepare the Vectors

The dataset contains vector representations stored as strings. We need to convert them back to lists for use with Weaviate.

```python
# Convert string vectors back to lists
article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
article_df['content_vector'] = article_df.content_vector.apply(literal_eval)

# Ensure vector_id is a string
article_df['vector_id'] = article_df['vector_id'].apply(str)

# Verify the data structure
print(article_df.info(show_counts=True))
```

## Step 2: Set Up Weaviate

Weaviate offers both self-hosted and cloud-managed options. We'll cover both approaches.

### Option A: Local Deployment with Docker

1. Create a `docker-compose.yml` file with the following content:

```yaml
version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:1.19.6
    restart: on-failure:0
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 20
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-openai'
      ENABLE_MODULES: 'text2vec-openai'
      OPENAI_APIKEY: '${OPENAI_API_KEY}'
      CLUSTER_HOSTNAME: 'node1'
```

2. Start Weaviate:
```bash
docker-compose up -d
```

### Option B: Weaviate Cloud Service (WCS)

For a managed solution:
1. Create a free account at [Weaviate Cloud Service](https://console.weaviate.io/)
2. Create a Sandbox cluster with OIDC authentication disabled
3. Note your cluster URL (e.g., `https://your-project-name.weaviate.network`)

### Initialize the Weaviate Client

Choose one of the following connection methods based on your deployment:

```python
# Option 1: Connect to local Weaviate
client = weaviate.Client(
    url="http://localhost:8080",
    additional_headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    }
)

# Option 2: Connect to Weaviate Cloud Service
# client = weaviate.Client(
#     url="https://your-wcs-instance-name.weaviate.network",
#     additional_headers={
#         "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
#     }
# )

# Verify connection
print("Weaviate ready:", client.is_ready())
```

## Step 3: Create the Schema

Schemas in Weaviate define the structure of your data. We'll create an `Article` class to store our Wikipedia articles.

```python
# Clear any existing schemas
client.schema.delete_all()

# Define the Article schema
article_schema = {
    "class": "Article",
    "description": "A collection of articles",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
            "model": "ada",
            "modelVersion": "002",
            "type": "text"
        }
    },
    "properties": [
        {
            "name": "title",
            "description": "Title of the article",
            "dataType": ["string"]
        },
        {
            "name": "content",
            "description": "Contents of the article",
            "dataType": ["text"],
            "moduleConfig": { 
                "text2vec-openai": { 
                    "skip": True  # We'll provide our own vectors
                } 
            }
        }
    ]
}

# Create the schema
client.schema.create_class(article_schema)

# Verify the schema was created
print("Current schema:", client.schema.get())
```

## Step 4: Import Data with Batch Processing

Weaviate's batch API optimizes bulk data imports. We'll configure it for efficient data loading.

```python
# Configure batch settings
client.batch.configure(
    batch_size=100,      # Process 100 objects at a time
    dynamic=True,        # Adjust batch size dynamically
    timeout_retries=3,   # Retry failed operations
)

print("Uploading data with vectors to Article schema...")

counter = 0
with client.batch as batch:
    for _, row in article_df.iterrows():
        
        # Progress indicator
        if counter % 100 == 0:
            print(f"Import {counter} / {len(article_df)}")
        
        # Prepare object properties
        properties = {
            "title": row["title"],
            "content": row["text"]
        }
        
        # Use pre-computed title vector
        vector = row["title_vector"]
        
        # Add to batch
        batch.add_data_object(properties, "Article", None, vector)
        counter += 1

print(f"Import complete. Uploaded {len(article_df)} articles.")
```

### Verify Data Import

```python
# Count imported objects
result = (
    client.query.aggregate("Article")
    .with_fields("meta { count }")
    .do()
)
print("Object count:", result["data"]["Aggregate"]["Article"][0]["meta"]["count"])

# Inspect a sample article
test_article = (
    client.query
    .get("Article", ["title", "content", "_additional {id}"])
    .with_limit(1)
    .do()
)["data"]["Get"]["Article"][0]

print("\nSample article:")
print(f"ID: {test_article['_additional']['id']}")
print(f"Title: {test_article['title']}")
print(f"Content preview: {test_article['content'][:100]}...")
```

## Step 5: Perform Vector Similarity Searches

Now that our data is indexed, we can search using semantic similarity.

### Method 1: Manual Vector Search

This approach generates embeddings for queries using OpenAI's API and searches with those vectors.

```python
def query_weaviate(query, collection_name, top_k=20):
    """
    Search Weaviate using manually generated query embeddings.
    
    Args:
        query: Search query string
        collection_name: Weaviate class to search
        top_k: Number of results to return
    
    Returns:
        Query results with titles, content, and similarity scores
    """
    # Generate embedding for the query
    embedded_query = openai.Embedding.create(
        input=query,
        model=EMBEDDING_MODEL,
    )["data"][0]['embedding']
    
    # Prepare search parameters
    near_vector = {"vector": embedded_query}
    
    # Execute search
    query_result = (
        client.query
        .get(collection_name, ["title", "content", "_additional {certainty distance}"])
        .with_near_vector(near_vector)
        .with_limit(top_k)
        .do()
    )
    
    return query_result

# Example search
query_result = query_weaviate("modern art in Europe", "Article")

print("Search results for 'modern art in Europe':")
for i, article in enumerate(query_result["data"]["Get"]["Article"], 1):
    print(f"{i}. {article['title']} (Certainty: {round(article['_additional']['certainty'], 3)})")

# Another example
query_result = query_weaviate("Famous battles in Scottish history", "Article")

print("\nSearch results for 'Famous battles in Scottish history':")
for i, article in enumerate(query_result["data"]["Get"]["Article"], 1):
    print(f"{i}. {article['title']} (Certainty: {round(article['_additional']['certainty'], 3)})")
```

### Method 2: Weaviate's Built-in Vectorization

Weaviate can handle vectorization automatically using its OpenAI module. This simplifies the search process.

```python
def near_text_weaviate(query, collection_name, top_k=20):
    """
    Search Weaviate using text queries (Weaviate handles vectorization).
    
    Args:
        query: Search query string
        collection_name: Weaviate class to search
        top_k: Number of results to return
    
    Returns:
        Query results with titles, content, and similarity scores
    """
    nearText = {
        "concepts": [query],
        "distance": 0.7,  # Similarity threshold
    }
    
    properties = [
        "title", "content",
        "_additional {certainty distance}"
    ]
    
    query_result = (
        client.query
        .get(collection_name, properties)
        .with_near_text(nearText)
        .with_limit(top_k)
        .do()
    )["data"]["Get"][collection_name]
    
    print(f"Objects returned: {len(query_result)}")
    return query_result

# Example searches using Weaviate's vectorization
print("Using Weaviate's built-in vectorization:")

query_result = near_text_weaviate("modern art in Europe", "Article")
for i, article in enumerate(query_result, 1):
    print(f"{i}. {article['title']} (Distance: {round(article['_additional']['distance'], 3)})")

query_result = near_text_weaviate("Famous battles in Scottish history", "Article")
for i, article in enumerate(query_result, 1):
    print(f"{i}. {article['title']} (Distance: {round(article['_additional']['distance'], 3)})")
```

## Key Takeaways

1. **Vector Database Benefits**: Weaviate enables efficient storage and retrieval of vector embeddings, making semantic search scalable and production-ready.

2. **Flexible Deployment**: Choose between self-hosted (Docker) or managed (WCS) deployments based on your infrastructure needs.

3. **Two Search Approaches**:
   - **Manual Vector Search**: Generate embeddings externally and search using vectors
   - **Built-in Vectorization**: Let Weaviate handle embedding generation automatically

4. **Batch Processing**: Weaviate's batch API efficiently handles large data imports with configurable settings.

## Next Steps

- Explore Weaviate's [hybrid search](https://weaviate.io/developers/weaviate/search/hybrid) capabilities combining vector and keyword search
- Implement [filters](https://weaviate.io/developers/weaviate/api/graphql/filters) to refine search results
- Set up [authentication](https://weaviate.io/developers/weaviate/config-refs/authentication) for production deployments
- Experiment with different [vectorizers](https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules) for multimodal data

This foundation enables you to build sophisticated AI applications like intelligent chatbots, recommendation systems, and knowledge bases using semantic search capabilities.