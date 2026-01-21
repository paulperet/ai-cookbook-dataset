# Building a Semantic Search Engine with Qdrant

This guide walks you through creating a semantic search engine using Qdrant, a high-performance vector database. You'll learn how to index and search embedded Wikipedia articles, enabling you to build production-ready applications like chatbots, recommendation systems, and intelligent search.

## Prerequisites

Ensure you have the following installed:
- Python 3.8+
- Docker (for running Qdrant locally)
- An OpenAI API key (for generating embeddings)

## Setup

First, install the required Python packages and configure your environment.

```bash
pip install qdrant-client openai pandas tqdm wget
```

```python
import openai
import pandas as pd
from ast import literal_eval
import qdrant_client
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set your embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"
```

## Step 1: Load and Prepare the Dataset

We'll use a pre-embedded Wikipedia dataset. This dataset contains article titles, content, and their corresponding vector embeddings.

```python
import wget
import zipfile

# Download the dataset
embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"
wget.download(embeddings_url)

# Extract the contents
with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
    zip_ref.extractall("../data")

# Load the CSV file
article_df = pd.read_csv('../data/vector_database_wikipedia_articles_embedded.csv')

# Convert vector strings back to lists
article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
article_df['content_vector'] = article_df.content_vector.apply(literal_eval)

# Ensure vector_id is a string
article_df['vector_id'] = article_df['vector_id'].apply(str)

# Verify the data structure
print(f"Dataset loaded with {len(article_df)} articles")
print(article_df.info(show_counts=True))
```

## Step 2: Set Up Qdrant Locally

Qdrant runs as a Docker container. We'll start a local instance and connect to it.

```bash
# Create a docker-compose.yml file with the following content:
# version: '3.8'
# services:
#   qdrant:
#     image: qdrant/qdrant:latest
#     ports:
#       - "6333:6333"
#     volumes:
#       - ./qdrant_storage:/qdrant/storage

# Start the Qdrant container
docker-compose up -d
```

```python
# Connect to the Qdrant instance
qdrant = qdrant_client.QdrantClient(host="localhost", port=6333)

# Verify the connection
collections = qdrant.get_collections()
print(f"Connected to Qdrant. Available collections: {collections}")
```

## Step 3: Create and Configure the Collection

In Qdrant, data is organized into collections. Each item in a collection contains vectors and optional metadata (payload).

```python
from qdrant_client.http import models as rest

# Determine vector dimensions from the first article
vector_size = len(article_df['content_vector'][0])

# Create the collection with separate vectors for titles and content
qdrant.recreate_collection(
    collection_name='Articles',
    vectors_config={
        'title': rest.VectorParams(
            distance=rest.Distance.COSINE,
            size=vector_size,
        ),
        'content': rest.VectorParams(
            distance=rest.Distance.COSINE,
            size=vector_size,
        ),
    }
)

print(f"Collection 'Articles' created with vector size {vector_size}")
```

## Step 4: Index the Articles

Now we'll populate the collection with our article data, including both vectors and metadata.

```python
from qdrant_client.models import PointStruct
from tqdm import tqdm

# Insert articles with progress tracking
for index, row in tqdm(article_df.iterrows(), desc="Indexing articles", total=len(article_df)):
    try:
        qdrant.upsert(
            collection_name='Articles',
            points=[
                PointStruct(
                    id=index,
                    vector={
                        'title': row['title_vector'],
                        'content': row['content_vector']
                    },
                    payload={
                        'id': row['id'],
                        'title': row['title'],
                        'url': row['url']
                    }
                )
            ]
        )
    except Exception as e:
        print(f"Failed to index row {index}: {e}")

# Verify the insertion
count = qdrant.count(collection_name='Articles')
print(f"Successfully indexed {count.count} articles")
```

## Step 5: Implement Semantic Search

Create a search function that converts queries into embeddings and finds the most relevant articles.

```python
def query_qdrant(query, collection_name, vector_name='title', top_k=20):
    """
    Search for articles similar to the query.
    
    Args:
        query: The search query string
        collection_name: Name of the Qdrant collection
        vector_name: Which vector to search ('title' or 'content')
        top_k: Number of results to return
    
    Returns:
        List of search results with scores
    """
    # Generate embedding for the query
    embedded_query = openai.embeddings.create(
        input=query,
        model=EMBEDDING_MODEL,
    ).data[0].embedding
    
    # Search the collection
    query_results = qdrant.search(
        collection_name=collection_name,
        query_vector=(vector_name, embedded_query),
        limit=top_k,
        query_filter=None
    )
    
    return query_results
```

## Step 6: Test Your Search Engine

Let's run some searches to see the system in action.

```python
# Search by article titles
print("Searching for 'modern art in Europe' by title:")
query_results = query_qdrant('modern art in Europe', 'Articles', 'title')

for i, article in enumerate(query_results):
    print(f'{i + 1}. {article.payload["title"]}')
    print(f'   URL: {article.payload["url"]}')
    print(f'   Score: {round(article.score, 3)}\n')

# Search by article content
print("\nSearching for 'Famous battles in Scottish history' by content:")
query_results = query_qdrant('Famous battles in Scottish history', 'Articles', 'content')

for i, article in enumerate(query_results[:5]):  # Show top 5 results
    print(f'{i + 1}. {article.payload["title"]}')
    print(f'   URL: {article.payload["url"]}')
    print(f'   Score: {round(article.score, 3)}\n')
```

## Next Steps

Congratulations! You've built a functional semantic search engine. Here are some ways to extend this system:

1. **Add Filters**: Implement metadata filtering (e.g., by date, category)
2. **Hybrid Search**: Combine vector search with traditional keyword matching
3. **Production Deployment**: Move from local Docker to Qdrant Cloud or a managed instance
4. **Real-time Updates**: Implement a pipeline for adding new articles dynamically
5. **API Wrapper**: Create a REST API around your search function

## Cleaning Up

When you're done, remember to stop the Qdrant container:

```bash
docker-compose down
```

This tutorial demonstrated the core concepts of vector databases and semantic search. You can now adapt this foundation to build more complex AI-powered applications.