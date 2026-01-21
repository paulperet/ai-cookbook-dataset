# Building a Vector Search Engine with Chroma and OpenAI Embeddings

This guide walks you through creating a semantic search engine using OpenAI embeddings and the Chroma vector database. You'll learn how to index and search through Wikipedia articles by both title and content.

## Prerequisites

Before starting, ensure you have Python installed. This tutorial uses several Python libraries that you'll need to install.

## Setup

First, install the required packages:

```bash
pip install openai chromadb wget numpy pandas
```

Now, import the necessary libraries and configure your environment:

```python
import openai
import pandas as pd
import os
import wget
from ast import literal_eval
import chromadb
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set your embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
```

## Step 1: Load and Prepare the Dataset

We'll use a pre-embedded Wikipedia dataset for this tutorial. The dataset contains article titles, content, and their corresponding vector embeddings.

### Download the Dataset

```python
# Download the embedded Wikipedia articles dataset
embeddings_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'
wget.download(embeddings_url)
```

### Extract and Load the Data

```python
import zipfile

# Extract the downloaded zip file
with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
    zip_ref.extractall("../data")

# Load the CSV file into a pandas DataFrame
article_df = pd.read_csv('../data/vector_database_wikipedia_articles_embedded.csv')
```

### Prepare the Data

The vectors are stored as strings in the CSV. We need to convert them back to Python lists and ensure the IDs are strings:

```python
# Convert string vectors back to lists
article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
article_df['content_vector'] = article_df.content_vector.apply(literal_eval)

# Ensure vector_id is a string
article_df['vector_id'] = article_df['vector_id'].apply(str)

# Verify the data structure
print(article_df.info(show_counts=True))
```

## Step 2: Initialize Chroma Client

Chroma is an open-source vector database that can run in-memory (ephemeral) or persist to disk. For this tutorial, we'll use the in-memory version:

```python
# Create an ephemeral Chroma client (in-memory)
chroma_client = chromadb.EphemeralClient()

# For persistent storage, use:
# chroma_client = chromadb.PersistentClient()
```

## Step 3: Configure OpenAI Embeddings

Chroma has built-in support for OpenAI's embedding functions. First, ensure your OpenAI API key is set:

```python
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Check if OpenAI API key is set
if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print("OPENAI_API_KEY is ready")
else:
    print("OPENAI_API_KEY environment variable not found")
    # Alternatively, set it directly:
    # os.environ["OPENAI_API_KEY"] = 'your-api-key-here'

# Create embedding function
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get('OPENAI_API_KEY'),
    model_name=EMBEDDING_MODEL
)
```

## Step 4: Create Collections

Collections in Chroma are similar to tables in traditional databases. We'll create two collections: one for article titles and one for article content:

```python
# Create collection for article content embeddings
wikipedia_content_collection = chroma_client.create_collection(
    name='wikipedia_content',
    embedding_function=embedding_function
)

# Create collection for article title embeddings
wikipedia_title_collection = chroma_client.create_collection(
    name='wikipedia_titles',
    embedding_function=embedding_function
)
```

## Step 5: Populate Collections

Now we'll add our embedded data to the collections. Each collection will store the vectors along with their corresponding IDs:

```python
# Add content vectors to the content collection
wikipedia_content_collection.add(
    ids=article_df.vector_id.tolist(),
    embeddings=article_df.content_vector.tolist(),
)

# Add title vectors to the title collection
wikipedia_title_collection.add(
    ids=article_df.vector_id.tolist(),
    embeddings=article_df.title_vector.tolist(),
)
```

## Step 6: Create a Search Function

Let's create a helper function to query collections and return formatted results:

```python
def query_collection(collection, query, max_results, dataframe):
    """
    Query a Chroma collection and return formatted results.
    
    Args:
        collection: Chroma collection to query
        query: Search query string
        max_results: Maximum number of results to return
        dataframe: Original DataFrame with article metadata
    
    Returns:
        DataFrame with search results
    """
    results = collection.query(
        query_texts=query,
        n_results=max_results,
        include=['distances']
    )
    
    # Create results DataFrame
    df = pd.DataFrame({
        'id': results['ids'][0],
        'score': results['distances'][0],
        'title': dataframe[dataframe.vector_id.isin(results['ids'][0])]['title'],
        'content': dataframe[dataframe.vector_id.isin(results['ids'][0])]['text'],
    })
    
    return df
```

## Step 7: Perform Semantic Searches

Now let's test our search engine with some example queries:

### Search by Article Title

```python
# Search for articles about modern art in Europe
title_query_result = query_collection(
    collection=wikipedia_title_collection,
    query="modern art in Europe",
    max_results=10,
    dataframe=article_df
)

print("Top 5 results for 'modern art in Europe':")
print(title_query_result.head())
```

### Search by Article Content

```python
# Search for articles about Scottish history battles
content_query_result = query_collection(
    collection=wikipedia_content_collection,
    query="Famous battles in Scottish history",
    max_results=10,
    dataframe=article_df
)

print("\nTop 5 results for 'Famous battles in Scottish history':")
print(content_query_result.head())
```

## Next Steps

Congratulations! You've successfully built a semantic search engine using Chroma and OpenAI embeddings. Here are some ways to extend this project:

1. **Add Metadata Filtering**: Chroma supports filtering by metadata. You could add publication dates, categories, or other metadata to your collections.

2. **Implement a REST API**: Wrap your search functionality in a Flask or FastAPI application.

3. **Add Hybrid Search**: Combine vector search with traditional keyword search for improved results.

4. **Deploy to Production**: Chroma supports persistent storage and can be deployed as a standalone server.

For more advanced features, refer to the [Chroma documentation](https://docs.trychroma.com/usage-guide) to learn about:
- Using where filters for metadata queries
- Updating and deleting data in collections
- Performance optimization techniques
- Deployment options for production environments

This foundation enables you to build various applications like chatbots, recommendation systems, and knowledge bases using semantic search capabilities.