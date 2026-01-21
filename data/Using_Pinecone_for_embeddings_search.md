# Building a Semantic Search Engine with Pinecone

This guide walks you through creating a semantic search engine using OpenAI embeddings and Pinecone, a managed vector database. You'll learn how to embed text data, store it in a vector database, and perform similarity searches—a foundational skill for building production AI applications like chatbots, recommendation systems, and question-answering tools.

## Prerequisites

Before you begin, ensure you have:
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Pinecone account](https://www.pinecone.io/) and API key
- Python 3.7 or higher installed

## Setup

First, install the required Python packages and import the necessary libraries.

```bash
pip install pinecone-client wget openai pandas numpy
```

```python
import openai
import pinecone
import pandas as pd
import numpy as np
import os
import wget
from ast import literal_eval
from typing import Iterator
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set your embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
```

## Step 1: Load and Prepare the Dataset

You'll use a pre-embedded dataset of Wikipedia articles. This dataset contains both article titles and content, each already converted into vector embeddings.

```python
# Download the dataset
embeddings_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'
wget.download(embeddings_url)

# Extract the downloaded file
import zipfile
with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
    zip_ref.extractall("../data")

# Load the dataset into a pandas DataFrame
article_df = pd.read_csv('../data/vector_database_wikipedia_articles_embedded.csv')

# Convert string vectors back to Python lists
article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
article_df['content_vector'] = article_df.content_vector.apply(literal_eval)

# Ensure the ID column is a string
article_df['vector_id'] = article_df['vector_id'].apply(str)

# Verify the data structure
print(article_df.info(show_counts=True))
```

## Step 2: Initialize Pinecone

Connect to Pinecone using your API key. Make sure to set your `PINECONE_API_KEY` as an environment variable.

```python
# Initialize Pinecone client
api_key = os.getenv("PINECONE_API_KEY")
pinecone.init(api_key=api_key)
```

## Step 3: Create a Batch Generator

To efficiently insert data into Pinecone, create a helper class that chunks the DataFrame into manageable batches.

```python
class BatchGenerator:
    """Generates batches from a DataFrame for efficient upsert operations."""
    
    def __init__(self, batch_size: int = 300) -> None:
        self.batch_size = batch_size
    
    def to_batches(self, df: pd.DataFrame) -> Iterator[pd.DataFrame]:
        splits = self.splits_num(df.shape[0])
        if splits <= 1:
            yield df
        else:
            for chunk in np.array_split(df, splits):
                yield chunk

    def splits_num(self, elements: int) -> int:
        return round(elements / self.batch_size)
    
    __call__ = to_batches

# Instantiate the batch generator
df_batcher = BatchGenerator(300)
```

## Step 4: Create a Pinecone Index

Create a new index in Pinecone. If an index with the same name already exists, it will be deleted first.

```python
index_name = 'wikipedia-articles'

# Delete existing index if it exists
if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

# Create a new index
# The dimension must match the length of your embedding vectors
pinecone.create_index(name=index_name, dimension=len(article_df['content_vector'][0]))

# Connect to the index
index = pinecone.Index(index_name=index_name)

# Confirm the index was created
print("Available indexes:", pinecone.list_indexes())
```

## Step 5: Upload Vectors to Pinecone

Upload the article content and title vectors to separate namespaces within the index. Namespaces allow you to partition an index for different use cases.

```python
# Upload content vectors to the 'content' namespace
print("Uploading vectors to content namespace...")
for batch_df in df_batcher(article_df):
    index.upsert(
        vectors=zip(batch_df.vector_id, batch_df.content_vector),
        namespace='content'
    )

# Upload title vectors to the 'title' namespace
print("Uploading vectors to title namespace...")
for batch_df in df_batcher(article_df):
    index.upsert(
        vectors=zip(batch_df.vector_id, batch_df.title_vector),
        namespace='title'
    )

# Verify the upload by checking index statistics
stats = index.describe_index_stats()
print("Index stats:", stats)
```

## Step 6: Prepare Mappings for Search Results

Create dictionaries to map vector IDs back to their original titles and content. This will allow you to display readable search results.

```python
# Create mappings from vector IDs to text
title_map = dict(zip(article_df.vector_id, article_df.title))
content_map = dict(zip(article_df.vector_id, article_df.text))
```

## Step 7: Implement the Search Function

Define a function that takes a query, converts it to an embedding, searches the specified namespace, and returns formatted results.

```python
def query_article(query: str, namespace: str, top_k: int = 5) -> pd.DataFrame:
    """
    Query the Pinecone index in the specified namespace and return similar articles.
    
    Args:
        query: The search query string
        namespace: The namespace to search ('title' or 'content')
        top_k: Number of results to return
    
    Returns:
        DataFrame containing search results
    """
    # Generate embedding for the query
    response = openai.Embedding.create(
        input=query,
        model=EMBEDDING_MODEL
    )
    embedded_query = response["data"][0]['embedding']
    
    # Query the Pinecone index
    query_result = index.query(
        vector=embedded_query,
        namespace=namespace,
        top_k=top_k
    )
    
    # Process and display results
    print(f'\nMost similar results to "{query}" in "{namespace}" namespace:\n')
    
    if not query_result.matches:
        print('No results found.')
        return pd.DataFrame()
    
    # Extract matches
    matches = query_result.matches
    ids = [match.id for match in matches]
    scores = [match.score for match in matches]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'id': ids,
        'score': scores,
        'title': [title_map[_id] for _id in ids],
        'content': [content_map[_id] for _id in ids]
    })
    
    # Print formatted results
    for _, row in results_df.iterrows():
        print(f"{row['title']} (score: {row['score']:.3f})")
    
    print()  # Add spacing
    return results_df
```

## Step 8: Test Your Search Engine

Now you can test semantic searches in both the title and content namespaces.

```python
# Search by article title
title_results = query_article('modern art in Europe', 'title')

# Search by article content
content_results = query_article('Famous battles in Scottish history', 'content')
```

## Conclusion

You've successfully built a semantic search engine using OpenAI embeddings and Pinecone. You learned how to:

1. Load and prepare embedded data
2. Set up a Pinecone index with multiple namespaces
3. Batch upload vectors for efficiency
4. Perform semantic searches using natural language queries

This foundation enables you to build more complex applications like chatbots, recommendation systems, or document search tools. For production deployments, consider implementing error handling, monitoring, and optimizing search performance based on your specific use case.

## Next Steps

- Experiment with different embedding models
- Implement filtering to narrow search results
- Build a frontend interface for your search engine
- Explore Pinecone's metadata filtering for more advanced queries
- Consider implementing hybrid search combining vector and keyword search