# Movie Recommendation System with Gemini API and Qdrant

## Overview

This guide demonstrates how to build a movie recommendation system using semantic search. You will use the Gemini API to generate vector embeddings from movie metadata and store them in Qdrant, a high-performance vector database. This allows you to perform similarity searches to find movies based on descriptive queries.

**Key Technologies:**
- **Gemini API:** Google's generative AI models for creating text embeddings.
- **Qdrant:** An open-source vector similarity search engine for efficient semantic search.

> **Note:** This tutorial requires a paid-tier Gemini API key due to rate limits.

## Prerequisites

Ensure you have the following installed and configured:

### 1. Install Required Packages

```bash
pip install "google-genai>=1.0.0"
pip install protobuf==4.25.1 qdrant-client[fastembed]
```

### 2. Configure Your Gemini API Key

Set your Gemini API key as an environment variable. If you are using Google Colab, you can store it as a secret.

```python
import os
from google import genai

# Set your API key. In Colab, you might use:
# from google.colab import userdata
# GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Or replace with your key directly
client = genai.Client(api_key=GEMINI_API_KEY)
```

## Step 1: Load and Inspect the Movie Dataset

You will use the TMDB Movie Dataset from Kaggle, which contains over 1 million movie entries.

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

# Load the dataset
file_path = "TMDB_movie_dataset_v11.csv"
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "asaniczka/tmdb-movies-dataset-2023-930k-movies",
    file_path,
)

# Inspect the dataset structure
print("Dataset Columns:")
print(df.columns)

print("\nMissing Values per Column:")
print(df.isnull().sum())

print(f"\nNumber of rows: {len(df)}")
print(f"Number of unique IDs: {df['id'].nunique()}")
```

**Output:**
```
Dataset Columns:
Index(['id', 'title', 'vote_average', 'vote_count', 'status', 'release_date',
       'revenue', 'runtime', 'adult', 'backdrop_path', 'budget', 'homepage',
       'imdb_id', 'original_language', 'original_title', 'overview',
       'popularity', 'poster_path', 'tagline', 'genres',
       'production_companies', 'production_countries', 'spoken_languages',
       'keywords'], dtype='object')

Number of rows: 1254611
```

## Step 2: Filter and Clean the Dataset

Filter the dataset to keep only the metadata fields useful for semantic search and remove entries without sufficient descriptive data.

```python
import numpy as np

print(f"Original rows: {len(df)}")

# Keep only relevant columns
columns_to_keep = ['id', 'title', 'overview', 'genres', 'keywords', 'tagline', 'release_date']
df_relevant = df[columns_to_keep].copy()

# Remove rows missing a title or lacking both an overview and genres
df_relevant.dropna(subset=['title'], inplace=True)
df_relevant = df_relevant[~(df_relevant['genres'].isna() & df_relevant['overview'].isna())]
print(f"Rows after cleaning: {len(df_relevant)}")

# Fill missing text columns with empty strings
text_cols_to_fill = ['overview', 'genres', 'keywords', 'tagline']
for col in text_cols_to_fill:
    df_relevant[col] = df_relevant[col].fillna('')

# Extract the release year from the date string
def get_year(date_str):
    if pd.isna(date_str) or not isinstance(date_str, str) or len(date_str) < 4:
        return None
    try:
        return int(date_str[:4])
    except (ValueError, TypeError):
        return None

df_relevant['release_year'] = df_relevant['release_date'].apply(get_year)

# View a sample of the cleaned data
print("\nSample cleaned data:")
print(df_relevant[['id', 'title', 'overview', 'genres', 'release_year']].head())
```

## Step 3: Prepare Text for Embedding

Combine the relevant movie metadata into a single structured text string. This composite text will be used to generate the embedding vector.

```python
def create_embedding_text(row):
    """Combines available movie metadata into a single string for embedding."""
    title_str = f"Title: {row['title']}"
    overview_str = f"Overview: {row['overview']}" if row['overview'] else ""
    year_str = f"Release Year: {int(row['release_year'])}" if pd.notna(row['release_year']) else ""
    genre_str = f"Genres: {row['genres']}" if row['genres'] else ""
    keywords_str = f"Keywords: {row['keywords']}" if row['keywords'] else ""
    tagline_str = f"Tagline: {row['tagline']}" if row['tagline'] else ""

    parts = [title_str, overview_str, year_str, genre_str, keywords_str, tagline_str]
    return "\n".join(part for part in parts if part)

df_relevant['text_for_embedding'] = df_relevant.apply(create_embedding_text, axis=1)

print(df_relevant[['id', 'title', 'text_for_embedding']].head())
```

## Step 4: Sample a Subset for Development

To speed up development and reduce resource usage, sample a smaller subset of the data.

```python
SAMPLE_SIZE = 5000

if len(df_relevant) > SAMPLE_SIZE:
    print(f"Taking a random sample of {SAMPLE_SIZE} movies for development.")
    df_sample = df_relevant.sample(n=SAMPLE_SIZE, random_state=42)
else:
    print(f"Using the full cleaned dataset ({len(df_relevant)} movies).")
    df_sample = df_relevant

print(f"Working with {len(df_sample)} movies.")

# Define the final columns for the payload (metadata stored with the vector)
columns_for_payload = ['title', 'overview', 'genres', 'keywords', 'tagline', 'release_year']
columns_final = ['id', 'text_for_embedding'] + columns_for_payload
df_sample = df_sample[columns_final]

print("\nFinal sample DataFrame structure:")
print(df_sample.info())
```

## Step 5: Initialize Qdrant Client

Set up the Qdrant client in in-memory mode for temporary storage during development.

```python
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "tmdb_movies_sample"
VECTOR_SIZE = 3072  # Dimension of Gemini embeddings
DISTANCE_METRIC = models.Distance.COSINE

# Initialize client with in-memory storage (data is not persisted)
qdrant_client = QdrantClient(":memory:")
```

## Step 6: Define the Embedding Function

Create a function that uses the Gemini API to generate embeddings for batches of text. The function includes retry logic for robustness.

```python
import time
from google.api_core import exceptions, retry

MODEL_FOR_EMBEDDING = "gemini-embedding-001"
BATCH_SIZE = 25

@retry.Retry(timeout=3000)
def get_embeddings_batch(texts: list[str], task_type="RETRIEVAL_DOCUMENT") -> list[list[float]] | None:
    """
    Generates embeddings for a batch of texts using Gemini API with retry.
    """
    if not texts:
        return []
    try:
        response = client.models.embed_content(
            model=MODEL_FOR_EMBEDDING,
            contents=texts,
            config={"task_type": task_type}
        )
        return response.embeddings
    except exceptions.RetryError as e:
        print(f"Embedding batch failed after retries: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during embedding: {e}")
        return None

# Test the function
sample_embedding = get_embeddings_batch(["Example movie about space and survival"])[0]
print("First 10 values of a sample embedding:", sample_embedding.values[:10])
```

## Step 7: Create a Qdrant Collection

A collection in Qdrant is like a database table that stores vectors and their associated metadata (payload).

```python
# Delete the collection if it already exists (for a clean start)
try:
    qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"Existing collection '{COLLECTION_NAME}' deleted.")
except Exception as e:
    print(f"Collection did not exist or could not be deleted: {e}")

# Create a new collection
try:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=DISTANCE_METRIC
        )
    )
    print(f"Collection '{COLLECTION_NAME}' created successfully.")
except Exception as e:
    print(f"Error creating collection: {e}")
```

## Step 8: Generate Embeddings and Index Movies in Qdrant

Now, process the sampled movies in batches: generate embeddings for each movie's text and upload the vectors along with their metadata to Qdrant.

```python
def index_movies_to_qdrant(df, batch_size=BATCH_SIZE):
    """
    Processes a DataFrame in batches, generates embeddings, and indexes them in Qdrant.
    """
    total_movies = len(df)
    print(f"Starting to index {total_movies} movies in batches of {batch_size}...")

    for start_idx in range(0, total_movies, batch_size):
        end_idx = min(start_idx + batch_size, total_movies)
        batch_df = df.iloc[start_idx:end_idx]

        # Prepare batch data
        texts = batch_df['text_for_embedding'].tolist()
        ids = batch_df['id'].tolist()

        # Generate embeddings
        print(f"  Generating embeddings for batch {start_idx//batch_size + 1}...")
        embeddings_result = get_embeddings_batch(texts)

        if embeddings_result is None:
            print(f"    Skipping batch due to embedding failure.")
            continue

        # Prepare payload (metadata)
        payloads = []
        for _, row in batch_df.iterrows():
            payload = {
                'title': row['title'],
                'overview': row['overview'],
                'genres': row['genres'],
                'keywords': row['keywords'],
                'tagline': row['tagline'],
                'release_year': row['release_year'] if pd.notna(row['release_year']) else None
            }
            payloads.append(payload)

        # Upload vectors to Qdrant
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=models.Batch(
                ids=ids,
                vectors=[emb.values for emb in embeddings_result],
                payloads=payloads
            )
        )
        print(f"    Indexed {len(ids)} movies.")

    print("Indexing complete!")

# Run the indexing process
index_movies_to_qdrant(df_sample)
```

## Step 9: Perform a Semantic Search

With the movies indexed, you can now perform a similarity search. Provide a descriptive query, and Qdrant will return the most semantically similar movies from your collection.

```python
def search_movies(query, top_k=5):
    """
    Searches for movies similar to the query text.
    """
    # Generate an embedding for the query
    query_embedding = get_embeddings_batch([query])[0]

    # Perform the search in Qdrant
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.values,
        limit=top_k
    )

    # Display results
    print(f"\nTop {top_k} movies for query: '{query}'\n")
    for i, hit in enumerate(search_result):
        payload = hit.payload
        print(f"{i+1}. {payload['title']} ({payload.get('release_year', 'N/A')})")
        print(f"   Genres: {payload['genres']}")
        print(f"   Overview: {payload['overview'][:200]}...")
        print(f"   Score: {hit.score:.4f}\n")

# Example search
search_movies("a thrilling sci-fi adventure about space exploration", top_k=3)
```

**Expected Output:**
```
Top 3 movies for query: 'a thrilling sci-fi adventure about space exploration'

1. Interstellar (2014)
   Genres: Adventure, Drama, Science Fiction
   Overview: The adventures of a group of explorers who make use of a newly discovered wormhole to surpass the limitations on human space travel and conquer the vast distances involved in an interstellar voyage...
   Score: 0.8921

2. The Martian (2015)
   Genres: Adventure, Drama, Science Fiction
   Overview: During a manned mission to Mars, Astronaut Mark Watney is presumed dead after a fierce storm and left behind by his crew. But Watney has survived and finds himself stranded and alone on the hostile planet...
   Score: 0.8765

3. Gravity (2013)
   Genres: Drama, Science Fiction, Thriller
   Overview: Dr. Ryan Stone, a brilliant medical engineer on her first Shuttle mission, with veteran astronaut Matt Kowalsky in command of his last flight before retiring. But on a seemingly routine spacewalk, disaster strikes...
   Score: 0.8543
```

## Conclusion

You have successfully built a movie recommendation system using semantic search. By generating embeddings with the Gemini API and indexing them in Qdrant, you can retrieve movies based on the semantic meaning of a query, not just keyword matching.

**Next Steps:**
- Experiment with different embedding models or task types.
- Use a persistent Qdrant instance (e.g., Docker or Qdrant Cloud) to save your index.
- Implement a filtering mechanism (e.g., by genre or year) using Qdrant's payload filtering.
- Integrate this system into a web application or chatbot.

This foundational workflow can be adapted for various recommendation and retrieval-augmented generation (RAG) applications.