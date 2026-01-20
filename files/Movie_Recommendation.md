##### Copyright 2025 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Movie Recommendation System with Gemini API and Qdrant

This notebook requires paid tier rate limits to run properly.  
(cf. pricing for more details).

## Overview

The Gemini API provides access to a family of generative AI models for generating content and solving problems. These models are designed and trained to handle both text and images as input.

Qdrant is an open-source vector similarity search engine designed for efficient and scalable semantic search. It offers a simple yet powerful API to store and search high-dimensional vectors, supports filtering with metadata (payloads), and integrates easily into production systems. Qdrant can be self-hosted or accessed via its managed cloud service, making it quick to set up and ideal for a wide range of AI applications that rely on semantic understanding and retrieval.

In this notebook, you'll learn how to perform a similarity search on data from a website with the help of Gemini API and Qdrant.

This notebook was contributed by Anand Roy.

LinkedIn - See Anand other notebooks here.

Have a cool Gemini example? Feel free to share it too!

## Setup

First, you must install the packages and set the necessary environment variables.

### Installation

Install google's python client SDK for the Gemini API, `google-genai`. Next, install Qdrant's Python client SDK, `qdrant-client`.


```
%pip install -q "google-genai>=1.0.0"
%pip install -q protobuf==4.25.1 qdrant-client[fastembed]
```

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GEMINI_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see Authentication for an example.


```
from google.colab import userdata
from google import genai

GEMINI_API_KEY=userdata.get('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)
```

## Building the Movie Vector Index
This section covers preparing the movie dataset, generating embeddings using Gemini, and indexing them in Qdrant for similarity search.

### 1. Load the Dataset from Kaggle

Begin by loading the dataset from Kaggle using the kagglehub library. The dataset used in this notebook is the TMDB Movie Dataset 2024, which contains approximately 1 Million+ movie entries.


```
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "TMDB_movie_dataset_v11.csv"

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "asaniczka/tmdb-movies-dataset-2023-930k-movies",
  file_path,
)
```

    /tmp/ipykernel_26130/2431045845.py:6: DeprecationWarning: load_dataset is deprecated and will be removed in a future version.
      df = kagglehub.load_dataset(


### 2. Inspect the Dataset Structure

Since the dataset is large, inspecting it helps you identify useful fields and filter out irrelevant data early on.


```
print("\nDataset Columns:")
print(df.columns)

print("\nMissing Values per Column:")
print(df.isnull().sum())

print(f"\nNumber of rows: {len(df)}")
print(f"Number of unique IDs: {df['id'].nunique()}")
```

    
    Dataset Columns:
    Index(['id', 'title', 'vote_average', 'vote_count', 'status', 'release_date',
           'revenue', 'runtime', 'adult', 'backdrop_path', 'budget', 'homepage',
           'imdb_id', 'original_language', 'original_title', 'overview',
           'popularity', 'poster_path', 'tagline', 'genres',
           'production_companies', 'production_countries', 'spoken_languages',
           'keywords'],
          dtype='object')
    
    Missing Values per Column:
    id                            0
    title                        13
    vote_average                  0
    vote_count                    0
    status                        0
    release_date             239109
    revenue                       0
    runtime                       0
    adult                         0
    backdrop_path            930599
    budget                        0
    homepage                1123663
    imdb_id                  624300
    original_language             0
    original_title               13
    overview                 270635
    popularity                    0
    poster_path              418486
    tagline                 1078824
    genres                   526517
    production_companies     702743
    production_countries     580808
    spoken_languages         557829
    keywords                 928832
    dtype: int64
    
    Number of rows: 1254611
    Number of unique IDs: 1253629


### 3. Filter and Clean the Dataset

This step filters the dataset to keep only metadata useful for semantic search: `id`, `title`, `overview`, `genres`, `keywords`, `tagline`, and `release_date`. These fields provide enough context to generate meaningful embeddings.

Entries (rows) missing a `title` or lacking both `overview` and `genres` are removed, as they don’t have enough descriptive data for accurate recommendations.



```
import pandas as pd
import numpy as np
import ast
print(f"Original rows: {len(df)}")

columns_to_keep = ['id', 'title', 'overview', 'genres', 'keywords', 'tagline', 'release_date']

df_relevant = df[columns_to_keep].copy()

print(f"Rows before dropping missing title: {len(df_relevant)}")
df_relevant.dropna(subset=['title'], inplace=True)
df_relevant = df_relevant[~(df_relevant['genres'].isna() & df_relevant['overview'].isna())]
print(f"Rows after dropping missing title and dropping missing (genres and overview): {len(df_relevant)}")

# Fill missing text columns with empty strings
text_cols_to_fill = ['overview', 'genres', 'keywords', 'tagline']
for col in text_cols_to_fill:
    df_relevant[col] = df_relevant[col].fillna('')


# Extract release year from the release_date string
def get_year(date_str):
    if pd.isna(date_str) or not isinstance(date_str, str) or len(date_str) < 4:
        return None
    try:
        return int(date_str[:4])
    except (ValueError, TypeError):
        return None

df_relevant['release_year'] = df_relevant['release_date'].apply(get_year)

print("\nSample data after cleaning (keeping missing overviews):")
print(df_relevant[['id', 'title', 'overview', 'genres', 'keywords', 'tagline', 'release_year']].head())
```

    Original rows: 1254611
    Rows before dropping missing title: 1254611
    Rows after dropping missing title and dropping missing (genres and overview): 1109811
    
    Sample data after cleaning (keeping missing overviews):
           id            title                                           overview  \
    0   27205        Inception  Cobb, a skilled thief who commits corporate es...   
    1  157336     Interstellar  The adventures of a group of explorers who mak...   
    2     155  The Dark Knight  Batman raises the stakes in his war on crime. ...   
    3   19995           Avatar  In the 22nd century, a paraplegic Marine is di...   
    4   24428     The Avengers  When an unexpected enemy emerges and threatens...   
    
                                            genres  \
    0           Action, Science Fiction, Adventure   
    1            Adventure, Drama, Science Fiction   
    2               Drama, Action, Crime, Thriller   
    3  Action, Adventure, Fantasy, Science Fiction   
    4           Science Fiction, Action, Adventure   
    
                                                keywords  \
    0  rescue, mission, dream, airplane, paris, franc...   
    1  rescue, future, spacecraft, race against time,...   
    2  joker, sadism, chaos, secret identity, crime f...   
    3  future, society, culture clash, space travel, ...   
    4  new york city, superhero, shield, based on com...   
    
                                                 tagline  release_year  
    0               Your mind is the scene of the crime.        2010.0  
    1  Mankind was born on Earth. It was never meant ...        2014.0  
    2                  Welcome to a world without rules.        2008.0  
    3                        Enter the world of Pandora.        2009.0  
    4                            Some assembly required.        2012.0  


### 4. Prepare Text for Embedding

This step prepares the movie metadata for embedding by combining relevant fields into a single structured text string. This representation includes the title, overview, genres, keywords, tagline, and release year (if available). The output is stored in a new column called `text_for_embedding`.

Embeddings are numerical vector representations of text that capture semantic meaning and relationships. These vectors can be used for tasks like similarity search and clustering.
Learn more about text embeddings and explore the Gemini embedding notebook.


```
def create_embedding_text(row):
    """Combines available movie metadata into a single string for embedding."""
    # Title is always present, so it can be included directly
    title_str = f"Title: {row['title']}"
    overview_str = f"Overview: {row['overview']}" if row['overview'] else ""
    year_str = f"Release Year: {int(row['release_year'])}" if pd.notna(row['release_year']) else ""
    genre_str = f"Genres: {row['genres']}" if row['genres'] else ""
    keywords_str = f"Keywords: {row['keywords']}" if row['keywords'] else ""
    tagline_str = f"Tagline: {row['tagline']}" if row['tagline'] else ""

    parts = [
        title_str,
        overview_str,
        year_str,
        genre_str,
        keywords_str,
        tagline_str
    ]
    return "\n".join(part for part in parts if part)

df_relevant['text_for_embedding'] = df_relevant.apply(create_embedding_text, axis=1)

# Use this to inspect how movie data has been transformed for embedding
print(df_relevant[['id', 'title', 'text_for_embedding']].head())
```

           id            title                                 text_for_embedding
    0   27205        Inception  Title: Inception\nOverview: Cobb, a skilled th...
    1  157336     Interstellar  Title: Interstellar\nOverview: The adventures ...
    2     155  The Dark Knight  Title: The Dark Knight\nOverview: Batman raise...
    3   19995           Avatar  Title: Avatar\nOverview: In the 22nd century, ...
    4   24428     The Avengers  Title: The Avengers\nOverview: When an unexpec...


### 5. Sample a Subset for Development

To keep the notebook easy to run and ensure efficient development, you’ll want to iterate quickly and minimize resource usage. Instead of using the full dataset, this step samples 5,000 movies from the cleaned data, unless the dataset is already smaller, in which case all entries are used.


```
SAMPLE_SIZE = 5000

if len(df_relevant) > SAMPLE_SIZE:
    print(f"\nTaking a random sample of {SAMPLE_SIZE} movies for development.")
    df_sample = df_relevant.sample(n=SAMPLE_SIZE, random_state=42)
else:
    print(f"\nCleaned dataset size ({len(df_relevant)}) is smaller than or equal to SAMPLE_SIZE. Using the full cleaned dataset.")
    df_sample = df_relevant

print(f"Working with {len(df_sample)} movies for the next steps.")
print(df_sample[['id', 'title', 'release_year']].head())

columns_for_payload = ['title', 'overview', 'genres', 'keywords', 'tagline', 'release_year']
columns_final = ['id', 'text_for_embedding'] + columns_for_payload
df_sample = df_sample[columns_final]

print("\nFinal sample DataFrame structure for embedding/indexing:")
print(df_sample.info())
```

    
    Taking a random sample of 5000 movies for development.
    Working with 5000 movies for the next steps.
                id               title  release_year
    95090   714945              Gather        2020.0
    198838  117602    California Girls        1985.0
    827619  648297          After Jake        2013.0
    801274  637062  Duas vezes Senzala        2017.0
    846938  342211                Rise           NaN
    
    Final sample DataFrame structure for embedding/indexing:
    <class 'pandas.core.frame.DataFrame'>
    Index: 5000 entries, 95090 to 650934
    Data columns (total 8 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   id                  5000 non-null   int64  
     1   text_for_embedding  5000 non-null   object 
     2   title               5000 non-null   object 
     3   overview            5000 non-null   object 
     4   genres              5000 non-null   object 
     5   keywords            5000 non-null   object 
     6   tagline             5000 non-null   object 
     7   release_year        4260 non-null   float64
    dtypes: float64(1), int64(1), object(6)
    memory usage: 351.6+ KB
    None


### 6. Initialize Qdrant for Vector Indexing


With the data prepared, the next step is to set up **Qdrant**, a vector similarity search engine optimized for storing and querying high-dimensional vectors. It supports fast indexing, filtering, and similarity search across millions of vectors.

Qdrant can run:

* Locally as a standalone service
* In the cloud for production deployments
* Or entirely **in-memory** for fast, temporary use during development

In this notebook, Qdrant is initialized using in-memory mode by passing `":memory:"` to the client. This stores data only in RAM, meaning it **will not persist after the session ends**. This is suitable for experimentation but not for saving results long-term.

You also configure the following:

* `COLLECTION_NAME`: The name of the Qdrant collection to store movie vectors
* `VECTOR_SIZE`: Set to `3072` to match the dimensionality of the text embeddings generated by Gemini
* `DISTANCE_METRIC`: Set to **cosine distance**, which is ideal for measuring semantic similarity between embedding vectors


```
from qdrant_client import QdrantClient, models
import time

COLLECTION_NAME = "tmdb_movies_sample"

VECTOR_SIZE = 3072
DISTANCE_METRIC = models.Distance.COSINE


# Initialize Qdrant client using in-memory storage
qdrant_client = QdrantClient(":memory:")
```

### 7. Define Batch Embedding Function

This step defines the `get_embeddings_batch` function, which generates text embeddings for batches of movie data using the Gemini embedding model (`gemini-embedding-001`) including automatic retries for robustness.



```
import time
from google.api_core import exceptions, retry

MODEL_FOR_EMBEDDING = "gemini-embedding-001" # @param ["gemini-embedding-001", "embedding-001", "text-embedding-004"] {"allow-input":true, isTemplate: true}

BATCH_SIZE = 25
QDRANT_BATCH_SIZE = 3072


@retry.Retry(timeout=3000)
def get_embeddings_batch(texts: list[str], task_type="RETRIEVAL_DOCUMENT") -> list[list[float]] | None:
    """
    Generates embeddings for a batch of texts using Gemini API with retry.

    Args:
        texts: A list of strings to embed.
        task_type: The task type for the embedding model.

    Returns:
        A list of embedding vectors (list of floats), or None if a non-retryable error occurs.
    """
    if not texts:
        return []
    try:
        response = client.models.embed_content(
          model=MODEL_FOR_EMBEDDING,
          contents=texts,
          config={
            "task_type":task_type,
          }
        )
        return response.embeddings
    except exceptions.RetryError as e:
        print(f"Embedding batch failed after retries: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during embedding: {e}")
        return None

# Example of what an embedding looks like
sample_embedding = get_embeddings_batch(["Example movie about space and survival"])[0]
print("Example embedding vector:", sample_embedding.values[:10])
```

    Example embedding vector: [-0.023393, 0.011092304, 0.007310852, -0.095236674, 0.012521885, 0.0069673047, -0.011544445, -0.02173588, 0.027118236, 0.004006482]


### 8. Create a Collection in Qdrant


A collection in Qdrant is like a table in a database, it stores vectors along with optional metadata (payload). Each collection has its own configuration, including vector size and similarity metric.


```
# In case someone tries running the whole notebook again they would want to create the collection again

try:
    qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"Existing collection '{COLLECTION_NAME}' deleted.")
except Exception as e:
    print(f"Error deleting collection (it might not exist): {e}")

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
    print(f"