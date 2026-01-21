# Vector Search on Hugging Face with the Hub as Backend

This tutorial demonstrates how to perform vector similarity search on datasets hosted on the Hugging Face Hub using DuckDB. You'll learn to create embeddings for a dataset, upload them to the Hub, and perform fast, indexed searches—all while using the Hub as your storage backend.

## Prerequisites

Before you begin, ensure you have the required libraries installed.

```bash
pip install datasets duckdb sentence-transformers model2vec -q
```

## Step 1: Import Libraries and Load the Embedding Model

First, import the necessary modules and load a sentence embedding model. We'll use `model2vec` to load a pre-trained embedding model.

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

# Load a static embedding model from model2vec
static_embedding = StaticEmbedding.from_model2vec("minishlab/potion-base-8M")
model = SentenceTransformer(modules=[static_embedding])
```

## Step 2: Load a Dataset from the Hugging Face Hub

We'll use the `ai-blueprint/fineweb-bbc-news` dataset for this example. This dataset contains news articles, which we'll embed for search.

```python
from datasets import load_dataset

ds = load_dataset("ai-blueprint/fineweb-bbc-news")
```

## Step 3: Create Embeddings for the Dataset

Define a function to generate embeddings for each text entry in the dataset using the loaded model. We'll apply this function to the entire dataset in a batched manner for efficiency.

```python
def create_embeddings(batch):
    embeddings = model.encode(batch["text"], convert_to_numpy=True)
    batch["embeddings"] = embeddings.tolist()
    return batch

ds = ds.map(create_embeddings, batched=True)
```

## Step 4: Upload the Dataset with Embeddings to the Hub

Now, push the enriched dataset—which now includes an `embeddings` column—back to the Hugging Face Hub. This makes the embeddings available for remote querying.

```python
ds.push_to_hub("ai-blueprint/fineweb-bbc-news-embeddings")
```

## Step 5: Perform Vector Search Without an Index

For smaller datasets (under ~100k rows), you can perform a direct similarity search without building an index. This method is slower but more precise.

We'll write a function that connects to the dataset via DuckDB, computes cosine distances between the query embedding and all dataset embeddings, and returns the top-k matches.

```python
import duckdb
from typing import List

def similarity_search_without_duckdb_index(
    query: str,
    k: int = 5,
    dataset_name: str = "ai-blueprint/fineweb-bbc-news-embeddings",
    embedding_column: str = "embeddings",
):    
    # Encode the query using the same model
    query_vector = model.encode(query)
    embedding_dim = model.get_sentence_embedding_dimension()

    # Construct and execute the SQL query
    sql = f"""
        SELECT 
            *,
            array_cosine_distance(
                {embedding_column}::float[{embedding_dim}], 
                {query_vector.tolist()}::float[{embedding_dim}]
            ) as distance
        FROM 'hf://datasets/{dataset_name}/**/*.parquet'
        ORDER BY distance
        LIMIT {k}
    """
    return duckdb.sql(sql).to_df()

# Example query
results = similarity_search_without_duckdb_index("What is the future of AI?")
print(results.head())
```

## Step 6: Perform Vector Search With an Index (Faster)

For larger datasets, building a local index dramatically speeds up queries. We'll use DuckDB's `vss` extension to create a Hierarchical Navigable Small World (HNSW) index.

### 6.1 Set Up Helper Functions

First, define helper functions to install the `vss` extension, manage tables, and create the index.

```python
def _setup_vss():
    duckdb.sql(
        query="""
        INSTALL vss;
        LOAD vss;
        """
    )

def _drop_table(table_name):
    duckdb.sql(
        query=f"""
        DROP TABLE IF EXISTS {table_name};
        """
    )

def _create_table(dataset_name, table_name, embedding_column):
    duckdb.sql(
        query=f"""
        CREATE TABLE {table_name} AS 
        SELECT *, {embedding_column}::float[{model.get_sentence_embedding_dimension()}] as {embedding_column}_float 
        FROM 'hf://datasets/{dataset_name}/**/*.parquet';
        """
    )

def _create_index(table_name, embedding_column):
    duckdb.sql(
        query=f"""
        CREATE INDEX my_hnsw_index ON {table_name} USING HNSW ({embedding_column}_float) WITH (metric = 'cosine');
        """
    )
```

### 6.2 Create the Index

Combine the helper functions to create a local table and index from the remote dataset.

```python
def create_index(dataset_name, table_name, embedding_column):
    _setup_vss()
    _drop_table(table_name)
    _create_table(dataset_name, table_name, embedding_column)
    _create_index(table_name, embedding_column)

create_index(
    dataset_name="ai-blueprint/fineweb-bbc-news-embeddings",
    table_name="fineweb_bbc_news_embeddings",
    embedding_column="embeddings"
)
```

### 6.3 Query Using the Index

Now, perform a similarity search using the indexed table. This returns results in sub-second time.

```python
def similarity_search_with_duckdb_index(
    query: str,
    k: int = 5,
    table_name: str = "fineweb_bbc_news_embeddings",
    embedding_column: str = "embeddings"
):
    embedding = model.encode(query).tolist()
    return duckdb.sql(
        query=f"""
        SELECT *, array_cosine_distance({embedding_column}_float, {embedding}::FLOAT[{model.get_sentence_embedding_dimension()}]) as distance 
        FROM {table_name}
        ORDER BY distance 
        LIMIT {k};
    """
    ).to_df()

# Example query
indexed_results = similarity_search_with_duckdb_index("What is the future of AI?")
print(indexed_results.head())
```

## Conclusion

You've successfully implemented vector search on a Hugging Face dataset using DuckDB. You learned to:

1.  Generate and store embeddings for a dataset on the Hub.
2.  Perform a direct, index-free similarity search suitable for smaller datasets.
3.  Create a local HNSW index for fast, scalable searches on larger datasets.

This approach provides a lightweight, efficient alternative to deploying a full-scale vector database, leveraging the Hugging Face Hub for storage and DuckDB for computation.

## Learn More

-   [Vector Search on Hugging Face](https://huggingface.co/docs/hub/en/datasets-duckdb)
-   [Vector Search Indexing with DuckDB](https://duckdb.org/docs/extensions/vss.html)