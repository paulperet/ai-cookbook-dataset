# Vector Search on Hugging Face with the Hub as Backend

Datasets on the Hugging Face Hub rely on parquet files. We can [interact with these files using DuckDB](https://huggingface.co/docs/hub/en/datasets-duckdb) as a fast in-memory database system. One of DuckDB's features is [vector similarity search](https://duckdb.org/docs/extensions/vss.html) which can be used with or without an index. 

## Install dependencies

```python
!pip install datasets duckdb sentence-transformers model2vec -q
```

## Create embeddings for the dataset

First, we need to create embeddings for the dataset to search over. We will use the `sentence-transformers` library to create embeddings for the dataset.

```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

static_embedding = StaticEmbedding.from_model2vec("minishlab/potion-base-8M")
model = SentenceTransformer(modules=[static_embedding])
```

Now, let's load the [ai-blueprint/fineweb-bbc-news](https://huggingface.co/datasets/ai-blueprint/fineweb-bbc-news) dataset from the Hub. 

```python
from datasets import load_dataset

ds = load_dataset("ai-blueprint/fineweb-bbc-news")
```

We can now create embeddings for the dataset. Normally, we might want to chunk our data into smaller batches to avoid losing precision, but for this example, we will just create embeddings for the full text of the dataset.

```python
def create_embeddings(batch):
    embeddings = model.encode(batch["text"], convert_to_numpy=True)
    batch["embeddings"] = embeddings.tolist()
    return batch

ds = ds.map(create_embeddings, batched=True)
```

We can now upload our dataset with embeddings back to the Hub.

```python
ds.push_to_hub("ai-blueprint/fineweb-bbc-news-embeddings")
```

## Vector Search the Hugging Face Hub

We can now perform vector search on the dataset using `duckdb`. When doing so, we can either use an index or not. Searching **without** an index is slower but more precise, whereas searching **with** an index is faster but less precise. 

### Without an index

To search without an index, we can use the `duckdb` library to connect to the dataset and perform a vector search. This is a slow operation, but normally works quick enough for small datasets up to let's say 100k rows. Meaning querying our dataset will be somewhat slower.

```python
import duckdb
from typing import List

def similarity_search_without_duckdb_index(
    query: str,
    k: int = 5,
    dataset_name: str = "ai-blueprint/fineweb-bbc-news-embeddings",
    embedding_column: str = "embeddings",
):    
    # Use same model as used for indexing
    query_vector = model.encode(query)
    embedding_dim = model.get_sentence_embedding_dimension()

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

similarity_search_without_duckdb_index("What is the future of AI?")
```

### With an index

This approach creates a local copy of the dataset and uses this to create an index. This has some minor overhead but it will significantly speed up the search once you've created it.

```python
import duckdb

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

Now we can perform a vector search with the index, which return the results instantly. 

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

similarity_search_with_duckdb_index("What is the future of AI?")
```

The query reduces from 30 seconds to sub-second response times and does not require you to deploy a heavy-weight vector search engine, while storage is handled by the Hub.

## Conclusion

We have seen how to perform vector search on the Hub using `duckdb`. For small datasets <100k rows, we can perform vector search without an index using the Hub as a vector search backend, but for larger datasets, we should create an index with the `vss` extension while doing local search and using the Hub as a storage backend. 

## Learn more

- [Vector Search on Hugging Face](https://huggingface.co/docs/hub/en/datasets-duckdb)
- [Vector Search Indexing with DuckDB](https://duckdb.org/docs/extensions/vss.html)