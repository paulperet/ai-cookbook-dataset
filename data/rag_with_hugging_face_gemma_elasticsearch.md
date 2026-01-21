# Building a RAG System with Gemma, Elasticsearch, and Hugging Face Models

This guide walks you through building a Retrieval-Augmented Generation (RAG) system powered by Elasticsearch (ES) and Hugging Face models. You will learn how to toggle between two vectorization strategies:
1.  **ES-Vectorization:** Your Elasticsearch cluster handles vectorization during data ingestion and querying.
2.  **Self-Vectorization:** You generate embeddings locally before sending data to Elasticsearch.

The choice depends on your infrastructure and needs. ES-vectorization simplifies client-side code but requires ML nodes in your cluster. Self-vectorization offers more control if you have a preferred embedding pipeline.

> **Tested with:** Elasticsearch 8.13.x and 8.14.x.

## Prerequisites & Setup

### 1. Install Required Libraries
Begin by installing the necessary Python packages.

```bash
pip install elasticsearch sentence_transformers transformers eland==8.12.1
pip install datasets==2.19.2
```

### 2. Configure Hugging Face Authentication
Authenticate with Hugging Face to download models and datasets.

```python
from huggingface_hub import notebook_login

notebook_login()
```

### 3. Connect to Your Elasticsearch Deployment
You need an active Elasticsearch deployment. If you don't have one, you can [create a deployment on Elastic Cloud](https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#creating-a-cloud-deployment).

Store your `CLOUD_ID` and `ELASTIC_DEPL_API_KEY` as environment variables or secrets (e.g., in Google Colab secrets).

```python
from google.colab import userdata
from elasticsearch import Elasticsearch

# Retrieve your credentials (Colab example)
CLOUD_ID = userdata.get("CLOUD_ID")
ELASTIC_API_KEY = userdata.get("ELASTIC_DEPL_API_KEY")

# Create and test the Elasticsearch client
client = Elasticsearch(cloud_id=CLOUD_ID, api_key=ELASTIC_API_KEY)
client.info()  # Should return a successful response
```

## Step 1: Source and Prepare Your Data

We'll use the `MongoDB/embedded_movies` dataset from Hugging Face, which contains movie metadata and plots.

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("MongoDB/embedded_movies")
print(dataset)
```

**Output:**
```
DatasetDict({
    train: Dataset({
        features: ['plot', 'fullplot', 'languages', 'writers', ...],
        num_rows: 1500
    })
})
```

Now, clean the data:
1.  Remove entries where the `fullplot` is missing, as this is our primary text for embeddings.
2.  Remove any pre-existing `plot_embedding` column, as we will generate new embeddings.

```python
# Filter out entries with no fullplot
dataset = dataset.filter(lambda x: x["fullplot"] is not None)

# Remove old embeddings if they exist
if "plot_embedding" in sum(dataset.column_names.values(), []):
    dataset = dataset.remove_columns("plot_embedding")

print(dataset["train"])
```

**Output:**
```
Dataset({
    features: ['plot', 'fullplot', 'languages', 'writers', ...],
    num_rows: 1452
})
```

## Step 2: Choose Your Vectorization Strategy

Here you decide how embeddings are generated. This choice affects the rest of the setup.

```python
# Set to True to let Elasticsearch handle vectorization.
# Requires your ES cluster to have at least 1 ML node.
USE_ELASTICSEARCH_VECTORISATION = True

# The embedding model to use (from Hugging Face)
EMBEDDING_MODEL_ID = "thenlper/gte-small"
# Update this based on your chosen model's output dimensions
EMBEDDING_DIMENSIONS = 384  # 384 for gte-small, 768 for gte-base, 1024 for gte-large
```

**Note:** If you change these values later, you must re-run the notebook from this step.

## Step 3: Configure the Embedding Pipeline

### Option A: Deploy Model to Elasticsearch (ES-Vectorization)
If `USE_ELASTICSEARCH_VECTORISATION` is `True`, deploy the Hugging Face model to your Elasticsearch cluster using Eland. This allows ES to vectorize data and queries internally.

```python
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# Use Eland to import the model into Elasticsearch
!(if [ "True" == $USE_ELASTICSEARCH_VECTORISATION ]; then \
  eland_import_hub_model --cloud-id $CLOUD_ID --hub-model-id $EMBEDDING_MODEL_ID --task-type text_embedding --es-api-key $ELASTIC_API_KEY --start --clear-previous; \
fi)
```

### Option B: Set Up Local Embedding (Self-Vectorization)
If `USE_ELASTICSEARCH_VECTORISATION` is `False`, load the model locally and generate embeddings for your dataset before ingestion.

```python
from sentence_transformers import SentenceTransformer

if not USE_ELASTICSEARCH_VECTORISATION:
    # Load the embedding model locally
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)

def get_embedding(text: str) -> list[float]:
    """Generate an embedding for a given text string locally."""
    if USE_ELASTICSEARCH_VECTORISATION:
        raise Exception("Local embedding is disabled when using ES vectorization.")
    if not text.strip():
        print("Warning: Attempted to get embedding for empty text.")
        return []
    embedding = embedding_model.encode(text)
    return embedding.tolist()

def add_fullplot_embedding(x):
    """Add an 'embedding' column to the dataset batch."""
    if USE_ELASTICSEARCH_VECTORISATION:
        raise Exception("Local embedding is disabled when using ES vectorization.")
    full_plots = x["fullplot"]
    return {"embedding": [get_embedding(fp) for fp in full_plots]}

# Apply the embedding function to the dataset
if not USE_ELASTICSEARCH_VECTORISATION:
    dataset = dataset.map(add_fullplot_embedding, batched=True)
    print(dataset["train"])
```

## Step 4: Create the Elasticsearch Index

Create an index with the correct mappings to support vector search. The mapping differs based on your vectorization choice.

```python
index_name = "movies"
# Format the model ID for Elasticsearch (replace '/' with '__')
model_id = EMBEDDING_MODEL_ID.replace("/", "__")

# Base index mapping
index_mapping = {
    "properties": {
        "fullplot": {"type": "text"},
        "plot": {"type": "text"},
        "title": {"type": "text"},
    }
}

# Add the embedding field mapping based on the chosen strategy
if USE_ELASTICSEARCH_VECTORISATION:
    # Mapping for ES-generated embeddings
    index_mapping["properties"]["embedding"] = {
        "properties": {
            "is_truncated": {"type": "boolean"},
            "model_id": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}},
            "predicted_value": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMENSIONS,
                "index": True,
                "similarity": "cosine",
            },
        }
    }
else:
    # Mapping for pre-computed embeddings
    index_mapping["properties"]["embedding"] = {
        "type": "dense_vector",
        "dims": EMBEDDING_DIMENSIONS,
        "index": "true",
        "similarity": "cosine",
    }

# Delete the index if it already exists (for a clean start)
if client.indices.exists(index=index_name):
    print(f"Deleting existing index: {index_name}")
    client.indices.delete(index=index_name, ignore=[400, 404])

print(f"Creating index: {index_name}")

# Configure an ingest pipeline for ES-vectorization
if USE_ELASTICSEARCH_VECTORISATION:
    pipeline_id = "vectorize_fullplots"
    client.ingest.put_pipeline(
        id=pipeline_id,
        processors=[
            {
                "inference": {
                    "model_id": model_id,
                    "target_field": "embedding",
                    "field_map": {"fullplot": "text_field"},
                }
            }
        ],
    )
    index_settings = {"index": {"default_pipeline": pipeline_id}}
else:
    index_settings = {}

# Create the index
client.options(ignore_status=[400, 404]).indices.create(
    index=index_name, mappings=index_mapping, settings=index_settings
)
```

**Output:**
```
Creating index movies
ObjectApiResponse({'acknowledged': True, 'shards_acknowledged': True, 'index': 'movies'})
```

## Step 5: Ingest Data into Elasticsearch

Ingest the movie data in batches. The process differs slightly depending on whether embeddings are generated by ES or provided pre-computed.

```python
from elasticsearch.helpers import bulk, BulkIndexError

def batch_to_bulk_actions(batch):
    """Convert a dataset batch into bulk index actions for Elasticsearch."""
    for record in batch:
        action = {
            "_index": "movies",
            "_source": {
                "title": record["title"],
                "fullplot": record["fullplot"],
                "plot": record["plot"],
            },
        }
        # Include pre-computed embedding only if not using ES-vectorization
        if not USE_ELASTICSEARCH_VECTORISATION:
            action["_source"]["embedding"] = record["embedding"]
        yield action

def bulk_index(ds):
    """Index the entire dataset using bulk operations."""
    batch_size = 100
    total_len = len(ds)

    # Use a longer timeout if ES is generating embeddings to avoid timeouts
    batch_client = client.options(request_timeout=600) if USE_ELASTICSEARCH_VECTORISATION else client

    for batch_start in range(0, total_len, batch_size):
        batch_end = min(batch_start + batch_size, total_len)
        print(f"Indexing batch: start [{batch_start}], end [{batch_end}]")

        batch = ds.select(range(batch_start, batch_end))
        actions = batch_to_bulk_actions(batch)
        bulk(batch_client, actions)

# Execute the bulk indexing
try:
    bulk_index(dataset["train"])
except BulkIndexError as e:
    print(f"Bulk indexing errors: {e.errors}")

print("Data ingestion into Elasticsearch complete!")
```

**Output:**
```
Indexing batch: start [0], end [100]
...
Indexing batch: start [1400], end [1452]
Data ingestion into Elasticsearch complete!
```

## Step 6: Implement Vector Search

Create a search function that queries the index. The function automatically adapts to your chosen vectorization strategy.

```python
def vector_search(plot_query):
    """
    Perform a k-NN vector search on the 'movies' index.
    The query is vectorized either by ES or locally.
    """
    if USE_ELASTICSEARCH_VECTORISATION:
        # ES will vectorize the query text using the deployed model
        knn = {
            "field": "embedding.predicted_value",
            "k": 10,
            "query_vector_builder": {
                "text_embedding": {
                    "model_id": model_id,
                    "model_text": plot_query,
                }
            },
            "num_candidates": 150,
        }
    else:
        # Vectorize the query locally
        question_embedding = get_embedding(plot_query)
        knn = {
            "field": "embedding",
            "query_vector": question_embedding,
            "k": 10,
            "num_candidates": 150,
        }

    # Execute the search
    response = client.search(index="movies", knn=knn, size=5)
    
    # Format the results
    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "id": hit["_id"],
            "_score": hit["_score"],
            "title": hit["_source"]["title"],
            "plot": hit["_source"]["plot"],
            "fullplot": hit["_source"]["fullplot"],
        })
    return results

def pretty_search(query):
    """Format the vector search results into a readable string."""
    knowledge = vector_search(query)
    search_result = ""
    for result in knowledge:
        search_result += f"Title: {result.get('title', 'N/A')}, Plot: {result.get('fullplot', 'N/A')}\n"
    return search_result
```

## Step 7: Prepare Queries for the LLM (Gemma)

Combine the user's query with retrieved search results to create a context-rich prompt for the LLM.

```python
def combined_query(query):
    """Retrieve relevant context and format it for the LLM."""
    source_information = pretty_search(query)
    return f"Query: {query}\nContinue to answer the query by using these Search Results:\n{source_information}."

# Test the pipeline
query = "What is the best romantic movie to watch and why?"
combined_results = combined_query(query)
print(combined_results)
```

**Output (truncated):**
```
Query: What is the best romantic movie to watch and why?
Continue to answer the query by using these Search Results:
Title: Shut Up and Kiss Me!, Plot: Ryan and Pete are 27-year old best friends...
Title: Titanic, Plot: The plot focuses on the romances of two couples upon the doomed ship's maiden voyage...
Title: Dark Blue World, Plot: March 15, 1939: Germany invades Czechoslovakia...
...
```

## Next Steps

You have successfully built the retrieval backbone of a RAG system. The output `combined_results` provides a context-augmented prompt ready to be fed into a Large Language Model like **Gemma** to generate a final, sourced answer.

To complete the system:
1.  Load the Gemma model (e.g., using `transformers`).
2.  Pass the `combined_results` string to the model as a prompt.
3.  Configure generation parameters (max tokens, temperature) to get a coherent response.

Your RAG pipeline is now ready to answer questions using the knowledge stored in your Elasticsearch index.