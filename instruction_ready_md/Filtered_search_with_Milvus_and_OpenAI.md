# Filtered Search with Milvus and OpenAI: Finding Your Next Movie

This guide demonstrates how to build a movie recommendation system using semantic search. You will generate embeddings for movie descriptions using OpenAI's API, store them in a Milvus vector database, and perform searches enhanced with metadata filtering to find relevant results.

## Prerequisites

Ensure you have Docker installed and running on your system, as we will use it to launch a Milvus instance. You will also need an OpenAI API key.

## Step 1: Environment Setup

Begin by installing the required Python packages.

```bash
pip install openai pymilvus datasets tqdm
```

## Step 2: Launch Milvus with Docker

We'll use Docker Compose to start a standalone Milvus service. Ensure you are in the directory containing the `docker-compose.yaml` file.

```bash
docker compose up -d
```

This command runs the Milvus service in the background.

## Step 3: Configure Global Variables and Connect to Milvus

Now, let's set up our connection parameters and initialize the Milvus client. Replace `'sk-your_key'` with your actual OpenAI API key.

```python
import openai
from pymilvus import connections, utility, FieldSchema, Collection, CollectionSchema, DataType

# Configuration
HOST = 'localhost'
PORT = 19530
COLLECTION_NAME = 'movie_search'
DIMENSION = 1536  # Dimension for text-embedding-3-small
OPENAI_ENGINE = 'text-embedding-3-small'
openai.api_key = 'sk-your_key'  # Replace with your key

# Index and Search Parameters
INDEX_PARAM = {
    'metric_type': 'L2',
    'index_type': "HNSW",
    'params': {'M': 8, 'efConstruction': 64}
}

QUERY_PARAM = {
    "metric_type": "L2",
    "params": {"ef": 64},
}

BATCH_SIZE = 1000  # Number of records to process in each batch

# Connect to Milvus
connections.connect(host=HOST, port=PORT)
```

## Step 4: Create the Milvus Collection

We'll create a new collection to store our movie data. If a collection with the same name already exists, we'll remove it first.

```python
# Remove the existing collection if it exists
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

# Define the schema for our collection
fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='type', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='release_year', dtype=DataType.INT64),
    FieldSchema(name='rating', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='description', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]

schema = CollectionSchema(fields=fields)
collection = Collection(name=COLLECTION_NAME, schema=schema)
```

## Step 5: Build an Index and Load the Collection

To enable efficient similarity search, we need to create an index on the embedding field and then load the collection into memory.

```python
# Create an HNSW index on the 'embedding' field
collection.create_index(field_name="embedding", index_params=INDEX_PARAM)

# Load the collection for searching
collection.load()
```

## Step 6: Load the Movie Dataset

We'll use the `netflix-shows` dataset from Hugging Face, which contains metadata for thousands of movies and TV shows.

```python
import datasets

# Download the dataset
dataset = datasets.load_dataset('hugginglearners/netflix-shows', split='train')
```

## Step 7: Create an Embedding Function

This function takes a list of text descriptions and uses the OpenAI API to convert them into vector embeddings.

```python
def embed(texts):
    """Convert a list of text strings into embeddings using OpenAI."""
    embeddings = openai.Embedding.create(
        input=texts,
        engine=OPENAI_ENGINE
    )
    # Extract the embedding vectors from the response
    return [x['embedding'] for x in embeddings['data']]
```

## Step 8: Insert Data into Milvus in Batches

To optimize performance, we will process the dataset in batches: we collect metadata, generate embeddings for a batch of descriptions, and then insert the batch into Milvus.

```python
from tqdm import tqdm

# Initialize lists to hold batch data
data = [
    [],  # title
    [],  # type
    [],  # release_year
    [],  # rating
    [],  # description
]

# Iterate through the dataset with a progress bar
for i in tqdm(range(0, len(dataset))):
    # Append metadata for the current movie
    data[0].append(dataset[i]['title'] or '')
    data[1].append(dataset[i]['type'] or '')
    data[2].append(dataset[i]['release_year'] or -1)
    data[3].append(dataset[i]['rating'] or '')
    data[4].append(dataset[i]['description'] or '')

    # When a batch is full, generate embeddings and insert
    if len(data[0]) % BATCH_SIZE == 0:
        data.append(embed(data[4]))  # Generate embeddings for the batch descriptions
        collection.insert(data)      # Insert the batch into Milvus
        data = [[], [], [], [], []]  # Reset for the next batch

# Process any remaining records in the final batch
if len(data[0]) != 0:
    data.append(embed(data[4]))
    collection.insert(data)
```

## Step 9: Perform a Filtered Semantic Search

With the data loaded, you can now query the database. The `query` function performs a vector similarity search combined with a metadata filter. The filter uses Milvus's boolean expression syntax.

```python
import textwrap

def query(search_query, top_k=5):
    """
    Perform a filtered vector search.
    
    Args:
        search_query: A tuple containing (search_text, filter_expression).
        top_k: Number of results to return.
    """
    text, expr = search_query
    
    # Generate embedding for the query text
    query_embedding = embed([text])
    
    # Execute the search
    results = collection.search(
        query_embedding,
        anns_field='embedding',
        expr=expr,
        param=QUERY_PARAM,
        limit=top_k,
        output_fields=['title', 'type', 'release_year', 'rating', 'description']
    )
    
    # Print the query and filter
    print('Search Description:', text)
    print('Filter Expression:', expr)
    print('\nResults:')
    
    # Display each result
    for hits in results:
        for rank, hit in enumerate(hits):
            print(f'\tRank: {rank + 1}, Score: {hit.score:.4f}, Title: {hit.entity.get("title")}')
            print(f'\t\tType: {hit.entity.get("type")}, '
                  f'Release Year: {hit.entity.get("release_year")}, '
                  f'Rating: {hit.entity.get("rating")}')
            # Wrap the description for readability
            wrapped_desc = textwrap.fill(hit.entity.get('description'), width=88)
            print(f'\t\tDescription: {wrapped_desc}\n')

# Example Query: Find movies about fluffy animals released before 2019 with a PG rating
my_query = ('movie about a fluffy animal', 'release_year < 2019 and rating like "PG%"')
query(my_query)
```

This query will return the top 5 movies whose descriptions are semantically similar to "movie about a fluffy animal", but only if they were released before 2019 and have a rating starting with "PG" (e.g., PG, PG-13).

## Summary

You have successfully built a movie search system that combines the power of OpenAI's semantic embeddings with Milvus's fast vector search and flexible metadata filtering. This pattern can be adapted for various recommendation and retrieval-augmented generation (RAG) applications.