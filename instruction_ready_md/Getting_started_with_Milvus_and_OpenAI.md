# Building a Book Search Engine with Milvus and OpenAI Embeddings

This guide walks you through creating a semantic book search engine. You'll generate embeddings for over a million book descriptions using OpenAI's API, store them in a Milvus vector database, and perform similarity searches to find relevant books based on natural language queries.

## Prerequisites

Ensure you have Docker installed and running on your system, as we'll use it to launch a Milvus instance. You'll also need an OpenAI API key.

## Step 1: Install Required Libraries

Begin by installing the necessary Python packages.

```bash
pip install openai pymilvus datasets tqdm
```

## Step 2: Launch the Milvus Service

We'll use Docker Compose to run a standalone Milvus instance. Ensure you are in the directory containing the `docker-compose.yaml` file.

```bash
docker compose up -d
```

This command starts the Milvus service in the background.

## Step 3: Configure Global Variables and Imports

Now, set up your environment by defining configuration variables and importing libraries.

```python
import openai

# Milvus connection settings
HOST = 'localhost'
PORT = 19530
COLLECTION_NAME = 'book_search'

# Embedding model configuration
DIMENSION = 1536  # Dimension for text-embedding-3-small
OPENAI_ENGINE = 'text-embedding-3-small'
openai.api_key = 'sk-your_key'  # Replace with your actual OpenAI API key

# Index parameters for Milvus (HNSW index for efficient similarity search)
INDEX_PARAM = {
    'metric_type': 'L2',
    'index_type': "HNSW",
    'params': {'M': 8, 'efConstruction': 64}
}

# Search parameters
QUERY_PARAM = {
    "metric_type": "L2",
    "params": {"ef": 64},
}

# Batch size for processing the dataset
BATCH_SIZE = 1000
```

## Step 4: Set Up the Milvus Database

In this step, you'll connect to Milvus, create a collection to store your data, and build an index for fast searching.

```python
from pymilvus import connections, utility, FieldSchema, Collection, CollectionSchema, DataType

# 1. Connect to the Milvus server
connections.connect(host=HOST, port=PORT)

# 2. Remove the collection if it already exists (for a clean start)
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

# 3. Define the schema for the collection
# The collection will store an auto-generated ID, the book title, description, and its vector embedding.
fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='description', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]
schema = CollectionSchema(fields=fields)

# 4. Create the collection
collection = Collection(name=COLLECTION_NAME, schema=schema)

# 5. Create an index on the embedding field and load the collection into memory
collection.create_index(field_name="embedding", index_params=INDEX_PARAM)
collection.load()
```

## Step 5: Load the Book Dataset

We'll use a public dataset from Hugging Face containing over a million book titles and descriptions.

```python
import datasets

# Download the dataset (the train split, which is approximately 800MB)
dataset = datasets.load_dataset('Skelebor/book_titles_and_descriptions_en_clean', split='train')
```

## Step 6: Create the Embedding Function

Define a helper function that sends text to the OpenAI API and returns the corresponding vector embeddings.

```python
def embed(texts):
    """Convert a list of text strings into embedding vectors."""
    embeddings = openai.Embedding.create(
        input=texts,
        engine=OPENAI_ENGINE
    )
    # Extract just the embedding vectors from the API response
    return [x['embedding'] for x in embeddings['data']]
```

## Step 7: Insert Data into Milvus in Batches

To efficiently process over a million records, we'll embed and insert data in batches.

```python
from tqdm import tqdm

# Initialize a list to batch titles and descriptions
data_batch = [
    [],  # For titles
    [],  # For descriptions
]

# Iterate through the dataset with a progress bar
for i in tqdm(range(0, len(dataset))):
    data_batch[0].append(dataset[i]['title'])
    data_batch[1].append(dataset[i]['description'])

    # When the batch reaches the defined size, embed and insert
    if len(data_batch[0]) % BATCH_SIZE == 0:
        # Generate embeddings for the batch of descriptions
        embeddings = embed(data_batch[1])
        # Append embeddings to the data batch
        data_batch.append(embeddings)
        # Insert the batch (titles, descriptions, embeddings) into Milvus
        collection.insert(data_batch)
        # Reset the batch for the next chunk of data
        data_batch = [[], []]

# Handle any remaining records that didn't fill a complete batch
if len(data_batch[0]) != 0:
    embeddings = embed(data_batch[1])
    data_batch.append(embeddings)
    collection.insert(data_batch)
```

**Note:** This insertion process will take some time due to the dataset size. You can interrupt the cell early if you wish to proceed with a smaller subset for testing. This will reduce result accuracy but allows for quicker validation.

## Step 8: Query the Search Engine

Finally, create a function to query your book database. It will take a natural language description, convert it to an embedding, and find the most similar books in Milvus.

```python
import textwrap

def query(queries, top_k=5):
    """
    Search for books similar to the provided query string(s).

    Args:
        queries (str or list): A query string or list of query strings.
        top_k (int): The number of top results to return.
    """
    # Ensure queries is a list
    if not isinstance(queries, list):
        queries = [queries]

    # Generate embeddings for the search queries
    query_embeddings = embed(queries)

    # Perform the vector similarity search in Milvus
    results = collection.search(
        query_embeddings,
        anns_field='embedding',
        param=QUERY_PARAM,
        limit=top_k,
        output_fields=['title', 'description']  # Return these fields in the results
    )

    # Print the results in a readable format
    for i, hits in enumerate(results):
        print('Query Description:', queries[i])
        print('Search Results:')
        for rank, hit in enumerate(hits):
            print(f'\tRank: {rank + 1}, Score: {hit.score:.4f}, Title: {hit.entity.get("title")}')
            # Wrap the description text for better readability
            print(textwrap.fill(hit.entity.get('description'), width=88))
            print()
```

## Step 9: Perform a Search

Test your search engine with a sample query.

```python
query('Book about a k-9 from europe')
```

This will output the top 5 most relevant books based on the semantic meaning of your query, including their titles, descriptions, and a similarity score.

## Summary

You've successfully built a semantic book search engine. You launched a Milvus vector database, populated it with embeddings from a large book dataset using OpenAI's API, and created a query interface to find books based on conceptual similarity rather than just keyword matching. You can extend this by building a front-end interface or experimenting with different embedding models and search parameters.