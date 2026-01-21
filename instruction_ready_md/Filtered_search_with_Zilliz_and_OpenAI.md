# Filtered Movie Search with Zilliz and OpenAI

## Overview
This guide demonstrates how to build a semantic movie search engine using OpenAI's embedding models and Zilliz's vector database. You'll learn to:
1.  Generate vector embeddings for movie descriptions.
2.  Store these embeddings and associated metadata in Zilliz.
3.  Perform filtered searches that combine semantic similarity with metadata constraints (e.g., release year, rating).

## Prerequisites & Setup

### 1. Install Required Libraries
Run the following command to install the necessary Python packages.

```bash
pip install openai pymilvus datasets tqdm
```

### 2. Configure Your Credentials and Parameters
Before running the code, you must set up your Zilliz database and obtain an OpenAI API key. Replace the placeholder values in the script below with your own credentials.

```python
import openai

# Zilliz Cloud Configuration
URI = 'your_uri'  # e.g., "https://your-cluster.aws.zillizcloud.com:443"
TOKEN = 'your_token'  # Format: "username:password" or "api_key"
COLLECTION_NAME = 'movie_search'

# OpenAI Configuration
OPENAI_ENGINE = 'text-embedding-3-small'
openai.api_key = 'sk-your_key_here'  # Your OpenAI API key

# Application Parameters
DIMENSION = 1536  # Dimension for text-embedding-3-small
BATCH_SIZE = 1000  # Number of records to process in each batch

# Zilliz Index and Search Parameters
INDEX_PARAM = {
    'metric_type': 'L2',
    'index_type': "AUTOINDEX",
    'params': {}
}

QUERY_PARAM = {
    "metric_type": "L2",
    "params": {},
}
```

## Step 1: Connect to Zilliz and Create a Collection

First, establish a connection to your Zilliz database and define the schema for your movie collection.

```python
from pymilvus import connections, utility, FieldSchema, Collection, CollectionSchema, DataType

# 1. Connect to Zilliz
connections.connect(uri=URI, token=TOKEN)

# 2. Remove the collection if it already exists (for a clean start)
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

# 3. Define the collection schema
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

# 4. Create the collection
collection = Collection(name=COLLECTION_NAME, schema=schema)

# 5. Create an index on the embedding field and load the collection into memory
collection.create_index(field_name="embedding", index_params=INDEX_PARAM)
collection.load()
```

## Step 2: Load the Movie Dataset

We'll use a Netflix shows dataset from Hugging Face Datasets, which contains metadata like title, description, type, release year, and rating.

```python
import datasets

# Download the dataset
dataset = datasets.load_dataset('hugginglearners/netflix-shows', split='train')
print(f"Loaded dataset with {len(dataset)} entries.")
```

## Step 3: Embed and Insert Data into Zilliz

### Define the Embedding Function
This function sends a batch of text descriptions to the OpenAI API and returns their vector embeddings.

```python
def embed(texts):
    """Convert a list of text strings into embeddings using OpenAI."""
    embeddings = openai.Embedding.create(
        input=texts,
        engine=OPENAI_ENGINE
    )
    return [x['embedding'] for x in embeddings['data']]
```

### Batch Insertion Process
To efficiently process thousands of records, we'll embed and insert data in batches.

```python
from tqdm import tqdm

# Initialize lists to hold batch data
data_batch = [
    [],  # titles
    [],  # types
    [],  # release_years
    [],  # ratings
    [],  # descriptions
]

# Iterate through the dataset
for i in tqdm(range(0, len(dataset))):
    # Append metadata for the current movie
    data_batch[0].append(dataset[i]['title'] or '')
    data_batch[1].append(dataset[i]['type'] or '')
    data_batch[2].append(dataset[i]['release_year'] or -1)
    data_batch[3].append(dataset[i]['rating'] or '')
    data_batch[4].append(dataset[i]['description'] or '')

    # When the batch reaches the specified size, embed and insert
    if len(data_batch[0]) % BATCH_SIZE == 0:
        # Generate embeddings for the descriptions in this batch
        data_batch.append(embed(data_batch[4]))
        # Insert the batch into Zilliz
        collection.insert(data_batch)
        # Reset the batch lists
        data_batch = [[], [], [], [], []]

# Insert any remaining records in the final, partial batch
if len(data_batch[0]) != 0:
    data_batch.append(embed(data_batch[4]))
    collection.insert(data_batch)
```

## Step 4: Perform Filtered Semantic Searches

Now you can query the database. The `query` function performs a vector similarity search combined with a metadata filter expression.

```python
import textwrap

def query(search_query, top_k=5):
    """
    Execute a filtered semantic search.
    
    Args:
        search_query (tuple): A tuple containing (search_text, filter_expression).
        top_k (int): Number of results to return.
    """
    text, expr = search_query
    
    # 1. Embed the search query
    query_embedding = embed([text])
    
    # 2. Perform the search in Zilliz
    results = collection.search(
        data=query_embedding,
        anns_field='embedding',
        expr=expr,
        param=QUERY_PARAM,
        limit=top_k,
        output_fields=['title', 'type', 'release_year', 'rating', 'description']
    )
    
    # 3. Print the results
    print(f'Query: "{text}"')
    print(f'Filter: {expr}')
    print('-' * 50)
    
    for i, hits in enumerate(results):
        for rank, hit in enumerate(hits):
            print(f'Rank: {rank + 1}, Score: {hit.score:.4f}')
            print(f'Title: {hit.entity.get("title")}')
            print(f'Type: {hit.entity.get("type")}, Year: {hit.entity.get("release_year")}, Rating: {hit.entity.get("rating")}')
            # Wrap the description for readability
            wrapped_desc = textwrap.fill(hit.entity.get('description'), width=88)
            print(f'Description: {wrapped_desc}')
            print()
```

### Example Query
Let's search for movies about fluffy animals that were released before 2019 and have a PG rating.

```python
my_query = ('movie about a fluffy animal', 'release_year < 2019 and rating like "PG%"')
query(my_query)
```

**Expected Output:**
```
Query: "movie about a fluffy animal"
Filter: release_year < 2019 and rating like "PG%"
--------------------------------------------------
Rank: 1, Score: 0.3009
Title: The Lamb
Type: Movie, Year: 2017, Rating: PG
Description: A big-dreaming donkey escapes his menial existence and befriends some
free-spirited animal pals in this imaginative retelling of the Nativity Story.

Rank: 2, Score: 0.3353
Title: Puss in Boots
Type: Movie, Year: 2011, Rating: PG
Description: The fabled feline heads to the Land of Giants with friends Humpty Dumpty
and Kitty Softpaws on a quest to nab its greatest treasure: the Golden Goose.

... (additional results)
```

## Summary
You have successfully built a filtered semantic search engine for movies. By combining OpenAI's powerful text embeddings with Zilliz's vector search and filtering capabilities, you can retrieve relevant results based on both meaning and specific metadata criteria. You can extend this by experimenting with different filter expressions or search queries.