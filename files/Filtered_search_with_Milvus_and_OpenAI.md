# Filtered Search with Milvus and OpenAI
### Finding your next movie

In this notebook we will be going over generating embeddings of movie descriptions with OpenAI and using those embeddings within Milvus to find relevant movies. To narrow our search results and try something new, we are going to be using filtering to do metadata searches. The dataset in this example is sourced from HuggingFace datasets, and contains a little over 8 thousand movie entries.

Lets begin by first downloading the required libraries for this notebook:
- `openai` is used for communicating with the OpenAI embedding service
- `pymilvus` is used for communicating with the Milvus server
- `datasets` is used for downloading the dataset
- `tqdm` is used for the progress bars

```python
! pip install openai pymilvus datasets tqdm
```

With the required packages installed we can get started. Lets begin by launching the Milvus service. The file being run is the `docker-compose.yaml` found in the folder of this file. This command launches a Milvus standalone instance which we will use for this test.  

```python
! docker compose up -d
```

[First Entry, ..., Last Entry]

With Milvus running we can setup our global variables:
- HOST: The Milvus host address
- PORT: The Milvus port number
- COLLECTION_NAME: What to name the collection within Milvus
- DIMENSION: The dimension of the embeddings
- OPENAI_ENGINE: Which embedding model to use
- openai.api_key: Your OpenAI account key
- INDEX_PARAM: The index settings to use for the collection
- QUERY_PARAM: The search parameters to use
- BATCH_SIZE: How many movies to embed and insert at once

```python
import openai

HOST = 'localhost'
PORT = 19530
COLLECTION_NAME = 'movie_search'
DIMENSION = 1536
OPENAI_ENGINE = 'text-embedding-3-small'
openai.api_key = 'sk-your_key'

INDEX_PARAM = {
    'metric_type':'L2',
    'index_type':"HNSW",
    'params':{'M': 8, 'efConstruction': 64}
}

QUERY_PARAM = {
    "metric_type": "L2",
    "params": {"ef": 64},
}

BATCH_SIZE = 1000
```

```python
from pymilvus import connections, utility, FieldSchema, Collection, CollectionSchema, DataType

# Connect to Milvus Database
connections.connect(host=HOST, port=PORT)
```

```python
# Remove collection if it already exists
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)
```

```python
# Create collection which includes the id, title, and embedding.
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

```python
# Create the index on the collection and load it.
collection.create_index(field_name="embedding", index_params=INDEX_PARAM)
collection.load()
```

## Dataset
With Milvus up and running we can begin grabbing our data. Hugging Face Datasets is a hub that holds many different user datasets, and for this example we are using HuggingLearners's netflix-shows dataset. This dataset contains movies and their metadata pairs for over 8 thousand movies. We are going to embed each description and store it within Milvus along with its title, type, release_year and rating.

```python
import datasets

# Download the dataset 
dataset = datasets.load_dataset('hugginglearners/netflix-shows', split='train')
```

## Insert the Data
Now that we have our data on our machine we can begin embedding it and inserting it into Milvus. The embedding function takes in text and returns the embeddings in a list format. 

```python
# Simple function that converts the texts to embeddings
def embed(texts):
    embeddings = openai.Embedding.create(
        input=texts,
        engine=OPENAI_ENGINE
    )
    return [x['embedding'] for x in embeddings['data']]

```

This next step does the actual inserting. We iterate through all the entries and create batches that we insert once we hit our set batch size. After the loop is over we insert the last remaning batch if it exists. 

```python
from tqdm import tqdm

data = [
    [], # title
    [], # type
    [], # release_year
    [], # rating
    [], # description
]

# Embed and insert in batches
for i in tqdm(range(0, len(dataset))):
    data[0].append(dataset[i]['title'] or '')
    data[1].append(dataset[i]['type'] or '')
    data[2].append(dataset[i]['release_year'] or -1)
    data[3].append(dataset[i]['rating'] or '')
    data[4].append(dataset[i]['description'] or '')
    if len(data[0]) % BATCH_SIZE == 0:
        data.append(embed(data[4]))
        collection.insert(data)
        data = [[],[],[],[],[]]

# Embed and insert the remainder 
if len(data[0]) != 0:
    data.append(embed(data[4]))
    collection.insert(data)
    data = [[],[],[],[],[]]

```

[First Entry, ..., Last Entry]

## Query the Database
With our data safely inserted in Milvus, we can now perform a query. The query takes in a tuple of the movie description you are searching for an the filter to use. More info about the filter can be found [here](https://milvus.io/docs/boolean.md). The search first prints out your description and filter expression. After that for each result we print the score, title, type, release year, rating, and description of the result movies. 

```python
import textwrap

def query(query, top_k = 5):
    text, expr = query
    res = collection.search(embed(text), anns_field='embedding', expr = expr, param=QUERY_PARAM, limit = top_k, output_fields=['title', 'type', 'release_year', 'rating', 'description'])
    for i, hit in enumerate(res):
        print('Description:', text, 'Expression:', expr)
        print('Results:')
        for ii, hits in enumerate(hit):
            print('\t' + 'Rank:', ii + 1, 'Score:', hits.score, 'Title:', hits.entity.get('title'))
            print('\t\t' + 'Type:', hits.entity.get('type'), 'Release Year:', hits.entity.get('release_year'), 'Rating:', hits.entity.get('rating'))
            print(textwrap.fill(hits.entity.get('description'), 88))
            print()

my_query = ('movie about a fluffly animal', 'release_year < 2019 and rating like \"PG%\"')

query(my_query)
```