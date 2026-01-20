# Getting Started with Milvus and OpenAI
### Finding your next book

In this notebook we will be going over generating embeddings of book descriptions with OpenAI and using those embeddings within Milvus to find relevant books. The dataset in this example is sourced from HuggingFace datasets, and contains a little over 1 million title-description pairs.

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

With Milvus running we can setup our global variables:
- HOST: The Milvus host address
- PORT: The Milvus port number
- COLLECTION_NAME: What to name the collection within Milvus
- DIMENSION: The dimension of the embeddings
- OPENAI_ENGINE: Which embedding model to use
- openai.api_key: Your OpenAI account key
- INDEX_PARAM: The index settings to use for the collection
- QUERY_PARAM: The search parameters to use
- BATCH_SIZE: How many texts to embed and insert at once

```python
import openai

HOST = 'localhost'
PORT = 19530
COLLECTION_NAME = 'book_search'
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

## Milvus
This segment deals with Milvus and setting up the database for this use case. Within Milvus we need to setup a collection and index the collection. 

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
With Milvus up and running we can begin grabbing our data. Hugging Face Datasets is a hub that holds many different user datasets, and for this example we are using Skelebor's book dataset. This dataset contains title-description pairs for over 1 million books. We are going to embed each description and store it within Milvus along with its title. 

```python
import datasets

# Download the dataset and only use the `train` portion (file is around 800Mb)
dataset = datasets.load_dataset('Skelebor/book_titles_and_descriptions_en_clean', split='train')
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

This next step does the actual inserting. Due to having so many datapoints, if you want to immidiately test it out you can stop the inserting cell block early and move along. Doing this will probably decrease the accuracy of the results due to less datapoints, but it should still be good enough. 

```python
from tqdm import tqdm

data = [
    [], # title
    [], # description
]

# Embed and insert in batches
for i in tqdm(range(0, len(dataset))):
    data[0].append(dataset[i]['title'])
    data[1].append(dataset[i]['description'])
    if len(data[0]) % BATCH_SIZE == 0:
        data.append(embed(data[1]))
        collection.insert(data)
        data = [[],[]]

# Embed and insert the remainder 
if len(data[0]) != 0:
    data.append(embed(data[1]))
    collection.insert(data)
    data = [[],[]]
```

## Query the Database
With our data safely inserted in Milvus, we can now perform a query. The query takes in a string or a list of strings and searches them. The resuts print out your provided description and the results that include the result score, the result title, and the result book description. 

```python
import textwrap

def query(queries, top_k = 5):
    if type(queries) != list:
        queries = [queries]
    res = collection.search(embed(queries), anns_field='embedding', param=QUERY_PARAM, limit = top_k, output_fields=['title', 'description'])
    for i, hit in enumerate(res):
        print('Description:', queries[i])
        print('Results:')
        for ii, hits in enumerate(hit):
            print('\t' + 'Rank:', ii + 1, 'Score:', hits.score, 'Title:', hits.entity.get('title'))
            print(textwrap.fill(hits.entity.get('description'), 88))
            print()
```

```python
query('Book about a k-9 from europe')
```