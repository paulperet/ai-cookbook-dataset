# Running Hybrid VSS Queries with Redis and OpenAI

This notebook provides an introduction to using Redis as a vector database with OpenAI embeddings and running hybrid queries that combine VSS and lexical search using Redis Query and Search capability. Redis is a scalable, real-time database that can be used as a vector database when using the [RediSearch Module](https://oss.redislabs.com/redisearch/). The Redis Query and Search capability allows you to index and search for vectors in Redis. This notebook will show you how to use the Redis Query and Search to index and search for vectors created by using the OpenAI API and stored in Redis.

Hybrid queries combine vector similarity with traditional Redis Query and Search filtering capabilities on GEO, NUMERIC, TAG or TEXT data simplifying application code. A common example of a hybrid query in an e-commerce use case is to find items visually similar to a given query image limited to items available in a GEO location and within a price range.

## Prerequisites

Before we start this project, we need to set up the following:

* start a Redis database with RediSearch (redis-stack)
* install libraries
    * [Redis-py](https://github.com/redis/redis-py)
* get your [OpenAI API key](https://beta.openai.com/account/api-keys)

===========================================================

### Start Redis

To keep this example simple, we will use the Redis Stack docker container which we can start as follows

```bash
$ docker-compose up -d
```

This also includes the [RedisInsight](https://redis.com/redis-enterprise/redis-insight/) GUI for managing your Redis database which you can view at [http://localhost:8001](http://localhost:8001) once you start the docker container.

You're all set up and ready to go! Next, we import and create our client for communicating with the Redis database we just created.

## Install Requirements

Redis-Py is the python client for communicating with Redis. We will use this to communicate with our Redis-stack database. 

```python
! pip install redis pandas openai
```

===========================================================
## Prepare your OpenAI API key

The `OpenAI API key` is used for vectorization of query data.

If you don't have an OpenAI API key, you can get one from [https://beta.openai.com/account/api-keys](https://beta.openai.com/account/api-keys).

Once you get your key, please add it to your environment variables as `OPENAI_API_KEY` by using following command:

```python
# Test that your OpenAI API key is correctly set as an environment variable
# Note. if you run this notebook locally, you will need to reload your terminal and the notebook for the env variables to be live.
import os
import openai

os.environ["OPENAI_API_KEY"] = '<YOUR_OPENAI_API_KEY>'

if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")
```

## Load data

In this section we'll load and clean an ecommerce dataset. We'll generate embeddings using OpenAI and use this data to create an index in Redis and then search for similar vectors.

```python
import pandas as pd
import numpy as np
from typing import List

from utils.embeddings_utils import (
    get_embeddings,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)

EMBEDDING_MODEL = "text-embedding-3-small"

# load in data and clean data types and drop null rows
df = pd.read_csv("../../data/styles_2k.csv", on_bad_lines='skip')
df.dropna(inplace=True)
df["year"] = df["year"].astype(int)
df.info()

# print dataframe
n_examples = 5
df.head(n_examples)
```

```python
df["product_text"] = df.apply(lambda row: f"name {row['productDisplayName']} category {row['masterCategory']} subcategory {row['subCategory']} color {row['baseColour']} gender {row['gender']}".lower(), axis=1)
df.rename({"id":"product_id"}, inplace=True, axis=1)

df.info()
```

```python
# check out one of the texts we will use to create semantic embeddings
df["product_text"][0]
```

## Connect to Redis

Now that we have our Redis database running, we can connect to it using the Redis-py client. We will use the default host and port for the Redis database which is `localhost:6379`.

```python
import redis
from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)
from redis.commands.search.query import Query
from redis.commands.search.field import (
    TagField,
    NumericField,
    TextField,
    VectorField
)

REDIS_HOST =  "localhost"
REDIS_PORT = 6379
REDIS_PASSWORD = "" # default for passwordless Redis

# Connect to Redis
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD
)
redis_client.ping()
```

## Creating a Search Index in Redis

The below cells will show how to specify and create a search index in Redis. We will:

1. Set some constants for defining our index like the distance metric and the index name
2. Define the index schema with RediSearch fields
3. Create the index

```python
# Constants
INDEX_NAME = "product_embeddings"           # name of the search index
PREFIX = "doc"                            # prefix for the document keys
DISTANCE_METRIC = "L2"                # distance metric for the vectors (ex. COSINE, IP, L2)
NUMBER_OF_VECTORS = len(df)
```

```python
# Define RediSearch fields for each of the columns in the dataset
name = TextField(name="productDisplayName")
category = TagField(name="masterCategory")
articleType = TagField(name="articleType")
gender = TagField(name="gender")
season = TagField(name="season")
year = NumericField(name="year")
text_embedding = VectorField("product_vector",
    "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": 1536,
        "DISTANCE_METRIC": DISTANCE_METRIC,
        "INITIAL_CAP": NUMBER_OF_VECTORS,
    }
)
fields = [name, category, articleType, gender, season, year, text_embedding]
```

```python
# Check if index exists
try:
    redis_client.ft(INDEX_NAME).info()
    print("Index already exists")
except:
    # Create RediSearch Index
    redis_client.ft(INDEX_NAME).create_index(
        fields = fields,
        definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
)
```

## Generate OpenAI Embeddings and Load Documents into the Index

Now that we have a search index, we can load documents into it. We will use the dataframe containing the styles dataset loaded previously. In Redis, either the HASH or JSON (if using RedisJSON in addition to RediSearch) data types can be used to store documents. We will use the HASH data type in this example. The cells below will show how to get OpenAI embeddings for the different products and load documents into the index.

```python
# Use OpenAI get_embeddings batch requests to speed up embedding creation
def embeddings_batch_request(documents: pd.DataFrame):
    records = documents.to_dict("records")
    print("Records to process: ", len(records))
    product_vectors = []
    docs = []
    batchsize = 1000

    for idx,doc in enumerate(records,start=1):
        # create byte vectors
        docs.append(doc["product_text"])
        if idx % batchsize == 0:
            product_vectors += get_embeddings(docs, EMBEDDING_MODEL)
            docs.clear()
            print("Vectors processed ", len(product_vectors), end='\r')
    product_vectors += get_embeddings(docs, EMBEDDING_MODEL)
    print("Vectors processed ", len(product_vectors), end='\r')
    return product_vectors
```

```python
def index_documents(client: redis.Redis, prefix: str, documents: pd.DataFrame):
    product_vectors = embeddings_batch_request(documents)
    records = documents.to_dict("records")
    batchsize = 500

    # Use Redis pipelines to batch calls and save on round trip network communication
    pipe = client.pipeline()
    for idx,doc in enumerate(records,start=1):
        key = f"{prefix}:{str(doc['product_id'])}"

        # create byte vectors
        text_embedding = np.array((product_vectors[idx-1]), dtype=np.float32).tobytes()

        # replace list of floats with byte vectors
        doc["product_vector"] = text_embedding

        pipe.hset(key, mapping = doc)
        if idx % batchsize == 0:
            pipe.execute()
    pipe.execute()
```

```python
%%time
index_documents(redis_client, PREFIX, df)
print(f"Loaded {redis_client.info()['db0']['keys']} documents in Redis search index with name: {INDEX_NAME}")
```

## Simple Vector Search Queries with OpenAI Query Embeddings

Now that we have a search index and documents loaded into it, we can run search queries. Below we will provide a function that will run a search query and return the results. Using this function we run a few queries that will show how you can utilize Redis as a vector database.

```python
def search_redis(
    redis_client: redis.Redis,
    user_query: str,
    index_name: str = "product_embeddings",
    vector_field: str = "product_vector",
    return_fields: list = ["productDisplayName", "masterCategory", "gender", "season", "year", "vector_score"],
    hybrid_fields = "*",
    k: int = 20,
    print_results: bool = True,
) -> List[dict]:

    # Use OpenAI to create embedding vector from user query
    embedded_query = openai.Embedding.create(input=user_query,
                                            model="text-embedding-3-small",
                                            )["data"][0]['embedding']

    # Prepare the Query
    base_query = f'{hybrid_fields}=>[KNN {k} @{vector_field} $vector AS vector_score]'
    query = (
        Query(base_query)
         .return_fields(*return_fields)
         .sort_by("vector_score")
         .paging(0, k)
         .dialect(2)
    )
    params_dict = {"vector": np.array(embedded_query).astype(dtype=np.float32).tobytes()}

    # perform vector search
    results = redis_client.ft(index_name).search(query, params_dict)
    if print_results:
        for i, product in enumerate(results.docs):
            score = 1 - float(product.vector_score)
            print(f"{i}. {product.productDisplayName} (Score: {round(score ,3) })")
    return results.docs
```

```python
# Execute a simple vector search in Redis
results = search_redis(redis_client, 'man blue jeans', k=10)
```

## Hybrid Queries with Redis

The previous examples showed how run vector search queries with RediSearch. In this section, we will show how to combine vector search with other RediSearch fields for hybrid search. In the example below, we will combine vector search with full text search.

```python
# improve search quality by adding hybrid query for "man blue jeans" in the product vector combined with a phrase search for "blue jeans"
results = search_redis(redis_client,
                       "man blue jeans",
                       vector_field="product_vector",
                       k=10,
                       hybrid_fields='@productDisplayName:"blue jeans"'
                       )
```

```python
# hybrid query for shirt in the product vector and only include results with the phrase "slim fit" in the title
results = search_redis(redis_client,
                       "shirt",
                       vector_field="product_vector",
                       k=10,
                       hybrid_fields='@productDisplayName:"slim fit"'
                       )
```

```python
# hybrid query for watch in the product vector and only include results with the tag "Accessories" in the masterCategory field
results = search_redis(redis_client,
                       "watch",
                       vector_field="product_vector",
                       k=10,
                       hybrid_fields='@masterCategory:{Accessories}'
                       )
```

```python
# hybrid query for sandals in the product vector and only include results within the 2011-2012 year range
results = search_redis(redis_client,
                       "sandals",
                       vector_field="product_vector",
                       k=10,
                       hybrid_fields='@year:[2011 2012]'
                       )
```

```python
# hybrid query for sandals in the product vector and only include results within the 2011-2012 year range from the summer season
results = search_redis(redis_client,
                       "blue sandals",
                       vector_field="product_vector",
                       k=10,
                       hybrid_fields='(@year:[2011 2012] @season:{Summer})'
                       )
```

```python
# hybrid query for a brown belt filtering results by a year (NUMERIC) with a specific article types (TAG) and with a brand name (TEXT)
results = search_redis(redis_client,
                       "brown belt",
                       vector_field="product_vector",
                       k=10,
                       hybrid_fields='(@year:[2012 2012] @articleType:{Shirts | Belts} @productDisplayName:"Wrangler")'
                       )
```