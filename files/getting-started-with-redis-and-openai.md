# Using Redis as a Vector Database with OpenAI

This notebook provides an introduction to using Redis as a vector database with OpenAI embeddings. Redis is a scalable, real-time database that can be used as a vector database when using the [RediSearch Module](https://oss.redislabs.com/redisearch/). The RediSearch module allows you to index and search for vectors in Redis. This notebook will show you how to use the RediSearch module to index and search for vectors created by using the OpenAI API and stored in Redis.

### What is Redis?

Most developers from a web services background are probably familiar with Redis. At it's core, Redis is an open-source key-value store that can be used as a cache, message broker, and database. Developers choice Redis because it is fast, has a large ecosystem of client libraries, and has been deployed by major enterprises for years.

In addition to the traditional uses of Redis. Redis also provides [Redis Modules](https://redis.io/modules) which are a way to extend Redis with new data types and commands. Example modules include [RedisJSON](https://redis.io/docs/stack/json/), [RedisTimeSeries](https://redis.io/docs/stack/timeseries/), [RedisBloom](https://redis.io/docs/stack/bloom/) and [RediSearch](https://redis.io/docs/stack/search/).

### What is RediSearch?

RediSearch is a [Redis module](https://redis.io/modules) that provides querying, secondary indexing, full-text search and vector search for Redis. To use RediSearch, you first declare indexes on your Redis data. You can then use the RediSearch clients to query that data. For more information on the feature set of RediSearch, see the [README](./README.md) or the [RediSearch documentation](https://redis.io/docs/stack/search/).

### Deployment options

There are a number of ways to deploy Redis. For local development, the quickest method is to use the [Redis Stack docker container](https://hub.docker.com/r/redis/redis-stack) which we will use here. Redis Stack contains a number of Redis modules that can be used together to create a fast, multi-model data store and query engine.

For production use cases, The easiest way to get started is to use the [Redis Cloud](https://redislabs.com/redis-enterprise-cloud/overview/) service. Redis Cloud is a fully managed Redis service. You can also deploy Redis on your own infrastructure using [Redis Enterprise](https://redislabs.com/redis-enterprise/overview/). Redis Enterprise is a fully managed Redis service that can be deployed in kubernetes, on-premises or in the cloud.

Additionally, every major cloud provider ([AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-e6y7ork67pjwg?sr=0-2&ref_=beagle&applicationId=AWSMPContessa), [Google Marketplace](https://console.cloud.google.com/marketplace/details/redislabs-public/redis-enterprise?pli=1), or [Azure Marketplace](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/garantiadata.redis_enterprise_1sp_public_preview?tab=Overview)) offers Redis Enterprise in a marketplace offering.

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
! pip install redis wget pandas openai
```

===========================================================
## Prepare your OpenAI API key

The `OpenAI API key` is used for vectorization of query data.

If you don't have an OpenAI API key, you can get one from [https://beta.openai.com/account/api-keys](https://beta.openai.com/account/api-keys).

Once you get your key, please add it to your environment variables as `OPENAI_API_KEY` by using following command:

```python
! export OPENAI_API_KEY="your API key"
```

```python
# Test that your OpenAI API key is correctly set as an environment variable
# Note. if you run this notebook locally, you will need to reload your terminal and the notebook for the env variables to be live.
import os
import openai

# Note. alternatively you can set a temporary env variable like this:
# os.environ["OPENAI_API_KEY"] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")
```

## Load data

In this section we'll load embedded data that has already been converted into vectors. We'll use this data to create an index in Redis and then search for similar vectors.

```python
import sys
import numpy as np
import pandas as pd
from typing import List

# use helper function in nbutils.py to download and read the data
# this should take from 5-10 min to run
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import nbutils

nbutils.download_wikipedia_data()
data = nbutils.read_wikipedia_data()

data.head()
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
VECTOR_DIM = len(data['title_vector'][0]) # length of the vectors
VECTOR_NUMBER = len(data)                 # initial number of vectors
INDEX_NAME = "embeddings-index"           # name of the search index
PREFIX = "doc"                            # prefix for the document keys
DISTANCE_METRIC = "COSINE"                # distance metric for the vectors (ex. COSINE, IP, L2)
```

```python
# Define RediSearch fields for each of the columns in the dataset
title = TextField(name="title")
url = TextField(name="url")
text = TextField(name="text")
title_embedding = VectorField("title_vector",
    "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC,
        "INITIAL_CAP": VECTOR_NUMBER,
    }
)
text_embedding = VectorField("content_vector",
    "FLAT", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC,
        "INITIAL_CAP": VECTOR_NUMBER,
    }
)
fields = [title, url, text, title_embedding, text_embedding]
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

## Load Documents into the Index

Now that we have a search index, we can load documents into it. We will use the same documents we used in the previous examples. In Redis, either the HASH or JSON (if using RedisJSON in addition to RediSearch) data types can be used to store documents. We will use the HASH data type in this example. The below cells will show how to load documents into the index.

```python
def index_documents(client: redis.Redis, prefix: str, documents: pd.DataFrame):
    records = documents.to_dict("records")
    for doc in records:
        key = f"{prefix}:{str(doc['id'])}"

        # create byte vectors for title and content
        title_embedding = np.array(doc["title_vector"], dtype=np.float32).tobytes()
        content_embedding = np.array(doc["content_vector"], dtype=np.float32).tobytes()

        # replace list of floats with byte vectors
        doc["title_vector"] = title_embedding
        doc["content_vector"] = content_embedding

        client.hset(key, mapping = doc)
```

```python
index_documents(redis_client, PREFIX, data)
print(f"Loaded {redis_client.info()['db0']['keys']} documents in Redis search index with name: {INDEX_NAME}")
```

## Simple Vector Search Queries with OpenAI Query Embeddings

Now that we have a search index and documents loaded into it, we can run search queries. Below we will provide a function that will run a search query and return the results. Using this function we run a few queries that will show how you can utilize Redis as a vector database.

```python
def search_redis(
    redis_client: redis.Redis,
    user_query: str,
    index_name: str = "embeddings-index",
    vector_field: str = "title_vector",
    return_fields: list = ["title", "url", "text", "vector_score"],
    hybrid_fields = "*",
    k: int = 20,
    print_results: bool = True,
) -> List[dict]:

    # Creates embedding vector from user query
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
        for i, article in enumerate(results.docs):
            score = 1 - float(article.vector_score)
            print(f"{i}. {article.title} (Score: {round(score ,3) })")
    return results.docs
```

```python
# For using OpenAI to generate query embedding
results = search_redis(redis_client, 'modern art in Europe', k=10)
```

```python
results = search_redis(redis_client, 'Famous battles in Scottish history', vector_field='content_vector', k=10)
```

## Hybrid Queries with Redis

The previous examples showed how run vector search queries with RediSearch. In this section, we will show how to combine vector search with other RediSearch fields for hybrid search. In the below example, we will combine vector search with full text search.

```python
def create_hybrid_field(field_name: str, value: str) -> str:
    return f'@{field_name}:"{value}"'

# search the content vector for articles about famous battles in Scottish history and only include results with Scottish in the title
results = search_redis(redis_client,
                       "Famous battles in Scottish history",
                       vector_field="title_vector",
                       k=5,
                       hybrid_fields=create_hybrid_field("title", "Scottish")
                       )
```

```python
# run a hybrid query for articles about Art in the title vector and only include results with the phrase "Leonardo da Vinci" in the text
results = search_redis(redis_client,
                       "Art",
                       vector_field="title_vector",
                       k=5,
                       hybrid_fields=create_hybrid_field("text", "Leonardo da Vinci")
                       )

# find specific mention of Leonardo da Vinci in the text that our full-text-search query returned
mention = [sentence for sentence in results[0].text.split("\n") if "Leonardo da Vinci" in sentence][0]
mention
```

## HNSW Index

Up until now, we've been using the ``FLAT`` or "brute-force" index to run our queries. Redis also supports the ``HNSW`` index which is a fast, approximate index. The ``HNSW`` index is a graph-based index that uses a hierarchical navigable small world graph to store vectors. The ``HNSW`` index is a good choice for large datasets where you want to run approximate queries.

``HNSW`` will take longer to build and consume more memory for most cases than ``FLAT`` but will be faster to run queries on, especially for large datasets.

The following cells will show how to create an ``HNSW`` index and run queries with it using the same data as before.

```python
# re-define RediSearch vector fields to use HNSW index
title_embedding = VectorField("title_vector",
    "HNSW", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC,
        "INITIAL_CAP": VECTOR_NUMBER
    }
)
text_embedding = VectorField("content_vector",
    "HNSW", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": DISTANCE_METRIC,
        "INITIAL_CAP": VECTOR_NUMBER
    }
)
fields = [title, url, text, title_embedding, text_embedding]
```

```python
import time
# Check if index exists
HNSW_INDEX_NAME = INDEX_NAME+ "_HNSW"

try:
    redis_client.ft(HNSW_INDEX_NAME).info()
    print("Index already exists")
except:
    # Create RediSearch Index
    redis_client.ft(HNSW_INDEX_NAME).create_index(
        fields = fields,
        definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
    )

# since RediSearch creates the index in the background for existing documents, we will wait until
# indexing is complete before running our queries. Although this is not necessary for the first query,
# some queries may take longer to run if the index is not fully built. In general, Redis will perform
# best when adding new documents to existing indices rather than new indices on existing documents.
while redis_client.ft(HNSW_INDEX_NAME).info()["indexing"] == "1":
    time.sleep(5)
```

```python
results = search_redis(redis_client, 'modern art in Europe', index_name=HNSW_INDEX_NAME, k=10)
```

```python
# compare the results of the HNSW index to the FLAT index and time both queries
def time_queries(iterations: int = 10):
    print(" ----- Flat Index ----- ")
    t0 = time.time()
    for i in range(iterations):
        results_flat = search_redis(redis_client, 'modern art in Europe', k=10, print_results=False)
    t0 = (time.time() - t0) / iterations
    results_flat = search_redis(redis_client, 'modern art in Europe', k=10, print_results=True)
    print(f"Flat index query time: {round(t0, 3)} seconds\n")
    time.sleep(1)
    print(" ----- HNSW Index ------ ")
    t1 = time.time()
    for i in range(iterations):
        results_hnsw = search_redis(redis_client, 'modern art in Europe', index_name=HNSW_INDEX_NAME, k=10, print_results=False)
    t1 = (time.time() - t1) / iterations
    results_hnsw = search_redis(redis_client, 'modern art in Europe', index_name=HNSW_INDEX_NAME, k=10, print_results=True)
    print(f"HNSW index query time: {round(t1, 3)} seconds")
    print(" ------------------------ ")
time_queries()
```

```python

```