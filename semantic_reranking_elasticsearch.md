# Semantic reranking with Elasticsearch and Hugging Face

_Authored by: [Liam Thompson](https://github.com/leemthompo)_

In this notebook we will learn how to implement semantic reranking in Elasticsearch by uploading a model from Hugging Face into an Elasticsearch cluster. We'll use the `retriever` abstraction, a simpler Elasticsearch syntax for crafting queries and combining different search operations.

You will:

- Choose a cross-encoder model from Hugging Face to perform semantic reranking
- Upload the model to your Elasticsearch deployment using [Eland](https://www.elastic.co/guide/en/elasticsearch/client/eland/current/machine-learning.html)— a Python client for machine learning with Elasticsearch 
- Create an inference endpoint to manage your `rerank` task
- Query your data using the `text_similarity_rerank` retriever

## 🧰 Requirements

For this example, you will need:

- An Elastic deployment on version 8.15.0 or above (for non-serverless deployments)
    
    - We'll be using Elastic Cloud for this example (available with a [free trial](https://cloud.elastic.co/registration)).
    - See our other [deployment options](https://www.elastic.co/guide/en/elasticsearch/reference/current/elasticsearch-intro.html#elasticsearch-intro-deploy)
-  You'll need to find your deployment's Cloud ID and create an API key. [Learn more](https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#finding-your-cloud-id).


## Install and import packages

ℹ️ The `eland` installation will take a couple of minutes.


```python
!pip install -qU elasticsearch
!pip install eland[pytorch]
from elasticsearch import Elasticsearch, helpers
```

## Initialize Elasticsearch Python client

First you need to connect to your Elasticsearch instance.


```python
from getpass import getpass

# https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#finding-your-cloud-id
ELASTIC_CLOUD_ID = getpass("Elastic Cloud ID: ")

# https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#creating-an-api-key
ELASTIC_API_KEY = getpass("Elastic Api Key: ")

# Create the client instance
client = Elasticsearch(
    # For local development
    # hosts=["http://localhost:9200"]
    cloud_id=ELASTIC_CLOUD_ID,
    api_key=ELASTIC_API_KEY,
)
```

    Elastic Cloud ID: ··········
    Elastic Api Key: ··········


## Test connection

Confirm that the Python client has connected to your Elasticsearch instance with this test.




```python
print(client.info())
```

This examples uses a small dataset of movies.


```python
from urllib.request import urlopen
import json
import time

url = "https://huggingface.co/datasets/leemthompo/small-movies/raw/main/small-movies.json"
response = urlopen(url)

# Load the response data into a JSON object
data_json = json.loads(response.read())

# Prepare the documents to be indexed
documents = []
for doc in data_json:
    documents.append(
        {
            "_index": "movies",
            "_source": doc,
        }
    )

# Use helpers.bulk to index
helpers.bulk(client, documents)

print("Done indexing documents into `movies` index!")
time.sleep(3)
```

    Done indexing documents into `movies` index!


## Upload Hugging Face model using Eland

Now we'll use Eland's `eland_import_hub_model` command to upload the model to Elasticsearch. For this example we've chosen the `cross-encoder/ms-marco-MiniLM-L-6-v2` text similarity model.


```python
!eland_import_hub_model \
  --cloud-id $ELASTIC_CLOUD_ID \
  --es-api-key $ELASTIC_API_KEY \
  --hub-model-id cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --task-type text_similarity \
  --clear-previous \
  --start
```

    [2024-08-13 17:04:12,386 INFO : Establishing connection to Elasticsearch, ..., 2024-08-13 17:05:18,238 INFO : Model successfully imported with id 'cross-encoder__ms-marco-minilm-l-6-v2']


## Create inference endpoint

Next we'll create an inference endpoint for the `rerank` task to deploy and manage our model and, if necessary, spin up the necessary ML resources behind the scenes.


```python
client.inference.put(
    task_type="rerank",
    inference_id="my-msmarco-minilm-model",
    inference_config={
        "service": "elasticsearch",
        "service_settings": {
            "model_id": "cross-encoder__ms-marco-minilm-l-6-v2",
            "num_allocations": 1,
            "num_threads": 1,
        },
    },
)
```




    ObjectApiResponse({'endpoints': [{'model_id': 'my-msmarco-minilm-model', 'inference_id': 'my-msmarco-minilm-model', 'task_type': 'rerank', 'service': 'elasticsearch', 'service_settings': {'num_allocations': 1, 'num_threads': 1, 'model_id': 'cross-encoder__ms-marco-minilm-l-6-v2'}, 'task_settings': {'return_documents': True}}]})



Run the following command to confirm your inference endpoint is deployed.


```python
client.inference.get()
```


⚠️ When you deploy your model, you might need to sync your ML saved objects in the Kibana (or Serverless) UI.
Go to **Trained Models** and select **Synchronize saved objects**.

## Lexical queries

First let's use a `standard` retriever to test out some lexical (or full-text) searches and then we'll compare the improvements when we layer in semantic reranking.

### Lexical match with `query_string` query

Let's say we vaguely remember that there is a famous movie about a killer who eats his victims. For the sake of argument, pretend we've momentarily forgotten the word "cannibal".

Let's perform a [`query_string` query](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-query-string-query.html) to find the phrase "flesh-eating bad guy" in the `plot` fields of our Elasticsearch documents.


```python
resp = client.search(
    index="movies",
    retriever={
        "standard": {
            "query": {
                "query_string": {
                    "query": "flesh-eating bad guy",
                    "default_field": "plot",
                }
            }
        }
    },
)

if resp["hits"]["hits"]:
    for hit in resp["hits"]["hits"]:
        title = hit["_source"]["title"]
        plot = hit["_source"]["plot"]
        print(f"Title: {title}\nPlot: {plot}\n")
else:
    print("No search results found")
```

    No search results found


No results! Unfortunately we don't have any near exact matches for "flesh-eating bad guy". Because we don't have any more specific information about the exact phrasing in the Elasticsearch data, we'll need to cast our search net wider.

### Simple `multi_match` query

This lexical query performs a standard keyword search for the term "crime" within the "plot" and "genre" fields of our Elasticsearch documents.


```python
resp = client.search(
    index="movies",
    retriever={
        "standard": {
            "query": {"multi_match": {"query": "crime", "fields": ["plot", "genre"]}}
        }
    },
)

for hit in resp["hits"]["hits"]:
    title = hit["_source"]["title"]
    plot = hit["_source"]["plot"]
    print(f"Title: {title}\nPlot: {plot}\n")
```

    Title: The Godfather
    Plot: An organized crime dynasty's aging patriarch transfers control of his clandestine empire to his reluctant son.
    
    Title: Goodfellas
    Plot: The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners Jimmy Conway and Tommy DeVito in the Italian-American crime syndicate.
    
    Title: The Silence of the Lambs
    Plot: A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer, a madman who skins his victims.
    
    Title: Pulp Fiction
    Plot: The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.
    
    Title: Se7en
    Plot: Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives.
    
    Title: The Departed
    Plot: An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in South Boston.
    
    Title: The Usual Suspects
    Plot: A sole survivor tells of the twisty events leading up to a horrific gun battle on a boat, which began when five criminals met at a seemingly random police lineup.
    
    Title: The Dark Knight
    Plot: When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.
    


That's better! At least we've got some results now. We broadened our search criteria to increase the chances of finding relevant results.

But these results aren't very precise in the context of our original query "flesh-eating bad guy". We can see that "The Silence of the Lambs" is returned in the middle of the results set with this generic `match` query. Let's see if we can use our semantic reranking model to get closer to the searcher's original intent.

## Semantic reranker

In the following `retriever` syntax, we wrap our standard query retriever in a `text_similarity_reranker`. This allows us to leverage the NLP model we deployed to Elasticsearch to rerank the results based on the phrase "flesh-eating bad guy".


```python
resp = client.search(
    index="movies",
    retriever={
        "text_similarity_reranker": {
            "retriever": {
                "standard": {
                    "query": {
                        "multi_match": {"query": "crime", "fields": ["plot", "genre"]}
                    }
                }
            },
            "field": "plot",
            "inference_id": "my-msmarco-minilm-model",
            "inference_text": "flesh-eating bad guy",
        }
    },
)

for hit in resp["hits"]["hits"]:
    title = hit["_source"]["title"]
    plot = hit["_source"]["plot"]
    print(f"Title: {title}\nPlot: {plot}\n")
```

    Title: The Silence of the Lambs
    Plot: A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer, a madman who skins his victims.
    
    Title: Pulp Fiction
    Plot: The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.
    
    Title: Se7en
    Plot: Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives.
    
    Title: Goodfellas
    Plot: The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners Jimmy Conway and Tommy DeVito in the Italian-American crime syndicate.
    
    Title: The Dark Knight
    Plot: When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.
    
    Title: The Godfather
    Plot: An organized crime dynasty's aging patriarch transfers control of his clandestine empire to his reluctant son.
    
    Title: The Departed
    Plot: An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in South Boston.
    
    Title: The Usual Suspects
    Plot: A sole survivor tells of the twisty events leading up to a horrific gun battle on a boat, which began when five criminals met at a seemingly random police lineup.
    


Success! "The Silence of the Lambs" is our top result. Semantic reranking helped us find the most relevant result by parsing a natural language query, overcoming the limitations of lexical search which relies more on exact matching.

Semantic reranking enables semantic search in a few steps, without the need for generating and storing embeddings. Being able to use open source models hosted on Hugging Face natively in your Elasticsearch cluster is great for prototyping, testing, and building search experiences.

## Learn more

- For this example we've chosen the [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) text similarity model. Refer to [the Elastic NLP model reference](https://www.elastic.co/guide/en/machine-learning/8.15/ml-nlp-model-ref.html#ml-nlp-model-ref-text-similarity) for a list of third-party text similarity models supported by Elasticsearch.
- Learn more about [integrating Hugging Face](https://www.elastic.co/search-labs/integrations/hugging-face) with Elasticsearch.
- Check out Elastic's catalogue of Python notebooks in the [`elasticsearch-labs` repo](https://github.com/elastic/elasticsearch-labs/tree/main/notebooks).
- Learn more about [retrievers and reranking in Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/retrievers-reranking-overview.html)