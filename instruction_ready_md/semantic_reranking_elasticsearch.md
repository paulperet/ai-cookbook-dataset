# Semantic Reranking with Elasticsearch and Hugging Face

_Authored by: [Liam Thompson](https://github.com/leemthompo)_

This guide demonstrates how to implement semantic reranking in Elasticsearch by uploading a cross-encoder model from Hugging Face. You will use the `retriever` abstraction to craft queries and combine search operations.

By the end of this tutorial, you will:
- Choose a cross-encoder model from Hugging Face for semantic reranking.
- Upload the model to your Elasticsearch deployment using Eland.
- Create an inference endpoint to manage the `rerank` task.
- Query your data using the `text_similarity_rerank` retriever.

## Prerequisites

For this example, you will need:
- An Elastic deployment on version 8.15.0 or above (for non-serverless deployments).
    - We'll be using Elastic Cloud for this example (available with a [free trial](https://cloud.elastic.co/registration)).
    - See other [deployment options](https://www.elastic.co/guide/en/elasticsearch/reference/current/elasticsearch-intro.html#elasticsearch-intro-deploy).
- Your deployment's Cloud ID and an API key. [Learn more](https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#finding-your-cloud-id).

## Step 1: Install and Import Packages

First, install the required Python packages. The `eland` installation may take a couple of minutes.

```bash
pip install -qU elasticsearch
pip install eland[pytorch]
```

Now, import the necessary modules.

```python
from elasticsearch import Elasticsearch, helpers
from getpass import getpass
from urllib.request import urlopen
import json
import time
```

## Step 2: Initialize the Elasticsearch Python Client

Connect to your Elasticsearch instance using your Cloud ID and API key.

```python
# https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#finding-your-cloud-id
ELASTIC_CLOUD_ID = getpass("Elastic Cloud ID: ")

# https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#creating-an-api-key
ELASTIC_API_KEY = getpass("Elastic Api Key: ")

# Create the client instance
client = Elasticsearch(
    cloud_id=ELASTIC_CLOUD_ID,
    api_key=ELASTIC_API_KEY,
)
```

Test the connection to confirm the client is properly connected.

```python
print(client.info())
```

## Step 3: Index Sample Data

This example uses a small dataset of movies. You will download and index this data into an Elasticsearch index named `movies`.

```python
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

## Step 4: Upload a Hugging Face Model Using Eland

Now, use Eland's `eland_import_hub_model` command to upload a model to Elasticsearch. For this example, we'll use the `cross-encoder/ms-marco-MiniLM-L-6-v2` text similarity model.

```bash
eland_import_hub_model \
  --cloud-id $ELASTIC_CLOUD_ID \
  --es-api-key $ELASTIC_API_KEY \
  --hub-model-id cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --task-type text_similarity \
  --clear-previous \
  --start
```

## Step 5: Create an Inference Endpoint

Next, create an inference endpoint for the `rerank` task. This deploys and manages your model, spinning up the necessary ML resources.

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

Confirm your inference endpoint is deployed.

```python
client.inference.get()
```

> **Note:** When you deploy your model, you might need to sync your ML saved objects in the Kibana (or Serverless) UI. Go to **Trained Models** and select **Synchronize saved objects**.

## Step 6: Perform Lexical Queries

First, let's test some lexical (full-text) searches to establish a baseline. Then, you'll compare the improvements when adding semantic reranking.

### Lexical Match with `query_string` Query

Imagine you vaguely remember a movie about a killer who eats his victims but forget the word "cannibal". Perform a `query_string` query to find the phrase "flesh-eating bad guy" in the `plot` field.

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

This query returns no results because there are no exact matches for "flesh-eating bad guy". You need to broaden the search.

### Simple `multi_match` Query

This lexical query performs a standard keyword search for the term "crime" within the "plot" and "genre" fields.

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

Now you get results, but they aren't precise for the original query. Notice "The Silence of the Lambs" appears in the middle of the results. Let's use semantic reranking to improve relevance.

## Step 7: Implement Semantic Reranking

In the following `retriever` syntax, you wrap the standard query retriever in a `text_similarity_reranker`. This leverages the NLP model you deployed to rerank results based on the phrase "flesh-eating bad guy".

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

Success! "The Silence of the Lambs" is now the top result. Semantic reranking parsed the natural language query and identified the most relevant document, overcoming the limitations of lexical search.

## Conclusion

Semantic reranking enables semantic search in a few steps without generating and storing embeddings. Using open-source models from Hugging Face natively in your Elasticsearch cluster is excellent for prototyping, testing, and building advanced search experiences.

## Learn More

- For this example, we used the [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) text similarity model. Refer to [the Elastic NLP model reference](https://www.elastic.co/guide/en/machine-learning/8.15/ml-nlp-model-ref.html#ml-nlp-model-ref-text-similarity) for a list of supported third-party text similarity models.
- Learn more about [integrating Hugging Face](https://www.elastic.co/search-labs/integrations/hugging-face) with Elasticsearch.
- Check out Elastic's catalogue of Python notebooks in the [`elasticsearch-labs` repo](https://github.com/elastic/elasticsearch-labs/tree/main/notebooks).
- Learn more about [retrievers and reranking in Elasticsearch](https://www.elastic.co/guide/en/elasticsearch/reference/current/retrievers-reranking-overview.html).