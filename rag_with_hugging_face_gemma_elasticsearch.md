# Building A RAG System with Gemma, Elasticsearch and Hugging Face Models

Authored By: [lloydmeta](https://huggingface.co/lloydmeta)

This notebook walks you through building a Retrieval-Augmented Generation (RAG) powered by Elasticsearch (ES) and Hugging Face models, letting you toggle between ES-vectorising (your ES cluster vectorises for you when ingesting and querying) vs self-vectorising (you vectorise all your data before sending it to ES).

What should you use for your use case? *It depends* 🤷‍♂️. ES-vectorising means your clients don't have to implement it, so that's the default here; however, if you don't have any ML nodes, or your own embedding setup is better/faster, feel free to set `USE_ELASTICSEARCH_VECTORISATION` to `False` in the `Choose data and query vectorisation options` section below!

> [!TIP]
> This notebook has been tested with ES 8.13.x, and 8.14.x

## Step 1: Installing Libraries

```python
!pip install elasticsearch sentence_transformers transformers eland==8.12.1 # accelerate # uncomment if using GPU
!pip install datasets==2.19.2 # Remove version lock if https://github.com/huggingface/datasets/pull/6978 has been released
```

## Step 2: Set up

### Hugging Face
This allows you to authenticate with Hugging Face to download models and datasets.

```python
from huggingface_hub import notebook_login

notebook_login()
```

#### Elasticsearch deployment

Let's make sure that you can access your Elasticsearch deployment. If you don't have one, create one at [Elastic Cloud](https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#creating-a-cloud-deployment).

Ensure you have `CLOUD_ID` and `ELASTIC_DEPL_API_KEY` saved as Colab secrets.

```python
from google.colab import userdata

# https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#finding-your-cloud-id
CLOUD_ID = userdata.get("CLOUD_ID") # or "<YOUR CLOUD_ID>"

# https://www.elastic.co/search-labs/tutorials/install-elasticsearch/elastic-cloud#creating-an-api-key
ELASTIC_API_KEY = userdata.get("ELASTIC_DEPL_API_KEY")  # or "<YOUR API KEY>"
```

Set up the client and make sure the credentials work.

```python
from elasticsearch import Elasticsearch, helpers

# Create the client instance
client = Elasticsearch(cloud_id=CLOUD_ID, api_key=ELASTIC_API_KEY)

# Successful response!
client.info()
```

## Step 3: Data sourcing and preparation

The data utilised in this tutorial is sourced from Hugging Face datasets, specifically the
[MongoDB/embedded_movies dataset](https://huggingface.co/datasets/MongoDB/embedded_movies).

```python
# Load Dataset
from datasets import load_dataset

# https://huggingface.co/datasets/MongoDB/embedded_movies
dataset = load_dataset("MongoDB/embedded_movies")

dataset
```

[Downloading readme: ..., Downloading data: ..., Generating train split: ...]

```
DatasetDict({
    train: Dataset({
        features: ['plot', 'fullplot', 'languages', 'writers', 'num_mflix_comments', 'genres', 'title', 'type', 'rated', 'cast', 'imdb', 'metacritic', 'awards', 'runtime', 'poster', 'countries', 'directors', 'plot_embedding'],
        num_rows: 1500
    })
})
```

The operations within the following code snippet below focus on enforcing data integrity and quality.
1. The first process ensures that each data point's `fullplot` attribute is not empty, as this is the primary data we utilise in the embedding process.
2. The second step also ensures we remove the `plot_embedding` attribute from all data points as this will be replaced by new embeddings created with a different embedding model, the `gte-large`.

```python
# Data Preparation

# Remove data point where plot coloumn is missing
dataset = dataset.filter(lambda x: x["fullplot"] is not None)

if "plot_embedding" in sum(dataset.column_names.values(), []):
    # Remove the plot_embedding from each data point in the dataset as we are going to create new embeddings with an open source embedding model from Hugging Face
    dataset = dataset.remove_columns("plot_embedding")

dataset["train"]
```

[Filter: ...]

```
Dataset({
    features: ['plot', 'fullplot', 'languages', 'writers', 'num_mflix_comments', 'genres', 'title', 'type', 'rated', 'cast', 'imdb', 'metacritic', 'awards', 'runtime', 'poster', 'countries', 'directors'],
    num_rows: 1452
})
```

## Step 4: Load Elasticsearch with vectorised data

### Choose data and query vectorisation options

Here, you need to make a decision: do you want Elasticsearch to vectorise your data and queries, or do you want to do it yourself?

Setting `USE_ELASTICSEARCH_VECTORISATION` to `True` will make the rest of this notebook set up and use ES-hosted-vectorisation for your data and your querying, but **BE AWARE** that this requires your ES deployment to have at least 1 ML node (I would recommend setting autoscaling to true on your Cloud deployment in case the model you choose is too big).

If `USE_ELASTICSEARCH_VECTORISATION` is `False`, this notebook will set up and use the provided model "locally" for data and query vectorisation.

Here, I've picked the [thenlper/gte-small](https://huggingface.co/thenlper/gte-small) model for really no other reason than it was used in another cookbook, and it worked well enough for me. Please feel free to try others if you'd like - the only important thing is that you update the `EMBEDDING_DIMENSIONS` according to the model.

**Note**: if you change these values, you'll likely need to re-run the notebook from this step.

```python
USE_ELASTICSEARCH_VECTORISATION = True

EMBEDDING_MODEL_ID = "thenlper/gte-small"
# https://huggingface.co/thenlper/gte-small's page shows the dimensions of the model
# If you use the `gte-base` or `gte-large` embedding models, the numDimension
# value in the vector search index must be set to 768 and 1024, respectively.
EMBEDDING_DIMENSIONS = 384
```

### Load Hugging Face model into Elasticsearch if needed

This step loads and deploys the Hugging Face model into Elasticsearch using [Eland](https://eland.readthedocs.io/en/v8.12.1/), if `USE_ELASTICSEARCH_VECTORISATION` is `True`. This allows Elasticsearch to vectorise your queries, and data in later steps.

```python
import locale
locale.getpreferredencoding = lambda: "UTF-8"
!(if [ "True" == $USE_ELASTICSEARCH_VECTORISATION ]; then \
  eland_import_hub_model --cloud-id $CLOUD_ID --hub-model-id $EMBEDDING_MODEL_ID --task-type text_embedding --es-api-key $ELASTIC_API_KEY --start --clear-previous; \
fi)
```

This step adds functions for creating embeddings for text locally, and enriches the dataset with embeddings, so that the data can be ingested into Elasticsearch as vectors. Does not run if `USE_ELASTICSEARCH_VECTORISATION` is True.

```python
from sentence_transformers import SentenceTransformer

if not USE_ELASTICSEARCH_VECTORISATION:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_ID)


def get_embedding(text: str) -> list[float]:
    if USE_ELASTICSEARCH_VECTORISATION:
        raise Exception(
            f"Disabled when USE_ELASTICSEARCH_VECTORISATION is [{USE_ELASTICSEARCH_VECTORISATION}]"
        )
    else:
        if not text.strip():
            print("Attempted to get embedding for empty text.")
            return []

        embedding = embedding_model.encode(text)
        return embedding.tolist()


def add_fullplot_embedding(x):
    if USE_ELASTICSEARCH_VECTORISATION:
        raise Exception(
            f"Disabled when USE_ELASTICSEARCH_VECTORISATION is [{USE_ELASTICSEARCH_VECTORISATION}]"
        )
    else:
        full_plots = x["fullplot"]
        return {"embedding": [get_embedding(full_plot) for full_plot in full_plots]}


if not USE_ELASTICSEARCH_VECTORISATION:
    dataset = dataset.map(add_fullplot_embedding, batched=True)
    dataset["train"]
```

## Step 5: Create a Search Index with vector search mappings.

At this point, we create an index in Elasticsearch with the right index mappings to handle vector searches.

Go here to read more about [Elasticsearch vector capabilities](https://www.elastic.co/what-is/vector-search).

```python
# Needs to match the id returned from Eland
# in general for Hugging Face models, you just replace the forward slash with
# double underscore
model_id = EMBEDDING_MODEL_ID.replace("/", "__")

index_name = "movies"

index_mapping = {
    "properties": {
        "fullplot": {"type": "text"},
        "plot": {"type": "text"},
        "title": {"type": "text"},
    }
}
# define index mapping
if USE_ELASTICSEARCH_VECTORISATION:
    index_mapping["properties"]["embedding"] = {
        "properties": {
            "is_truncated": {"type": "boolean"},
            "model_id": {
                "type": "text",
                "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
            },
            "predicted_value": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMENSIONS,
                "index": True,
                "similarity": "cosine",
            },
        }
    }
else:
    index_mapping["properties"]["embedding"] = {
        "type": "dense_vector",
        "dims": EMBEDDING_DIMENSIONS,
        "index": "true",
        "similarity": "cosine",
    }

# flag to check if index has to be deleted before creating
should_delete_index = True

# check if we want to delete index before creating the index
if should_delete_index:
    if client.indices.exists(index=index_name):
        print("Deleting existing %s" % index_name)
        client.indices.delete(index=index_name, ignore=[400, 404])

print("Creating index %s" % index_name)


# ingest pipeline definition
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

    index_settings = {
        "index": {
            "default_pipeline": pipeline_id,
        }
    }
else:
    index_settings = {}

client.options(ignore_status=[400, 404]).indices.create(
    index=index_name, mappings=index_mapping, settings=index_settings
)
```

Creating index movies

```
ObjectApiResponse({'acknowledged': True, 'shards_acknowledged': True, 'index': 'movies'})
```

Ingesting data into a Elasticsearch is best done in batches. Luckily `helpers` offers an easy way to do this.

```python
from elasticsearch.helpers import BulkIndexError

def batch_to_bulk_actions(batch):
    for record in batch:
        action = {
            "_index": "movies",
            "_source": {
                "title": record["title"],
                "fullplot": record["fullplot"],
                "plot": record["plot"],
            },
        }
        if not USE_ELASTICSEARCH_VECTORISATION:
            action["_source"]["embedding"] = record["embedding"]
        yield action


def bulk_index(ds):
    start = 0
    end = len(ds)
    batch_size = 100
    if USE_ELASTICSEARCH_VECTORISATION:
        # If using auto-embedding, bulk requests can take a lot longer,
        # so pass a longer request_timeout here (defaults to 10s), otherwise
        # we could get Connection timeouts
        batch_client = client.options(request_timeout=600)
    else:
        batch_client = client
    for batch_start in range(start, end, batch_size):
        batch_end = min(batch_start + batch_size, end)
        print(f"batch: start [{batch_start}], end [{batch_end}]")
        batch = ds.select(range(batch_start, batch_end))
        actions = batch_to_bulk_actions(batch)
        helpers.bulk(batch_client, actions)


try:
    bulk_index(dataset["train"])
except BulkIndexError as e:
    print(f"{e.errors}")

print("Data ingestion into Elasticsearch complete!")
```

[batch: start [0], end [100], ..., batch: start [1400], end [1452]]
Data ingestion into Elasticsearch complete!

## Step 6: Perform Vector Search on User Queries

The following step implements a function that returns a vector search result.

If `USE_ELASTICSEARCH_VECTORISATION` is true, the text query is sent directly to
ES where the uploaded model will be used to vectorise it first before doing a vector search. If `USE_ELASTICSEARCH_VECTORISATION` is false, then we do the
vectorising locally before sending a query with the vectorised form of the query.

```python
def vector_search(plot_query):
    if USE_ELASTICSEARCH_VECTORISATION:
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
        question_embedding = get_embedding(plot_query)
        knn = {
            "field": "embedding",
            "query_vector": question_embedding,
            "k": 10,
            "num_candidates": 150,
        }

    response = client.search(index="movies", knn=knn, size=5)
    results = []
    for hit in response["hits"]["hits"]:
        id = hit["_id"]
        score = hit["_score"]
        title = hit["_source"]["title"]
        plot = hit["_source"]["plot"]
        fullplot = hit["_source"]["fullplot"]
        result = {
            "id": id,
            "_score": score,
            "title": title,
            "plot": plot,
            "fullplot": fullplot,
        }
        results.append(result)
    return results

def pretty_search(query):

    get_knowledge = vector_search(query)

    search_result = ""
    for result in get_knowledge:
        search_result += f"Title: {result.get('title', 'N/A')}, Plot: {result.get('fullplot', 'N/A')}\n"

    return search_result
```

## Step 7: Handling user queries and loading Gemma

```python
# Conduct query with retrival of sources, combining results into something that
# we can feed to Gemma
def combined_query(query):
    source_information = pretty_search(query)
    return f"Query: {query}\nContinue to answer the query by using these Search Results:\n{source_information}."


query = "What is the best romantic movie to watch and why?"
combined_results = combined_query(query)

print(combined_results)
```

```
Query: What is the best romantic movie to watch and why?
Continue to answer the query by using these Search Results:
Title: Shut Up and Kiss Me!, Plot: Ryan and Pete are 27-year old best friends in Miami, born on the same day and each searching for the perfect woman. Ryan is a rookie stockbroker living with his psychic Mom. Pete is a slick surfer dude yet to find commitment. Each meets the women of their dreams on the same day. Ryan knocks heads in an elevator with the gorgeous Jessica, passing out before getting her number. Pete falls for the insatiable Tiara, but Tiara's uncle is mob boss Vincent Bublione, charged with her protection. This high-energy romantic comedy asks to what extent will you go for true love?
Title: Titanic, Plot: The plot focuses on the romances of two couples upon the doomed ship's maiden voyage. Isabella Paradine (Catherine Zeta-Jones) is a wealthy woman mourning the loss of her aunt, who reignites a romance with former flame Wynn Park (Peter Gallagher). Meanwhile, a charming ne'er-do-well named Jamie Perse (Mike Doyle) steals a ticket for the ship, and falls for a sweet innocent Irish girl on board. But their romance is threatened by the villainous Simon Doonan (Tim Curry), who has discovered about the ticket and makes Jamie his unwilling accomplice, as well as having sinister plans for the girl.
Title: Dark Blue World, Plot: March 15, 1939: Germany invades Czechoslovakia. Czech and Slovak pilots flee to England, joining the RAF. After the war, back home, they are put in labor camps, suspected of anti-Communist ideas. This film cuts between a post-war camp where Franta is a prisoner and England during the war, where Franta is like a big brother to Karel, a very young pilot. On maneuvers, Karel crash lands by the rural home of Susan, an English woman whose husband is MIA. She spends one night with Karel, and he thinks he's found the love of his life. It's complicated by Susan's attraction to Franta. How will the three handle innocence, Eros, friendship, and the heat of battle? When war ends, what then?
Title: Dark Blue World, Plot: March 15, 1939: Germany invades Czechoslovakia.