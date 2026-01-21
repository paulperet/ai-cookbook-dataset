# Semantic Search with Elasticsearch and OpenAI

This guide demonstrates how to build a semantic search system using Elasticsearch as a vector database and OpenAI's embeddings. You will learn how to index a dataset of pre-computed embeddings, encode a search query, and retrieve the most semantically relevant documents.

## Prerequisites

Ensure you have the following installed and ready:

1.  **Python 3.7+**
2.  An **Elastic Cloud deployment** (or a local Elasticsearch cluster). [Sign up for a free trial](https://cloud.elastic.co/registration?utm_source=github&utm_content=openai-cookbook) if needed.
3.  An **OpenAI API key**.

## Step 1: Setup and Installation

Begin by installing the required Python packages and importing the necessary modules.

```bash
pip install -qU openai pandas wget elasticsearch
```

```python
from getpass import getpass
from elasticsearch import Elasticsearch, helpers
import wget
import zipfile
import pandas as pd
import json
from openai import OpenAI
```

## Step 2: Connect to Elasticsearch

You need to connect to your Elasticsearch deployment. If you're using Elastic Cloud, you will need your **Cloud ID** and password (or API key).

1.  Find your Cloud ID in the Elastic Cloud console under your deployment's details.
2.  Run the following code, entering your credentials when prompted.

```python
# Securely input your Elastic Cloud credentials
CLOUD_ID = getpass("Elastic deployment Cloud ID")
CLOUD_PASSWORD = getpass("Elastic deployment Password")

# Create the Elasticsearch client
client = Elasticsearch(
    cloud_id=CLOUD_ID,
    basic_auth=("elastic", CLOUD_PASSWORD)  # Use `api_key` instead of `basic_auth` if preferred
)

# Test the connection
print(client.info())
```

A successful connection will print your cluster's information.

## Step 3: Download and Prepare the Dataset

We'll use a sample Wikipedia dataset with pre-computed embeddings from OpenAI.

```python
# Download the dataset
embeddings_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'
wget.download(embeddings_url)

# Extract the contents
with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
    zip_ref.extractall("data")

# Load the data into a Pandas DataFrame for easier handling
wikipedia_dataframe = pd.read_csv("data/vector_database_wikipedia_articles_embedded.csv")
```

## Step 4: Create an Elasticsearch Index with Vector Mappings

To store our vector embeddings, we must create a dedicated index with the correct field mappings. We use the `dense_vector` field type for the embedding fields.

```python
# Define the index mapping
index_mapping = {
    "properties": {
        "title_vector": {
            "type": "dense_vector",
            "dims": 1536,  # Dimension of the OpenAI embeddings
            "index": "true",
            "similarity": "cosine"
        },
        "content_vector": {
            "type": "dense_vector",
            "dims": 1536,
            "index": "true",
            "similarity": "cosine"
        },
        "text": {"type": "text"},
        "title": {"type": "text"},
        "url": {"type": "keyword"},
        "vector_id": {"type": "long"}
    }
}

# Create the index
client.indices.create(index="wikipedia_vector_index", mappings=index_mapping)
```

## Step 5: Index the Data into Elasticsearch

Indexing all data at once can be resource-intensive. We'll create a helper function to format the data and then index it in manageable batches.

```python
def dataframe_to_bulk_actions(df):
    """Generator function to format DataFrame rows for Elasticsearch's Bulk API."""
    for index, row in df.iterrows():
        yield {
            "_index": 'wikipedia_vector_index',
            "_id": row['id'],
            "_source": {
                'url': row["url"],
                'title': row["title"],
                'text': row["text"],
                'title_vector': json.loads(row["title_vector"]),
                'content_vector': json.loads(row["content_vector"]),
                'vector_id': row["vector_id"]
            }
        }

# Index data in batches of 100
start = 0
end = len(wikipedia_dataframe)
batch_size = 100

for batch_start in range(start, end, batch_size):
    batch_end = min(batch_start + batch_size, end)
    batch_dataframe = wikipedia_dataframe.iloc[batch_start:batch_end]
    actions = dataframe_to_bulk_actions(batch_dataframe)
    helpers.bulk(client, actions)

print("Data indexing complete.")
```

Let's verify the index works with a simple text search.

```python
# Test with a standard match query
test_response = client.search(
    index="wikipedia_vector_index",
    body={
        "_source": {"excludes": ["title_vector", "content_vector"]},
        "query": {
            "match": {
                "text": {
                    "query": "Hummingbird"
                }
            }
        }
    }
)
print(f"Found {test_response['hits']['total']['value']} documents.")
```

## Step 6: Encode a Search Query with OpenAI

To perform semantic search, we must encode our search query using the same embedding model that created the document vectors (`text-embedding-3-small`).

```python
# Initialize the OpenAI client (set your API key in your environment variables)
openai_client = OpenAI()

# Define your search question
question = 'Is the Atlantic the biggest ocean in the world?'

# Generate the embedding for the question
question_embedding_response = openai_client.embeddings.create(
    input=question,
    model="text-embedding-3-small"
)

# Extract the embedding vector
query_vector = question_embedding_response.data[0].embedding
```

## Step 7: Perform Semantic (k-NN) Search

Now we can query Elasticsearch using a k-Nearest Neighbors (kNN) search on the `content_vector` field with our encoded query vector.

First, let's create a helper function to display the results cleanly.

```python
def pretty_response(response):
    """Prints search results in a readable format."""
    for hit in response['hits']['hits']:
        doc_id = hit['_id']
        score = hit['_score']
        title = hit['_source']['title']
        text = hit['_source']['text'][:500]  # Show first 500 chars
        print(f"\nID: {doc_id}\nTitle: {title}\nSummary: {text}...\nScore: {score}\n{'-'*50}")
```

Execute the kNN search.

```python
# Execute the semantic search
semantic_response = client.search(
    index="wikipedia_vector_index",
    knn={
        "field": "content_vector",
        "query_vector": query_vector,
        "k": 10,
        "num_candidates": 100
    }
)

# Display the results
print(f"Semantic search results for: '{question}'")
pretty_response(semantic_response)
```

## Conclusion and Next Steps

You have successfully built a semantic search pipeline! You learned how to:
1.  Set up an Elasticsearch index for vector data.
2.  Bulk index a dataset containing embeddings.
3.  Generate embeddings for a text query using the OpenAI API.
4.  Retrieve the most semantically similar documents using Elasticsearch's kNN search.

**To explore further:**
*   Experiment with different queries.
*   Adjust the `k` and `num_candidates` parameters in the kNN search to balance speed and accuracy.
*   Try indexing your own data using the same embedding model.
*   Build upon this example to create a **Retrieval-Augmented Generation (RAG)** system by feeding the search results into an OpenAI chat model for answer synthesis.