# Retrieval Augmented Generation (RAG) with Elasticsearch and OpenAI

This guide demonstrates how to build a Retrieval Augmented Generation (RAG) pipeline. You will learn how to:
1.  Index a vector dataset into Elasticsearch.
2.  Embed a user query using OpenAI's API.
3.  Perform a semantic (vector) search in Elasticsearch to find relevant context.
4.  Use the retrieved context to generate an informed answer with OpenAI's Chat Completions API.

## Prerequisites

Before you begin, ensure you have the following:

*   An **OpenAI API key**.
*   An **Elastic Cloud deployment** (sign up for a [free trial](https://cloud.elastic.co/registration?utm_source=github&utm_content=openai-cookbook) if you don't have one).
*   The **Cloud ID** and password for your Elastic deployment.

## Step 1: Environment Setup

First, install the required Python packages and import the necessary modules.

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
import openai
```

## Step 2: Connect to Elasticsearch

You will connect to your Elasticsearch deployment using its Cloud ID and credentials.

1.  Find your deployment's **Cloud ID** in the [Elastic Cloud console](https://cloud.elastic.co/deployments).
2.  Run the following code, providing your Cloud ID and password when prompted.

```python
# Securely input your Elastic Cloud credentials
CLOUD_ID = getpass("Elastic deployment Cloud ID")
CLOUD_PASSWORD = getpass("Elastic deployment Password")

# Create the Elasticsearch client
client = Elasticsearch(
    cloud_id=CLOUD_ID,
    basic_auth=("elastic", CLOUD_PASSWORD)  # Or use `api_key` for API key authentication
)

# Test the connection
print(client.info())
```

## Step 3: Download and Prepare the Dataset

You will use a sample Wikipedia embeddings dataset provided by OpenAI.

```python
# Download the dataset
embeddings_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'
wget.download(embeddings_url)

# Extract the downloaded ZIP file
with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
    zip_ref.extractall("data")

# Load the data into a Pandas DataFrame for easier handling
wikipedia_dataframe = pd.read_csv("data/vector_database_wikipedia_articles_embedded.csv")
```

## Step 4: Create an Elasticsearch Index with Vector Mappings

To perform vector search, you must create an index with specific mappings for the `dense_vector` field type.

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

Indexing the entire DataFrame at once can be resource-intensive. You will batch the process for efficiency.

First, define a helper function to format the DataFrame rows for Elasticsearch's Bulk API.

```python
def dataframe_to_bulk_actions(df):
    """Generator function to yield documents for bulk indexing."""
    for index, row in df.iterrows():
        yield {
            "_index": 'wikipedia_vector_index',
            "_id": row['id'],
            "_source": {
                'url': row["url"],
                'title': row["title"],
                'text': row["text"],
                'title_vector': json.loads(row["title_vector"]),  # Parse the stored JSON string
                'content_vector': json.loads(row["content_vector"]),
                'vector_id': row["vector_id"]
            }
        }
```

Now, index the data in batches.

```python
start = 0
end = len(wikipedia_dataframe)
batch_size = 100

for batch_start in range(start, end, batch_size):
    batch_end = min(batch_start + batch_size, end)
    batch_dataframe = wikipedia_dataframe.iloc[batch_start:batch_end]
    actions = dataframe_to_bulk_actions(batch_dataframe)
    helpers.bulk(client, actions)
    print(f"Indexed rows {batch_start} to {batch_end-1}")
```

Let's verify the index works with a simple text search.

```python
# Test query: search for documents about "Hummingbird"
test_response = client.search(
    index="wikipedia_vector_index",
    body={
        "_source": {"excludes": ["title_vector", "content_vector"]},  # Exclude vectors from the response
        "query": {
            "match": {
                "text": {
                    "query": "Hummingbird"
                }
            }
        }
    }
)

print(f"Found {test_response['hits']['total']['value']} document(s).")
```

## Step 6: Implement the RAG Pipeline

Now for the core RAG workflow. You will:
1.  Set up the OpenAI client.
2.  Create an embedding for a user's question.
3.  Use that embedding to find the most relevant documents via vector search in Elasticsearch.
4.  Construct a prompt with the retrieved context and the original question.
5.  Send the prompt to OpenAI's Chat Completions API to generate a final answer.

### 6.1 Configure the OpenAI Client

```python
# Securely input your OpenAI API key
openai.api_key = getpass("OpenAI API Key")
```

### 6.2 Define the Embedding and Search Functions

```python
def get_embedding(text, model="text-embedding-ada-002"):
    """Generate an embedding for a given text string using OpenAI."""
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

def elastic_vector_search(query, client, index="wikipedia_vector_index", k=5):
    """Perform a kNN vector search in Elasticsearch."""
    # 1. Generate the query embedding
    query_embedding = get_embedding(query)

    # 2. Define the Elasticsearch kNN search query
    search_body = {
        "knn": {
            "field": "content_vector",
            "query_vector": query_embedding,
            "k": k,
            "num_candidates": 100
        },
        "_source": ["title", "text", "url"]  # Fields to return
    }

    # 3. Execute the search
    response = client.search(index=index, body=search_body)
    return response['hits']['hits']
```

### 6.3 Perform Retrieval Augmented Generation

```python
def rag_pipeline(question, client, index="wikipedia_vector_index"):
    """Orchestrates the full RAG process: Retrieve context and generate an answer."""
    # 1. Retrieve relevant context from Elasticsearch
    print("🔍 Retrieving relevant context...")
    search_results = elastic_vector_search(question, client, index=index, k=3)

    # 2. Format the retrieved context
    context = ""
    for i, hit in enumerate(search_results):
        source = hit["_source"]
        context += f"Document {i+1} | Title: {source['title']}\n"
        context += f"Content: {source['text'][:500]}...\n"  # Truncate for brevity
        context += "-" * 50 + "\n"

    print(f"Retrieved {len(search_results)} document(s).\n")

    # 3. Construct the prompt for the LLM
    prompt = f"""You are a helpful assistant. Answer the user's question based only on the provided context.
If the context does not contain enough information to answer, say "I cannot answer based on the provided context."

Context:
{context}

Question: {question}

Answer:"""

    # 4. Generate the final answer using OpenAI
    print("🤖 Generating answer...\n")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    answer = response.choices[0].message.content
    return answer
```

### 6.4 Run the Pipeline

Test your RAG system with a question.

```python
# Ask a question
user_question = "What are the key characteristics of hummingbirds?"

# Get the answer
final_answer = rag_pipeline(user_question, client, index="wikipedia_vector_index")
print("💬 Question:", user_question)
print("\n📄 Answer:\n", final_answer)
```

## Summary

You have successfully built a RAG pipeline. The key steps were:
1.  **Data Ingestion:** Loading and indexing a vector dataset into Elasticsearch.
2.  **Vector Search:** Using OpenAI embeddings to find semantically relevant documents.
3.  **Contextual Generation:** Augmenting an LLM prompt with retrieved context to produce a grounded, informative answer.

This architecture is foundational for building AI applications that require access to a private or extensive knowledge base. You can extend it by using different embedding models, tuning the search parameters (`k`, `num_candidates`), or experimenting with the prompt structure.