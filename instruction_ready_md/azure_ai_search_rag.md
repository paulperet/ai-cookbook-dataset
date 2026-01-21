# RAG with Mistral AI, Azure AI Search, and Azure AI Studio: A Step-by-Step Guide

This guide walks you through building a Retrieval-Augmented Generation (RAG) pipeline. You will use Mistral AI's embedding model to create vector representations of a text dataset, store and search those vectors using Azure AI Search, and then ground the retrieved results in a Mistral Large language model to generate accurate, context-aware answers.

## Prerequisites

Before you begin, ensure you have the following:

1.  **Mistral AI API Key** (or an Azure AI Studio-deployed Mistral model with its API key).
2.  An **Azure AI Search** service provisioned in your Azure subscription.
3.  A **Python 3.x** environment.

## Step 1: Environment Setup

First, install the required Python libraries.

```bash
pip install azure-search-documents==11.5.1
pip install azure-identity==1.16.0 datasets==2.19.1 mistralai==1.0.1
```

## Step 2: Load and Prepare Your Dataset

We'll use a sample dataset of AI research paper chunks from Hugging Face.

```python
from datasets import load_dataset

# Load 10,000 text chunks from the dataset
data = load_dataset(
    "jamescalam/ai-arxiv2-semantic-chunks",
    split="train[:10000]"
)

# Remove unnecessary columns to simplify our data
data = data.remove_columns(["prechunk_id", "postchunk_id", "references"])
print(f"Loaded {len(data)} documents.")
```

Each document in the dataset now contains `id`, `title`, `content`, and `arxiv_id` fields. The `content` field is the text we will embed.

## Step 3: Initialize the Mistral AI Client

You'll need your Mistral AI API key to generate embeddings.

```python
import os
from mistralai import Mistral
import getpass

# Securely fetch your API key
mistral_api_key = os.getenv("MISTRAL_API_KEY") or getpass.getpass("Enter your Mistral API key: ")

# Initialize the client
mistral = Mistral(api_key=mistral_api_key)

# Test the embedding model
embed_model = "mistral-embed"
test_embed = mistral.embeddings.create(model=embed_model, inputs=["this is a test"])
dims = len(test_embed.data[0].embedding)
print(f"Embedding dimensionality: {dims}")
```

Take note of the `dims` value (e.g., 1024). You'll need it when configuring your vector index.

## Step 4: Configure Azure AI Search Authentication

You can authenticate to Azure AI Search using either an API key or Azure Active Directory (AAD). This example supports both methods.

```python
from getpass import getpass
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
import os

# Configuration - Set to False to use an API key
USE_AAD_FOR_SEARCH = True
SEARCH_SERVICE_ENDPOINT = os.getenv("SEARCH_SERVICE_ENDPOINT") or getpass("Enter your Azure AI Search Service Endpoint: ")

def authenticate_azure_search(use_aad_for_search=False):
    if use_aad_for_search:
        print("Using AAD for authentication.")
        credential = DefaultAzureCredential()
    else:
        print("Using API keys for authentication.")
        api_key = os.getenv("SEARCH_SERVICE_API_KEY") or getpass("Enter your Azure AI Search Service API Key: ")
        if api_key is None:
            raise ValueError("API key must be provided if not using AAD for authentication.")
        credential = AzureKeyCredential(api_key)
    return credential

azure_search_credential = authenticate_azure_search(
    use_aad_for_search=USE_AAD_FOR_SEARCH
)
```

## Step 5: Create a Vector Search Index

Now, create an index in Azure AI Search designed to store your text and its corresponding vector embeddings.

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField, SearchFieldDataType, SearchableField, SearchField,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SemanticConfiguration, SemanticPrioritizedFields, SemanticField,
    SemanticSearch, SearchIndex
)

DIMENSIONS = 1024  # Must match the embedding model's output dimension
HNSW_PARAMETERS = {"m": 4, "metric": "cosine", "ef_construction": 400, "ef_search": 500}
INDEX_NAME = "ai-arxiv2-semantic-chunks"

# Define the schema for the index
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
    SearchableField(name="title", type=SearchFieldDataType.String),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SearchableField(name="arxiv_id", type=SearchFieldDataType.String, filterable=True),
    SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=DIMENSIONS, vector_search_profile_name="myHnswProfile")
]

# Configure vector search
vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name="myHnsw", parameters=HNSW_PARAMETERS)],
    profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")]
)

# (Optional) Configure semantic search for improved relevance
semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="title"),
        keywords_fields=[SemanticField(field_name="arxiv_id")],
        content_fields=[SemanticField(field_name="content")]
    )
)
semantic_search = SemanticSearch(configurations=[semantic_config])

# Create the index
index_client = SearchIndexClient(endpoint=SEARCH_SERVICE_ENDPOINT, credential=azure_search_credential)
index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search, semantic_search=semantic_search)
result = index_client.create_or_update_index(index)
print(f"Index '{result.name}' created or updated successfully.")
```

## Step 6: Transform Data and Generate Embeddings

Before uploading, you must transform your dataset and generate vector embeddings for the `content` field.

### 6.1 Transform the Dataset

```python
# Function to transform your dataset into the format required by Azure AI Search
def transform_to_search_document(record):
    return {
        "id": record["id"],
        "arxiv_id": record["arxiv_id"],
        "title": record["title"],
        "content": record["content"]
    }

# Transform all documents
transformed_documents = [transform_to_search_document(doc) for doc in data]
```

### 6.2 Generate Embeddings in Batches

To manage costs and API limits, generate embeddings in batches.

```python
def generate_embeddings(documents, model, batch_size=20):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        contents = [doc['content'] for doc in batch]
        embeds = mistral.embeddings.create(model=model, inputs=contents)
        for j, document in enumerate(batch):
            document['embedding'] = embeds.data[j].embedding
    return documents

embed_model = "mistral-embed"
generate_embeddings(transformed_documents, embed_model)
print(f"Generated embeddings for {len(transformed_documents)} documents.")
```

### 6.3 Encode Document IDs

Azure AI Search has restrictions on document ID characters. A safe practice is to base64 encode them.

```python
import base64

def encode_key(key):
    return base64.urlsafe_b64encode(key.encode()).decode()

for document in transformed_documents:
    document['id'] = encode_key(document['id'])
```

## Step 7: Upload Documents to the Index

Upload the transformed documents, complete with their embeddings, to your Azure AI Search index.

```python
from azure.search.documents import SearchIndexingBufferedSender

def upload_documents(index_name, endpoint, credential, documents):
    buffered_sender = SearchIndexingBufferedSender(endpoint=endpoint, index_name=index_name, credential=credential)
    for document in documents:
        buffered_sender.merge_or_upload_documents(documents=[document])
    buffered_sender.flush()
    print(f"Uploaded {len(documents)} documents in total")

upload_documents(INDEX_NAME, SEARCH_SERVICE_ENDPOINT, azure_search_credential, transformed_documents)
```

## Step 8: Perform a Vector Search

With your data indexed, you can now perform semantic searches. Convert a user's natural language query into an embedding and find the most relevant documents.

```python
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

# Initialize the search client
search_client = SearchClient(endpoint=SEARCH_SERVICE_ENDPOINT, index_name=INDEX_NAME, credential=azure_search_credential)

def generate_query_embedding(query):
    embed = mistral.embeddings.create(model="mistral-embed", inputs=[query])
    return embed.data[0].embedding

# Define your search query
query = "where is Mistral AI headquartered?"

# Generate the query embedding and perform the search
vector_query = VectorizedQuery(vector=generate_query_embedding(query), k_nearest_neighbors=3, fields="embedding")
results = search_client.search(search_text=None, vector_queries=[vector_query], select=["id", "arxiv_id", "title", "content"])

# Display the top results
for result in results:
    print(f"Title: {result['title']}")
    print(f"Score: {result['@search.score']:.3f}")
    print(f"Content Preview: {result['content'][:150]}...\n{'-'*50}")
```

## Step 9: Ground Results in the Mistral Large Model (Direct API)

Use the context retrieved from your search to generate a precise answer using Mistral's chat model via their direct API.

```python
from mistralai import SystemMessage, UserMessage

# Initialize the chat client (reuses the earlier 'mistral' client)
client = Mistral(api_key=mistral_api_key)

# Format the search results into a context string
context = "\n---\n".join([
    f"Title: {result['title']}\nContent: {result['content']}"
    for result in results
])

# Create the system and user messages
system_message = SystemMessage(
    content=f"You are a helpful assistant that answers questions using the provided context.\n\nCONTEXT:\n{context}"
)
user_message = UserMessage(content=query)

# Generate the grounded response
messages = [system_message, user_message]
chat_response = client.chat.complete(model="mistral-large-latest", messages=messages, max_tokens=100)

print("Answer:", chat_response.choices[0].message.content)
```

## Step 10: Ground Results in Mistral Large (Azure AI Studio)

If you have deployed a Mistral model via Azure AI Studio, you can use that endpoint instead. The process is nearly identical, requiring only a different client configuration.

```python
import getpass
import os

# Fetch your Azure AI Studio endpoint and key
azure_ai_studio_mistral_base_url = os.getenv("AZURE_AI_STUDIO_MISTRAL_BASE_URL") or getpass.getpass("Enter your Azure Mistral Deployed Endpoint Base URL: ")
azure_ai_studio_mistral_api_key = os.getenv("AZURE_AI_STUDIO_MISTRAL_API_KEY") or getpass.getpass("Enter your Azure Mistral API Key: ")

# Initialize the client for Azure AI Studio
client_azure = Mistral(endpoint=azure_ai_studio_mistral_base_url, api_key=azure_ai_studio_mistral_api_key)

# Reuse the same context and messages
messages = [system_message, user_message]
chat_response_azure = client_azure.chat.complete(model="azureai", messages=messages, max_tokens=100)

print("Answer (Azure AI Studio):", chat_response_azure.choices[0].message.content)
```

## Summary

You have successfully built an end-to-end RAG pipeline. You learned how to:
1.  Load and prepare a text dataset.
2.  Generate vector embeddings using Mistral AI.
3.  Create and populate a vector search index in Azure AI Search.
4.  Perform semantic search to find relevant context.
5.  Use that context to ground responses from a large language model, both via Mistral's direct API and an Azure AI Studio deployment.

This pipeline can be adapted for various use cases, such as building intelligent Q&A systems, research assistants, or enterprise knowledge bases.