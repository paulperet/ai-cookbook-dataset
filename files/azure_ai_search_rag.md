# RAG with Mistral AI, Azure AI Search and Azure AI Studio

## Overview

This notebook demonstrates how to integrate Mistral Embeddings with Azure AI Search as a vector store, and use the results to ground responses in the Mistral Chat Completion Model.

## Prerequisites

- Mistral AI API Key OR Azure AI Studio Deployed Mistral Chat Completion Model and Azure AI Studio API Key
- Azure AI Search service
- Python 3.x environment with necessary libraries installed

## Steps

1. Install required packages
2. Load data and generate Mistral embeddings
3. Index embeddings in Azure AI Search
4. Perform search using Azure AI Search
5. Ground search results in Mistral Chat Completion Model

## Install Required Packages

```python
# Install Required Packages
!pip install azure-search-documents==11.5.1
!pip install azure-identity==1.16.0 datasets==2.19.1 mistralai==1.0.1
```

## Load Data and Generate Mistral Embeddings

```python
from datasets import load_dataset

data = load_dataset(
    "jamescalam/ai-arxiv2-semantic-chunks",
    split="train[:10000]"
)
data
```

We have 10K chunks, where each chunk is roughly the length of 1-2 paragraphs in length. Here is an example of a single record:

```python
data[0]
```

Format the data into the format we need, this will contain `id`, `title`, `content` (which we will embed), and `arxiv_id`.

```python
data = data.remove_columns(["prechunk_id", "postchunk_id", "references"])
data
```

We need to define an embedding model to create our embedding vectors for retrieval, for that we will be using Mistral AI's `mistral-embed`. There is some cost associated with this model, so be aware of that (costs for running this notebook are <$1).

```python
import os
from mistralai import Mistral

import getpass  # for securely inputting API key

# Fetch the API key from environment variable or prompt the user
mistral_api_key = os.getenv("MISTRAL_API_KEY") or getpass.getpass("Enter your Mistral API key: ")

# Initialize the Mistral client
mistral = Mistral(api_key=mistral_api_key)

```

```python
embed_model = "mistral-embed"

embeds = mistral.embeddings.create(
    model=embed_model, inputs=["this is a test"]
)
```

We can view the dimensionality of our returned embeddings, which we'll need soon when initializing our vector index:

```python
dims = len(embeds.data[0].embedding)
dims
```

## Index Embeddings into Azure AI Search

Now we create our vector DB to store our vectors. For this, we need to set up an [Azure AI Search service](https://portal.azure.com/#create/Microsoft.Search).

There are two ways to authenticate to Azure AI Search:

1. **Service Key**: The service key can be found in the "Settings -> Keys" section in the left navbar of the Azure portal dashboard. Make sure to select the ADMIN key.
2. **Managed Identity**: Using Microsoft Entra ID (f.k.a. Azure Active Directory) is a more secure and recommended way to authenticate. You can follow the instructions in the [official Microsoft documentation](https://learn.microsoft.com/azure/search/search-security-rbac) to set up Managed Identity.

For more detailed instructions on creating an Azure AI Search service, please refer to the [official Microsoft documentation](https://learn.microsoft.com/azure/search/search-create-service-portal).

### Authenticate into Azure AI Search

```python
from getpass import getpass
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
import os

# Configuration variable
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

### Create a vector index

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField, SearchFieldDataType, SearchableField, SearchField,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SemanticConfiguration, SemanticPrioritizedFields, SemanticField,
    SemanticSearch, SearchIndex
)

DIMENSIONS = 1024
HNSW_PARAMETERS = {"m": 4, "metric": "cosine", "ef_construction": 400, "ef_search": 500}
INDEX_NAME = "ai-arxiv2-semantic-chunks"

# Create a search index
index_client = SearchIndexClient(endpoint=SEARCH_SERVICE_ENDPOINT, credential=azure_search_credential)
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=False, filterable=True, facetable=False),
    SearchableField(name="title", type=SearchFieldDataType.String),
    SearchableField(name="content", type=SearchFieldDataType.String),
    SearchableField(name="arxiv_id", type=SearchFieldDataType.String, filterable=True),
    SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=DIMENSIONS, vector_search_profile_name="myHnswProfile", hidden=False)
]

vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name="myHnsw", parameters=HNSW_PARAMETERS)],
    profiles=[VectorSearchProfile(name="myHnswProfile", algorithm_configuration_name="myHnsw")]
)

semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        title_field=SemanticField(field_name="title"),
        keywords_fields=[SemanticField(field_name="arxiv_id")],
        content_fields=[SemanticField(field_name="content")]
    )
)

semantic_search = SemanticSearch(configurations=[semantic_config])
index = SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search, semantic_search=semantic_search)
result = index_client.create_or_update_index(index)
print(f"{result.name} created")
```

### Estimate Cost for Embedding Generation

As per the information from [Lunary.ai's Mistral Tokenizer](https://lunary.ai/mistral-tokenizer), one token is approximately equivalent to five characters of text.

According to [Mistral's Pricing](https://mistral.ai/technology/#pricing), the cost for using `mistral-embed` is $0.1 per 1M tokens for both inputs and outputs.

In the following code block, we will calculate the estimated cost for generating embeddings based on the size of our dataset and these pricing details.

```python
# Estimate cost for generating embeddings
def estimate_cost(data, cost_per_million_tokens=0.1):
    total_characters = sum(len(entry['content']) for entry in data)
    total_tokens = total_characters / 5  # 1 token is approximately 5 characters
    total_cost = (total_tokens / 1_000_000) * cost_per_million_tokens
    return total_cost

estimated_cost = estimate_cost(data)
print(f"Estimated cost for generating embeddings: ${estimated_cost}")
```

### Transform Dataset for Azure AI Search Upload

```python
# Function to transform your dataset into the format required by Azure AI Search
def transform_to_search_document(record):
    return {
        "id": record["id"],
        "arxiv_id": record["arxiv_id"],
        "title": record["title"],
        "content": record["content"]
    }
    
# Transform all documents in the dataset
transformed_documents = [transform_to_search_document(doc) for doc in data]
```

### Generate Embeddings

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
```

Azure AI Search doesn't allow certain unsafe keys so we'll base64 encode `id` here

```python
import base64

# Base64 encode IDs for Azure AI Search compatibility
def encode_key(key):
    return base64.urlsafe_b64encode(key.encode()).decode()

for document in transformed_documents:
    document['id'] = encode_key(document['id'])
```

### Upload Documents

```python
from azure.search.documents import SearchIndexingBufferedSender

# Upload documents
def upload_documents(index_name, endpoint, credential, documents):
    buffered_sender = SearchIndexingBufferedSender(endpoint=endpoint, index_name=index_name, credential=credential)
    for document in documents:
        buffered_sender.merge_or_upload_documents(documents=[document])
    buffered_sender.flush()
    print(f"Uploaded {len(documents)} documents in total")

upload_documents(INDEX_NAME, SEARCH_SERVICE_ENDPOINT, azure_search_credential, transformed_documents)
```

## Perform a Vector Search

```python
from azure.search.documents import SearchClient

search_client = SearchClient(endpoint=SEARCH_SERVICE_ENDPOINT, index_name=INDEX_NAME, credential=azure_search_credential)
from azure.search.documents.models import VectorizedQuery

# Generate query embedding and perform search
def generate_query_embedding(query):
    embed = mistral.embeddings.create(model="mistral-embed", inputs=[query])
    return embed.data[0].embedding

query = "where is Mistral AI headquartered?"
vector_query = VectorizedQuery(vector=generate_query_embedding(query), k_nearest_neighbors=3, fields="embedding")
results = search_client.search(search_text=None, vector_queries=[vector_query], select=["id", "arxiv_id", "title", "content"])

for result in results:
    print(f"ID: {result['id']}\nArxiv ID: {result['arxiv_id']}\nTitle: {result['title']}\nScore: {result['@search.score']}\nContent: {result['content']}\n{'-' * 50}")
```

## Ground retrieved results from Azure AI Search to Mistral-Large LLM

```python
from mistralai import Mistral, SystemMessage, UserMessage

# Initialize the client
client = Mistral(api_key=mistral_api_key)
context = "\n---\n".join([f"ID: {result['id']}\nArxiv ID: {result['arxiv_id']}\nTitle: {result['title']}\nScore: {result['@search.score']}\nContent: {result['content']}" for result in results])
system_message = SystemMessage(content="You are a helpful assistant that answers questions about AI using the context provided below.\n\nCONTEXT:\n" + context)
user_message = UserMessage(content="where is Mistral AI headquartered?")

# Generate the response
messages = [system_message, user_message]
chat_response = client.chat.complete(model="mistral-large-latest", messages=messages, max_tokens=50)

print(chat_response.choices[0].message.content)
```

## Ground Results to Mistral-Large hosted in Azure AI Studio

```python
import getpass
from mistralai import Mistral, SystemMessage, UserMessage

azure_ai_studio_mistral_base_url = os.getenv("AZURE_AI_STUDIO_MISTRAL_BASE_URL") or getpass.getpass("Enter your Azure Mistral Deployed Endpoint Base URL: ")
azure_ai_studio_mistral_api_key = os.getenv("AZURE_AI_STUDIO_MISTRAL_API_KEY") or getpass.getpass("Enter your Azure Mistral API Key: ")

# Initialize the client for Azure AI Studio
client = Mistral(endpoint=azure_ai_studio_mistral_base_url, api_key=azure_ai_studio_mistral_api_key)
context = "\n---\n".join([f"ID: {result['id']}\nArxiv ID: {result['arxiv_id']}\nTitle: {result['title']}\nScore: {result['@search.score']}\nContent: {result['content']}" for result in results])
system_message = SystemMessage(content="You are a helpful assistant that answers questions about AI using the context provided below.\n\nCONTEXT:\n" + context)
user_message = UserMessage(content="where is Mistral AI headquartered?")

# Generate the response
messages = [system_message, user_message]
chat_response = client.chat.complete(model="azureai", messages=messages, max_tokens=50)

print(chat_response.choices[0].message.content)
```