# Building a RAG System with Azure AI Search and Azure Functions

## Overview
This guide walks you through creating a Retrieval-Augmented Generation (RAG) system using Azure AI Search as a vector database and Azure Functions as a serverless API layer. You'll learn how to embed documents, store them in Azure AI Search, and expose them through an Azure Function that can be integrated with ChatGPT's Custom GPTs.

## Prerequisites

Before starting, ensure you have:
- An Azure subscription with permissions to create Azure AI Search and Azure Function resources
- An OpenAI API key
- Python 3.8 or higher

## Step 1: Environment Setup

### Install Required Libraries
First, install the necessary Python packages:

```bash
pip install azure-search-documents azure-identity openai azure-mgmt-search pandas azure-mgmt-resource azure-mgmt-storage pyperclip PyPDF2 tiktoken python-dotenv
```

### Import Libraries
Create a new Python file and import the required libraries:

```python
# Standard Libraries
import json
import os
import uuid
from itertools import islice

# Third-Party Libraries
from PyPDF2 import PdfReader
import tiktoken
import pandas as pd

# OpenAI
from openai import OpenAI

# Azure Identity and Credentials
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.core.credentials import AzureKeyCredential

# Azure Search Documents
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchableField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

# Azure Management Clients
from azure.mgmt.search import SearchManagementClient
from azure.mgmt.resource import ResourceManagementClient, SubscriptionClient
from azure.mgmt.storage import StorageManagementClient
```

## Step 2: Configure OpenAI Settings

Set up your OpenAI client and embedding model:

```python
# Configure OpenAI
openai_api_key = os.environ.get("OPENAI_API_KEY", "<your-openai-api-key>")
openai_client = OpenAI(api_key=openai_api_key)
embeddings_model = "text-embedding-3-small"  # Can change to text-embedding-3-large if desired
```

## Step 3: Configure Azure AI Search

### Set Azure Credentials and Resource Details
Configure your Azure subscription details:

```python
# Update with your Azure details
subscription_id = "<your-subscription-id>"
resource_group = "<your-resource-group>"
region = "eastus"  # Choose a supported region

# Initialize Azure credentials
credential = InteractiveBrowserCredential()
```

### Create Azure AI Search Service
Now, create the search service programmatically:

```python
# Initialize Search Management Client
search_management_client = SearchManagementClient(
    credential=credential,
    subscription_id=subscription_id,
)

# Generate unique service name
generated_uuid = str(uuid.uuid4())
search_service_name = f"search-service-gpt-demo-{generated_uuid}"
search_service_endpoint = f"https://{search_service_name}.search.windows.net"

# Create the search service
response = search_management_client.services.begin_create_or_update(
    resource_group_name=resource_group,
    search_service_name=search_service_name,
    service={
        "location": region,
        "properties": {"hostingMode": "default", "partitionCount": 1, "replicaCount": 1},
        "sku": {"name": "free"},  # Free tier for demo
        "tags": {"app-name": "Search service demo"},
    },
).result()

print(f"Search Service Name: {search_service_name}")
print(f"Search Service Endpoint: {search_service_endpoint}")
```

### Retrieve Search Service API Key
Get the API key needed for authentication:

```python
# Retrieve admin keys
response = search_management_client.admin_keys.get(
    resource_group_name=resource_group,
    search_service_name=search_service_name,
)
search_service_api_key = response.primary_key
print("Successfully retrieved the API key.")
```

## Step 4: Prepare Your Data

### Define Helper Functions for Text Processing
Create functions to handle text chunking and tokenization:

```python
def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch

def chunked_tokens(text, chunk_length, encoding_name='cl100k_base'):
    """Tokenize text and break into chunks."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator
```

### Define Embedding Generation Functions
Create functions to generate embeddings with proper chunking:

```python
# Embedding model constants
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

def generate_embeddings(text, model):
    """Generate embeddings for a single text chunk."""
    embeddings_response = openai_client.embeddings.create(model=model, input=text)
    return embeddings_response.data[0].embedding

def len_safe_get_embedding(text, model=embeddings_model, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING):
    """Safely generate embeddings for long texts by chunking."""
    chunk_embeddings = []
    chunk_texts = []
    
    for chunk in chunked_tokens(text, chunk_length=max_tokens, encoding_name=encoding_name):
        chunk_embeddings.append(generate_embeddings(chunk, model=model))
        chunk_texts.append(tiktoken.get_encoding(encoding_name).decode(chunk))
    
    return chunk_embeddings, chunk_texts
```

### Add Metadata Categorization
Categorize documents for better filtering:

```python
categories = ['authentication', 'models', 'techniques', 'tools', 'setup', 'billing_limits', 'other']

def categorize_text(text, categories):
    """Categorize text using GPT-4."""
    messages = [
        {"role": "system", "content": f"""You are an expert in LLMs, and you will be given text that corresponds to an article in OpenAI's documentation.
         Categorize the document into one of these categories: {', '.join(categories)}. Only respond with the category name and nothing else."""},
        {"role": "user", "content": text}
    ]
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error categorizing text: {str(e)}")
        return None
```

### Process Documents
Create a function to process your documents:

```python
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF files."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def process_file(file_path, idx, categories, embeddings_model):
    """Process a single file and generate embeddings."""
    file_name = os.path.basename(file_path)
    print(f"Processing file {idx + 1}: {file_name}")
    
    # Read text content
    if file_name.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    elif file_name.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    
    # Generate embeddings
    title = file_name
    title_vectors, title_text = len_safe_get_embedding(title, embeddings_model)
    content_vectors, content_text = len_safe_get_embedding(text, embeddings_model)
    
    # Categorize
    category = categorize_text(' '.join(content_text), categories)
    print(f"Categorized {file_name} as {category}")
    
    # Prepare data for each chunk
    data = []
    for i, content_vector in enumerate(content_vectors):
        data.append({
            "id": f"{idx}_{i}",
            "vector_id": f"{idx}_{i}",
            "title": title_text[0],
            "text": content_text[i],
            "title_vector": json.dumps(title_vectors[0]),
            "content_vector": json.dumps(content_vector),
            "category": category
        })
    
    return data
```

### Process All Documents
Now process your document folder:

```python
# Set your document folder path
folder_name = "../../../data/oai_docs"  # Update with your path
files = [os.path.join(folder_name, f) for f in os.listdir(folder_name) 
         if f.endswith('.txt') or f.endswith('.pdf')]

data = []

# Process files (you can add concurrent processing here if needed)
for idx, file_path in enumerate(files):
    try:
        result = process_file(file_path, idx, categories, embeddings_model)
        data.extend(result)
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")

# Save to CSV for inspection
import pandas as pd
df = pd.DataFrame(data)
df.to_csv('processed_documents.csv', index=False)
print(f"Processed {len(data)} document chunks")
```

## Step 5: Create Azure AI Search Index

### Define the Search Index Schema
Create a schema that includes vector fields for semantic search:

```python
def create_search_index(index_name, search_service_endpoint, search_service_api_key):
    """Create a search index with vector support."""
    
    # Initialize the search index client
    credential = AzureKeyCredential(search_service_api_key)
    index_client = SearchIndexClient(
        endpoint=search_service_endpoint,
        credential=credential
    )
    
    # Define fields
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="vector_id", type=SearchFieldDataType.String),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="text", type=SearchFieldDataType.String),
        SearchField(
            name="title_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,  # OpenAI embedding dimension
            vector_search_profile_name="myHnswProfile"
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,  # OpenAI embedding dimension
            vector_search_profile_name="myHnswProfile"
        ),
        SimpleField(name="category", type=SearchFieldDataType.String, filterable=True)
    ]
    
    # Configure vector search
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters=HnswParameters(
                    m=4,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE
                )
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw"
            )
        ]
    )
    
    # Create the index
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search
    )
    
    try:
        index_client.create_index(index)
        print(f"Index '{index_name}' created successfully.")
    except Exception as e:
        print(f"Failed to create index: {e}")
        raise

# Create the index
index_name = "gpt-demo-index"
create_search_index(index_name, search_service_endpoint, search_service_api_key)
```

### Upload Documents to the Index
Upload your processed documents:

```python
def upload_documents_to_index(documents, index_name, search_service_endpoint, search_service_api_key):
    """Upload documents to the search index."""
    
    credential = AzureKeyCredential(search_service_api_key)
    search_client = SearchClient(
        endpoint=search_service_endpoint,
        index_name=index_name,
        credential=credential
    )
    
    # Prepare documents for upload
    upload_docs = []
    for doc in documents:
        upload_doc = {
            "id": doc["id"],
            "vector_id": doc["vector_id"],
            "title": doc["title"],
            "text": doc["text"],
            "title_vector": json.loads(doc["title_vector"]),
            "content_vector": json.loads(doc["content_vector"]),
            "category": doc["category"]
        }
        upload_docs.append(upload_doc)
    
    # Upload in batches
    batch_size = 100
    for i in range(0, len(upload_docs), batch_size):
        batch = upload_docs[i:i + batch_size]
        result = search_client.upload_documents(batch)
        
        # Check for errors
        for res in result:
            if not res.succeeded:
                print(f"Failed to upload document {res.key}: {res.error_message}")
    
    print(f"Uploaded {len(upload_docs)} documents to index '{index_name}'")

# Upload your processed data
upload_documents_to_index(data, index_name, search_service_endpoint, search_service_api_key)
```

## Step 6: Test the Search Functionality

Test that your vector search works correctly:

```python
def search_documents(query, index_name, search_service_endpoint, search_service_api_key, category_filter=None):
    """Search documents using vector similarity."""
    
    # Generate query embedding
    query_embedding = generate_embeddings(query, embeddings_model)
    
    # Initialize search client
    credential = AzureKeyCredential(search_service_api_key)
    search_client = SearchClient(
        endpoint=search_service_endpoint,
        index_name=index_name,
        credential=credential
    )
    
    # Create vector query
    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=3,
        fields="content_vector"
    )
    
    # Build search filter
    filter_expression = None
    if category_filter:
        filter_expression = f"category eq '{category_filter}'"
    
    # Execute search
    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        filter=filter_expression,
        select=["title", "text", "category"],
        top=3
    )
    
    # Process results
    search_results = []
    for result in results:
        search_results.append({
            "title": result["title"],
            "text": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"],
            "category": result["category"],
            "score": result["@search.score"]
        })
    
    return search_results

# Test the search
test_query = "How do I authenticate with the OpenAI API?"
results = search_documents(test_query, index_name, search_service_endpoint, search_service_api_key)

print("Search Results:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. Title: {result['title']}")
    print(f"   Category: {result['category']}")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Text: {result['text']}")
```

## Next Steps

You've successfully:
1. Set up Azure AI Search as a vector database
2. Processed and embedded your documents
3. Created a search index with vector support
4. Uploaded your documents
5. Tested vector search functionality

In the next part of this guide, you'll learn how to:
- Create an Azure Function to expose this search as an API
- Integrate the function with ChatGPT Custom GPTs
- Add authentication and monitoring

The Azure Function will serve as a bridge between your vector database and ChatGPT, allowing you to create a fully functional RAG system within Azure's ecosystem.