# Azure AI Search as a vector database + Azure Functions for GPT integration in ChatGPT

This notebook provides step by step instuctions on using Azure AI Search (f.k.a Azure Cognitive Search) as a vector database with OpenAI embeddings, then creating an Azure Function on top to plug into a Custom GPT in ChatGPT. 

This can be a solution for customers looking to set up RAG infrastructure contained within Azure, and exposing it as an endpoint to integrate that with other platforms such as ChatGPT.

Azure AI Search is a cloud search service that gives developers infrastructure, APIs, and tools for building a rich search experience over private, heterogeneous content in web, mobile, and enterprise applications. 

Azure Functions is a serverless compute service that runs event-driven code, automatically managing infrastructure, scaling, and integrating with other Azure services.

## Prerequisites:
For the purposes of this exercise you must have the following:
- Azure user with permission to create [Azure AI Search Service](https://learn.microsoft.com/azure/search/) and Azure Function Apps
- Azure subscription ID and a resource group.
- [OpenAI Key](https://platform.openai.com/account/api-keys) 

# Architecture
Below is a diagram of the architecture of this solution, which we'll walk through step-by-step.

> Note: This architecture pattern of vector data store + serverless functions can be extrapolated to other vector data stores. For example, if you would want to use something like Postgres within Azure, you'd change the [Configure Azure AI Search Settings](#configure-azure-ai-search-settings) step to set-up the requirements for Postgres, you'd modify the [Create Azure AI Vector Search](#create-azure-ai-vector-search) to create the database and table in Postgres instead, and you'd update the `function_app.py` code in this repository to query Postgres instead of Azure AI Search. The data preparation and creation of the Azure Function would stay consistent. 


# Table of Contents:

1. **[Setup of Environment](#set-up-environment)**
    Setup environment by installing and importing the required libraries and configuring our Azure settings. Includes:
     - [Install and Import Required Libraries](#install-and-import-required-libraries)
     - [Configure OpenAI Settings](#configure-openai-settings)
     - [Configure Azure AI Search Settings](#configure-azure-ai-search-settings)
 

2. **[Prepare Data](#prepare-data)** Prepare the data for uploading by embedding the documents, as well as capturing additional metadata. We will use a subset of OpenAI's docs as example data for this.
 
3. **[Create Azure AI Vector Search](#create-azure-ai-vector-search)** Create an Azure AI Vector Search and upload the data we've prepared. Includes:
     - [Create Index](#create-index): Steps to create an index in Azure AI Search.
     - [Upload Data](#upload-data): Instructions to upload data to Azure AI Search.
     - [Test Search](#test-search): Steps to test the search functionality.
 
4. **[Create Azure Function](#create-azure-function)** Create an Azure Function to interact with the Azure AI Vector Search. Includes:
     - [Create Storage Account](#create-storage-account): Steps to create a storage account for the Azure Function.
     - [Create Function App](#create-function-app): Instructions to create a function app in Azure.
 
5. **[Input in a Custom GPT in ChatGPT](#input-in-a-custom-gpt-in-chatgpt)** Integrate the Azure Function with a Custom GPT in ChatGPT. Includes:
     - [Create OpenAPI Spec](#create-openapi-spec): Steps to create an OpenAPI specification for the Azure Function.
     - [Create GPT Instructions](#create-gpt-instructions): Instructions to create GPT-specific instructions for the integration.




# Set up environment
We'll set up our environment by importing the required libraries and configuring our Azure settings.

## Install and import required libraries
We categorize these libraries into standard Python libraries, third-party libraries, and Azure-related libraries for readability.


```python
! pip install -q wget
! pip install -q azure-search-documents 
! pip install -q azure-identity
! pip install -q openai
! pip install -q azure-mgmt-search
! pip install -q pandas
! pip install -q azure-mgmt-resource 
! pip install -q azure-mgmt-storage
! pip install -q pyperclip
! pip install -q PyPDF2
! pip install -q tiktoken
```


```python
# Standard Libraries
import json  
import os
import platform
import subprocess
import csv
from itertools import islice
import uuid
import shutil
import concurrent.futures

# Third-Party Libraries
import pandas as pd
from PyPDF2 import PdfReader
import tiktoken
from dotenv import load_dotenv
import pyperclip

# OpenAI Libraries (note we use OpenAI directly here, but you can replace with Azure OpenAI as needed)
from openai import OpenAI

# Azure Identity and Credentials
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.core.credentials import AzureKeyCredential  
from azure.core.exceptions import HttpResponseError

# Azure Search Documents
from azure.search.documents import SearchClient, SearchIndexingBufferedSender  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import (
    VectorizedQuery
)
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

## Configure OpenAI settings

Before going through this section, make sure you have your OpenAI API key.



```python
openai_api_key = os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>") # Saving this as a variable to reference in function app in later step
openai_client = OpenAI(api_key=openai_api_key)
embeddings_model = "text-embedding-3-small" # We'll use this by default, but you can change to your text-embedding-3-large if desired
```

## Configure Azure AI Search Settings
You can locate your Azure AI Search service details in the Azure Portal or programmatically via the [Search Management SDK](https://learn.microsoft.com/rest/api/searchmanagement/).


#### Prerequisites:
- Subscription ID from Azure
- Resource Group name from Azure
- Region in Azure


```python
# Update the below with your values
subscription_id="<enter_your_subscription_id>"
resource_group="<enter_your_resource_group>"

## Make sure to choose a region that supports the proper products. We've defaulted to "eastus" below. https://azure.microsoft.com/en-us/explore/global-infrastructure/products-by-region/#products-by-region_tab5
region = "eastus"
credential = InteractiveBrowserCredential()
subscription_client = SubscriptionClient(credential)
subscription = next(subscription_client.subscriptions.list())
```

#### Create and Configure Azure AI Search Service
Below we'll generate a unique name for the search service, set up the service properties, and create the search service.


```python
# Initialize the SearchManagementClient with the provided credentials and subscription ID
search_management_client = SearchManagementClient(
    credential=credential,
    subscription_id=subscription_id,
)

# Generate a unique name for the search service using UUID, but you can change this if you'd like.
generated_uuid = str(uuid.uuid4())
search_service_name = "search-service-gpt-demo" + generated_uuid
## The below is the default endpoint structure that is created when you create a search service. This may differ based on your Azure settings.
search_service_endpoint = 'https://'+search_service_name+'.search.windows.net'

# Create or update the search service with the specified parameters
response = search_management_client.services.begin_create_or_update(
    resource_group_name=resource_group,
    search_service_name=search_service_name,
    service={
        "location": region,
        "properties": {"hostingMode": "default", "partitionCount": 1, "replicaCount": 1},
        # We are using the free pricing tier for this demo. You are only allowed one free search service per subscription.
        "sku": {"name": "free"},
        "tags": {"app-name": "Search service demo"},
    },
).result()

# Convert the response to a dictionary and then to a pretty-printed JSON string
response_dict = response.as_dict()
response_json = json.dumps(response_dict, indent=4)

print(response_json)
print("Search Service Name:" + search_service_name)
print("Search Service Endpoint:" + search_service_endpoint)

```

#### Get the Search Service API Key
Now that we have the search service up and running, we need the [Search Service API Key](https://learn.microsoft.com/en-us/azure/search/search-security-api-keys?tabs=rest-use,portal-find,portal-query), which we'll use to initiate the index creation, and later to execute the search.


```python
# Retrieve the admin keys for the search service
try:
    response = search_management_client.admin_keys.get(
        resource_group_name=resource_group,
        search_service_name=search_service_name,
    )
    # Extract the primary API key from the response and save as a variable to be used later
    search_service_api_key = response.primary_key
    print("Successfully retrieved the API key.")
except Exception as e:
    print(f"Failed to retrieve the API key: {e}")
```

# Prepare data
We're going to embed and store a few pages of the OpenAI docs in the oai_docs folder. We'll first embed each, add it to a CSV, and then use that CSV to upload to the index.

In order to handle longer text files beyond the context of 8191 tokens, we can either use the chunk embeddings separately, or combine them in some way, such as averaging (weighted by the size of each chunk).

We will take a function from Python's own cookbook that breaks up a sequence into chunks.


```python
def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch

```

Now we define a function that encodes a string into tokens and then breaks it up into chunks. We'll use tiktoken, a fast open-source tokenizer by OpenAI.

To read more about counting tokens with Tiktoken, check out [this cookbook](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken). 




```python
def chunked_tokens(text, chunk_length, encoding_name='cl100k_base'):
    # Get the encoding object for the specified encoding name. OpenAI's tiktoken library, which is used in this notebook, currently supports two encodings: 'bpe' and 'cl100k_base'. The 'bpe' encoding is used for GPT-3 and earlier models, while 'cl100k_base' is used for newer models like GPT-4.
    encoding = tiktoken.get_encoding(encoding_name)
    # Encode the input text into tokens
    tokens = encoding.encode(text)
    # Create an iterator that yields chunks of tokens of the specified length
    chunks_iterator = batched(tokens, chunk_length)
    # Yield each chunk from the iterator
    yield from chunks_iterator
```

Finally, we can write a function that safely handles embedding requests, even when the input text is longer than the maximum context length, by chunking the input tokens and embedding each chunk individually. The average flag can be set to True to return the weighted average of the chunk embeddings, or False to simply return the unmodified list of chunk embeddings.

> Note: there are other, more sophisticated techniques you can take here, including:
> - using GPT-4o to capture images/chart descriptions for embedding.
> - keeping text overlap between the chunks to minimize cutting off important context.
> - chunking based on paragraphs or sections.
> - adding more descriptive metadata about each article.


```python
## Change the below based on model. The below is for the latest embeddings models from OpenAI, so you can leave as is unless you are using a different embedding model..
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING='cl100k_base'
```


```python
def generate_embeddings(text, model):
    # Generate embeddings for the provided text using the specified model
    embeddings_response = openai_client.embeddings.create(model=model, input=text)
    # Extract the embedding data from the response
    embedding = embeddings_response.data[0].embedding
    return embedding

def len_safe_get_embedding(text, model=embeddings_model, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING):
    # Initialize lists to store embeddings and corresponding text chunks
    chunk_embeddings = []
    chunk_texts = []
    # Iterate over chunks of tokens from the input text
    for chunk in chunked_tokens(text, chunk_length=max_tokens, encoding_name=encoding_name):
        # Generate embeddings for each chunk and append to the list
        chunk_embeddings.append(generate_embeddings(chunk, model=model))
        # Decode the chunk back to text and append to the list
        chunk_texts.append(tiktoken.get_encoding(encoding_name).decode(chunk))
    # Return the list of chunk embeddings and the corresponding text chunks
    return chunk_embeddings, chunk_texts
```

Next, we can define a helper function that will capture additional metadata about the documents. This is useful to use as a metadata filter for search queries, and capturing richer data for search. 

In this example, I'll choose from a list of categories to use later on in a metadata filter.


```python
## These are the categories I will be using for the categorization task. You can change these as needed based on your use case.
categories = ['authentication','models','techniques','tools','setup','billing_limits','other']

def categorize_text(text, categories):
    # Create a prompt for categorization
    messages = [
        {"role": "system", "content": f"""You are an expert in LLMs, and you will be given text that corresponds to an article in OpenAI's documentation.
         Categorize the document into one of these categories: {', '.join(categories)}. Only respond with the category name and nothing else."""},
        {"role": "user", "content": text}
    ]
    try:
        # Call the OpenAI API to categorize the text
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        # Extract the category from the response
        category = response.choices[0].message.content
        return category
    except Exception as e:
        print(f"Error categorizing text: {str(e)}")
        return None
```

Now, we can define some helper functions to process the .txt files in the oai_docs folder within the data folder. You can use this with your own data as well and supports both .txt and .pdf files.


```python
def extract_text_from_pdf(pdf_path):
    # Initialize the PDF reader
    reader = PdfReader(pdf_path)
    text = ""
    # Iterate through each page in the PDF and extract text
    for page in reader.pages:
        text += page.extract_text()
    return text

def process_file(file_path, idx, categories, embeddings_model):
    file_name = os.path.basename(file_path)
    print(f"Processing file {idx + 1}: {file_name}")
    
    # Read text content from .txt files
    if file_name.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    # Extract text content from .pdf files
    elif file_name.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    
    title = file_name
    # Generate embeddings for the title
    title_vectors, title_text = len_safe_get_embedding(title, embeddings_model)
    print(f"Generated title embeddings for {file_name}")
    
    # Generate embeddings for the content
    content_vectors, content_text = len_safe_get_embedding(text, embeddings_model)
    print(f"Generated content embeddings for {file_name}")
    
    category = categorize_text(' '.join(content_text), categories)
    print(f"Categorized {file_name} as {category}")
    
    # Prepare the data to be appended
    data = []
    for i, content_vector in enumerate(content_vectors):
        data.append({
            "id": f"{idx}_{i}",
            "vector_id": f"{idx}_{i}",
            "title": title_text[0],
            "text": content_text[i],
            "title_vector": json.dumps(title_vectors[0]),  # Assuming title is short and has only one chunk
            "content_vector": json.dumps(content_vector),
            "category": category
        })
        print(f"Appended data for chunk {i + 1}/{len(content_vectors)} of {file_name}")
    
    return data


```

We'll now use this helper function to process our OpenAI documentation. Feel free to update this to use your own data by changing the folder in `process_files` below.

Note that this will process the documents in chosen folder concurrently, so this should take <30 seconds if using txt files, and slightly longer if using PDFs.


```python
## Customize the location below if you are using different data besides the OpenAI documentation. Note that if you are using a different dataset, you will need to update the categories list as well.
folder_name = "../../../data/oai_docs"

files = [os.path.join(folder_name, f) for f in os.listdir(folder_name) if f.endswith('.txt') or f.endswith('.pdf')]
data = []

# Process each file concurrently
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_file, file_path, idx, categories, embeddings_model): idx for idx, file_path in enumerate(files)}
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            data.extend(result)
        except Exception as e:
            print(f"Error processing file: {str(e)}")

# Write the data to a CSV file
csv_file = os.path