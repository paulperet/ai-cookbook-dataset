# Using Azure AI Search as a Vector Database for OpenAI Embeddings

This guide provides step-by-step instructions for using Azure AI Search as a vector database with OpenAI embeddings. Azure AI Search is a cloud search service that provides infrastructure, APIs, and tools for building rich search experiences over private, heterogeneous content.

## Prerequisites

Before you begin, ensure you have the following:
- An [Azure AI Search Service](https://learn.microsoft.com/azure/search/)
- An [OpenAI Key](https://platform.openai.com/account/api-keys) or [Azure OpenAI credentials](https://learn.microsoft.com/azure/cognitive-services/openai/)

## Setup and Installation

First, install the required Python packages:

```bash
pip install wget
pip install azure-search-documents
pip install azure-identity
pip install openai
```

Now, import the necessary libraries:

```python
import json
import wget
import pandas as pd
import zipfile
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient, SearchIndexingBufferedSender
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery,
)
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchableField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
```

## Step 1: Configure OpenAI Settings

This section guides you through setting up authentication for Azure OpenAI. You can use either Azure Active Directory (AAD) or an API key. For detailed AAD setup instructions, refer to the [official documentation](https://learn.microsoft.com/azure/ai-services/openai/how-to/managed-identity).

Replace the placeholder values with your own credentials:

```python
endpoint: str = "YOUR_AZURE_OPENAI_ENDPOINT"
api_key: str = "YOUR_AZURE_OPENAI_KEY"
api_version: str = "2023-05-15"
deployment = "YOUR_AZURE_OPENAI_DEPLOYMENT_NAME"
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)

# Set this flag to True if you are using Azure Active Directory
use_aad_for_aoai = True

if use_aad_for_aoai:
    # Use Azure Active Directory (AAD) authentication
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_ad_token_provider=token_provider,
    )
else:
    # Use API key authentication
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )
```

## Step 2: Configure Azure AI Search Vector Store Settings

Next, set up the Azure AI Search client. You can find your service details in the Azure Portal or via the [Search Management SDK](https://learn.microsoft.com/rest/api/searchmanagement/).

```python
# Configuration
search_service_endpoint: str = "YOUR_AZURE_SEARCH_ENDPOINT"
search_service_api_key: str = "YOUR_AZURE_SEARCH_ADMIN_KEY"
index_name: str = "azure-ai-search-openai-cookbook-demo"

# Set this flag to True if you are using Azure Active Directory
use_aad_for_search = True

if use_aad_for_search:
    # Use Azure Active Directory (AAD) authentication
    credential = DefaultAzureCredential()
else:
    # Use API key authentication
    credential = AzureKeyCredential(search_service_api_key)

# Initialize the SearchClient with the selected authentication method
search_client = SearchClient(
    endpoint=search_service_endpoint, index_name=index_name, credential=credential
)
```

## Step 3: Load and Prepare the Data

You'll work with a sample dataset of Wikipedia articles that already contain pre-computed embeddings.

First, download the dataset:

```python
embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"

# The file is ~700 MB so this will take some time
wget.download(embeddings_url)
```

Extract the downloaded ZIP file:

```python
with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
    zip_ref.extractall("../../data")
```

Load the data into a pandas DataFrame and prepare the vector columns:

```python
article_df = pd.read_csv("../../data/vector_database_wikipedia_articles_embedded.csv")

# Read vectors from strings back into a list using json.loads
article_df["title_vector"] = article_df.title_vector.apply(json.loads)
article_df["content_vector"] = article_df.content_vector.apply(json.loads)
article_df["vector_id"] = article_df["vector_id"].apply(str)
article_df.head()
```

## Step 4: Create a Search Index

Now, create a search index that incorporates both vector search and semantic ranker capabilities. For more details, see the documentation on [creating a vector index](https://learn.microsoft.com/azure/search/vector-search-how-to-create-index).

```python
# Initialize the SearchIndexClient
index_client = SearchIndexClient(
    endpoint=search_service_endpoint, credential=credential
)

# Define the fields for the index
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String),
    SimpleField(name="vector_id", type=SearchFieldDataType.String, key=True),
    SimpleField(name="url", type=SearchFieldDataType.String),
    SearchableField(name="title", type=SearchFieldDataType.String),
    SearchableField(name="text", type=SearchFieldDataType.String),
    SearchField(
        name="title_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        vector_search_dimensions=1536,
        vector_search_profile_name="my-vector-config",
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        vector_search_dimensions=1536,
        vector_search_profile_name="my-vector-config",
    ),
]

# Configure the vector search configuration
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="my-hnsw",
            kind=VectorSearchAlgorithmKind.HNSW,
            parameters=HnswParameters(
                m=4,
                ef_construction=400,
                ef_search=500,
                metric=VectorSearchAlgorithmMetric.COSINE,
            ),
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="my-vector-config",
            algorithm_configuration_name="my-hnsw",
        )
    ],
)

# Configure the semantic search configuration
semantic_search = SemanticSearch(
    configurations=[
        SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                keywords_fields=[SemanticField(field_name="url")],
                content_fields=[SemanticField(field_name="text")],
            ),
        )
    ]
)

# Create the search index with the vector search and semantic search configurations
index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search,
    semantic_search=semantic_search,
)

# Create or update the index
result = index_client.create_or_update_index(index)
print(f"{result.name} created")
```

## Step 5: Upload Data to the Index

Upload the Wikipedia articles from the DataFrame to your Azure AI Search index. For more on data import strategies, see [Data Import in Azure AI Search](https://learn.microsoft.com/azure/search/search-what-is-data-import).

```python
from azure.core.exceptions import HttpResponseError

# Convert the 'id' and 'vector_id' columns to string so one of them can serve as our key field
article_df["id"] = article_df["id"].astype(str)
article_df["vector_id"] = article_df["vector_id"].astype(str)
# Convert the DataFrame to a list of dictionaries
documents = article_df.to_dict(orient="records")

# Create a SearchIndexingBufferedSender
batch_client = SearchIndexingBufferedSender(
    search_service_endpoint, index_name, credential
)

try:
    # Add upload actions for all documents in a single call
    batch_client.upload_documents(documents=documents)

    # Manually flush to send any remaining documents in the buffer
    batch_client.flush()
except HttpResponseError as e:
    print(f"An error occurred: {e}")
finally:
    # Clean up resources
    batch_client.close()

print(f"Uploaded {len(documents)} documents in total")
```

### Generating Embeddings (Optional)

If your dataset doesn't contain pre-computed embeddings, you can generate them using the OpenAI API. This same function will also be used later to generate query embeddings for vector searches.

```python
# Example function to generate document embedding
def generate_embeddings(text, model):
    # Generate embeddings for the provided text using the specified model
    embeddings_response = client.embeddings.create(model=model, input=text)
    # Extract the embedding data from the response
    embedding = embeddings_response.data[0].embedding
    return embedding


first_document_content = documents[0]["text"]
print(f"Content: {first_document_content[:100]}")

content_vector = generate_embeddings(first_document_content, deployment)
print("Content vector generated")
```

## Step 6: Perform a Vector Similarity Search

Now, let's perform a pure vector search. This query finds documents whose content vectors are most similar to the query's embedding.

```python
# Pure Vector Search
query = "modern art in Europe"

search_client = SearchClient(search_service_endpoint, index_name, credential)
vector_query = VectorizedQuery(vector=generate_embeddings(query, deployment), k_nearest_neighbors=3, fields="content_vector")

results = search_client.search(
    search_text=None,
    vector_queries=[vector_query],
    select=["title", "text", "url"]
)

for result in results:
    print(f"Title: {result['title']}")
    print(f"Score: {result['@search.score']}")
    print(f"URL: {result['url']}\n")
```

## Step 7: Perform a Hybrid Search

Hybrid search combines traditional keyword-based search with vector-based similarity search. This approach is useful for complex queries that benefit from understanding both exact keyword matches and semantic meaning.

```python
# Hybrid Search
query = "Famous battles in Scottish history"

search_client = SearchClient(search_service_endpoint, index_name, credential)
vector_query = VectorizedQuery(vector=generate_embeddings(query, deployment), k_nearest_neighbors=3, fields="content_vector")

results = search_client.search(
    search_text=query,
    vector_queries=[vector_query],
    select=["title", "text", "url"],
    top=3
)

for result in results:
    print(f"Title: {result['title']}")
    print(f"Score: {result['@search.score']}")
    print(f"URL: {result['url']}\n")
```

## Step 8: Perform a Hybrid Search with Semantic Reranking

The semantic ranker uses language understanding to rerank search results, improving relevance. It can also provide extractive captions, answers, and highlights. Learn more about [semantic search](https://learn.microsoft.com/azure/search/semantic-search-overview).

```python
# Semantic Hybrid Search
query = "What were the key technological advancements during the Industrial Revolution?"

search_client = SearchClient(search_service_endpoint, index_name, credential)
vector_query = VectorizedQuery(
    vector=generate_embeddings(query, deployment),
    k_nearest_neighbors=3,
    fields="content_vector",
)

results = search_client.search(
    search_text=query,
    vector_queries=[vector_query],
    select=["title", "text", "url"],
    query_type=QueryType.SEMANTIC,
    semantic_configuration_name="my-semantic-config",
    query_caption=QueryCaptionType.EXTRACTIVE,
    query_answer=QueryAnswerType.EXTRACTIVE,
    top=3,
)

semantic_answers = results.get_answers()
for answer in semantic_answers:
    if answer.highlights:
        print(f"Semantic Answer: {answer.highlights}")
    else:
        print(f"Semantic Answer: {answer.text}")
    print(f"Semantic Answer Score: {answer.score}\n")

for result in results:
    print(f"Title: {result['title']}")
    print(f"Reranker Score: {result['@search.reranker_score']}")
    print(f"URL: {result['url']}")
    captions = result["@search.captions"]
    if captions:
        caption = captions[0]
        if caption.highlights:
            print(f"Caption: {caption.highlights}\n")
        else:
            print(f"Caption: {caption.text}\n")
```

## Summary

You've successfully set up Azure AI Search as a vector database for OpenAI embeddings. You learned how to:
1. Configure authentication for Azure OpenAI and Azure AI Search
2. Load and prepare a dataset with pre-computed embeddings
3. Create a search index with vector and semantic search capabilities
4. Upload documents to the index
5. Perform various types of searches: pure vector, hybrid, and semantic hybrid search

This setup provides a powerful foundation for building intelligent search applications that understand both the literal and semantic meaning of your queries.