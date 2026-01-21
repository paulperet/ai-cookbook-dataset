# Building a RAG Pipeline with NVIDIA NIM, LlamaIndex, and Azure AI Search

This guide demonstrates how to construct a Retrieval-Augmented Generation (RAG) pipeline by integrating NVIDIA's high-performance AI models with Azure AI Search as a vector store, using LlamaIndex as the orchestration layer.

## Benefits
- **Scalability**: Leverage NVIDIA's large language models and Azure AI Search for scalable, efficient retrieval.
- **Cost Efficiency**: Optimize search and retrieval with efficient vector storage and hybrid search techniques.
- **High Performance**: Combine powerful LLMs with vectorized search for faster, more accurate responses.
- **Quality**: Ground LLM responses with relevant retrieved documents to maintain high answer quality.

## Prerequisites
- Python 3.9 or higher
- An [Azure AI Search Service](https://learn.microsoft.com/azure/search/)
- An NVIDIA API Key for accessing NVIDIA's LLMs and Embeddings via the NVIDIA NIM microservices

## Setup and Installation

First, install the required Python packages.

```bash
pip install azure-search-documents==11.5.1
pip install --upgrade llama-index
pip install --upgrade llama-index-core
pip install --upgrade llama-index-readers-file
pip install --upgrade llama-index-llms-nvidia
pip install --upgrade llama-index-embeddings-nvidia
pip install --upgrade llama-index-postprocessor-nvidia-rerank
pip install --upgrade llama-index-vector-stores-azureaisearch
pip install python-dotenv
```

## Step 1: Configure Your API Keys

You need to set your `NVIDIA_API_KEY`. To obtain one:
1. Create a free account on the [NVIDIA AI Foundation Models platform](https://build.nvidia.com/explore/discover).
2. Select a model, navigate to the Python tab under Input, and click **Get API Key** > **Generate Key**.
3. Copy and save the generated key (it starts with `nvapi-`).

Now, load your environment variables and securely prompt for the key if it's not already set.

```python
import getpass
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Securely set the NVIDIA API key if not present
if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
    assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key
```

## Step 2: Initialize the LLM and Embedding Model

We'll configure the global `Settings` in LlamaIndex to use NVIDIA's models.

### 2.1 Initialize the LLM
We'll use the `microsoft/phi-3.5-moe-instruct` model from NVIDIA's API catalog.

```python
from llama_index.core import Settings
from llama_index.llms.nvidia import NVIDIA

# Set the global LLM to a Phi-3.5-MOE model
Settings.llm = NVIDIA(model="microsoft/phi-3.5-moe-instruct", api_key=os.getenv("NVIDIA_API_KEY"))
```

### 2.2 Initialize the Embedding Model
We'll use the `nvidia/nv-embedqa-e5-v5` model for generating text embeddings.

```python
from llama_index.embeddings.nvidia import NVIDIAEmbedding

# Set the global embedding model
Settings.embed_model = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5", api_key=os.getenv("NVIDIA_API_KEY"))
```

## Step 3: Set Up the Azure AI Search Vector Store

Now, configure the connection to your Azure AI Search service and create a vector store index.

```python
import os
import getpass
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore, IndexManagement

# Collect Azure AI Search credentials
search_service_api_key = os.getenv('AZURE_SEARCH_ADMIN_KEY') or getpass.getpass('Enter your Azure Search API key: ')
search_service_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT') or getpass.getpass('Enter your Azure Search service endpoint: ')
search_service_api_version = "2024-07-01"

credential = AzureKeyCredential(search_service_api_key)

# Define your index name
index_name = "llamaindex-nvidia-azureaisearch-demo"

# Create an index client
index_client = SearchIndexClient(
    endpoint=search_service_endpoint,
    credential=credential,
)

# Instantiate the vector store. It will create the index if it doesn't exist.
vector_store = AzureAISearchVectorStore(
    search_or_index_client=index_client,
    index_name=index_name,
    index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
    id_field_key="id",
    chunk_field_key="chunk",
    embedding_field_key="embedding",
    embedding_dimensionality=1024, # Dimensionality for the nv-embedqa-e5-v5 model
    metadata_string_field_key="metadata",
    doc_id_field_key="doc_id",
    language_analyzer="en.lucene",
    vector_algorithm_type="exhaustiveKnn",
)
```

## Step 4: Load, Chunk, and Index Documents

We'll load a sample text document, split it into chunks, and upload it to our Azure AI Search index.

```python
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.text_splitter import TokenTextSplitter

# Configure a text splitter. The nv-embedqa-e5-v5 model has a limit of 512 tokens.
text_splitter = TokenTextSplitter(separator=" ", chunk_size=500, chunk_overlap=10)

# Load a sample document (ensure you have a 'data/txt' directory with the file)
documents = SimpleDirectoryReader(
    input_files=["data/txt/state_of_the_union.txt"]
).load_data()

# Create a storage context linked to our Azure vector store
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create the index, applying the text splitter to the documents
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[text_splitter],
    storage_context=storage_context,
)
```

## Step 5: Query Your Data with Different Retrieval Strategies

With the data indexed, you can create query engines to ask questions. LlamaIndex and Azure AI Search support multiple retrieval modes.

### 5.1 Basic Vector Search Query

First, let's perform a simple query using pure vector search.

```python
from IPython.display import Markdown, display

# Create a standard query engine (defaults to vector search)
query_engine = index.as_query_engine()
response = query_engine.query("Who did the speaker mention as being present in the chamber?")

display(Markdown(f"{response}"))
```

**Expected Output:**
> The speaker mentioned the Ukrainian Ambassador to the United States, along with other members of Congress, the Cabinet, and various officials such as the Vice President, the First Lady, and the Second Gentleman, as being present in the chamber.

### 5.2 Hybrid Search Query

Hybrid search combines vector (semantic) search with traditional keyword search. This can improve recall for certain queries.

```python
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.core.schema import MetadataMode

# Initialize a retriever configured for hybrid search
hybrid_retriever = index.as_retriever(vector_store_query_mode=VectorStoreQueryMode.HYBRID)
hybrid_query_engine = RetrieverQueryEngine(retriever=hybrid_retriever)

# Execute a query
query = "What were the exact economic consequences mentioned in relation to Russia's stock market?"
response = hybrid_query_engine.query(query)

display(Markdown(f"{response}"))
print("\n")

# Inspect the source nodes (chunks) used to generate the answer
print("Source Nodes:")
for node in response.source_nodes:
    print(node.get_content(metadata_mode=MetadataMode.LLM))
```

**Analysis:**
The LLM accurately summarizes the information from the retrieved source nodes, stating that the Russian stock market lost 40% of its value and trading was suspended. This demonstrates how hybrid search can effectively ground the LLM's response in the source material.

### 5.3 When Hybrid Search Lacks Context

Let's examine a case where the indexed documents don't contain the specific information requested.

```python
query = "What was the precise date when Russia invaded Ukraine?"
response = hybrid_query_engine.query(query)

display(Markdown(f"{response}"))
print("\n")

print("Source Nodes:")
for node in response.source_nodes:
    print(node.get_content(metadata_mode=MetadataMode.LLM))
```

**Output Analysis:**
The LLM correctly identifies that the precise date is not present in the source text. It provides a cautious response, suggesting the user consult external historical records. The source nodes confirm this, containing discussions about the invasion's consequences but not its exact date.

### 5.4 Hybrid Search with Semantic Reranking

Azure AI Search offers a built-in semantic reranker that can improve result relevance. Let's use the `SEMANTIC_HYBRID` mode.

```python
# Initialize a retriever with semantic hybrid search (includes reranking)
semantic_reranker_retriever = index.as_retriever(vector_store_query_mode=VectorStoreQueryMode.SEMANTIC_HYBRID)
semantic_reranker_query_engine = RetrieverQueryEngine(retriever=semantic_reranker_retriever)

# Execute the same query
query = "What was the precise date when Russia invaded Ukraine?"
response = semantic_reranker_query_engine.query(query)

display(Markdown(f"{response}"))
print("\n")

print("Source Nodes:")
for node in response.source_nodes:
    print(node.get_content(metadata_mode=MetadataMode.LLM))
```

**Output Analysis:**
With semantic reranking, the LLM's response improves. It now infers from the context that the invasion occurred "six days before the speech was given." While still not a calendar date, this provides valuable temporal context that was not highlighted in the standard hybrid search results. The reranker successfully surfaced a source node containing this relative timing information.

## Conclusion and Next Steps

You have successfully built a RAG pipeline using:
- **NVIDIA NIM** for state-of-the-art LLM (`Phi-3.5-MOE`) and embedding (`nv-embedqa-e5-v5`) models.
- **Azure AI Search** as a scalable, production-ready vector database.
- **LlamaIndex** to orchestrate document loading, chunking, indexing, and querying.

You experimented with different Azure AI Search retrieval modes:
- **Vector Search**: Pure semantic similarity search.
- **Hybrid Search**: Combines vector and keyword search for broader recall.
- **Semantic Hybrid Search**: Enhances hybrid search with a built-in reranker for improved relevance.

### Using Self-Hosted NVIDIA NIM Microservices

The NVIDIA connectors used in this guide can also point to self-hosted NIM microservices. Simply provide the `base_url` parameter.

```python
# Example for a self-hosted Llama 3 model
from llama_index.llms.nvidia import NVIDIA

self_hosted_llm = NVIDIA(
    model="meta/llama3-8b-instruct",
    base_url="http://your-nim-host-address:8000/v1",
    api_key="your-api-key-if-required" # API key may not be needed for local deployments
)
```

This pipeline provides a robust foundation for building production-grade AI search applications that deliver accurate, context-grounded answers.