# Azure AI Search with NVIDIA NIM and LlamaIndex Integration

In this notebook, we'll demonstrate how to leverage NVIDIA's AI models and LlamaIndex to create a powerful Retrieval-Augmented Generation (RAG) pipeline. We'll use NVIDIA's LLMs and embeddings, integrate them with Azure AI Search as the vector store, and perform RAG to enhance search quality and efficiency.

## Benefits
- **Scalability**: Leverage NVIDIA's large language models and Azure AI Search for scalable and efficient retrieval.
- **Cost Efficiency**: Optimize search and retrieval with efficient vector storage and hybrid search techniques.
- **High Performance**: Combine powerful LLMs with vectorized search for faster and more accurate responses.
- **Quality**: Maintain high search quality by grounding LLM responses with relevant retrieved documents.

## Prerequisites
- 🐍 Python 3.9 or higher
- 🔗 [Azure AI Search Service](https://learn.microsoft.com/azure/search/)
- 🔗 NVIDIA API Key for access to NVIDIA's LLMs and Embeddings via the NVIDIA NIM microservices

## Features Covered
- ✅ NVIDIA LLM Integration (we'll use [Phi-3.5-MOE](https://build.nvidia.com/microsoft/phi-3_5-moe))
- ✅ NVIDIA Embeddings (we'll use [nv-embedqa-e5-v5](https://build.nvidia.com/nvidia/nv-embedqa-e5-v5))
- ✅ Azure AI Search Advanced Retreival Modes
- ✅ Document Indexing with LlamaIndex
- ✅ RAG using Azure AI Search and LlamaIndex with NVIDIA LLMs

Let's get started!



```python
!pip install azure-search-documents==11.5.1
!pip install --upgrade llama-index
!pip install --upgrade llama-index-core
!pip install --upgrade llama-index-readers-file
!pip install --upgrade llama-index-llms-nvidia
!pip install --upgrade llama-index-embeddings-nvidia
!pip install --upgrade llama-index-postprocessor-nvidia-rerank
!pip install --upgrade llama-index-vector-stores-azureaisearch
!pip install python-dotenv
```

## Installation and Requirements
Create a Python environment using Python version >3.10.

## Getting Started!

To get started, you need a `NVIDIA_API_KEY` to use NVIDIA AI Foundation models:
1) Create a free account with [NVIDIA](https://build.nvidia.com/explore/discover).
2) Click on your model of choice.
3) Under Input, select the Python tab, and click **Get API Key** and then click **Generate Key**.
4) Copy and save the generated key as NVIDIA_API_KEY. From there, you should have access to the endpoints.



```python
import getpass
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
    assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key

```

## RAG Example using LLM and Embedding
### 1) Initialize the LLM
`llama-index-llms-nvidia`, also known as NVIDIA's LLM connector, allows you to connect to and generate from compatible models available on the NVIDIA API catalog. See here for a list of chat completion models: https://build.nvidia.com/search?term=Text-to-Text

Here we will use **mixtral-8x7b-instruct-v0.1**



```python
from llama_index.core import Settings
from llama_index.llms.nvidia import NVIDIA

# Here we are using mixtral-8x7b-instruct-v0.1 model from API Catalog
Settings.llm = NVIDIA(model="microsoft/phi-3.5-moe-instruct", api_key=os.getenv("NVIDIA_API_KEY"))
```

### 2) Initialize the Embedding
`llama-index-embeddings-nvidia`, also known as NVIDIA's Embeddings connector, allows you to connect to and generate from compatible models available on the NVIDIA API catalog. We selected `nvidia/nv-embedqa-e5-v5` as the embedding model. See here for a list of text embedding models: https://build.nvidia.com/nim?filters=usecase%3Ausecase_text_to_embedding%2Cusecase%3Ausecase_image_to_embedding



```python
from llama_index.embeddings.nvidia import NVIDIAEmbedding

Settings.embed_model = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5", api_key=os.getenv("NVIDIA_API_KEY"))
```

### 3) Create an Azure AI Search Vector Store


```python
import logging
import sys
import os
import getpass
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from IPython.display import Markdown, display
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore, IndexManagement


search_service_api_key = os.getenv('AZURE_SEARCH_ADMIN_KEY') or getpass.getpass('Enter your Azure Search API key: ')
search_service_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT') or getpass.getpass('Enter your Azure Search service endpoint: ')
search_service_api_version = "2024-07-01"
credential = AzureKeyCredential(search_service_api_key)

# Index name to use
index_name = "llamaindex-nvidia-azureaisearch-demo"

# Use index client to demonstrate creating an index
index_client = SearchIndexClient(
    endpoint=search_service_endpoint,
    credential=credential,
)

# Use search client to demonstrate using existing index
search_client = SearchClient(
    endpoint=search_service_endpoint,
    index_name=index_name,
    credential=credential,
)
```


```python
vector_store = AzureAISearchVectorStore(
    search_or_index_client=index_client,
    index_name=index_name,
    index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
    id_field_key="id",
    chunk_field_key="chunk",
    embedding_field_key="embedding",
    embedding_dimensionality=1024, # dimensionality for nv-embedqa-e5-v5 model
    metadata_string_field_key="metadata",
    doc_id_field_key="doc_id",
    language_analyzer="en.lucene",
    vector_algorithm_type="exhaustiveKnn",
    # compression_type="binary" # Option to use "scalar" or "binary". NOTE: compression is only supported for HNSW
)
```

### 4) Load, Chunk, and Upload Documents


```python
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.text_splitter import TokenTextSplitter

# Configure text splitter (nv-embedqa-e5-v5 model has a limit of 512 tokens per input size)
text_splitter = TokenTextSplitter(separator=" ", chunk_size=500, chunk_overlap=10)

# Load documents
documents = SimpleDirectoryReader(
    input_files=["data/txt/state_of_the_union.txt"]
).load_data()
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index with text splitter
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[text_splitter],
    storage_context=storage_context,
)
```

### 5) Create a Query Engine to ask questions over your data

Here is a query using pure vector search in Azure AI Search and grounding the response to our LLM (Phi-3.5-MOE)



```python
query_engine = index.as_query_engine()
response = query_engine.query("Who did the speaker mention as being present in the chamber?")
display(Markdown(f"{response}"))
```


 The speaker mentioned the Ukrainian Ambassador to the United States, along with other members of Congress, the Cabinet, and various officials such as the Vice President, the First Lady, and the Second Gentleman, as being present in the chamber.


Here is a query using hybrid search in Azure AI Search.


```python
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from IPython.display import Markdown, display
from llama_index.core.schema import MetadataMode

# Initialize hybrid retriever and query engine
hybrid_retriever = index.as_retriever(vector_store_query_mode=VectorStoreQueryMode.HYBRID)
hybrid_query_engine = RetrieverQueryEngine(retriever=hybrid_retriever)

# Query execution
query = "What were the exact economic consequences mentioned in relation to Russia's stock market?"
response = hybrid_query_engine.query(query)

# Display the response
display(Markdown(f"{response}"))
print("\n")

# Print the source nodes
print("Source Nodes:")
for node in response.source_nodes:
    print(node.get_content(metadata_mode=MetadataMode.LLM))
```


 The Russian stock market experienced a significant drop, losing 40% of its value. Additionally, trading had to be suspended due to the ongoing situation.


    
    
    Source Nodes:
    [First Entry, ..., Last Entry]

#### Vector Search Analysis
The LLM response accurately captures the key economic consequences mentioned in the source text regarding Russia's stock market. Specifically, it states that the Russian stock market experienced a significant drop, losing 40% of its value, and that trading was suspended due to the ongoing situation. This response aligns well with the information provided in the source, indicating that the LLM correctly identified and summarized the relevant details regarding the stock market impact as a result of Russia's actions and the sanctions imposed.

#### Source Nodes Commentary
The source nodes provide a detailed account of the economic consequences that Russia faced due to international sanctions. The text highlights that the Russian stock market lost 40% of its value, and trading was suspended. Additionally, it mentions other economic repercussions, such as the devaluation of the Ruble and the broader isolation of Russia's economy. The LLM response effectively distilled the critical points from these nodes, focusing on the stock market impact as requested by the query.


Now, let's take a look at a query where Hybrid Search doesn't give a well-grounded answer:



```python
# Query execution
query = "What was the precise date when Russia invaded Ukraine?"
response = hybrid_query_engine.query(query)

# Display the response
display(Markdown(f"{response}"))
print("\n")

# Print the source nodes
print("Source Nodes:")
for node in response.source_nodes:
    print(node.get_content(metadata_mode=MetadataMode.LLM))

```


 The provided context does not specify the exact date of Russia's invasion of Ukraine. However, it does mention that the events discussed are happening in the current era and that the actions taken are in response to Putin's aggression. For the precise date, one would need to refer to external sources or historical records.


    
    
    Source Nodes:
    [First Entry, ..., Last Entry]

### Hybrid Search: LLM Response Analysis
The LLM response in the Hybrid Search example indicates that the provided context does not specify the exact date of Russia's invasion of Ukraine. This response suggests that the LLM is leveraging the information available in the source documents but acknowledges the absence of precise details in the text.

The response is accurate in identifying that the context mentions events related to Russia's aggression but does not pinpoint the specific invasion date. This showcases the LLM's ability to comprehend the information provided while recognizing gaps in the content. The LLM effectively prompts the user to seek external sources or historical records for the exact date, displaying a level of cautiousness when information is incomplete.

### Source Nodes Analysis
The source nodes in the Hybrid Search example contain excerpts from a speech discussing the U.S. response to Russia's actions in Ukraine. These nodes emphasize the broader geopolitical impact and the steps taken by the U.S. and its allies in response to the invasion, but they do not mention the specific invasion date. This aligns with the LLM response, which correctly identifies that the context lacks the precise date information.



```python
# Initialize hybrid retriever and query engine
semantic_reranker_retriever = index.as_retriever(vector_store_query_mode=VectorStoreQueryMode.SEMANTIC_HYBRID)
semantic_reranker_query_engine = RetrieverQueryEngine(retriever=semantic_reranker_retriever)

# Query execution
query = "What was the precise date when Russia invaded Ukraine?"
response = semantic_reranker_query_engine.query(query)

# Display the response
display(Markdown(f"{response}"))
print("\n")

# Print the source nodes
print("Source Nodes:")
for node in response.source_nodes:
    print(node.get_content(metadata_mode=MetadataMode.LLM))

```


 The provided context does not specify the exact date of Russia's invasion of Ukraine. However, it mentions that the event occurred six days before the speech was given. To determine the precise date, one would need to know the date of the speech.


    
    
    Source Nodes:
    [First Entry, ..., Last Entry]

### Hybrid w/Reranking: LLM Response Analysis
In the Hybrid w/Reranking example, the LLM response provides additional context by noting that the event occurred six days before the speech was given. This indicates that the LLM is able to infer the invasion date based on the timing of the speech, even though it still requires knowing the exact date of the speech for precision.

This response demonstrates an improved ability to use context clues to provide a more informative answer. It highlights the advantage of reranking, where the LLM can access and prioritize more relevant information to give a closer approximation of the desired detail (i.e., the invasion date).

### Source Nodes Analysis
The source nodes in this example include references to the timing of Russia's invasion, specifically mentioning that it occurred six days before the speech. While the exact date is still not explicitly stated, the nodes provide temporal context that allows the LLM to give a more nuanced response. The inclusion of this detail showcases how reranking can improve the LLM's ability to extract and infer information from the provided context, resulting in a more accurate and informative response.


**Note:**
In this notebook, we used NVIDIA NIM microservices from the NVIDIA API Catalog.
The above APIs, `NVIDIA (llms)`, `NVIDIAEmbedding`, and [Azure AI Search Semantic Hybrid Retrieval (built-in reranking)](https://learn.microsoft.com/azure/search/semantic-search-overview). Note, that the above APIs can also support self-hosted microservices. 

**Example:**
```python
NVIDIA(model="meta/llama3-8b-instruct", base_url="http://your-nim-host-address:8000/v1")

```