# Build a RAG application with Milvus Lite, Mistral and Llama-index

In this notebook, we are showing how you can build a Retrieval Augmented Generation (RAG) application to interact with data from the French Parliament. It uses Ollama with Mistral for LLM operations, Llama-index for orchestration, and [Milvus](https://milvus.io/) for vector storage.

## Install Ollama

Make sure to have Ollama installed and Running on your laptop --> https://ollama.com/

### Install the different dependencies 

```python
!pip install -U pymilvus ollama llama-index-llms-ollama llama-index-vector-stores-milvus llama-index-readers-file llama-index-embeddings-mistralai llama-index-llms-mistralai
```

### Download data

Note: Run this cell only if you haven't cloned the repository.

```python
!wget 'https://raw.githubusercontent.com/mistralai/cookbook/main/third_party/Milvus/data/french_parliament_discussion.xml' -O './data/french_parliament_discussion.xml'
```

### Use Mistral Embedding

Make sure to create an [API Key](https://console.mistral.ai/api-keys/) on Mistral's platform and load it as an environment variable.

On this tutorial, we are loading the environment variable stored in our `.env` file.

```python
from dotenv import load_dotenv
import os
load_dotenv()

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
```

```python
from llama_index.embeddings.mistralai import MistralAIEmbedding

model_name = "mistral-embed"
embed_model = MistralAIEmbedding(model_name=model_name, api_key=MISTRAL_API_KEY)
```

### Prepare out data to be stored in Milvus

This code makes it possible to process text embeddings using Mistral Embed & Mistral-7B and store those in Milvus.

**!!Make sure to have Ollama running on your laptop!!**

* Initialises Mistral-7B model using Ollama
* Service Context: Configures a service context with Mistral and the embedding model defined above
* Vector Store: Sets up a collection in Milvus to store text embeddings, specifying the database file, collection name, vector dimensions
* Storage Context: Configures a storage context with the Milvus vector store

This makes it possible to have efficient storage and retrieval of vector embeddings for text data.

```python
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.milvus import MilvusVectorStore

from llama_index.core import StorageContext, Settings

llm = Ollama(model="mistral", request_timeout=120.0)

Settings.llm = Ollama(model="mistral", request_timeout=120.0)
Settings.embed_model = embed_model
Settings.chunk_size = 350
Settings.chunk_overlap = 20

vector_store = MilvusVectorStore(
    uri="milvus_mistral_rag.db",
    collection_name="mistral_french_parliament",
    dim=1024, 
    overwrite=True  # drop table if exist and then create
    
    )
storage_context = StorageContext.from_defaults(vector_store=vector_store)
```

### Using Mistral AI API

If you prefer not to run models locally or need more powerful models, you can use Mistral's API instead of Ollama. The API offers:
- Access to more powerful models like `mistral-large` and `mistral-small`
- No local GPU/CPU requirements
- Consistent performance and reliability
- Production-ready deployment

Make sure to create an [API Key](https://console.mistral.ai/api-keys/) on Mistral's platform first.
```python
from llama_index.llms.mistralai import MistralAI

# Initialize Mistral LLM
mistral_llm = MistralAI(api_key=MISTRAL_API_KEY, model="mistral-7B")

# Configure settings for Mistral
Settings.llm = mistral_llm
```

The rest of the setup using Milvus would stay the same.

### Process and load the Data 

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

docs = SimpleDirectoryReader(input_files=['data/french_parliament_discussion.xml']).load_data()
vector_index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
```

```python
from llama_index.core.tools import RetrieverTool, ToolMetadata

milvus_tool_openai = RetrieverTool(
    retriever=vector_index.as_retriever(similarity_top_k=3),  # retrieve top_k results
    metadata=ToolMetadata(
        name="CustomRetriever",
        description='Retrieve relevant information from provided documents.'
    ),
)
```

### Finally, ask questions to our RAG system

```python
query_engine = vector_index.as_query_engine()
response = query_engine.query("What did the French parliament talk about the last time?")
print(response)
```

 The conversation in the French parliament centered around a motion and a method for action regarding the seventh wave of some issue. There was criticism towards the chosen method being considered as "peu efficace" (ineffective) and "très disproportionnée" (highly disproportionate). Additionally, there were comments about the parliament not acting democratically and without consulting other parties when it comes to implementing certain measures like the passe sanitaire or vaccinal. The session ended with applause from some groups, specifically LFI-NUPES.

---

#### If you like this tutorial, feel free to reach out on [LinkedIn](https://www.linkedin.com/in/stephen-batifol/), check out [Milvus](https://github.com/milvus-io/milvus) and join our [Discord](https://discord.gg/FG6hMJStWu).