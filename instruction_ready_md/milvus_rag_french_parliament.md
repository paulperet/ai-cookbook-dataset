# Build a RAG Application with Milvus Lite, Mistral, and LlamaIndex

This guide walks you through building a Retrieval Augmented Generation (RAG) application to query data from the French Parliament. You will use Ollama with the Mistral model for local LLM operations, LlamaIndex for orchestration, and Milvus Lite for local vector storage.

## Prerequisites

Ensure you have the following set up before beginning:

1.  **Ollama:** Install and run Ollama on your machine. Follow the instructions at [https://ollama.com/](https://ollama.com/).
2.  **Mistral API Key (Optional):** If you choose to use the Mistral API instead of a local model, create an API key at [https://console.mistral.ai/api-keys/](https://console.mistral.ai/api-keys/).

## Step 1: Install Dependencies

Install the required Python libraries.

```bash
pip install -U pymilvus ollama llama-index-llms-ollama llama-index-vector-stores-milvus llama-index-readers-file llama-index-embeddings-mistralai llama-index-llms-mistralai python-dotenv
```

## Step 2: Download the Dataset

Download the sample XML file containing French Parliament discussion data.

```bash
wget 'https://raw.githubusercontent.com/mistralai/cookbook/main/third_party/Milvus/data/french_parliament_discussion.xml' -O './data/french_parliament_discussion.xml'
```

## Step 3: Configure the Embedding Model

You will use Mistral's embedding model to convert text into vector representations. First, load your API key from an environment variable.

```python
from dotenv import load_dotenv
import os

load_dotenv()
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
```

Next, initialize the Mistral embedding model.

```python
from llama_index.embeddings.mistralai import MistralAIEmbedding

model_name = "mistral-embed"
embed_model = MistralAIEmbedding(model_name=model_name, api_key=MISTRAL_API_KEY)
```

## Step 4: Set Up the Vector Store and LLM

This step configures the core components: the language model for generation and Milvus for storing and retrieving vector embeddings.

### Option A: Using Ollama (Local)

This option runs the Mistral model locally via Ollama. Ensure the Ollama service is running.

```python
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext, Settings

# Initialize the local LLM
llm = Ollama(model="mistral", request_timeout=120.0)

# Configure global settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 350
Settings.chunk_overlap = 20

# Initialize the Milvus vector store
vector_store = MilvusVectorStore(
    uri="milvus_mistral_rag.db",  # Local database file
    collection_name="mistral_french_parliament",
    dim=1024,  # Dimension of Mistral embeddings
    overwrite=True  # Recreates the collection if it exists
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
```

### Option B: Using the Mistral AI API

If you prefer not to run models locally, you can use Mistral's hosted API. It provides access to more powerful models and requires no local GPU.

```python
from llama_index.llms.mistralai import MistralAI

# Initialize the Mistral LLM via API
mistral_llm = MistralAI(api_key=MISTRAL_API_KEY, model="mistral-7B")

# Update the global LLM setting
Settings.llm = mistral_llm
# The embed_model and vector_store configuration from Option A remains the same.
```

## Step 5: Load and Index Your Documents

Now, load the XML data file and create a vector index. LlamaIndex will chunk the text, generate embeddings using the Mistral model, and store them in the Milvus collection you configured.

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Load documents from the XML file
docs = SimpleDirectoryReader(input_files=['data/french_parliament_discussion.xml']).load_data()

# Create a vector index from the documents
vector_index = VectorStoreIndex.from_documents(
    docs,
    storage_context=storage_context
)
```

## Step 6: Create a Retrieval Tool (Optional)

For more advanced use cases, such as building an agent, you can wrap the index's retriever into a tool.

```python
from llama_index.core.tools import RetrieverTool, ToolMetadata

milvus_retriever_tool = RetrieverTool(
    retriever=vector_index.as_retriever(similarity_top_k=3),
    metadata=ToolMetadata(
        name="FrenchParliamentRetriever",
        description='Retrieve relevant information from French Parliament discussion documents.'
    ),
)
```

## Step 7: Query Your RAG System

Finally, create a query engine from the index and ask questions. The system will retrieve the most relevant context from the stored documents and use the LLM to generate an answer.

```python
query_engine = vector_index.as_query_engine()
response = query_engine.query("What did the French parliament talk about the last time?")
print(response)
```

**Example Output:**
```
The conversation in the French parliament centered around a motion and a method for action regarding the seventh wave of some issue. There was criticism towards the chosen method being considered as "peu efficace" (ineffective) and "très disproportionnée" (highly disproportionate). Additionally, there were comments about the parliament not acting democratically and without consulting other parties when it comes to implementing certain measures like the passe sanitaire or vaccinal. The session ended with applause from some groups, specifically LFI-NUPES.
```

## Conclusion

You have successfully built a functional RAG application. You can now query the French Parliament dataset using natural language. To extend this project, consider:
*   Experimenting with different chunk sizes or overlap.
*   Integrating the retriever tool into a LlamaIndex agent for more complex reasoning.
*   Using a different dataset by changing the file path in the `SimpleDirectoryReader`.

If you enjoyed this tutorial, feel free to connect on [LinkedIn](https://www.linkedin.com/in/stephen-batifol/), check out the [Milvus project](https://github.com/milvus-io/milvus), and join the community on [Discord](https://discord.gg/FG6hMJStWu).