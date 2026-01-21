# Building a Knowledge Graph with LlamaIndex and Mistral AI

This guide walks you through creating and querying a Property Graph Index using LlamaIndex and Mistral AI. You'll learn how to extract a knowledge graph from unstructured text and use it for powerful semantic queries.

## Prerequisites

Ensure you have the necessary packages installed and your API key configured.

### 1. Install Required Libraries

```bash
pip install llama-index-core
pip install llama-index-llms-mistralai
pip install llama-index-embeddings-mistralai
```

### 2. Configure Your Environment

```python
import os
os.environ['MISTRAL_API_KEY'] = 'YOUR_MISTRAL_API_KEY'
```

### 3. Import Libraries and Initialize Models

```python
import nest_asyncio
nest_asyncio.apply()

from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI

llm = MistralAI(model='mistral-large-latest')
embed_model = MistralAIEmbedding()
```

## Step 1: Load Your Data

First, download and load the text document you'll use to build the knowledge graph.

```python
# Download the sample essay
!mkdir -p 'data/paul_graham/'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'
```

```python
from llama_index.core import SimpleDirectoryReader

# Load documents from the directory
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
```

## Step 2: Create the Property Graph Index

The `PropertyGraphIndex` automatically processes your documents through several stages:

1. **Parsing Nodes**: Documents are split into manageable text chunks (nodes)
2. **Extracting Knowledge Paths**: An LLM identifies relationships between entities, creating graph triples (subject → relationship → object)
3. **Inferring Implicit Paths**: Additional relationships are extracted from node metadata
4. **Generating Embeddings**: Both text nodes and graph nodes receive vector embeddings for semantic search

```python
from llama_index.core import PropertyGraphIndex

# Build the knowledge graph from your documents
index = PropertyGraphIndex.from_documents(
    documents,
    llm=llm,
    embed_model=embed_model,
    show_progress=True,
)
```

### Visualize Your Graph (Optional)

For debugging and exploration, you can save a visual representation of your graph:

```python
# Save graph visualization to HTML file
index.property_graph_store.save_networkx_graph(name="./kg.html")
```

## Step 3: Configure Global Settings

Set default models for consistent behavior across your application:

```python
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
```

## Step 4: Query Your Knowledge Graph

The Property Graph Index supports multiple querying strategies that work together:

- **Synonym/Keyword Expansion**: The LLM generates related terms from your query
- **Vector Retrieval**: Semantic search finds relevant nodes using embeddings

### Basic Retrieval

Retrieve graph paths related to your query:

```python
# Create a retriever that returns only graph paths (no source text)
retriever = index.as_retriever(include_text=False)

# Query the graph
nodes = retriever.retrieve("What happened at Interleaf and Viaweb?")

# Display the retrieved paths
for node in nodes:
    print(node.text)
```

This returns knowledge graph triples like:
```
Viaweb -> Launch date -> January 1996
Viaweb -> Growth rate -> 7x a year
I -> Got a job at -> Interleaf
Interleaf -> Made software for -> Creating documents
```

### Full Query Engine

For comprehensive answers that combine graph knowledge with source text:

```python
from IPython.display import Markdown, display

# Create a query engine that includes source text
query_engine = index.as_query_engine(include_text=True)

# Get a synthesized answer
response = query_engine.query("What happened at Interleaf and Viaweb?")

display(Markdown(f"{response.response}"))
```

The response synthesizes information from multiple graph paths and source documents into a coherent answer.

## Step 5: Persist and Load Your Index

Save your graph and vector stores to disk for later use:

```python
# Save the entire index to disk
index.storage_context.persist(persist_dir="./storage")

# Load it back later
from llama_index.core import StorageContext, load_index_from_storage

index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="./storage")
)

# Continue querying as before
query_engine = index.as_query_engine(include_text=True)
response = query_engine.query("What happened at Interleaf and Viaweb?")
```

## Step 6: Customize Your Vector Store (Advanced)

By default, LlamaIndex uses simple in-memory stores. For production use, you can integrate specialized vector databases like ChromaDB.

### Install ChromaDB Integration

```bash
pip install llama-index-vector-stores-chroma
```

### Build Index with ChromaDB

```python
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Set up ChromaDB client and collection
client = chromadb.PersistentClient("./chroma_db")
collection = client.get_or_create_collection("my_graph_vector_db")

# Build index with custom vector store
index = PropertyGraphIndex.from_documents(
    documents,
    llm=llm,
    embed_model=embed_model,
    property_graph_store=SimplePropertyGraphStore(),
    vector_store=ChromaVectorStore(chroma_collection=collection),
    show_progress=True,
)

# Persist the configuration
index.storage_context.persist(persist_dir="./storage")
```

### Load and Query with Custom Storage

```python
# Load the index with both graph and vector stores
index = PropertyGraphIndex.from_existing(
    SimplePropertyGraphStore.from_persist_dir("./storage"),
    vector_store=ChromaVectorStore(chroma_collection=collection),
    llm=llm,
)

# Query as usual
query_engine = index.as_query_engine(include_text=True)
response = query_engine.query("why did author do at YC?")
```

## Key Takeaways

1. **Property Graph Index** automatically extracts entities and relationships from unstructured text
2. **Dual Retrieval Strategy** combines semantic search with graph traversal for comprehensive answers
3. **Flexible Storage** supports both simple in-memory stores and production-ready databases
4. **Persistent Storage** allows saving and reloading your knowledge graph

The Property Graph Index transforms unstructured documents into queryable knowledge, enabling you to ask complex questions about relationships and events in your text corpus.