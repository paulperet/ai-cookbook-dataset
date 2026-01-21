# Building a Property Graph with Neo4j and LlamaIndex

This guide walks you through constructing a knowledge graph (Property Graph) using Neo4j and LlamaIndex. You'll learn how to extract structured relationships from unstructured text, store them in a Neo4j graph database, and perform intelligent retrieval and querying.

## Prerequisites

Ensure you have the following installed and set up before beginning:

*   **Python 3.8+**
*   **Docker** (for running Neo4j locally)
*   A **Mistral AI API key**

## Step 1: Install Required Libraries

First, install the necessary Python packages.

```bash
pip install llama-index-core
pip install llama-index-graph-stores-neo4j
pip install llama-index-llms-mistralai
pip install llama-index-embeddings-mistralai
```

## Step 2: Set Up a Local Neo4j Instance with Docker

We'll use Docker to run a Neo4j database locally with the APOC (Awesome Procedures on Cypher) plugin enabled for extended functionality.

1.  Run the following command to start a Neo4j container:

    ```bash
    docker run \
        -p 7474:7474 -p 7687:7687 \
        -v $PWD/data:/data -v $PWD/plugins:/plugins \
        --name neo4j-apoc \
        -e NEO4J_apoc_export_file_enabled=true \
        -e NEO4J_apoc_import_file_enabled=true \
        -e NEO4J_apoc_import_file_use__neo4j__config=true \
        -e NEO4JLABS_PLUGINS=\[\"apoc\"\] \
        neo4j:latest
    ```

2.  Once the container is running, you can access the Neo4j Browser at `http://localhost:7474`.
3.  Log in with the default credentials:
    *   **Username:** `neo4j`
    *   **Password:** `neo4j`
4.  You will be prompted to change the password on first login. For this tutorial, set the new password to `llamaindex`.

## Step 3: Configure Your Python Environment

Now, let's set up the Python environment, configure the LLM, and prepare for asynchronous operations.

```python
import nest_asyncio
import os

# Apply nest_asyncio to allow async operations in environments like Jupyter
nest_asyncio.apply()

# Set your Mistral AI API key as an environment variable
os.environ['MISTRAL_API_KEY'] = 'YOUR_MISTRAL_API_KEY_HERE'

# Import and initialize the LLM and embedding model
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI

llm = MistralAI(model='mistral-large-latest')
embed_model = MistralAIEmbedding()
```

## Step 4: Download and Load Your Data

We'll use Paul Graham's essay as our sample text data.

```python
# Download the sample essay
!mkdir -p 'data/paul_graham/'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# Load the document using LlamaIndex's reader
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
```

## Step 5: Connect to the Neo4j Graph Store

Create a connection to your running Neo4j database. This `graph_store` object will be the interface through which LlamaIndex writes and reads graph data.

```python
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="llamaindex",  # The password you set in Step 2
    url="bolt://localhost:7687",  # Default Neo4j Bolt protocol port
)
```

## Step 6: Construct the Property Graph Index

This is the core step where the magic happens. The `PropertyGraphIndex` will process your documents, use an LLM to extract entities and relationships (Knowledge Graph triples), and store them in Neo4j.

```python
from llama_index.core import PropertyGraphIndex
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=embed_model,
    kg_extractors=[
        SimpleLLMPathExtractor(
            llm=llm  # The LLM used to parse text and extract graph relationships
        )
    ],
    property_graph_store=graph_store,  # Where to persist the graph
    show_progress=True,
)
```

## Step 7: Configure Global Settings (Optional)

Setting global `Settings` can simplify future object creation by providing default LLM and embedding models.

```python
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
```

## Step 8: Configure Specialized Retrievers

LlamaIndex provides specialized retrievers for graph databases. Here we set up two:
1.  **`LLMSynonymRetriever`**: Uses the LLM to expand your query with synonyms before searching the graph.
2.  **`VectorContextRetriever`**: Performs a vector search on node text to find the most semantically relevant graph nodes.

```python
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
)

llm_synonym = LLMSynonymRetriever(
    index.property_graph_store,
    llm=llm,
    include_text=False,
)
vector_context = VectorContextRetriever(
    index.property_graph_store,
    embed_model=embed_model,
    include_text=False,
)
```

## Step 9: Query Your Knowledge Graph

### 9.1 Using a Retriever for Raw Nodes

A retriever fetches relevant nodes (and their relationships) from the graph based on your query. Here we combine both synonym and vector retrieval strategies.

```python
# Create a composite retriever that uses both strategies
retriever = index.as_retriever(
    sub_retrievers=[
        llm_synonym,
        vector_context,
    ],
)

# Retrieve nodes for a specific query
nodes = retriever.retrieve("What did the author do at Viaweb?")

# Inspect the text of the retrieved nodes
for node in nodes:
    print(node.text)
```

### 9.2 Using a Query Engine for Summarized Answers

A `QueryEngine` goes a step further. It retrieves relevant context from the graph and uses the LLM to synthesize a coherent, natural language answer.

```python
from IPython.display import Markdown, display

# Create a query engine that includes source text in the context
query_engine = index.as_query_engine(include_text=True)

# Ask a question and get a generated response
response = query_engine.query("What did the author do at Viaweb?")

# Display the formatted response
display(Markdown(f"{response.response}"))
```

**Example Output:**
> The author, along with Robert Morris, started Viaweb, a company that aimed to build online stores. The author's role involved writing software to generate websites for galleries initially, and later, developing a new site generator for online stores using Lisp. The author also had the innovative idea of running the software on the server and letting users control it by clicking on links, eliminating the need for any client software or command line interaction on the server. This led to the creation of a web app, which at the time was a novel concept.

## Conclusion

You have successfully built a Property Graph knowledge base from a text document. The graph is stored in a production-ready Neo4j database, allowing for efficient and complex queries. By combining graph retrieval with vector search and LLM-powered synonym expansion, you created a robust system for extracting precise and contextually rich answers from unstructured data.