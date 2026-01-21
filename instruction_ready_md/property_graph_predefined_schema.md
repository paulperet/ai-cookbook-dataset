# Building a Property Graph with Pre-defined Schemas

This guide walks you through constructing a knowledge graph using Neo4j, Ollama, and Hugging Face. You'll use the `SchemaLLMPathExtractor` to define a precise schema for entity and relationship extraction, ensuring your graph is structured and consistent.

## Prerequisites

Ensure you have the following installed:
- Python 3.8+
- Docker (for running Neo4j)
- Basic understanding of knowledge graphs and LLMs

## Step 1: Install Required Libraries

Begin by installing the necessary Python packages.

```bash
pip install llama-index-core
pip install llama-index-llms-ollama
pip install llama-index-embeddings-huggingface
pip install llama-index-graph-stores-neo4j
```

## Step 2: Import Dependencies and Apply Async Fix

Import the required modules and apply a patch for asynchronous operations, which is often needed in notebook environments.

```python
import nest_asyncio

nest_asyncio.apply()

from IPython.display import Markdown, display
```

## Step 3: Download the Sample Data

You'll use Paul Graham's essay as sample text. Download it to a local directory.

```bash
mkdir -p 'data/paul_graham/'
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'
```

## Step 4: Load the Documents

Use LlamaIndex's `SimpleDirectoryReader` to load the text file into document objects.

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
```

## Step 5: Define the Graph Schema

The `SchemaLLMPathExtractor` allows you to define a strict schema for entity and relationship extraction. This ensures the LLM extracts only the types you specify, leading to a cleaner, more predictable graph.

First, define the possible entity and relationship types using Python's `Literal` type for validation.

```python
from typing import Literal

entities = Literal["PERSON", "PLACE", "ORGANIZATION"]
relations = Literal["HAS", "PART_OF", "WORKED_ON", "WORKED_WITH", "WORKED_AT"]
```

Next, define the validation schema. This is a list of triples specifying which entity types can be connected by which relation types.

```python
validation_schema = [
    ("ORGANIZATION", "HAS", "PERSON"),
    ("PERSON", "WORKED_AT", "ORGANIZATION"),
    ("PERSON", "WORKED_WITH", "PERSON"),
    ("PERSON", "WORKED_ON", "ORGANIZATION"),
    ("PERSON", "PART_OF", "ORGANIZATION"),
    ("ORGANIZATION", "PART_OF", "ORGANIZATION"),
    ("PERSON", "WORKED_AT", "PLACE"),
]
```

## Step 6: Initialize the Knowledge Graph Extractor

Create an instance of `SchemaLLMPathExtractor`. You'll use the Ollama LLM with the Mistral 7B model, set to output JSON. The `strict=True` parameter enforces the validation schema.

```python
from llama_index.llms.ollama import Ollama
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

kg_extractor = SchemaLLMPathExtractor(
    llm=Ollama(model="mistral:7b", json_mode=True, request_timeout=3600),
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    strict=True,
)
```

## Step 7: Set Up Neo4j with Docker

You'll use Neo4j as your graph database. Run it in a Docker container with the APOC plugin enabled for extended functionality.

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

## Step 8: Connect to Neo4j and Initialize Stores

Create a connection to your Neo4j instance and initialize the graph and vector stores.

```python
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.vector_stores.simple import SimpleVectorStore

graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="llamaindex",
    url="bolt://localhost:7687",
)
vec_store = SimpleVectorStore()
```

## Step 9: Build the Property Graph Index

Now, construct the `PropertyGraphIndex` from your loaded documents. This step uses the schema extractor, an embedding model for vector representations, and the stores you just set up. It will parse the text, extract entities and relationships according to your schema, generate embeddings, and populate the graph.

```python
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[kg_extractor],
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    property_graph_store=graph_store,
    vector_store=vec_store,
    show_progress=True,
)
```

## Step 10: Configure Retrievers

To query the graph effectively, you'll set up two specialized retrievers:
1.  **`LLMSynonymRetriever`**: Uses the LLM to understand synonyms and related terms for your query.
2.  **`VectorContextRetriever`**: Uses vector similarity to find textually relevant nodes.

```python
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
)

llm_synonym = LLMSynonymRetriever(
    index.property_graph_store,
    llm=Ollama(model="mistral:instruct", request_timeout=3600),
    include_text=False,
)
vector_context = VectorContextRetriever(
    index.property_graph_store,
    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    include_text=False,
)
```

## Step 11: Query the Graph

### Using a Retriever Directly

First, create a composite retriever that uses both strategies, then query it to fetch relevant nodes.

```python
retriever = index.as_retriever(
    sub_retrievers=[
        llm_synonym,
        vector_context,
    ]
)

nodes = retriever.retrieve("What happened at Interleaf?")
for node in nodes:
    print(node.text)
```

### Using a Full Query Engine

For a more powerful question-answering experience, create a `QueryEngine`. This uses the retrievers to fetch context and an LLM to synthesize a final answer.

```python
query_engine = index.as_query_engine(
    sub_retrievers=[
        llm_synonym,
        vector_context,
    ],
    llm=Ollama(model="mistral:instruct", request_timeout=3600),
)

response = query_engine.query("What happened at Interleaf?")
display(Markdown(f"{response.response}"))
```

**Example Output:**
> At Interleaf, a software company, they developed software for creating documents and added a scripting language. Paul Graham worked on this project for them. Eventually, something occurred that led to the significant impact of Interleaf being crushed by Moore's law.

## Summary

You have successfully built a property graph with a pre-defined schema. By controlling the entity and relationship types, you created a structured knowledge base from unstructured text. You then connected this graph to Neo4j and implemented a hybrid retrieval system combining synonym expansion and vector search to answer questions effectively.