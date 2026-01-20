# Property Graph with Pre-defined Schemas

In this notebook, we guide you through using Neo4j, Ollama, and Huggingface to build a property graph.

Specifically, we will utilize the SchemaLLMPathExtractor, which enables us to define a precise schema containing possible entity types, relation types, and how they can be interconnected.

## Setup

```python
%pip install llama-index-core
%pip install llama-index-llms-ollama
%pip install llama-index-embeddings-huggingface
%pip install llama-index-graph-stores-neo4j
```

```python
import nest_asyncio

nest_asyncio.apply()

from IPython.display import Markdown, display
```

## Download Data

```python
!mkdir -p 'data/paul_graham/'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'
```

## Load Data

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
```

## Graph Construction

To build our graph, we will use the SchemaLLMPathExtractor. 

This tool allows us to specify a schema for the graph, enabling us to extract entities and relations that adhere to this predefined schema, rather than allowing the LLM to randomly determine entities and relations.

```python
from typing import Literal
from llama_index.llms.ollama import Ollama
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

# best practice to use upper-case
entities = Literal["PERSON", "PLACE", "ORGANIZATION"]
relations = Literal["HAS", "PART_OF", "WORKED_ON", "WORKED_WITH", "WORKED_AT"]

# define which entities can have which relations
# validation_schema = {
#     "PERSON": ["HAS", "PART_OF", "WORKED_ON", "WORKED_WITH", "WORKED_AT"],
#     "PLACE": ["HAS", "PART_OF", "WORKED_AT"],
#     "ORGANIZATION": ["HAS", "PART_OF", "WORKED_WITH"],
# }
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

```python
kg_extractor = SchemaLLMPathExtractor(
    llm=Ollama(model="mistral:7b", json_mode=True, request_timeout=3600),
    possible_entities=entities,
    possible_relations=relations,
    kg_validation_schema=validation_schema,
    # if false, allows for values outside of the schema
    # useful for using the schema as a suggestion
    strict=True,
)
```

## Neo4j setup

```python
!docker run \
    -p 7474:7474 -p 7687:7687 \
    -v $PWD/data:/data -v $PWD/plugins:/plugins \
    --name neo4j-apoc \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4JLABS_PLUGINS=\[\"apoc\"\] \
    neo4j:latest
```

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

## PropertyGraphIndex construction

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

[Parsing nodes: 100%|██████████| 1/1 [00:00<00:00, 28.70it/s], Extracting paths from text with schema: 100%|██████████| 22/22 [08:14<00:00, 22.46s/it], Generating embeddings: 100%|██████████| 3/3 [00:00<00:00,  3.37it/s], ..., Generating embeddings: 100%|██████████| 9/9 [00:01<00:00,  5.87it/s]]

## Setup Retrievers

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

## Querying

### Retriever

```python
retriever = index.as_retriever(
    sub_retrievers=[
        llm_synonym,
        vector_context,
    ]
)
```

```python
nodes = retriever.retrieve("What happened at Interleaf?")

for node in nodes:
    print(node.text)
```

### QueryEngine

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

At Interleaf, a software company, they developed software for creating documents and added a scripting language. Paul Graham worked on this project for them. Eventually, something occurred that led to the significant impact of Interleaf being crushed by Moore's law.

```python

```