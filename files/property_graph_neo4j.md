# PropertyGraph using Neo4j

In this notebook we will demonstrate building PropertyGraph using Neo4j

Neo4j is a production-grade graph database that excels in storing property graphs, performing vector searches, filtering, and more.

The simplest way to begin is by using a cloud-hosted instance through Neo4j Aura. However, for the purposes of this notebook, we will focus on how to run the database locally using Docker.

```python
%pip install llama-index-core 
%pip install llama-index-graph-stores-neo4j
%pip install llama-index-llms-mistralai
%pip install llama-index-embeddings-mistralai
```

## Docker Setup

You need to login and set password for the first time.

1. username: neo4j

2. password: neo4j

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

## Setup

```python
import nest_asyncio

nest_asyncio.apply()

from IPython.display import Markdown, display
```

```python
import os
os.environ['MISTRAL_API_KEY'] = 'YOUR MISTRAL API KEY'
```

```python
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI

llm = MistralAI(model='mistral-large-latest')
embed_model = MistralAIEmbedding()
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

## Index Construction

```python
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

# Note: used to be `Neo4jPGStore`
graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="llamaindex",
    url="bolt://localhost:7687",
)
```

```python
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor

index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=embed_model,
    kg_extractors=[
        SimpleLLMPathExtractor(
            llm=llm
        )
    ],
    property_graph_store=graph_store,
    show_progress=True,
)
```

```python
from llama_index.core import Settings
Settings.llm = llm
Settings.embed_model = embed_model
```

## Retrievers

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

## Querying

### Retrieving

```python
retriever = index.as_retriever(
    sub_retrievers=[
        llm_synonym,
        vector_context,
    ],
)

nodes = retriever.retrieve("What did author do at Viaweb?")

for node in nodes:
    print(node.text)
```

### QueryEngine

```python
query_engine = index.as_query_engine(include_text=True)

response = query_engine.query("What did author do at Viaweb?")

display(Markdown(f"{response.response}"))
```

The author, along with Robert Morris, started Viaweb, a company that aimed to build online stores. The author's role involved writing software to generate websites for galleries initially, and later, developing a new site generator for online stores using Lisp. The author also had the innovative idea of running the software on the server and letting users control it by clicking on links, eliminating the need for any client software or command line interaction on the server. This led to the creation of a web app, which at the time was a novel concept.

```python

```