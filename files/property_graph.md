# PropertyGraph Index with Mistral AI and LlamaIndex 

In this notebook, we demonstrate the basic usage of the PropertyGraphIndex in LlamaIndex.

The property graph index will process unstructured documents, extract a property graph from them, and offer various methods for querying this graph.

## Setup


```python
%pip install llama-index-core
%pip install llama-index-llms-mistralai
%pip install llama-index-embeddings-mistralai
```


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

    [First Entry, ..., Last Entry]


## Load Data


```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
```

## Create PropertyGraphIndex

The following steps occur during the creation of a PropertyGraph:

1.	PropertyGraphIndex.from_documents(): We load documents into an index.

2.	Parsing Nodes: The index parses the documents into nodes.

3.	Extracting Paths from Text: The nodes are passed to an LLM, which is prompted to generate knowledge graph triples (i.e., paths).

4.	Extracting Implicit Paths: The node.relationships property is used to infer implicit paths.

5.	Generating Embeddings: Embeddings are generated for each text node and graph node, occurring twice during the process.


```python
from llama_index.core import PropertyGraphIndex


index = PropertyGraphIndex.from_documents(
    documents,
    llm=llm,
    embed_model=embed_model,
    show_progress=True,
)
```

    [First Entry, ..., Last Entry]


For debugging purposes, the default SimplePropertyGraphStore includes a helper to save a networkx representation of the graph to an html file.


```python
index.property_graph_store.save_networkx_graph(name="./kg.html")
```


```python
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
```

## Querying

Querying a property graph index typically involves using one or more sub-retrievers and combining their results. The process of graph retrieval includes:

1.	Selecting Nodes: Identifying the initial nodes of interest within the graph.
2.	Traversing: Moving from the selected nodes to explore connected elements.

By default, two primary types of retrieval are employed simultaneously:

•	Synonym/Keyword Expansion: Utilizing an LLM to generate synonyms and keywords derived from the query.

•	Vector Retrieval: Employing embeddings to locate nodes within your graph.


Once nodes are identified, you can choose to:

•	Return Paths: Provide the paths adjacent to the selected nodes, typically in the form of triples.

•	Return Paths and Source Text: Provide both the paths and the original source text of the chunk, if available.


### Retreival


```python
retriever = index.as_retriever(
    include_text=False,  # include source text, default True
)

nodes = retriever.retrieve("What happened at Interleaf and Viaweb?")

for node in nodes:
    print(node.text)
```

    Viaweb -> Launch date -> January 1996
    Viaweb -> Growth rate -> 7x a year
    Viaweb -> Received seed funding from -> Julian
    Viaweb -> Is -> Online store builder
    Viaweb -> User growth -> 70 stores at the end of 1996 and about 500 at the end of 1997
    Viaweb -> Pricing -> $100 a month for a small store and $300 a month for a big one
    Viaweb -> Acquisition -> Bought by yahoo in the summer of 1998
    Viaweb -> Has -> Code editor
    Viaweb -> Strategy -> Doing things that don't scale
    Viaweb -> Developed by -> Robert and trevor
    Viaweb -> Founded by -> Paul graham and robert
    Viaweb -> Reached breakeven -> Summer of 1998
    Viaweb -> Was -> One of the best general-purpose site builders
    Viaweb -> Started by -> I
    Viaweb -> Service -> Building stores for users
    Viaweb -> Investors -> Had significant influence on company decisions
    Viaweb -> Software -> Works via the web
    Viaweb -> Was founded by -> I and robert morris
    Viaweb -> Bought by -> Yahoo
    Viaweb -> Initial product -> Wysiwyg site builder
    Viaweb -> Had -> Handful of employees
    Viaweb -> Started for -> Needing money
    Viaweb -> Hosts -> Stores
    Viaweb -> Status before acquisition -> Not profitable
    I -> Got a job at -> Interleaf
    Interleaf -> Made software for -> Creating documents
    Interleaf -> Added a scripting language -> Lisp
    I -> Arranged to do freelance work for -> Interleaf


### QueryEngine


```python
query_engine = index.as_query_engine(
    include_text=True
)

response = query_engine.query("What happened at Interleaf and Viaweb?")

display(Markdown(f"{response.response}"))
```

At Interleaf, the company made software for creating documents and added a scripting language, Lisp, inspired by Emacs. The narrator worked there for a year but admits to being a bad employee, not understanding most of the software and being irresponsible. However, they learned some valuable lessons about what not to do in a technology company.

Viaweb, on the other hand, was an online store builder that hosted stores for users. Before its public launch, it had to recruit an initial set of users and ensure they had decent-looking stores. Viaweb had a code editor for users to define their own page styles, which was essentially editing Lisp expressions, but it wasn't an app editor. The company was not profitable before it was bought by Yahoo in the summer of 1998. Viaweb's strategy was to do things that don't scale, and it was one of the best general-purpose site builders at the time.


## Storage

By default, storage is managed using our straightforward in-memory abstractions—SimpleVectorStore for embeddings and SimplePropertyGraphStore for the property graph.

We can save and load these structures to and from disk.


```python
index.storage_context.persist(persist_dir="./storage")

from llama_index.core import StorageContext, load_index_from_storage

index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="./storage")
)

query_engine = index.as_query_engine(
    include_text=True
)

response = query_engine.query("What happened at Interleaf and Viaweb?")

display(Markdown(f"{response.response}"))
```

At Interleaf, the company made software for creating documents and added a scripting language, Lisp, inspired by Emacs. The narrator worked there for a year but admits to being a bad employee, not understanding most of the software and being irresponsible. However, they learned some valuable lessons about what not to do in a technology company.

Viaweb, on the other hand, was an online store builder that hosted stores for users. Before its public launch, it had to recruit an initial set of users and ensure they had decent-looking stores. Viaweb had a code editor for users to define their own page styles, which was essentially editing Lisp expressions, but it wasn't an app editor. The company was not profitable before it was bought by Yahoo in the summer of 1998. Viaweb's strategy was to do things that don't scale, and it was one of the best general-purpose site builders at the time.


## Vector Stores

While some graph databases, such as Neo4j, support vectors, you can still specify which vector store to use with your graph in cases where vectors are not supported, or when you want to override the default settings.

Below, we will demonstrate how to combine ChromaVectorStore with the default SimplePropertyGraphStore.


```python
%pip install llama-index-vector-stores-chroma
```

### Build and Save Index


```python
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

client = chromadb.PersistentClient("./chroma_db")
collection = client.get_or_create_collection("my_graph_vector_db")

index = PropertyGraphIndex.from_documents(
    documents,
    llm=llm,
    embed_model=embed_model,
    property_graph_store=SimplePropertyGraphStore(),
    vector_store=ChromaVectorStore(chroma_collection=collection),
    show_progress=True,
)

index.storage_context.persist(persist_dir="./storage")
```

    [First Entry, ..., Last Entry]


### Load Index


```python
index = PropertyGraphIndex.from_existing(
    SimplePropertyGraphStore.from_persist_dir("./storage"),
    vector_store=ChromaVectorStore(chroma_collection=collection),
    llm=llm,
)

query_engine = index.as_query_engine(
    include_text=True
)

response = query_engine.query("why did author do at YC?")

display(Markdown(f"{response.response}"))
```

The author was involved in YC, where they encountered various problems, including having to address misinterpretations of their essays on a forum they managed. They also worked closely with someone named Jessica. However, the specific roles or tasks the author undertook at YC are not detailed in the provided context.



```python

```