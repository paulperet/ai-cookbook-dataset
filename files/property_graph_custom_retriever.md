# Custom Retriever in PropertyGraph

In this notebook we demonstrate how to define a custom retriever for a property graph. Although this approach is more complex than using standard graph retrievers, it offers detailed control over the retrieval process, allowing for customization that aligns closely with your specific application needs.

Also, we will walk you through an advanced retrieval workflow using the property graph store directly. We’ll conduct both vector search and text-to-Cypher retrieval, and subsequently integrate the results using a reranking module.

We will be using cohere reranker, so we will need cohere-api-key

```python
%pip install llama-index-core
%pip install llama-index-llms-mistralai
%pip install llama-index-embeddings-mistralai
%pip install llama-index-graph-stores-neo4j
%pip install llama-index-postprocessor-cohere-rerank
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

## Setup Neo4j GraphStore

```python
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="llamaindex",
    url="bolt://localhost:7687",
)
```

## PropertyGraphIndex Construction

```python
from llama_index.core import PropertyGraphIndex

index = PropertyGraphIndex.from_documents(
    documents,
    llm=llm,
    embed_model=embed_model,
    property_graph_store=graph_store,
    show_progress=True,
)
```

## CustomRetriever

Define a custom retriever that combines VectorContextRetriever, TextToCypherRetriever and Reranker.

```python
from llama_index.core.retrievers import (
    CustomPGRetriever,
    VectorContextRetriever,
    TextToCypherRetriever,
)
from llama_index.core.graph_stores import PropertyGraphStore
from llama_index.core.vector_stores.types import VectorStore
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import LLM
from llama_index.postprocessor.cohere_rerank import CohereRerank


from typing import Optional, Any, Union


class CustomRetriever(CustomPGRetriever):
    """Custom retriever with cohere reranking."""

    def init(
        self,
        ## vector context retriever params
        embed_model: Optional[BaseEmbedding] = None,
        vector_store: Optional[VectorStore] = None,
        similarity_top_k: int = 4,
        path_depth: int = 1,
        ## text-to-cypher params
        llm: Optional[LLM] = None,
        text_to_cypher_template: Optional[Union[PromptTemplate, str]] = None,
        ## cohere reranker params
        cohere_api_key: Optional[str] = None,
        cohere_top_n: int = 2,
        **kwargs: Any,
    ) -> None:
        """Uses any kwargs passed in from class constructor."""

        self.vector_retriever = VectorContextRetriever(
            self.graph_store,
            include_text=self.include_text,
            embed_model=embed_model,
            vector_store=vector_store,
            similarity_top_k=similarity_top_k,
            path_depth=path_depth,
        )

        self.cypher_retriever = TextToCypherRetriever(
            self.graph_store,
            llm=llm,
            text_to_cypher_template=text_to_cypher_template
            ## NOTE: you can attach other parameters here if you'd like
        )

        self.reranker = CohereRerank(
            api_key=cohere_api_key, top_n=cohere_top_n
        )

    def custom_retrieve(self, query_str: str) -> str:
        """Define custom retriever with reranking.

        Could return `str`, `TextNode`, `NodeWithScore`, or a list of those.
        """
        nodes_1 = self.vector_retriever.retrieve(query_str)
        nodes_2 = self.cypher_retriever.retrieve(query_str)
        reranked_nodes = self.reranker.postprocess_nodes(
            nodes_1 + nodes_2, query_str=query_str
        )

        ## TMP: please change
        final_text = "\n\n".join(
            [n.get_content(metadata_mode="llm") for n in reranked_nodes]
        )

        return final_text

    # optional async method
    # async def acustom_retrieve(self, query_str: str) -> str:
    #     ...
```

```python
from llama_index.core import Settings
Settings.llm = llm
Settings.embed_model=embed_model
```

## Setup CustomRetriever

```python
custom_sub_retriever = CustomRetriever(
    index.property_graph_store,
    include_text=True,
    vector_store=index.vector_store,
    cohere_api_key="YOUR COHERE API KEY",
)
```

## QueryEngine

```python
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(
    index.as_retriever(sub_retrievers=[custom_sub_retriever]), llm=llm
)
```

```python
response = query_engine.query("What did author do at Interleaf?")
display(Markdown(f"{response.response}"))
```

The author worked at Interleaf, a software company that made software for creating documents, similar to Microsoft Word. He was hired as a Lisp hacker to write in their scripting language, which was a dialect of Lisp and inspired by Emacs. However, he admits to being a bad employee due to his lack of knowledge and unwillingness to learn C, as their Lisp was just a thin layer on top of a large C codebase. He also found the conventional office hours unnatural, which led to friction. Towards the end of his time there, he spent much of his time working on his own project, "On Lisp". Despite his shortcomings as an employee, he was well compensated, which helped him save enough money to return to RISD and pay off his college loans.

```python

```