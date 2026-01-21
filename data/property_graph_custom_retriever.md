# Building a Custom Retriever for Property Graphs

This guide walks you through creating a custom retriever for a property graph. While more complex than using standard graph retrievers, this approach provides fine-grained control over the retrieval process, enabling you to tailor it precisely to your application's needs.

We'll implement an advanced retrieval workflow that performs both vector search and text-to-Cypher retrieval, then merges and reranks the results using Cohere's reranker.

## Prerequisites

Ensure you have the following installed and set up:

```bash
pip install llama-index-core
pip install llama-index-llms-mistralai
pip install llama-index-embeddings-mistralai
pip install llama-index-graph-stores-neo4j
pip install llama-index-postprocessor-cohere-rerank
```

## 1. Initial Setup

First, apply necessary patches for async operations and configure your environment.

```python
import nest_asyncio
import os

nest_asyncio.apply()

# Set your Mistral AI API key
os.environ['MISTRAL_API_KEY'] = 'YOUR_MISTRAL_API_KEY'
```

Now, initialize the language model and embedding model.

```python
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI

llm = MistralAI(model='mistral-large-latest')
embed_model = MistralAIEmbedding()
```

## 2. Load Your Data

We'll use a sample essay for this tutorial. Download and load it.

```python
# Download the sample data
!mkdir -p 'data/paul_graham/'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

# Load the document
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
```

## 3. Set Up Neo4j with Docker

We'll use Neo4j as our property graph store. Run it via Docker.

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

**Note:** On first run, log into the Neo4j browser at `http://localhost:7474` with username `neo4j` and password `neo4j`. You will be prompted to set a new password. Use `llamaindex` for this tutorial.

## 4. Configure the Property Graph Store

Connect to your Neo4j instance from Python.

```python
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

graph_store = Neo4jPropertyGraphStore(
    username="neo4j",
    password="llamaindex",
    url="bolt://localhost:7687",
)
```

## 5. Build the Property Graph Index

Create an index from your documents. This step extracts entities, relationships, and properties, storing them in the graph and generating vector embeddings.

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

## 6. Define the Custom Retriever Class

This is the core of the tutorial. We'll create a custom retriever that combines a vector search, a text-to-Cypher query, and a reranking step.

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
    """Custom retriever that combines vector search, Cypher query, and Cohere reranking."""

    def init(
        self,
        ## Vector retriever parameters
        embed_model: Optional[BaseEmbedding] = None,
        vector_store: Optional[VectorStore] = None,
        similarity_top_k: int = 4,
        path_depth: int = 1,
        ## Text-to-Cypher parameters
        llm: Optional[LLM] = None,
        text_to_cypher_template: Optional[Union[PromptTemplate, str]] = None,
        ## Cohere reranker parameters
        cohere_api_key: Optional[str] = None,
        cohere_top_n: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialize the sub-retrievers and reranker using provided parameters."""

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
            # You can attach other parameters here if needed
        )

        self.reranker = CohereRerank(
            api_key=cohere_api_key, top_n=cohere_top_n
        )

    def custom_retrieve(self, query_str: str) -> str:
        """Define the custom retrieval logic with reranking."""
        # Step 1: Perform vector similarity search
        nodes_1 = self.vector_retriever.retrieve(query_str)
        # Step 2: Perform a text-to-Cypher graph query
        nodes_2 = self.cypher_retriever.retrieve(query_str)
        # Step 3: Combine and rerank all retrieved nodes
        reranked_nodes = self.reranker.postprocess_nodes(
            nodes_1 + nodes_2, query_str=query_str
        )

        # Combine the content of the top reranked nodes
        final_text = "\n\n".join(
            [n.get_content(metadata_mode="llm") for n in reranked_nodes]
        )
        return final_text

    # Optional: Define an async version
    # async def acustom_retrieve(self, query_str: str) -> str:
    #     ...
```

## 7. Configure Global Settings

Set the default LLM and embedding model for the application.

```python
from llama_index.core import Settings
Settings.llm = llm
Settings.embed_model = embed_model
```

## 8. Instantiate the Custom Retriever

Create an instance of your `CustomRetriever`, providing it with the graph store, vector store, and your Cohere API key.

```python
custom_sub_retriever = CustomRetriever(
    index.property_graph_store,
    include_text=True,
    vector_store=index.vector_store,
    cohere_api_key="YOUR_COHERE_API_KEY",  # Replace with your key
)
```

## 9. Create and Test the Query Engine

Finally, build a query engine that uses your custom retriever and test it with a sample query.

```python
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(
    index.as_retriever(sub_retrievers=[custom_sub_retriever]), llm=llm
)

# Execute a query
response = query_engine.query("What did author do at Interleaf?")
print(response.response)
```

**Expected Output:**
The author worked at Interleaf, a software company that made software for creating documents, similar to Microsoft Word. He was hired as a Lisp hacker to write in their scripting language, which was a dialect of Lisp and inspired by Emacs. However, he admits to being a bad employee due to his lack of knowledge and unwillingness to learn C, as their Lisp was just a thin layer on top of a large C codebase. He also found the conventional office hours unnatural, which led to friction. Towards the end of his time there, he spent much of his time working on his own project, "On Lisp". Despite his shortcomings as an employee, he was well compensated, which helped him save enough money to return to RISD and pay off his college loans.

## Summary

You have successfully built a custom retriever for a property graph that:
1.  Performs a hybrid search using both vector similarity and graph-based Cypher queries.
2.  Combines the results from both retrieval methods.
3.  Uses Cohere's reranker to select the most relevant information.
4.  Integrates seamlessly into a LlamaIndex query engine.

This pattern provides a powerful template you can adapt, extending the `custom_retrieve` method to incorporate other retrievers, filters, or post-processing logic specific to your use case.