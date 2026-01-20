# Router Query Engine with Mistral AI and LlamaIndex

A `VectorStoreIndex` is designed to handle queries related to specific contexts, while a `SummaryIndex` is optimized for answering summarization queries. However, in real-world scenarios, user queries may require either context-specific responses or summarizations. To address this, the system must effectively route user queries to the appropriate index to provide relevant answers.

In this notebook, we will utilize the `RouterQueryEngine` to direct user queries to the appropriate index based on the query type.

### Installation

```python
!pip install llama-index
!pip install llama-index-llms-mistralai
!pip install llama-index-embeddings-mistralai
```

### Setup API Key

```python
import os
os.environ['MISTRAL_API_KEY'] = 'YOUR MISTRAL API KEY'
```

### Set LLM and Embedding Model

```python
import nest_asyncio

nest_asyncio.apply()
```

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Settings

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import LLMSingleSelector
```

```python
llm = MistralAI(model='mistral-large')
embed_model = MistralAIEmbedding()

Settings.llm = llm
Settings.embed_model = embed_model
```

### Download Data

We will use `Uber 10K SEC Filings`.

```python
!mkdir -p 'data/10k/'
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
```

### Load Data

```python
uber_docs = SimpleDirectoryReader(input_files=["./data/10k/uber_2021.pdf"]).load_data()
```

### Index and Query Engine creation

 1. VectorStoreIndex -> Specific context queries
 2. SummaryIndex -> Summarization queries

```python
uber_vector_index = VectorStoreIndex.from_documents(uber_docs)

uber_summary_index = VectorStoreIndex.from_documents(uber_docs)
```

```python
uber_vector_query_engine = uber_vector_index.as_query_engine(similarity_top_k = 5)
uber_summary_query_engine = uber_summary_index.as_query_engine()
```

### Create Tools

```python
query_engine_tools = [
    QueryEngineTool(
        query_engine=uber_vector_query_engine,
        metadata=ToolMetadata(
            name="vector_engine",
            description=(
                "Provides information about Uber financials for year 2021."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_summary_query_engine,
        metadata=ToolMetadata(
            name="summary_engine",
            description=(
                "Provides Summary about Uber financials for year 2021."
            ),
        ),
    ),
]
```

### Create Router Query Engine

```python
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=query_engine_tools,
    verbose = True
)
```

### Querying

#### Summarization Query

You can see that it uses `SummaryIndex` to provide answer to the summarization query.

```python
response = query_engine.query("What is the summary of the Uber Financials in 2021?")
print(response)
```

#### Specific Context Query

You can see it uses `VectorStoreIndex` to answer specific context type query.

```python
response = query_engine.query("What is the the revenue of Uber in 2021?")
print(response)
```