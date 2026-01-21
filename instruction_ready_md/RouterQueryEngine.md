# Building a Router Query Engine with Mistral AI and LlamaIndex

## Introduction

When building a question-answering system, different types of queries require different retrieval strategies. A `VectorStoreIndex` excels at finding specific, context-rich information, while a `SummaryIndex` is optimized for generating high-level summaries. In real-world applications, users ask both types of questions, so we need a system that can intelligently route each query to the appropriate engine.

In this tutorial, you'll build a `RouterQueryEngine` using LlamaIndex and Mistral AI. This system will analyze incoming queries and automatically direct them to the most suitable index, ensuring users get the best possible answer for their question type.

## Prerequisites

First, let's install the necessary packages and set up our environment.

### Step 1: Install Required Packages

```bash
pip install llama-index
pip install llama-index-llms-mistralai
pip install llama-index-embeddings-mistralai
```

### Step 2: Configure Your API Key

```python
import os

# Replace with your actual Mistral AI API key
os.environ['MISTRAL_API_KEY'] = 'YOUR_MISTRAL_API_KEY'
```

### Step 3: Import Libraries and Configure Settings

```python
import nest_asyncio

# Required for running async operations in Jupyter/notebook environments
nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Settings

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import LLMSingleSelector
```

### Step 4: Initialize the LLM and Embedding Model

```python
# Initialize the Mistral AI LLM and embedding model
llm = MistralAI(model='mistral-large')
embed_model = MistralAIEmbedding()

# Set these as the default models for the LlamaIndex Settings
Settings.llm = llm
Settings.embed_model = embed_model
```

## Prepare Your Data

We'll use Uber's 2021 10-K SEC filing as our sample document. This financial document contains detailed information perfect for testing both specific and summary queries.

### Step 5: Download the Data File

```python
import os

# Create the directory if it doesn't exist
os.makedirs('data/10k/', exist_ok=True)

# Download the Uber 10-K filing
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
```

### Step 6: Load the Document

```python
# Load the PDF document
uber_docs = SimpleDirectoryReader(input_files=["./data/10k/uber_2021.pdf"]).load_data()
```

## Create the Query Engines

Now we'll create two different query engines from the same document, each optimized for a different type of query.

### Step 7: Build Both Indexes

```python
# Create a VectorStoreIndex for specific, context-based queries
uber_vector_index = VectorStoreIndex.from_documents(uber_docs)

# Create another VectorStoreIndex that will function as our summary engine
uber_summary_index = VectorStoreIndex.from_documents(uber_docs)
```

### Step 8: Instantiate the Query Engines

```python
# Create a vector-based query engine that returns the top 5 most relevant chunks
uber_vector_query_engine = uber_vector_index.as_query_engine(similarity_top_k=5)

# Create a summary query engine (using default settings)
uber_summary_query_engine = uber_summary_index.as_query_engine()
```

## Create Tools for the Router

The router needs to know about the available query engines. We'll wrap each engine in a `QueryEngineTool` with descriptive metadata.

### Step 9: Define the Query Engine Tools

```python
query_engine_tools = [
    QueryEngineTool(
        query_engine=uber_vector_query_engine,
        metadata=ToolMetadata(
            name="vector_engine",
            description=(
                "Provides specific, detailed information about Uber financials for year 2021. "
                "Use this for questions about exact numbers, specific facts, or detailed context."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_summary_query_engine,
        metadata=ToolMetadata(
            name="summary_engine",
            description=(
                "Provides high-level summaries about Uber financials for year 2021. "
                "Use this for questions asking for overviews, summaries, or general understanding."
            ),
        ),
    ),
]
```

## Build the Router Query Engine

Now we'll create the intelligent router that will select the appropriate tool based on the user's query.

### Step 10: Initialize the Router Query Engine

```python
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),  # Uses the LLM to choose the best tool
    query_engine_tools=query_engine_tools,
    verbose=True  # Set to True to see the routing decision process
)
```

## Test the Router

Let's test our router with different types of queries to see it in action.

### Step 11: Test a Summarization Query

When you ask for a summary, the router should select the `summary_engine`.

```python
response = query_engine.query("What is the summary of the Uber Financials in 2021?")
print(response)
```

The router will analyze this query, recognize it asks for a summary, and route it to the summary engine. You'll see output similar to:

```
The summary engine provides an overview of Uber's 2021 financial performance...
```

### Step 12: Test a Specific Context Query

When you ask for specific information, the router should select the `vector_engine`.

```python
response = query_engine.query("What is the revenue of Uber in 2021?")
print(response)
```

The router will recognize this as a specific factual question and route it to the vector engine. You'll see output similar to:

```
Uber's revenue in 2021 was $17.5 billion...
```

## How It Works

The `RouterQueryEngine` uses the `LLMSingleSelector` to analyze each incoming query. The selector:
1. Examines the query text
2. Compares it against the descriptions of available tools
3. Selects the most appropriate query engine
4. Routes the query to that engine for processing

The `verbose=True` setting lets you see this decision-making process in real time, showing which tool was selected and why.

## Conclusion

You've successfully built an intelligent query routing system that can handle different types of questions appropriately. This architecture is particularly useful when:

- You have multiple documents or data sources
- Different query types require different retrieval strategies
- You want to optimize response quality based on question type

You can extend this pattern by adding more specialized query engines (for tables, code, images, etc.) and letting the router handle increasingly complex query routing scenarios.