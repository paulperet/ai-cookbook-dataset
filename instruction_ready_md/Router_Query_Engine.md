# Building a Router Query Engine: A Step-by-Step Guide

This guide walks you through creating a `RouterQueryEngine`—a powerful LlamaIndex component that intelligently routes user queries to the most appropriate query engine tool. You'll learn how to set up multiple indices on the same document and let an LLM decide which one to use for each query.

## Prerequisites

Before you begin, ensure you have the necessary packages installed.

```bash
pip install llama-index
pip install llama-index-llms-anthropic
pip install llama-index-embeddings-huggingface
```

## Step 1: Initial Setup and Configuration

First, let's set up the environment. This includes handling async operations for Jupyter notebooks and configuring logging.

```python
# Required for running async operations in Jupyter notebooks
import nest_asyncio
nest_asyncio.apply()

import logging
import sys

# Configure logging to output to the console
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []  # Clear existing handlers

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
```

Next, set your Anthropic API key to use the Claude model.

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY_HERE"
```

## Step 2: Configure the LLM and Embedding Model

We'll use Anthropic's Claude 3 Opus as our language model and a Hugging Face embedding model.

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings

# Initialize the LLM and embedding model
llm = Anthropic(temperature=0.0, model="claude-3-opus-20240229")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Apply global settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
```

## Step 3: Load Your Document

For this tutorial, we'll use Paul Graham's essay "What I Worked On" as our sample document.

```python
from llama_index.core import SimpleDirectoryReader

# Load the document from the 'data/paul_graham' directory
documents = SimpleDirectoryReader("data/paul_graham").load_data()
```

> **Note:** Ensure the document file (`paul_graham_essay.txt`) is present in the `data/paul_graham` directory. You can download it from the provided URL if needed.

## Step 4: Create Multiple Indices and Query Engines

The core idea is to create different types of indices on the same document, each optimized for a different kind of query.

```python
from llama_index.core import SummaryIndex, VectorStoreIndex

# Create a Summary Index for summarization tasks
summary_index = SummaryIndex.from_documents(documents)

# Create a Vector Store Index for retrieving specific context
vector_index = VectorStoreIndex.from_documents(documents)
```

Now, create query engines from these indices.

```python
# Configure the Summary Index query engine for summarization
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)

# Configure the Vector Index query engine for retrieval
vector_query_engine = vector_index.as_query_engine()
```

## Step 5: Wrap Query Engines as Tools

To allow the router to choose between them, we wrap each query engine in a `QueryEngineTool` with a clear description.

```python
from llama_index.core.tools.query_engine import QueryEngineTool

# Tool for summarization questions
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Useful for summarization questions related to Paul Graham essay on What I Worked On.",
)

# Tool for specific context retrieval
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context from Paul Graham essay on What I Worked On.",
)
```

## Step 6: Build the Router Query Engine

The `RouterQueryEngine` uses a selector (an LLM in this case) to analyze the user's query and pick the most suitable tool.

```python
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import LLMSingleSelector

# Instantiate the router with the two tools
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[summary_tool, vector_tool],
)
```

## Step 7: Test the Router with Different Queries

Let's verify that the router correctly directs queries to the appropriate engine.

**Test 1: A Summarization Query**
This query asks for a high-level overview, which should be routed to the `summary_tool`.

```python
response = query_engine.query("What is the summary of the document?")
print(response.response)
```

**Expected Output:**
The LLM will provide a concise summary of Paul Graham's essay, covering his journey from childhood through his work on Viaweb, Y Combinator, and the Bel programming language.

**Test 2: A Specific Retrieval Query**
This query asks for a specific detail from the text, which should be routed to the `vector_tool`.

```python
response = query_engine.query("What did Paul Graham do growing up?")
print(response.response)
```

**Expected Output:**
The response will cite specific passages from the essay, noting that Paul Graham worked on writing short stories and programming on an IBM 1401 computer during his youth.

## Conclusion

You have successfully built a `RouterQueryEngine` that dynamically routes queries between a summarization engine and a retrieval engine. This pattern is highly extensible—you can add more tools (e.g., for code analysis, translation, or different document sets) to handle a wider variety of queries. The key is providing clear, descriptive `tool.description` fields so the LLM selector can make informed routing decisions.