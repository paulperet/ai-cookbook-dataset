# Building a RAG Pipeline with Ollama, Mistral, and LlamaIndex

This guide walks you through constructing a Retrieval-Augmented Generation (RAG) pipeline using Ollama, the Mistral model, and LlamaIndex. You will learn to integrate these components, implement basic RAG, and then enhance your system with advanced query routing and decomposition techniques.

## Prerequisites

Before you begin, ensure you have [Ollama installed](https://ollama.com/) and have pulled the Mistral model:
```bash
ollama pull mistral:instruct
```

## 1. Setup and Initial Configuration

First, install the necessary Python packages and configure your environment.

```python
# Apply nest_asyncio for async operations in environments like Jupyter
import nest_asyncio
nest_asyncio.apply()

# Import core components
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
```

### 1.1 Configure the Language Model (LLM)

Initialize the Mistral model via Ollama, setting a generous timeout for responses.

```python
llm = Ollama(model="mistral:instruct", request_timeout=60.0)
```

Let's test the LLM with a simple chat query.

```python
from llama_index.core.llms import ChatMessage

messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="What is the capital city of France?"),
]
response = llm.chat(messages)
print(response)
```
**Output:**
```
assistant: The capital city of France is Paris...
```

### 1.2 Configure the Embedding Model

We'll use a lightweight, effective embedding model from Hugging Face.

```python
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

### 1.3 Apply Global Settings

Set the LLM and embedding model as the default for all LlamaIndex components.

```python
Settings.llm = llm
Settings.embed_model = embed_model
```

## 2. Prepare Your Data

For this tutorial, we will use Uber and Lyft's 2021 SEC 10-K filings as example documents.

### 2.1 Download the Data

Download the PDF files to your local directory.

```bash
curl -o ./uber_2021.pdf https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf
curl -o ./lyft_2021.pdf https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf
```

### 2.2 Load the Documents

Use LlamaIndex's document loader to read the PDFs.

```python
from llama_index.core import SimpleDirectoryReader

uber_docs = SimpleDirectoryReader(input_files=["./uber_2021.pdf"]).load_data()
lyft_docs = SimpleDirectoryReader(input_files=["./lyft_2021.pdf"]).load_data()
```

## 3. Build a Basic RAG Pipeline

Create vector indexes and query engines for each document set.

```python
from llama_index.core import VectorStoreIndex

# Create indexes
uber_vector_index = VectorStoreIndex.from_documents(uber_docs)
lyft_vector_index = VectorStoreIndex.from_documents(lyft_docs)

# Create query engines
uber_vector_query_engine = uber_vector_index.as_query_engine(similarity_top_k=2)
lyft_vector_query_engine = lyft_vector_index.as_query_engine(similarity_top_k=2)
```

### 3.1 Test the Basic Query Engines

Query each engine to verify it retrieves correct information.

```python
# Query Uber's revenue
response = uber_vector_query_engine.query("What is the revenue of uber in 2021 in millions?")
print(f"Uber 2021 Revenue: {response.response}")
```
**Output:**
```
Uber 2021 Revenue: The revenue of Uber in 2021 was $17,455 million.
```

```python
# Query Lyft's revenue
response = lyft_vector_query_engine.query("What is the revenue of lyft in 2021 in millions?")
print(f"Lyft 2021 Revenue: {response.response}")
```
**Output:**
```
Lyft 2021 Revenue: ... the revenue for Lyft ... was approximately $3,208 million...
```

## 4. Implement Query Routing with RouterQueryEngine

Manually selecting the correct query engine is inefficient. The `RouterQueryEngine` automatically directs a user's question to the most appropriate data source.

### 4.1 Define Query Engine Tools

Wrap your query engines into tools with descriptive metadata.

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_vector_query_engine,
        metadata=ToolMetadata(
            name="vector_lyft_10k",
            description="Provides information about Lyft financials for year 2021",
        ),
    ),
    QueryEngineTool(
        query_engine=uber_vector_query_engine,
        metadata=ToolMetadata(
            name="vector_uber_10k",
            description="Provides information about Uber financials for year 2021",
        ),
    ),
]
```

### 4.2 Create the Router Query Engine

Initialize the router, which uses an LLM to select the best tool for a given query.

```python
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import LLMSingleSelector

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=query_engine_tools,
    verbose=True
)
```

### 4.3 Test the Router

Ask questions about either company. The router will log its selection process and return an answer from the correct document.

```python
response = query_engine.query("What are the investments made by Uber?")
print(f"Router Response: {response.response}")
```
**Output (with logs):**
```
Selecting query engine 1: ... to find out about Uber's investments...
Router Response: Uber invests in a variety of financial instruments...
```

The router successfully identified the query was about Uber and used the correct tool.

## 5. Handle Complex Queries with SubQuestionQueryEngine

Some questions require synthesizing information from multiple sources. The `SubQuestionQueryEngine` breaks down a complex query into sub-questions, routes each to the appropriate tool, and combines the answers.

### 5.1 Create the Sub-Question Query Engine

```python
from llama_index.core.query_engine import SubQuestionQueryEngine

sub_question_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    verbose=True
)
```

### 5.2 Test with a Comparative Query

Ask a question that requires data from both Uber and Lyft documents.

```python
response = sub_question_query_engine.query("Compare the revenues of Uber and Lyft in 2021?")
print(f"Comparative Analysis: {response.response}")
```
**Output (with logs):**
```
Generated 2 sub questions.
[vector_uber_10k] Q: What is the revenue of Uber for year 2021
[vector_lyft_10k] Q: What is the revenue of Lyft for year 2021
...
Comparative Analysis: In the year 2021, the revenue for Uber was $17,455 million and for Lyft it was $3,208,323 thousand...
```

The engine decomposed the single question into two sub-questions, queried the respective tools, and provided a synthesized comparison.

### 5.3 Test with a Multi-Source Fact Gathering Query

```python
response = sub_question_query_engine.query("What are the investments made by Uber and Lyft in 2021?")
print(f"Investment Summary: {response.response}")
```
The engine will generate sub-questions for each company's investments and return a consolidated summary.

## Conclusion

You have successfully built a RAG pipeline that progresses from basic retrieval to intelligent query routing and complex multi-document analysis. You learned to:

1.  Set up Ollama with the Mistral model and configure LlamaIndex.
2.  Create vector indexes and query engines from document data.
3.  Implement a `RouterQueryEngine` to automatically direct queries.
4.  Leverage a `SubQuestionQueryEngine` to decompose and answer complex, multi-faceted questions.

This pipeline provides a robust foundation for building sophisticated, document-aware AI applications.