# Financial Document Analysis with LlamaIndex

This guide demonstrates how to perform financial analysis over **10-K** documents using the **LlamaIndex** framework. You will build a Retrieval-Augmented Generation (RAG) system to extract and compare information from lengthy annual reports with just a few lines of code.

## Introduction

### About LlamaIndex
LlamaIndex is a data framework for LLM applications. It enables you to quickly build RAG systems and offers advanced tooling for data ingestion, indexing, retrieval, and re-ranking. For more details, see the [full documentation](https://gpt-index.readthedocs.io/en/latest/).

### The Task: Analyzing 10-K Forms
A 10-K is an annual report required by the U.S. Securities and Exchange Commission (SEC) that provides a comprehensive summary of a company's financial performance. These documents are often hundreds of pages long and contain domain-specific terminology, making manual analysis time-consuming. This tutorial shows how LlamaIndex can help a financial analyst quickly extract and synthesize insights across multiple documents.

## Prerequisites

First, install the required libraries.

```bash
pip install llama-index pypdf
```

## Setup

Import the necessary modules and configure the LLM.

```python
from langchain import OpenAI
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index import set_global_service_context
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
```

Configure the language model. This example uses `gpt-3.5-turbo-instruct` from OpenAI.

```python
llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct", max_tokens=-1)
```

Create a `ServiceContext` and set it as the global default. This ensures all subsequent LLM operations use your configured model.

```python
service_context = ServiceContext.from_defaults(llm=llm)
set_global_service_context(service_context=service_context)
```

## Step 1: Load and Index the Documents

Load the PDFs for Lyft's and Uber's 2021 10-K reports. The `SimpleDirectoryReader` converts each page of the PDFs into a plain text `Document` object.

> **Note:** This operation may take a moment as each document is over 100 pages.

```python
lyft_docs = SimpleDirectoryReader(input_files=["../data/10k/lyft_2021.pdf"]).load_data()
uber_docs = SimpleDirectoryReader(input_files=["../data/10k/uber_2021.pdf"]).load_data()

print(f'Loaded Lyft 10-K with {len(lyft_docs)} pages')
print(f'Loaded Uber 10-K with {len(uber_docs)} pages')
```

**Output:**
```
Loaded Lyft 10-K with 238 pages
Loaded Uber 10-K with 307 pages
```

Now, build a `VectorStoreIndex` for each set of documents. This index will enable semantic search over the content.

> **Note:** This step calls the OpenAI API to compute vector embeddings and may take some time.

```python
lyft_index = VectorStoreIndex.from_documents(lyft_docs)
uber_index = VectorStoreIndex.from_documents(uber_docs)
```

## Step 2: Perform Simple Queries

To query an index, you first create a `QueryEngine`. For a `VectorStoreIndex`, a key parameter is `similarity_top_k`, which controls how many document chunks (nodes) are retrieved as context for the LLM.

```python
lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)
```

You can now run queries against each company's data.

```python
response = await lyft_engine.aquery('What is the revenue of Lyft in 2021? Answer in millions with page reference')
print(response)
```

**Output:**
```
$3,208.3 million (page 63)
```

```python
response = await uber_engine.aquery('What is the revenue of Uber in 2021? Answer in millions, with page reference')
print(response)
```

**Output:**
```
$17,455 (page 53)
```

## Step 3: Perform Advanced Compare-and-Contrast Queries

For complex analysis involving multiple documents, use the `SubQuestionQueryEngine`. It decomposes a complex query into simpler sub-questions, routes each to the appropriate document index, and synthesizes the answers.

First, wrap your individual query engines into tools.

```python
query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(name='lyft_10k', description='Provides information about Lyft financials for year 2021')
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(name='uber_10k', description='Provides information about Uber financials for year 2021')
    ),
]

s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)
```

### Example 1: Compare Customer and Geographic Growth

Ask a question that requires analyzing both documents.

```python
response = await s_engine.aquery('Compare and contrast the customer segments and geographies that grew the fastest')
print(response)
```

**Output (Condensed):**
```
The customer segments that grew the fastest for Uber in 2021 were its Mobility Drivers, Couriers, Riders, and Eaters...
Uber experienced the most growth in large metropolitan areas, such as Chicago, Miami, New York City...

The customer segments that grew the fastest for Lyft were ridesharing, light vehicles, and public transit...
It is not possible to answer the question of which geographies grew the fastest for Lyft with the given context information.
```

### Example 2: Compare Revenue Growth

Ask a quantitative comparison question.

```python
response = await s_engine.aquery('Compare revenue growth of Uber and Lyft from 2020 to 2021')
print(response)
```

**Output:**
```
The revenue growth of Uber from 2020 to 2021 was 57%, or 54% on a constant currency basis, while the revenue growth of Lyft from 2020 to 2021 was 36%. This means that Uber had a higher revenue growth than Lyft from 2020 to 2021.
```

## Summary

You have successfully built a financial analysis system using LlamaIndex. You learned how to:
1. Load and index lengthy 10-K documents.
2. Create query engines to answer specific questions from a single document.
3. Use a `SubQuestionQueryEngine` to perform complex, multi-document compare-and-contrast analysis.

This workflow can be extended to analyze other financial documents, incorporate more data sources, or adjust retrieval parameters for higher precision.