# Building a Sub-Question Query Engine with Mistral AI and LlamaIndex

A standard `VectorStoreIndex` excels at answering questions about specific contexts within a single document or a focused collection. However, real-world user queries are often complex, requiring information to be synthesized from multiple, disparate sources. In these cases, a simple vector index may fall short.

The solution is to decompose the complex query into smaller, manageable sub-questions. This guide demonstrates how to use the `SubQuestionQueryEngine` from LlamaIndex to handle such multi-document queries effectively.

## Prerequisites

Ensure you have the necessary Python packages installed.

```bash
pip install llama-index
pip install llama-index-llms-mistralai
pip install llama-index-embeddings-mistralai
```

## Step 1: Configure Your Environment

First, set your Mistral AI API key as an environment variable.

```python
import os
os.environ['MISTRAL_API_KEY'] = '<YOUR MISTRAL API KEY>'
```

## Step 2: Initialize LLM and Embedding Model

Configure the core language model and embedding model from Mistral AI. We also apply `nest_asyncio` to manage event loops in a notebook environment.

```python
import nest_asyncio
nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Settings

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

# Initialize the LLM and Embedding model
llm = MistralAI(model='mistral-large')
embed_model = MistralAIEmbedding()

# Apply global settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
```

## Step 3: Download the Sample Data

We'll use three distinct documents to demonstrate multi-source querying: Uber's 2021 10-K filing, Lyft's 2021 10-K filing, and an essay by Paul Graham.

```bash
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O './uber_2021.pdf'
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O './lyft_2021.pdf'
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O './paul_graham_essay.txt'
```

## Step 4: Load the Documents

Use LlamaIndex's `SimpleDirectoryReader` to load each document into memory.

```python
# Load Uber document
uber_docs = SimpleDirectoryReader(input_files=["./uber_2021.pdf"]).load_data()

# Load Lyft document
lyft_docs = SimpleDirectoryReader(input_files=["./lyft_2021.pdf"]).load_data()

# Load Paul Graham essay
paul_graham_docs = SimpleDirectoryReader(input_files=["./paul_graham_essay.txt"]).load_data()
```

## Step 5: Create Vector Indexes and Query Engines

Create a separate vector index for each document set, then build a query engine on top of each index.

```python
# Create indexes
uber_vector_index = VectorStoreIndex.from_documents(uber_docs)
lyft_vector_index = VectorStoreIndex.from_documents(lyft_docs)
paul_graham_vector_index = VectorStoreIndex.from_documents(paul_graham_docs)

# Create query engines
uber_vector_query_engine = uber_vector_index.as_query_engine(similarity_top_k=5)
lyft_vector_query_engine = lyft_vector_index.as_query_engine(similarity_top_k=5)
paul_graham_vector_query_engine = paul_graham_vector_index.as_query_engine(similarity_top_k=5)
```

## Step 6: Wrap Query Engines as Tools

The `SubQuestionQueryEngine` requires each query engine to be wrapped as a `QueryEngineTool` with descriptive metadata.

```python
query_engine_tools = [
    QueryEngineTool(
        query_engine=uber_vector_query_engine,
        metadata=ToolMetadata(
            name="uber_vector_query_engine",
            description="Provides information about Uber financials for year 2021.",
        ),
    ),
    QueryEngineTool(
        query_engine=lyft_vector_query_engine,
        metadata=ToolMetadata(
            name="lyft_vector_query_engine",
            description="Provides information about Lyft financials for year 2021.",
        ),
    ),
    QueryEngineTool(
        query_engine=paul_graham_vector_query_engine,
        metadata=ToolMetadata(
            name="paul_graham_vector_query_engine",
            description="Provides information about Paul Graham.",
        ),
    ),
]
```

## Step 7: Instantiate the Sub-Question Query Engine

Create the main engine that will decompose complex queries and route sub-questions to the appropriate tools.

```python
sub_question_query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools)
```

## Step 8: Execute Complex Queries

Now you can ask complex, multi-part questions. The engine will automatically generate relevant sub-questions, query the appropriate tools, and synthesize a final answer.

### Example 1: Compare Uber and Lyft Revenue

This query involves two sub-documents (Uber and Lyft filings).

```python
response = sub_question_query_engine.query("Compare the revenue of uber and lyft?")
print(response.response)
```

**Output:**
```
In 2021, Uber's total revenue was significantly higher than Lyft's. Uber generated $17,455 million from various services like Mobility, Delivery, Freight, and other revenue streams. On the other hand, Lyft's revenue for the same year was $3,208,323 in thousands, which translates to $3,208.323 million. Therefore, Uber's revenue was approximately five times that of Lyft in 2021.
```

### Example 2: Query Across Disparate Sources

This query combines a financial question (Uber) with a biographical one (Paul Graham).

```python
response = sub_question_query_engine.query("What is the revenue of uber and why did paul graham start YC?")
print(response.response)
```

**Output:**
```
Uber's revenue for the year 2021 was reported to be $17,455 million. As for the reason behind Paul Graham starting Y Combinator, it was a result of three main factors. Firstly, he wanted to stop delaying his plans for angel investing. Secondly, he wished to collaborate with Robert and Trevor on projects. Lastly, he was frustrated with venture capitalists who took too long to make decisions. Therefore, he decided to start his own investment firm...
```

### Example 3: A Three-Part Query

This query combines all three data sources.

```python
response = sub_question_query_engine.query("Compare revenue of uber with lyft and why did paul graham start YC?")
print(response.response)
```

**Output:**
```
In 2021, Uber's total revenue was significantly higher than Lyft's. Uber generated a total of $17,455 million... On the other hand, Lyft's revenue for the same year was $3,208,323 (in thousands). Paul Graham started Y Combinator due to a combination of factors. Firstly, he wanted to stop delaying his plans to become an angel investor...
```

## Conclusion

You have successfully built a `SubQuestionQueryEngine` that can intelligently decompose complex, multi-source questions into sub-questions, route them to specialized query engines, and synthesize a coherent final answer. This pattern is powerful for building advanced RAG systems that need to reason across multiple documents or data domains.