# Building a Multi-Document Query Engine with LlamaIndex

This guide demonstrates how to handle complex queries spanning multiple documents using LlamaIndex's `SubQuestionQueryEngine`. You'll learn to break down broad questions into targeted sub-queries and synthesize answers from different data sources.

## Prerequisites

First, install the required packages:

```bash
pip install llama-index
pip install llama-index-llms-anthropic
pip install llama-index-embeddings-huggingface
```

## Setup

### 1. Configure API Key

Set your Anthropic API key to use Claude models:

```python
import os

os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"
```

### 2. Initialize Models

We'll use Claude 3 Opus as our LLM and a Hugging Face embedding model:

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

llm = Anthropic(temperature=0.0, model="claude-opus-4-1")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
```

### 3. Configure Global Settings

Set the default models and chunk size for all LlamaIndex operations:

```python
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
```

### 4. Enable Async Operations in Notebooks

*Note: This step is only necessary in Jupyter notebooks.*

```python
import nest_asyncio
import logging
import sys

nest_asyncio.apply()

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
```

## Load and Index Documents

### 5. Download Sample Data

We'll use Uber and Lyft's 2021 10-K SEC filings as example documents. Place these PDF files in your working directory:
- `lyft_2021.pdf`
- `uber_2021.pdf`

### 6. Load Documents

```python
from llama_index.core import SimpleDirectoryReader

lyft_docs = SimpleDirectoryReader(input_files=["lyft_2021.pdf"]).load_data()
uber_docs = SimpleDirectoryReader(input_files=["uber_2021.pdf"]).load_data()

print(f"Loaded Lyft 10-K with {len(lyft_docs)} pages")
print(f"Loaded Uber 10-K with {len(uber_docs)} pages")
```

### 7. Create Vector Indexes

Create separate vector indexes for each company's document:

```python
from llama_index.core import VectorStoreIndex

# Process first 100 pages for efficiency
lyft_index = VectorStoreIndex.from_documents(lyft_docs[:100])
uber_index = VectorStoreIndex.from_documents(uber_docs[:100])
```

## Create Query Engines

### 8. Initialize Basic Query Engines

Create query engines from each index with retrieval settings:

```python
lyft_engine = lyft_index.as_query_engine(similarity_top_k=5)
uber_engine = uber_index.as_query_engine(similarity_top_k=5)
```

### 9. Test Individual Query Engines

Verify each engine works correctly by querying them separately:

```python
# Query Lyft data
response = await lyft_engine.aquery(
    "What is the revenue of Lyft in 2021? Answer in millions with page reference"
)
print(f"Lyft Revenue: {response.response}")

# Query Uber data
response = await uber_engine.aquery(
    "What is the revenue of Uber in 2021? Answer in millions, with page reference"
)
print(f"Uber Revenue: {response.response}")
```

## Build the Sub-Question Query Engine

### 10. Create Query Tools

Wrap each query engine as a tool with descriptive metadata:

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description="Provides information about Lyft financials for year 2021"
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description="Provides information about Uber financials for year 2021"
        ),
    ),
]
```

### 11. Initialize SubQuestionQueryEngine

Create the multi-document query engine that can decompose complex questions:

```python
from llama_index.core.query_engine import SubQuestionQueryEngine

sub_question_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools
)
```

## Execute Complex Queries

### 12. Query Across Multiple Documents

Now you can ask questions that require information from both documents:

```python
# Compare revenue growth between companies
response = await sub_question_query_engine.aquery(
    "Compare revenue growth of Uber and Lyft from 2020 to 2021"
)
print(f"Revenue Comparison:\n{response.response}\n")

# Compare investment strategies
response = await sub_question_query_engine.aquery(
    "Compare the investments made by Uber and Lyft"
)
print(f"Investment Comparison:\n{response.response}")
```

## How It Works

The `SubQuestionQueryEngine` follows this process:

1. **Question Decomposition**: The LLM analyzes your complex query and breaks it into simpler sub-questions
2. **Tool Selection**: Each sub-question is routed to the appropriate query engine tool
3. **Parallel Execution**: Sub-queries are executed concurrently against their respective documents
4. **Answer Synthesis**: Results from all sub-queries are combined into a coherent final answer

## Key Benefits

- **Handles Complex Queries**: Automatically decomposes multi-faceted questions
- **Parallel Processing**: Executes sub-queries simultaneously for faster responses
- **Tool-Based Routing**: Intelligently routes questions to the most relevant data source
- **Synthesized Answers**: Combines information from multiple sources into a single response

This approach enables you to build sophisticated query systems that can answer complex, cross-document questions with minimal manual intervention.