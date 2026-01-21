# Adaptive RAG with LlamaIndex: A Practical Guide

User queries can range from simple to complex. Not every question requires a sophisticated Retrieval-Augmented Generation (RAG) system. Inspired by the [Adaptive RAG](https://arxiv.org/abs/2403.14403) concept, this guide demonstrates how to build a system that intelligently routes queries to the most appropriate tool—whether that's a simple LLM call, a single-document query engine, or a multi-document agent.

We'll implement this using Lyft's 10-K SEC filings from 2020, 2021, and 2022 as our document corpus. The system will use a `RouterQueryEngine` powered by MistralAI's function-calling capabilities to decide the best path for each query.

## Prerequisites & Setup

First, install the required packages.

```bash
pip install llama-index
pip install llama-index-llms-mistralai
pip install llama-index-embeddings-mistralai
```

Set your MistralAI API key as an environment variable.

```python
import os
os.environ['MISTRAL_API_KEY'] = '<YOUR MISTRAL API KEY>'
```

Now, import the necessary modules and configure the LLM and embedding model. We use `mistral-large-latest` as it supports function calling.

```python
import nest_asyncio
nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Settings

# Configure global settings
llm = MistralAI(model='mistral-large-latest')
embed_model = MistralAIEmbedding()

Settings.llm = llm
Settings.embed_model = embed_model
```

## Step 1: Download and Load the Data

We'll download three years of Lyft's 10-K filings.

```python
# Download the PDFs
!wget "https://www.dropbox.com/scl/fi/ywc29qvt66s8i97h1taci/lyft-10k-2020.pdf?rlkey=d7bru2jno7398imeirn09fey5&dl=0" -q -O ./lyft_10k_2020.pdf
!wget "https://www.dropbox.com/scl/fi/lpmmki7a9a14s1l5ef7ep/lyft-10k-2021.pdf?rlkey=ud5cwlfotrii6r5jjag1o3hvm&dl=0" -q -O ./lyft_10k_2021.pdf
!wget "https://www.dropbox.com/scl/fi/iffbbnbw9h7shqnnot5es/lyft-10k-2022.pdf?rlkey=grkdgxcrib60oegtp4jn8hpl8&dl=0" -q -O ./lyft_10k_2022.pdf

# Load the documents
lyft_2020_docs = SimpleDirectoryReader(input_files=["./lyft_10k_2020.pdf"]).load_data()
lyft_2021_docs = SimpleDirectoryReader(input_files=["./lyft_10k_2021.pdf"]).load_data()
lyft_2022_docs = SimpleDirectoryReader(input_files=["./lyft_10k_2022.pdf"]).load_data()
```

## Step 2: Create Vector Indices

Create a separate vector index for each year's document. This allows for efficient, targeted retrieval.

```python
lyft_2020_index = VectorStoreIndex.from_documents(lyft_2020_docs)
lyft_2021_index = VectorStoreIndex.from_documents(lyft_2021_docs)
lyft_2022_index = VectorStoreIndex.from_documents(lyft_2022_docs)
```

## Step 3: Create Query Engines

From each index, create a query engine. These will serve as our primary tools for answering questions about a specific year.

```python
lyft_2020_query_engine = lyft_2020_index.as_query_engine(similarity_top_k=5)
lyft_2021_query_engine = lyft_2021_index.as_query_engine(similarity_top_k=5)
lyft_2022_query_engine = lyft_2022_index.as_query_engine(similarity_top_k=5)
```

We also need a simple LLM query engine for general knowledge questions unrelated to our documents.

```python
from llama_index.core.query_engine import CustomQueryEngine

class LLMQueryEngine(CustomQueryEngine):
    """A simple query engine that uses only the LLM."""
    llm: MistralAI

    def custom_query(self, query_str: str):
        response = self.llm.complete(query_str)
        return str(response)

llm_query_engine = LLMQueryEngine(llm=llm)
```

## Step 4: Build an Agent for Complex Queries

For questions that span multiple years, we need an agent that can coordinate calls to multiple tools. We'll create a `FunctionCallingAgentWorker` with the three yearly query engines as its tools.

```python
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# Define tools for the agent
agent_tools = [
    QueryEngineTool(
        query_engine=lyft_2020_query_engine,
        metadata=ToolMetadata(
            name="lyft_2020_10k_form",
            description="Annual report of Lyft's financial activities in 2020",
        ),
    ),
    QueryEngineTool(
        query_engine=lyft_2021_query_engine,
        metadata=ToolMetadata(
            name="lyft_2021_10k_form",
            description="Annual report of Lyft's financial activities in 2021",
        ),
    ),
    QueryEngineTool(
        query_engine=lyft_2022_query_engine,
        metadata=ToolMetadata(
            name="lyft_2022_10k_form",
            description="Annual report of Lyft's financial activities in 2022",
        ),
    )
]

# Create the agent
agent_worker = FunctionCallingAgentWorker.from_tools(
    agent_tools,
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=True,
)
agent = AgentRunner(agent_worker)
```

## Step 5: Define the Complete Toolset for the Router

Our `RouterQueryEngine` will choose from this set of tools. It includes the three single-year tools, the multi-year agent, and the general LLM engine.

```python
query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_2020_query_engine,
        metadata=ToolMetadata(
            name="lyft_2020_10k_form",
            description="Queries related to only 2020 Lyft's financial activities.",
        ),
    ),
    QueryEngineTool(
        query_engine=lyft_2021_query_engine,
        metadata=ToolMetadata(
            name="lyft_2021_10k_form",
            description="Queries related to only 2021 Lyft's financial activities.",
        ),
    ),
    QueryEngineTool(
        query_engine=lyft_2022_query_engine,
        metadata=ToolMetadata(
            name="lyft_2022_10k_form",
            description="Queries related to only 2022 Lyft's financial activities.",
        ),
    ),
    QueryEngineTool(
        query_engine=agent,
        metadata=ToolMetadata(
            name="lyft_2020_2021_2022_10k_form",
            description=(
                "Useful for queries that span multiple years from 2020 to 2022 for Lyft's financial activities."
            )
        )
    ),
    QueryEngineTool(
        query_engine=llm_query_engine,
        metadata=ToolMetadata(
            name="general_queries",
            description=(
                "Provides information about general queries other than Lyft."
            )
        )
    )
]
```

## Step 6: Create the Router Query Engine

The `RouterQueryEngine` uses an `LLMSingleSelector` to evaluate the user's query and pick the most appropriate tool from the list above.

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=query_engine_tools,
    verbose=True
)
```

## Step 7: Query the System

Let's test our adaptive RAG system with various queries.

### Simple Queries

**General Knowledge Question**
```python
response = query_engine.query("What is the capital of France?")
print(response.response)
```
**Output:**
```
The capital of France is Paris. Known as the "City of Light," Paris is famous for its iconic landmarks like the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and more. It is a global center for art, fashion, gastronomy, and culture.
```
*The router correctly selected the `general_queries` tool (the LLM) for this unrelated question.*

**Single-Year Lyft Question (2022)**
```python
response = query_engine.query("What did Lyft do in R&D in 2022?")
print(response.response)
```
**Output:**
```
In 2022, Lyft made focused investments in research and development, though the specific details of these investments are not provided in the context. These investments are part of their ongoing commitment to launch new innovations on their platform. Additionally, Lyft completed multiple strategic acquisitions, one of which is PBSC Urban Solutions. This acquisition added over 100,000 bikes to their bikeshare systems across 46 markets in 15 countries.
```
*The router selected the `lyft_2022_10k_form` tool.*

**Single-Year Lyft Question (2021)**
```python
response = query_engine.query("What did Lyft do in R&D in 2021?")
print(response.response)
```
**Output:**
```
In 2021, Lyft's research and development expenses increased slightly by $2.8 million. This increase was primarily due to a $51.6 million rise in stock-based compensation. However, there was also a $25.4 million benefit from a restructuring event in the second quarter of 2020, which included a stock-based compensation benefit and severance and benefits costs that did not recur in 2021. These increases were offset by a $37.5 million decrease in personnel-related costs and a $4.6 million decrease in autonomous vehicle research costs.
```
*The router selected the `lyft_2021_10k_form` tool.*

**Single-Year Lyft Question (2020)**
```python
response = query_engine.query("What did Lyft do in R&D in 2020?")
print(response.response)
```
**Output:**
```
In 2020, Lyft continued to invest heavily in research and development as part of its mission to revolutionize transportation. The company made focused investments to launch new innovations on its platform. One notable example is the acquisition of Flexdrive, LLC, a longstanding partner in Lyft's Express Drive program. This acquisition is expected to contribute to the growth of Lyft's business and help expand the range of its use cases. Additionally, Lyft invested in the expansion of its network of Light Vehicles and autonomous open platform technology to remain at the forefront of transportation innovation. Despite the impact of COVID-19, Lyft plans to continue investing in the future, both organically and through acquisitions of complementary businesses.
```
*The router selected the `lyft_2020_10k_form` tool.*

### Complex Query

**Multi-Year Comparative Question**
```python
response = query_engine.query("What did Lyft do in R&D in 2022 vs 2020?")
print(response.response)
```
**Output (truncated for brevity):**
```
assistant: In 2020, Lyft's Research and Development expenses were $25,376 million... In 2022, these expenses decreased to $856,777 thousand, a decrease of $55.2 million, or 6%...
```
*The router correctly identified this as a multi-year query and selected the `lyft_2020_2021_2022_10k_form` tool (the agent). The agent then called the 2020 and 2022 tools to gather information and synthesize a comparative answer.*

## Conclusion

You've successfully built an Adaptive RAG system using LlamaIndex. This system intelligently routes queries:
1. **General knowledge questions** to a standard LLM.
2. **Specific, single-document questions** to the corresponding yearly query engine.
3. **Complex, multi-document questions** to a function-calling agent that can use multiple tools.

This approach optimizes cost and latency by avoiding unnecessary RAG processing for simple queries while providing powerful multi-hop reasoning for complex ones. You can extend this pattern by adding more documents, tools, or refining the routing logic.