# Building a ReAct Agent with LlamaIndex

This guide demonstrates how to build a Reasoning and Acting (ReAct) agent using LlamaIndex. You will learn to create an agent that uses tools to perform calculations and answer complex questions by querying financial documents.

## Prerequisites

First, install the required packages.

```bash
pip install llama-index
pip install llama-index-llms-anthropic
pip install llama-index-embeddings-huggingface
```

## Initial Setup

### Configure Environment and Imports

Since some components are async-first, we need to apply `nest_asyncio` for use in a notebook environment. Then, set your API key and import necessary modules.

```python
import nest_asyncio
nest_asyncio.apply()

import os

# Set your Anthropic API key
os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"

from IPython.display import HTML, display
```

### Configure LLM and Embedding Model

We'll use Anthropic's Claude 3 Opus as the language model and a Hugging Face embedding model.

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

llm = Anthropic(temperature=0.0, model="claude-3-opus-20240229")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
```

Apply these settings globally within LlamaIndex.

```python
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
```

## Part 1: ReAct Agent with Simple Tools

In this section, you'll create an agent that can perform basic arithmetic using custom tools.

### Step 1: Define Your Tools

Tools are functions the agent can call. We'll create two simple arithmetic tools.

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b

# Create tool objects from the functions
add_tool = FunctionTool.from_defaults(fn=add)
multiply_tool = FunctionTool.from_defaults(fn=multiply)
```

### Step 2: Instantiate the ReAct Agent

Create the agent by passing it the list of available tools and the configured LLM.

```python
agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)
```

### Step 3: Query the Agent

Now, ask the agent a question that requires multi-step reasoning. The `verbose=True` flag lets you see the agent's internal thought process.

```python
response = agent.chat("What is 20+(2*4)? Calculate step by step.")
```

**Agent's Reasoning Process (Verbose Output):**
```
Thought: I need to use the multiply tool to calculate 2*4 first, then use the add tool to add that result to 20.
Action: multiply
Action Input: {'a': 2, 'b': 4}
Observation: 8
Thought: Now I can use the add tool to add 20 to 8.
Action: add
Action Input: {'a': 20, 'b': 8}
Observation: 28
Thought: I can answer without using any more tools.
Answer: 20+(2*4) equals 28.
```

**Final Answer:**
```python
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```
> 20+(2*4) equals 28.
>
> To calculate it step-by-step:
> 1. First, calculate 2*4 which equals 8.
> 2. Then, add 20 to that result of 8.
> 3. 20 + 8 = 28
>
> Therefore, 20+(2*4) = 28.

### Step 4: Inspect the Agent's Prompts (Optional)

You can examine the system prompt that guides the agent's reasoning and tool usage.

```python
prompt_dict = agent.get_prompts()
for k, v in prompt_dict.items():
    print(f"Prompt: {k}\n\nValue: {v.template}\n")
```

## Part 2: ReAct Agent with RAG Tools

Now, let's build a more powerful agent that can answer questions by querying financial documents (10-K filings) using Retrieval-Augmented Generation (RAG).

### Step 1: Prepare the Data

First, download the sample 10-K filings for Uber and Lyft.

```bash
mkdir -p 'data/10k/'
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'
```

### Step 2: Load Documents and Build Indexes

Load the PDFs and create vector indexes for efficient retrieval.

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# Load the documents
lyft_docs = SimpleDirectoryReader(input_files=["./data/10k/lyft_2021.pdf"]).load_data()
uber_docs = SimpleDirectoryReader(input_files=["./data/10k/uber_2021.pdf"]).load_data()

# Build vector indexes
lyft_index = VectorStoreIndex.from_documents(lyft_docs)
uber_index = VectorStoreIndex.from_documents(uber_docs)
```

### Step 3: Create Query Engines and Tools

Convert each index into a query engine, then wrap those engines into tools the agent can use.

```python
# Create query engines
lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

# Wrap engines into tools with descriptive metadata
query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]
```

### Step 4: Create the Document Agent

Instantiate a new ReAct agent with the document query tools.

```python
agent = ReActAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
)
```

### Step 5: Query the Document Agent

#### Example 1: Simple Fact Retrieval

Ask a straightforward question about Lyft's financials.

```python
response = agent.chat("What was Lyft's revenue growth in 2021?")
```

**Agent's Process:**
```
Thought: I need to use a tool to help me answer the question.
Action: lyft_10k
Action Input: {'input': "What was Lyft's revenue growth in 2021?"}
Observation: Lyft's revenue grew by $843.6 million, or 36%, in 2021 compared to 2020...
Thought: The provided observation directly answers the question... I have enough information...
Answer: Lyft's revenue grew by $843.6 million, or 36%, in 2021 compared to 2020...
```

**Final Answer:**
```python
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```
> Lyft's revenue grew by $843.6 million, or 36%, in 2021 compared to 2020. The growth was mainly driven by a significant increase in Active Riders as COVID-19 vaccines became more widely available and communities reopened. However, the revenue growth was partially offset by higher driver incentives which were recorded as a reduction to revenue.

#### Example 2: Comparative Analysis

Ask a more complex question requiring data from both documents.

```python
response = agent.chat(
    "Compare and contrast the revenue growth of Uber and Lyft in 2021, then give an analysis"
)
```

**Agent's Process:**
The agent will:
1.  Use the `lyft_10k` tool to get Lyft's growth figures.
2.  Use the `uber_10k` tool to get Uber's growth figures.
3.  Synthesize the information into a comparative analysis.

**Final Answer:**
```python
display(HTML(f'<p style="font-size:20px">{response.response}</p>'))
```
> In comparing Lyft and Uber's revenue growth in 2021:
>
> Lyft's revenue grew by $843.6 million, or 36%, from $2.4 billion in 2020 to $3.2 billion in 2021.
>
> Uber's revenue grew by a much larger $8.5 billion, or 57%, from $14.1 billion in 2020 to $22.6 billion in 2021.
>
> So while both companies saw strong revenue growth as they recovered from the impacts of the pandemic in 2020, Uber's growth was significantly higher than Lyft's in both dollar and percentage terms.
>
> A few key factors likely contributed to Uber's higher growth rate:
> 1) Uber has a more diversified business with significant food delivery and freight segments in addition to ridesharing. These segments grew rapidly in 2021.
> 2) Uber operates in many more international markets than Lyft. As global travel recovered in 2021, this provided a boost to Uber.
> 3) Uber's overall scale is much larger than Lyft's, so similar percentage growth translates to much larger absolute growth.

## Summary

You have successfully built two types of ReAct agents:
1.  A **calculation agent** that uses simple function tools to perform step-by-step arithmetic.
2.  A **document analysis agent** that uses RAG-based query tools to retrieve and synthesize information from financial documents.

The ReAct framework enables the agent to reason about which tools to use and in what sequence, making it capable of handling complex, multi-step queries. You can extend this pattern by adding more specialized tools to create agents for various domains and tasks.