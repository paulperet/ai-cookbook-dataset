# Building ReAct and Function-Calling Agents with Mistral AI and LlamaIndex

This guide demonstrates how to build intelligent agents using LlamaIndex and Mistral AI's LLMs. You'll learn to create both Function-Calling and ReAct agents that can use custom tools and interact with a RAG (Retrieval-Augmented Generation) pipeline.

## Prerequisites

Before starting, ensure you have the necessary packages installed:

```bash
pip install llama-index
pip install llama-index-llms-mistralai
pip install llama-index-embeddings-mistralai
```

You'll also need a Mistral AI API key. Set it as an environment variable:

```python
import os
os.environ['MISTRAL_API_KEY'] = 'your-api-key-here'
```

## 1. Setting Up the Environment

First, import the required modules and set up your environment:

```python
import json
from typing import Sequence, List

from llama_index.llms.mistralai import MistralAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool

import nest_asyncio

# Required for async operations in Jupyter environments
nest_asyncio.apply()
```

## 2. Creating Simple Calculator Tools

Let's start by defining some basic calculator functions that our agents can use:

```python
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result"""
    return a * b

def add(a: int, b: int) -> int:
    """Add two integers and return the result"""
    return a + b

# Convert functions to tools
multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)
```

## 3. Initializing the Mistral AI LLM

Initialize the Mistral AI language model that will power our agents:

```python
llm = MistralAI(model="mistral-large-latest")
```

## 4. Building a Function-Calling Agent

Function-Calling agents can directly invoke tools based on the LLM's understanding of the query.

### 4.1 Basic Function-Calling Agent

Create an agent with sequential tool calling:

```python
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    [multiply_tool, add_tool],
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=False,
)
agent = AgentRunner(agent_worker)
```

Test the agent with a mathematical query:

```python
response = agent.chat("What is (121 + 2) * 5?")
print(str(response))
```

The agent will execute the calculation step by step:
1. First, it calls the `add` function with arguments `a=121` and `b=2`
2. Then, it calls the `multiply` function with the result `123` and `b=5`
3. Finally, it returns the answer: `615`

You can inspect the tool execution sources:

```python
print(response.sources)
```

### 4.2 Parallel Function-Calling Agent

Enable parallel tool execution for improved performance:

```python
# Enable parallel function calling
agent_worker = FunctionCallingAgentWorker.from_tools(
    [multiply_tool, add_tool],
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=True,
)
agent = AgentRunner(agent_worker)
```

Test with an async query:

```python
response = await agent.achat("What is (121 * 3) + (5 * 8)?")
print(str(response))
```

With parallel execution, the agent can call both multiplication operations simultaneously before performing the addition.

## 5. Building a ReAct Agent

ReAct (Reasoning + Acting) agents use a thought-action-observation loop to solve problems step by step.

```python
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=llm, verbose=True)

response = agent.chat("What is (121 * 3) + (5 * 8)?")
print(str(response))
```

The ReAct agent will show its reasoning process:
1. **Thought**: Analyze the problem and decide which tool to use
2. **Action**: Call the appropriate tool with specific inputs
3. **Observation**: Process the tool's output
4. Repeat until the problem is solved

## 6. Building an Agent with RAG Pipeline

Now let's create a more sophisticated agent that can query documents through a RAG pipeline.

### 6.1 Setting Up the RAG Pipeline

First, download a sample document (Uber's 2021 10-K filing):

```bash
mkdir -p 'data/10k/'
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
```

Now build the RAG pipeline:

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI

# Initialize embedding and query models
embed_model = MistralAIEmbedding()
query_llm = MistralAI(model="mistral-medium")

# Load and index the document
uber_docs = SimpleDirectoryReader(
    input_files=["./data/10k/uber_2021.pdf"]
).load_data()

uber_index = VectorStoreIndex.from_documents(
    uber_docs, embed_model=embed_model
)

# Create query engine
uber_engine = uber_index.as_query_engine(similarity_top_k=3, llm=query_llm)

# Wrap the query engine as a tool
query_engine_tool = QueryEngineTool(
    query_engine=uber_engine,
    metadata=ToolMetadata(
        name="uber_10k",
        description=(
            "Provides information about Uber financials for year 2021. "
            "Use a detailed plain text question as input to the tool."
        ),
    ),
)
```

### 6.2 Function-Calling Agent with RAG

Create a Function-Calling agent that can use the RAG tool:

```python
agent_worker = FunctionCallingAgentWorker.from_tools(
    [query_engine_tool], llm=llm, verbose=True
)
agent = AgentRunner(agent_worker)
```

Query the agent about Uber's risk factors:

```python
response = agent.chat(
    "What are the risk factors for Uber in 2021?"
)
print(str(response))
```

The agent will use the RAG pipeline to retrieve relevant information from the document and provide a comprehensive answer.

### 6.3 ReAct Agent with RAG

Create a ReAct agent with the same RAG tool:

```python
agent = ReActAgent.from_tools([query_engine_tool], llm=llm, verbose=True)

response = agent.chat("What are the risk factors for Uber in 2021?")
print(str(response))
```

The ReAct agent will show its reasoning process as it decides to use the RAG tool and processes the retrieved information.

## Key Takeaways

1. **Function-Calling Agents** are efficient for straightforward tool usage where the LLM can directly map queries to function calls.

2. **ReAct Agents** provide transparent reasoning, making them ideal for complex problems requiring step-by-step analysis.

3. **RAG Integration** allows agents to access and reason over large document collections, combining LLM capabilities with specific domain knowledge.

4. **Mistral AI Integration** provides high-quality LLM performance for both reasoning and embeddings in your agent workflows.

Both agent types can be extended with additional tools, custom logic, and more sophisticated RAG pipelines to handle increasingly complex tasks.