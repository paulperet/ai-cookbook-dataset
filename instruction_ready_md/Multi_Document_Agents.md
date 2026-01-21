# Multi-Document Agents: A Hierarchical RAG Tutorial

This guide demonstrates how to build a Retrieval-Augmented Generation (RAG) system for a large collection of documents using a hierarchical agent architecture. You'll learn to create specialized "Document Agents" for each document and a top-level router that intelligently selects the appropriate agent for a given query.

## Prerequisites & Setup

First, install the required packages.

```bash
pip install llama-index
pip install llama-index-llms-anthropic
pip install llama-index-embeddings-huggingface
```

Now, configure your environment and import the necessary modules.

```python
# Required for async operations in Jupyter notebooks
import nest_asyncio
nest_asyncio.apply()

import logging
import sys
import os

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# Set your Anthropic API key
os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY_HERE"
```

## Step 1: Configure LLM and Embedding Model

We'll use Anthropic's Claude 3 Opus as our language model and a Hugging Face embedding model.

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings

# Initialize models
llm = Anthropic(temperature=0.0, model="claude-3-opus-20240229")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Configure global settings
Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
```

## Step 2: Download and Load Documents

We'll use Wikipedia articles for five major cities as our document collection.

```python
wiki_titles = ["Toronto", "Seattle", "Chicago", "Boston", "Houston"]

from pathlib import Path
import requests

# Download Wikipedia articles
for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        },
        timeout=30,
    ).json()
    
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]
    
    # Save to file
    data_path = Path("data")
    data_path.mkdir(exist_ok=True)
    
    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

print("Documents downloaded successfully.")
```

Now, load the documents using LlamaIndex's document loader.

```python
from llama_index.core import SimpleDirectoryReader

# Load all documents into a dictionary
city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()

print(f"Loaded documents for {len(city_docs)} cities.")
```

## Step 3: Build Specialized Agents for Each City

For each city, we'll create a ReAct agent equipped with two query tools: one for vector-based retrieval and another for summarization.

```python
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# Dictionary to store our city agents
agents = {}

for wiki_title in wiki_titles:
    print(f"Building agent for {wiki_title}...")
    
    # Create both vector and summary indices
    vector_index = VectorStoreIndex.from_documents(city_docs[wiki_title])
    summary_index = SummaryIndex.from_documents(city_docs[wiki_title])
    
    # Create query engines from the indices
    vector_query_engine = vector_index.as_query_engine()
    summary_query_engine = summary_index.as_query_engine()
    
    # Define tools for the agent
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=f"Useful for retrieving specific context from {wiki_title}",
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=f"Useful for summarization questions related to {wiki_title}",
            ),
        ),
    ]
    
    # Create the ReAct agent with the defined tools
    agent = ReActAgent.from_tools(
        query_engine_tools,
        llm=llm,
        verbose=True,
    )
    
    agents[wiki_title] = agent

print("All city agents built successfully.")
```

## Step 4: Create Index Nodes for Agent Routing

We'll create IndexNode objects that serve as references to our city agents. These nodes will be used by the top-level retriever to route queries to the appropriate agent.

```python
from llama_index.core.schema import IndexNode

# Create IndexNode objects for each city agent
objects = []
for wiki_title in wiki_titles:
    wiki_summary = (
        f"This content contains Wikipedia articles about {wiki_title}. Use"
        " this index if you need to lookup specific facts about"
        f" {wiki_title}.\nDo not use this index if you want to analyze"
        " multiple cities."
    )
    node = IndexNode(text=wiki_summary, index_id=wiki_title, obj=agents[wiki_title])
    objects.append(node)

print(f"Created {len(objects)} index nodes for agent routing.")
```

## Step 5: Build the Top-Level Retriever

This retriever will examine incoming queries and select the most appropriate city agent to handle them.

```python
# Create a vector index of our IndexNode objects
vector_index = VectorStoreIndex(objects=objects)

# Create a query engine that selects the best agent
query_engine = vector_index.as_query_engine(similarity_top_k=1, verbose=True)

print("Top-level retriever configured successfully.")
```

## Step 6: Test the Hierarchical RAG System

Now let's test our system with various queries to see how it routes to different agents and tools.

### Test 1: Specific Fact Retrieval (Toronto)

```python
response = query_engine.query("What is the population of Toronto?")
print("Query: What is the population of Toronto?")
print(f"Response: {response.response}\n")
```

**Expected Output:**
```
The population of Toronto is 2,794,356 as of 2021. It is the fourth-most populous city in North America.
```

### Test 2: Historical Fact Retrieval (Houston)

```python
response = query_engine.query("Who and when was Houston founded?")
print("Query: Who and when was Houston founded?")
print(f"Response: {response.response}\n")
```

**Expected Output:**
```
Houston was founded by land investors on August 30, 1836. The city was named after Sam Houston, who was serving as the president of the Republic of Texas at that time.
```

### Test 3: Summarization Request (Boston)

```python
response = query_engine.query("Summarize about the sports teams in Boston")
print("Query: Summarize about the sports teams in Boston")
print(f"Response: {response.response}\n")
```

**Expected Output Summary:**
Boston has a rich sports tradition with successful professional teams across multiple leagues including the Red Sox (MLB), Celtics (NBA), Bruins (NHL), Patriots (NFL), and Revolution (MLS), with a total of 39 championships across these leagues.

### Test 4: Positive Aspects Summary (Chicago)

```python
response = query_engine.query("Give me a summary on all the positive aspects of Chicago")
print("Query: Give me a summary on all the positive aspects of Chicago")
print(f"Response: {response.response}")
```

**Expected Output Summary:**
Chicago is praised for its diverse economy, cultural vibrancy, tourist attractions, transportation infrastructure, educational institutions, and cosmopolitan character.

## How It Works

1. **Query Reception**: The top-level query engine receives a user query.
2. **Agent Selection**: It retrieves the most relevant IndexNode based on the query content.
3. **Agent Activation**: The selected city's ReAct agent is activated.
4. **Tool Selection**: The agent decides whether to use the vector tool (for specific facts) or summary tool (for overviews).
5. **Response Generation**: The tool processes the query and the agent formulates the final response.

## Key Benefits

- **Scalability**: Easily add new documents by creating additional specialized agents.
- **Efficiency**: Each agent only needs to process queries relevant to its domain.
- **Flexibility**: Different agents can use different tools or configurations as needed.
- **Maintainability**: Agents can be updated or replaced independently.

## Next Steps

To extend this system:
1. Add more document types beyond city descriptions
2. Implement caching for frequently asked questions
3. Add cross-document comparison capabilities
4. Integrate with real-time data sources

This hierarchical approach provides a robust foundation for building RAG systems with large, diverse document collections.