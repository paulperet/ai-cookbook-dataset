# Building a Multi-Document RAG Agent with LlamaIndex and Mistral AI

This guide demonstrates how to build a ReAct agent using LlamaIndex and Mistral AI to answer complex questions across multiple documents. You'll create separate vector indexes for three annual reports and equip an agent with tools to query each one, enabling it to synthesize information from different sources.

## Prerequisites

Ensure you have the following installed. Run these commands in your terminal or notebook environment.

```bash
pip install llama-index-core
pip install llama-index-embeddings-mistralai
pip install llama-index-llms-mistralai
pip install llama-index-readers-file
pip install mistralai pypdf
```

## Step 1: Configure the Mistral AI Services

First, import the necessary modules and configure the Mistral AI LLM and embedding model. You'll need a valid Mistral AI API key.

```python
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core.settings import Settings

# Replace with your actual API key
api_key = "YOUR_MISTRAL_API_KEY"

# Initialize the LLM and embedding model
llm = MistralAI(api_key=api_key, model="mistral-large-latest")
embed_model = MistralAIEmbedding(model_name='mistral-embed', api_key=api_key)

# Set them as the global defaults for LlamaIndex
Settings.llm = llm
Settings.embed_model = embed_model
```

## Step 2: Download the Sample Documents

You will use three Lyft annual reports (10-K forms) from 2020 to 2022 as your dataset.

```python
!wget "https://www.dropbox.com/scl/fi/ywc29qvt66s8i97h1taci/lyft-10k-2020.pdf?rlkey=d7bru2jno7398imeirn09fey5&dl=0" -q -O ./lyft_10k_2020.pdf
!wget "https://www.dropbox.com/scl/fi/lpmmki7a9a14s1l5ef7ep/lyft-10k-2021.pdf?rlkey=ud5cwlfotrii6r5jjag1o3hvm&dl=0" -q -O ./lyft_10k_2021.pdf
!wget "https://www.dropbox.com/scl/fi/iffbbnbw9h7shqnnot5es/lyft-10k-2022.pdf?rlkey=grkdgxcrib60oegtp4jn8hpl8&dl=0" -q -O ./lyft_10k_2022.pdf
```

## Step 3: Create Vector Indexes and Query Engines

For each document, you will:
1. Load the PDF text.
2. Create a vector index from the document.
3. Create a query engine from that index.

This process embeds the document content and prepares a retriever that can fetch relevant text chunks based on a query.

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Process the 2020 report
lyft_2020_docs = SimpleDirectoryReader(input_files=["./lyft_10k_2020.pdf"]).load_data()
lyft_2020_index = VectorStoreIndex.from_documents(lyft_2020_docs)
lyft_2020_engine = lyft_2020_index.as_query_engine()

# Process the 2021 report
lyft_2021_docs = SimpleDirectoryReader(input_files=["./lyft_10k_2021.pdf"]).load_data()
lyft_2021_index = VectorStoreIndex.from_documents(lyft_2021_docs)
lyft_2021_engine = lyft_2021_index.as_query_engine()

# Process the 2022 report
lyft_2022_docs = SimpleDirectoryReader(input_files=["./lyft_10k_2022.pdf"]).load_data()
lyft_2022_index = VectorStoreIndex.from_documents(lyft_2022_docs)
lyft_2022_engine = lyft_2022_index.as_query_engine()
```

Let's test one of the query engines to ensure it works.

```python
response = lyft_2022_engine.query("What was Lyft's profit in 2022?")
print(response)
```

Expected output:
```
Lyft did not make a profit in 2022. Instead, they incurred a net loss of $1,584,511.
```

## Step 4: Create Tools for the Agent

To enable the agent to use these query engines, you must wrap each one in a `QueryEngineTool`. The tool's metadata (name and description) helps the LLM understand when to use it.

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_2020_engine,
        metadata=ToolMetadata(
            name="lyft_2020_10k_form",
            description="Annual report of Lyft's financial activities in 2020",
        ),
    ),
    QueryEngineTool(
        query_engine=lyft_2021_engine,
        metadata=ToolMetadata(
            name="lyft_2021_10k_form",
            description="Annual report of Lyft's financial activities in 2021",
        ),
    ),
    QueryEngineTool(
        query_engine=lyft_2022_engine,
        metadata=ToolMetadata(
            name="lyft_2022_10k_form",
            description="Annual report of Lyft's financial activities in 2022",
        ),
    ),
]
```

## Step 5: Initialize the ReAct Agent

Create a ReAct agent using the tools you just defined. The agent will use the LLM to reason step-by-step, deciding which tool to call and how to combine the results.

```python
from llama_index.core.agent import ReActAgent

lyft_agent = ReActAgent.from_tools(query_engine_tools, llm=llm, verbose=True)
```

## Step 6: Query the Agent

Now you can ask the agent complex questions that may require information from multiple documents.

### Example 1: Query a Single Document

First, ask a question that should be answered by a single tool.

```python
response = lyft_agent.chat("What are the risk factors in 2022?")
print(response)
```

The agent will use the `lyft_2022_10k_form` tool and return a summary of the risk factors mentioned in that year's report.

### Example 2: Compare Information Across Years

Ask a question that requires fetching and comparing data from two different tools.

```python
response = lyft_agent.chat("What is Lyft's profit in 2022 vs 2020?")
print(response)
```

The agent will:
1. Use the `lyft_2020_10k_form` tool to find profit/loss for 2020.
2. Use the `lyft_2022_10k_form` tool to find profit/loss for 2022.
3. Synthesize both results into a comparative answer.

### Example 3: Analyze Trends Over Time

Ask about changes in a specific area, like Research & Development (R&D).

```python
response = lyft_agent.chat("What did Lyft do in R&D in 2022 versus 2021?")
print(response)
```

The agent will query the 2022 and 2021 tools, retrieve the relevant text about R&D expenses and activities, and provide a comparison.

## Conclusion

You have successfully built a ReAct agent capable of answering complex, multi-document questions using LlamaIndex and Mistral AI. The key steps were:

1.  **Setting up** the Mistral AI LLM and embedding service.
2.  **Indexing** individual documents into separate vector stores.
3.  **Wrapping** each query engine into a tool with descriptive metadata.
4.  **Creating** an agent that can reason about and use these tools.

This pattern is extensible. You can add more documents by creating additional tools, or even create hierarchical agents where a query engine tool is itself an agent for more complex sub-tasks.