# Building a Claude 3 RAG Agent with LangChain v1

This guide walks you through building a Retrieval-Augmented Generation (RAG) agent using LangChain v1. We'll use Claude 3 as our LLM, Voyage AI for embeddings, and Pinecone as our vector database to create an agent capable of answering technical questions using AI research papers from arXiv.

## Prerequisites

Before starting, ensure you have the following:
- Python 3.10+
- API keys for:
  - [Anthropic Claude](https://docs.claude.com/claude/reference/getting-started-with-the-api)
  - [Pinecone](https://docs.pinecone.io/docs/quickstart)
  - [Voyage AI](https://docs.voyageai.com/install/)

## Setup

First, install the required packages:

```bash
pip install -qU \
    langchain==0.1.11 \
    langchain-core==0.1.30 \
    langchain-community==0.0.27 \
    langchain-anthropic==0.1.4 \
    langchainhub==0.1.15 \
    anthropic==0.19.1 \
    voyageai==0.2.1 \
    pinecone-client==3.1.0 \
    datasets==2.16.1
```

Configure your API keys:

```python
ANTHROPIC_API_KEY = "<YOUR_ANTHROPIC_API_KEY>"
PINECONE_API_KEY = "<YOUR_PINECONE_API_KEY>"
VOYAGE_API_KEY = "<YOUR_VOYAGE_API_KEY>"
```

## Step 1: Load the Knowledge Dataset

We'll use a pre-chunked version of the AI ArXiv dataset containing research paper abstracts and metadata. This dataset provides the knowledge base for our RAG system.

```python
from datasets import load_dataset

# Load 20,000 chunks from the dataset
dataset = load_dataset("jamescalam/ai-arxiv2-chunks", split="train[:20000]")
print(f"Loaded dataset with {len(dataset)} chunks")
```

Each chunk contains metadata like DOI, title, authors, and the actual text content. Let's examine one sample:

```python
sample = dataset[1]
print(f"Title: {sample['title']}")
print(f"Authors: {sample['authors']}")
print(f"Chunk preview: {sample['chunk'][:200]}...")
```

## Step 2: Initialize Embeddings and Vector Database

### 2.1 Set Up Voyage AI Embeddings

We'll use Voyage AI's embedding model to convert our text chunks into vector representations.

```python
from langchain_community.embeddings import VoyageEmbeddings

embed = VoyageEmbeddings(voyage_api_key=VOYAGE_API_KEY, model="voyage-2")

# Test the embedding model
test_embedding = embed.embed_documents(["test"])
embedding_dimension = len(test_embedding[0])
print(f"Embedding dimension: {embedding_dimension}")
```

### 2.2 Configure Pinecone Vector Database

```python
from pinecone import Pinecone, ServerlessSpec
import time

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define serverless specification
spec = ServerlessSpec(cloud="aws", region="us-west-2")

# Create or connect to index
index_name = "claude-3-rag"

if index_name not in pc.list_indexes().names():
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        index_name,
        dimension=embedding_dimension,
        metric="dotproduct",
        spec=spec,
    )
    
    # Wait for index to be ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# Connect to the index
index = pc.Index(index_name)
time.sleep(1)

# Check index statistics
stats = index.describe_index_stats()
print(f"Index stats: {stats}")
```

## Step 3: Populate the Vector Database

Now we'll embed our dataset chunks and store them in Pinecone along with their metadata.

```python
from tqdm.auto import tqdm
import pandas as pd

# Convert dataset to pandas for easier batch processing
data = dataset.to_pandas()

batch_size = 100

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i + batch_size)
    
    # Get batch of data
    batch = data.iloc[i:i_end]
    
    # Generate unique IDs for each chunk
    ids = [f"{row['doi']}-{row['chunk-id']}" for _, row in batch.iterrows()]
    
    # Get text to embed
    texts = [row["chunk"] for _, row in batch.iterrows()]
    
    # Embed text
    embeds = embed.embed_documents(texts)
    
    # Prepare metadata
    metadata = [
        {
            "text": row["chunk"],
            "source": row["source"],
            "title": row["title"]
        }
        for _, row in batch.iterrows()
    ]
    
    # Add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata, strict=False))

print("Vector database populated successfully!")
```

## Step 4: Create the Search Tool

We'll create a tool that our agent can use to search the knowledge base when answering questions.

```python
from langchain.agents import tool

@tool
def arxiv_search(query: str) -> str:
    """Use this tool when answering questions about AI, machine learning, data
    science, or other technical questions that may be answered using arXiv
    papers.
    """
    # Create query vector
    query_embedding = embed.embed_query(query)
    
    # Perform similarity search
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    
    # Format results as a string
    results_str = "\n\n".join([
        match["metadata"]["text"] for match in results["matches"]
    ])
    
    return results_str

# Create tools list for the agent
tools = [arxiv_search]
```

Let's test our search tool:

```python
# Test the search tool
test_query = "What are the key features of Llama 2?"
search_results = arxiv_search.run(tool_input={"query": test_query})
print(f"Search results preview: {search_results[:500]}...")
```

## Step 5: Configure the Claude 3 LLM

Now we'll set up Claude 3 as our language model. The XML agent format works particularly well with Anthropic models.

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0,
    anthropic_api_key=ANTHROPIC_API_KEY,
    max_tokens=4096
)
```

## Step 6: Set Up the XML Agent

Anthropic models are trained to work with XML tags for tool usage. We'll use a pre-built prompt template from LangChain Hub.

```python
from langchain import hub

# Pull the XML agent conversation prompt
prompt = hub.pull("hwchase17/xml-agent-convo")
print("Prompt template loaded successfully")
```

The prompt template teaches the agent to use tools with XML syntax:
- `<tool>{tool_name}</tool>` to specify which tool to use
- `<tool_input>{input}</tool_input>` to provide input to the tool
- `<observation>{result}</observation>` for tool responses
- `<final_answer>{answer}</final_answer>` for the final response

## Step 7: Create and Run the Agent

Now we'll assemble all components into a working agent.

```python
from langchain.agents import create_xml_agent

# Create the XML agent
agent = create_xml_agent(llm, tools, prompt)

# Test the agent with a question
question = "What are the main differences between Llama 1 and Llama 2?"
response = agent.invoke({"input": question})

print("Agent response:")
print(response["output"])
```

The agent will:
1. Receive your question
2. Decide if it needs to search the knowledge base
3. Use the `arxiv_search` tool with XML syntax if needed
4. Process the search results
5. Generate a final answer based on the retrieved information

## Step 8: Interactive Conversation

You can run the agent in a loop for interactive conversations:

```python
def chat_with_agent():
    print("Starting conversation with Claude 3 RAG Agent (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        response = agent.invoke({"input": user_input})
        print(f"\nAgent: {response['output']}")

# Uncomment to start interactive chat
# chat_with_agent()
```

## Key Features of This Implementation

1. **Efficient Retrieval**: The agent searches through 20,000 AI research paper chunks to find relevant information
2. **XML Format Compatibility**: Uses Anthropic's preferred XML syntax for tool usage
3. **Scalable Architecture**: Pinecone's serverless infrastructure handles vector storage and similarity search
4. **High-Quality Embeddings**: Voyage AI embeddings provide accurate semantic search capabilities
5. **Modular Design**: Easy to swap components (LLM, embeddings, vector DB) as needed

## Troubleshooting Tips

1. **API Key Issues**: Ensure all API keys are correctly set and have sufficient credits
2. **Index Creation**: If the Pinecone index creation fails, check your region availability
3. **Embedding Dimension**: Verify the embedding dimension matches between Voyage AI and Pinecone
4. **Rate Limiting**: Add delays between API calls if you encounter rate limits
5. **Memory Management**: For larger datasets, consider streaming or chunking the embedding process

## Next Steps

To enhance this agent, consider:
1. Adding more specialized tools (e.g., code execution, web search)
2. Implementing conversation memory for multi-turn dialogues
3. Adding citation tracking to reference specific papers
4. Implementing confidence scoring for retrieved information
5. Creating a web interface for easier interaction

This RAG agent provides a solid foundation for building intelligent assistants that can leverage external knowledge sources while maintaining the conversational abilities of Claude 3.