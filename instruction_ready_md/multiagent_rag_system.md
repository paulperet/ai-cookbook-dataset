# Building a Multi-Agent RAG System

## Introduction

This guide walks you through building a **Multi-Agent Retrieval-Augmented Generation (RAG) System** where multiple specialized agents collaborate to handle complex tasks. You'll create a system with a central orchestrator that delegates work to three specialized agents:

1. **Web Search Agent** - Searches the web for current information
2. **Retriever Agent** - Queries structured knowledge bases
3. **Image Generation Agent** - Creates images from text prompts

## Prerequisites

Before starting, ensure you have:
- Basic understanding of agents and RAG systems
- A Hugging Face account with API access
- Python 3.8 or higher

## Setup

### 1. Install Required Packages

```bash
pip install smolagents markdownify duckduckgo-search spaces gradio-tools langchain langchain-community langchain-huggingface faiss-cpu --upgrade
```

### 2. Authenticate with Hugging Face

```python
from huggingface_hub import notebook_login

notebook_login()
```

### 3. Initialize the Base Model

We'll use Qwen2.5-72B-Instruct as our base LLM for all agents:

```python
from smolagents import InferenceClientModel

model_id = "Qwen/Qwen2.5-72B-Instruct"
model = InferenceClientModel(model_id)
```

## Step 1: Create the Web Search Agent

The Web Search Agent handles internet queries using DuckDuckGo search and webpage access tools.

### 1.1 Build the Agent

```python
from smolagents import ToolCallingAgent, ManagedAgent, DuckDuckGoSearchTool, VisitWebpageTool

# Create the web search agent with search tools
web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    model=model
)

# Wrap it for management by the central agent
managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="search_agent",
    description="Runs web searches for you. Give it your query as an argument.",
)
```

**Why this works**: The `ToolCallingAgent` is ideal for web search tasks because its JSON action format handles simple arguments well in sequential chains, making it perfect for search-and-retrieve workflows.

## Step 2: Create the Retriever Agent

The Retriever Agent queries two knowledge bases: Hugging Face documentation and PEFT GitHub issues.

### 2.1 Set Up the First Knowledge Base (Hugging Face Docs)

First, load and process the Hugging Face documentation dataset:

```python
import datasets
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

# Load the documentation dataset
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

# Convert to LangChain documents
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

# Configure text splitting
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained("thenlper/gte-small"),
    chunk_size=200,
    chunk_overlap=20,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

# Split documents and remove duplicates
print("Splitting documents...")
docs_processed = []
unique_texts = {}
for doc in tqdm(source_docs):
    new_docs = text_splitter.split_documents([doc])
    for new_doc in new_docs:
        if new_doc.page_content not in unique_texts:
            unique_texts[new_doc.page_content] = True
            docs_processed.append(new_doc)

# Create embeddings and vector store
print("Embedding documents...")
embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
huggingface_doc_vector_db = FAISS.from_documents(
    documents=docs_processed,
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)
```

### 2.2 Create the Retriever Tool Class

Now create a reusable tool class for querying vector databases:

```python
from smolagents import Tool
from langchain_core.vectorstores import VectorStore

class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vectordb.similarity_search(
            query,
            k=7,
        )

        return "\nRetrieved documents:\n" + "".join(
            [
                f"===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )
```

### 2.3 Create the First Retriever Tool

```python
huggingface_doc_retriever_tool = RetrieverTool(huggingface_doc_vector_db)
```

### 2.4 Set Up the Second Knowledge Base (PEFT Issues)

Now create a second retriever for PEFT GitHub issues:

```python
# Set your GitHub token (store securely in practice)
GITHUB_ACCESS_TOKEN = "your_github_token_here"

from langchain.document_loaders import GitHubIssuesLoader

# Load PEFT issues from GitHub
loader = GitHubIssuesLoader(
    repo="huggingface/peft", 
    access_token=GITHUB_ACCESS_TOKEN, 
    include_prs=False, 
    state="all"
)
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
chunked_docs = splitter.split_documents(docs)

# Create vector store
peft_issues_vector_db = FAISS.from_documents(chunked_docs, embedding=embedding_model)

# Create the second retriever tool
peft_issues_retriever_tool = RetrieverTool(peft_issues_vector_db)
```

### 2.5 Build the Retriever Agent

Combine both retriever tools into a single agent:

```python
retriever_agent = ToolCallingAgent(
    tools=[huggingface_doc_retriever_tool, peft_issues_retriever_tool], 
    model=model, 
    max_iterations=4, 
    verbose=2
)

managed_retriever_agent = ManagedAgent(
    agent=retriever_agent,
    name="retriever_agent",
    description="Retrieves documents from the knowledge base for you that are close to the input query. Give it your query as an argument. The knowledge base includes Hugging Face documentation and PEFT issues.",
)
```

## Step 3: Create the Image Generation Agent

This agent optimizes prompts and generates images using specialized tools.

### 3.1 Set Up the Tools

```python
from transformers import load_tool, CodeAgent
from smolagents import Tool

# Create a prompt optimization tool
prompt_generator_tool = Tool.from_space(
    "sergiopaniego/Promptist", 
    name="generator_tool", 
    description="Optimizes user input into model-preferred prompts"
)

# Create an image generation tool
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)
```

### 3.2 Build the Image Generation Agent

```python
image_generation_agent = CodeAgent(
    tools=[prompt_generator_tool, image_generation_tool], 
    model=model
)

managed_image_generation_agent = ManagedAgent(
    agent=image_generation_agent,
    name="image_generation_agent",
    description="Generates images from text prompts. Give it your prompt as an argument.",
    additional_prompting="\n\nYour final answer MUST BE only the generated image location."
)
```

**Note**: We use `CodeAgent` here because image generation typically involves executing multiple tool calls in one sequence.

## Step 4: Create the Central Manager Agent

The manager agent orchestrates all specialized agents and decides which one to use based on the query.

```python
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[
        managed_web_agent, 
        managed_retriever_agent, 
        managed_image_generation_agent
    ],
    additional_authorized_imports=["time", "datetime", "PIL"],
)
```

**How it works**: The manager analyzes each query and delegates to the appropriate specialized agent based on the task type.

## Step 5: Test the System

Now let's test our multi-agent system with different types of queries.

### 5.1 Test Web Search Capabilities

```python
result = manager_agent.run("How many years ago was Stripe founded?")
print(f"Answer: {result}")
```

**Expected behavior**: The manager should delegate this to the Web Search Agent, which will search DuckDuckGo and return the answer.

### 5.2 Test Image Generation

```python
result = manager_agent.run(
    "Improve this prompt, then generate an image of it.", 
    prompt='A rabbit wearing a space suit'
)
print(f"Image generated at: {result}")

# Display the image
from IPython.display import Image, display
display(Image(filename=result))
```

**Expected behavior**: The manager delegates to the Image Generation Agent, which first optimizes the prompt using Promptist, then generates an image.

### 5.3 Test Documentation Retrieval

```python
result = manager_agent.run("How can I push a model to the Hub?")
print(f"Answer: {result}")
```

**Expected behavior**: The manager uses the Retriever Agent to query the Hugging Face documentation knowledge base.

### 5.4 Test Issue Retrieval

```python
result = manager_agent.run("How do you combine multiple adapters in peft?")
print(f"Answer: {result}")
```

**Expected behavior**: The manager uses the Retriever Agent to query the PEFT issues knowledge base.

## Conclusion

You've successfully built a multi-agent RAG system that demonstrates:

1. **Specialized Agent Creation**: Each agent has specific tools and capabilities
2. **Intelligent Orchestration**: The manager agent routes queries to the appropriate specialist
3. **Flexible Knowledge Integration**: Combining web search, structured documentation, and image generation
4. **Scalable Architecture**: Easy to add more agents or knowledge bases

### Key Takeaways

- **ToolCallingAgent** works well for sequential tool execution (like web search)
- **CodeAgent** is better for parallel or complex tool execution (like image generation)
- **ManagedAgent** wrapper enables agent-to-agent communication
- The central manager pattern allows for clean separation of concerns

### Next Steps

Consider enhancing your system by:
1. Adding more specialized agents (e.g., code execution, data analysis)
2. Implementing agent memory for conversation history
3. Adding validation and error handling between agents
4. Creating a user interface for easier interaction

This architecture provides a foundation for building sophisticated AI systems that can handle diverse tasks through specialized collaboration.