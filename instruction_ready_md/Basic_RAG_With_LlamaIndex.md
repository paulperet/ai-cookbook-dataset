# Building a Basic RAG Pipeline with LlamaIndex

This guide walks you through creating a Retrieval-Augmented Generation (RAG) pipeline using LlamaIndex. You'll learn how to set up a language model and embedding model, load and index documents, and query them to get intelligent, context-aware answers.

## Prerequisites

Ensure you have the following installed. Run these commands in your terminal or notebook environment.

```bash
pip install llama-index
pip install llama-index-llms-anthropic
pip install llama-index-embeddings-huggingface
```

## Step 1: Configure Your API Key

First, set your Anthropic API key as an environment variable to authenticate requests to the Claude model.

```python
import os

os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY_HERE"
```

## Step 2: Initialize the LLM and Embedding Model

You'll use Claude 3 Opus as the language model and a Hugging Face embedding model for creating vector representations of your text.

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

# Initialize the Claude 3 Opus model with deterministic output
llm = Anthropic(temperature=0.0, model="claude-opus-4-1")

# Initialize the embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
```

## Step 3: Configure Global Settings

LlamaIndex uses a global `Settings` object to manage default configurations. Here, you set the LLM, embedding model, and chunk size for document processing.

```python
from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512  # Size of text chunks for indexing
```

## Step 4: Download the Sample Data

For this tutorial, you'll use Paul Graham's essay as sample data. The following commands create a directory and download the text file.

```bash
mkdir -p 'data/paul_graham/'
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'
```

## Step 5: Load the Documents

Import the `SimpleDirectoryReader` to load all text files from your data directory.

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/paul_graham").load_data()
```

## Step 6: Create a Vector Store Index

Index the loaded documents. This step processes the text, generates embeddings using your configured model, and stores them in a vector index for efficient retrieval.

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)
```

## Step 7: Create a Query Engine

The query engine is your interface for asking questions. It retrieves the most relevant text chunks (here, the top 3) and uses the LLM to synthesize an answer.

```python
query_engine = index.as_query_engine(similarity_top_k=3)
```

## Step 8: Query Your Data

Now, test the pipeline by asking a question about the content.

```python
response = query_engine.query("What did author do growing up?")
print(response)
```

**Expected Output:**
```
Based on the information provided, the author worked on two main things outside of school before college: writing and programming.

For writing, he wrote short stories as a beginning writer, though he felt they were awful, with hardly any plot and just characters with strong feelings.

In terms of programming, in 9th grade he tried writing his first programs on an IBM 1401 computer that his school district used. He and his friend got permission to use it, programming in an early version of Fortran using punch cards. However, he had difficulty figuring out what to actually do with the computer at that stage given the limited inputs available.
```

## Summary

You've successfully built a basic RAG pipeline. The system retrieved relevant passages from Paul Graham's essay and used Claude 3 Opus to generate a coherent answer. You can extend this pipeline by adding more documents, experimenting with different LLMs or embedding models, and adjusting retrieval parameters like `similarity_top_k`.