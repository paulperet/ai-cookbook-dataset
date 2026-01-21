# Building a Local RAG Ebook Librarian with LlamaIndex

## Introduction

This guide walks you through building a lightweight, locally-run Retrieval-Augmented Generation (RAG) system that acts as a "librarian" for your personal ebook collection. By leveraging open-source tools like LlamaIndex and Ollama, you'll create a system that can answer questions about your ebooks without relying on cloud services.

## Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher
- Basic familiarity with Python and command-line tools
- A system capable of running local LLMs (Apple Silicon Macs work well)

## Step 1: Install Dependencies

First, install the required Python packages:

```bash
pip install llama-index EbookLib html2text llama-index-embeddings-huggingface llama-index-llms-ollama
```

## Step 2: Set Up Ollama

Ollama allows you to run large language models locally. Follow these steps to install and configure it:

### Install System Dependencies
```bash
apt install pciutils lshw
```

### Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Start the Ollama Service
Run this in a separate terminal or as a background process:
```bash
ollama serve &
```

### Download the Llama 2 Model
```bash
ollama pull llama2
```

## Step 3: Prepare Your Test Library

Create a sample ebook library with two classic novels from Project Gutenberg:

```bash
mkdir -p "./test/library/jane-austen"
mkdir -p "./test/library/victor-hugo"
wget https://www.gutenberg.org/ebooks/1342.epub.noimages -O "./test/library/jane-austen/pride-and-prejudice.epub"
wget https://www.gutenberg.org/ebooks/135.epub.noimages -O "./test/library/victor-hugo/les-miserables.epub"
```

This creates a directory structure with two EPUB files that will serve as your test library.

## Step 4: Load Your Ebooks with LlamaIndex

Now you'll use LlamaIndex to load and process your ebook files. LlamaIndex's `SimpleDirectoryReader` makes this straightforward:

```python
from llama_index.core import SimpleDirectoryReader

# Create a loader that recursively searches for EPUB files
loader = SimpleDirectoryReader(
    input_dir="./test/",
    recursive=True,
    required_exts=[".epub"],
)

# Load the documents
documents = loader.load_data()
```

The `load_data()` method converts your EPUB files into `Document` objects that LlamaIndex can work with. Note that the documents haven't been chunked yet—that happens during the indexing phase.

## Step 5: Configure the Embedding Model

Instead of using OpenAI's embeddings (the LlamaIndex default), you'll use a lightweight, open-source model from Hugging Face:

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Use a compact but effective embedding model
embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

This model provides good performance while being small enough to run locally.

## Step 6: Index Your Documents

Indexing prepares your documents for efficient querying by creating vector embeddings:

```python
from llama_index.core import VectorStoreIndex

# Create an index with your documents and custom embedding model
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embedding_model,
)
```

By default, `VectorStoreIndex` uses an in-memory store and chunks documents with a size of 1024 tokens and an overlap of 20 tokens. This creates a searchable index of your ebook content.

## Step 7: Configure the Query Engine

Now you'll set up the query interface using Llama 2 via Ollama:

```python
from llama_index.llms.ollama import Ollama

# Initialize the Llama 2 model through Ollama
llama = Ollama(
    model="llama2",
    request_timeout=40.0,
)

# Create a query engine from your index
query_engine = index.as_query_engine(llm=llama)
```

The `request_timeout` parameter ensures the model has enough time to generate responses.

## Step 8: Query Your Library

With everything set up, you can now ask questions about your ebook collection:

```python
# Ask about available books
response = query_engine.query("What are the titles of all the books available?")
print(response)
```

Example output:
```
Based on the context provided, there are two books available:
1. "Pride and Prejudice" by Jane Austen
2. "Les Misérables" by Victor Hugo
```

```python
# Ask about specific content
response = query_engine.query("Who is the main character of 'Pride and Prejudice'?")
print(response)
```

Example output:
```
The main character of 'Pride and Prejudice' is Elizabeth Bennet.
```

## Conclusion

You've successfully built a local RAG system that can answer questions about your ebook library. This solution runs entirely on your local machine, protecting your privacy and eliminating cloud service dependencies.

## Future Improvements

Here are some ways you could enhance this system:

### 1. Add Citations
Implement citation tracking to verify where information comes from and reduce hallucinations.

### 2. Incorporate Metadata
Extend the system to read metadata from library management tools like Calibre, which could provide additional context like publisher information or edition details.

### 3. Optimize Indexing
Instead of re-indexing on every run, implement persistent storage for embeddings and only update when your library changes. This would significantly improve performance for larger collections.

### 4. Expand File Support
While EPUB files work well, you could extend support to other ebook formats like PDF, MOBI, or AZW3.

### 5. Add User Interface
Create a web or desktop interface to make the system more accessible to non-technical users.

The foundation you've built is flexible and can be adapted to various personal knowledge management use cases beyond ebook libraries.