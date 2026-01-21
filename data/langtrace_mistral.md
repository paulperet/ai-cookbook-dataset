# RAG Observability with Mistral AI and Langtrace

This guide demonstrates how to build a simple Retrieval-Augmented Generation (RAG) application that allows you to chat with the United States Constitution. We will integrate observability using Langtrace to monitor and trace all operations, from document ingestion to AI inference.

## Prerequisites

Before you begin, ensure you have the following:
- A [Mistral AI API key](https://console.mistral.ai/)
- A [Langtrace API key](https://langtrace.ai/)
- The URL or local path to a PDF of the U.S. Constitution (a public URL is provided in the code)

## Step 1: Environment Setup

Install the required Python libraries.

```bash
pip install mistralai langtrace-python-sdk chromadb pypdf langchain langchain-community
```

## Step 2: Initialize Clients and SDK

Import the necessary modules and initialize the Mistral AI client, ChromaDB client, and the Langtrace SDK with your API keys.

```python
import chromadb
from mistralai import Mistral
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from langtrace_python_sdk import langtrace, with_langtrace_root_span

# Initialize Langtrace for observability
langtrace.init(api_key='<your_langtrace_api_key>')

# Initialize the Mistral AI client
mistral = Mistral(api_key='<your_mistral_api_key>')

# Initialize a persistent ChromaDB client
client = chromadb.Client()
```

**Note:** Replace `<your_langtrace_api_key>` and `<your_mistral_api_key>` with your actual keys.

## Step 3: Load and Split the PDF Document

We'll use LangChain's `PyPDFLoader` to load the PDF and a `RecursiveCharacterTextSplitter` to break it into manageable chunks for retrieval.

```python
def load_and_split_pdf(pdf_path):
    """
    Load a PDF from a given path or URL and split it into text chunks.
    """
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    # Split documents into chunks of 1000 characters with a 200-character overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(data)
    return chunks
```

## Step 4: Set Up the Vector Database

Create a ChromaDB collection to store the document chunks. We'll use the default embedding function for simplicity.

```python
def setup_chroma():
    """
    Create or retrieve a ChromaDB collection for storing document embeddings.
    """
    return client.get_or_create_collection(
        name="mistral-rag",
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    )

def add_documents_to_collection(collection, chunks):
    """
    Insert document chunks into the specified ChromaDB collection.
    """
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk.page_content],
            metadatas=[{"source": chunk.metadata["source"]}],
            ids=[str(i)],
        )
```

## Step 5: Implement the RAG Query Function

This function handles the core RAG logic: retrieving relevant context and querying the Mistral AI model.

```python
def query_pdf(collection, query):
    """
    Query the vector database for relevant context and generate an answer using Mistral AI.
    """
    # 1. Retrieve the top 3 most relevant document chunks
    results = collection.query(query_texts=[query], n_results=3)

    # 2. Construct the prompt with the retrieved context
    context = "\n".join(results["documents"][0])
    prompt = f"""Use the following context to answer the question. If you cannot answer based on the context, say so.

Context:
{context}

Question: {query}

Answer:"""

    # 3. Send the prompt to Mistral AI for completion
    response = mistral.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

## Step 6: Execute the Full Pipeline with Observability

Wrap the main execution logic in a function decorated with `@with_langtrace_root_span`. This creates a trace in Langtrace that captures the entire workflow.

```python
@with_langtrace_root_span("main")
def main():
    print("Creating collection...")
    collection = setup_chroma()

    print("Loading and splitting PDF...")
    # Using a public URL for the U.S. Constitution PDF
    pdf_url = "https://www.govinfo.gov/content/pkg/CDOC-112hdoc129/pdf/CDOC-112hdoc129.pdf"
    chunks = load_and_split_pdf(pdf_url)

    print("Adding documents to collection...")
    add_documents_to_collection(collection, chunks)

    print("Querying PDF...")
    answer = query_pdf(collection, "What is the purpose of the constitution?")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
```

## Step 7: View Traces in Langtrace

After running the script, navigate to your [Langtrace dashboard](https://langtrace.ai/). You should see a detailed trace named "main" that includes:

1.  **The full RAG application span structure**, showing the document loading, chunking, database insertion, and query steps.
2.  **Detailed spans for Mistral AI inference**, where you can inspect the exact prompt (including the context retrieved from ChromaDB) and the model's response.

This observability allows you to debug performance, verify retrieved context, and optimize your RAG pipeline.

## Summary

You have successfully built an observable RAG application. The integration with Langtrace provides full visibility into each step, from document processing to AI-generated answers, enabling you to monitor, debug, and improve your system effectively.