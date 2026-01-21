# Building a RAG Pipeline with Mistral AI and Haystack

This guide walks you through building two essential pipelines for working with documents: an indexing pipeline to create embeddings from web content, and a retrieval-augmented generation (RAG) pipeline to chat with that content using Mistral AI models.

## Prerequisites

First, install the required packages:

```bash
pip install mistral-haystack trafilatura
```

Then, set up your Mistral API key:

```python
import os
from getpass import getpass

os.environ["MISTRAL_API_KEY"] = getpass("Mistral API Key:")
```

## Part 1: Indexing URLs with Mistral Embeddings

In this section, you'll create a pipeline that fetches web content, converts it to documents, generates embeddings using Mistral's embedding model, and stores them in a vector database.

### Step 1: Import Required Components

```python
from haystack import Pipeline
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.mistral.document_embedder import MistralDocumentEmbedder
```

### Step 2: Initialize Components

Create the document store and pipeline components:

```python
# Initialize the document store (InMemoryDocumentStore is simplest for getting started)
document_store = InMemoryDocumentStore()

# Create pipeline components
fetcher = LinkContentFetcher()
converter = HTMLToDocument()
embedder = MistralDocumentEmbedder()
writer = DocumentWriter(document_store=document_store)
```

> **Note:** `InMemoryDocumentStore` is used here for simplicity. For production use, consider switching to one of Haystack's [supported vector databases](https://haystack.deepset.ai/integrations?type=Document+Store) like Weaviate, Chroma, or AstraDB.

### Step 3: Build the Indexing Pipeline

Assemble the components into a pipeline with proper connections:

```python
indexing = Pipeline()

# Add components to the pipeline
indexing.add_component(name="fetcher", instance=fetcher)
indexing.add_component(name="converter", instance=converter)
indexing.add_component(name="embedder", instance=embedder)
indexing.add_component(name="writer", instance=writer)

# Connect the components
indexing.connect("fetcher", "converter")
indexing.connect("converter", "embedder")
indexing.connect("embedder", "writer")
```

### Step 4: Run the Indexing Pipeline

Now you can index content from any URLs. Let's index two Mistral AI blog posts:

```python
urls = ["https://mistral.ai/news/la-plateforme/", "https://mistral.ai/news/mixtral-of-experts"]
indexing.run({"fetcher": {"urls": urls}})
```

The pipeline will:
1. Fetch the HTML content from the URLs
2. Convert HTML to document objects
3. Generate embeddings using Mistral's embedding model
4. Store the documents with their embeddings in the document store

## Part 2: Chatting with Indexed Content Using RAG

Now that you have indexed documents, you can create a RAG pipeline to query them using Mistral's generative models.

### Step 1: Define the Chat Template

First, create a template that structures how the model should use retrieved documents:

```python
from haystack.dataclasses import ChatMessage

chat_template = """Answer the following question based on the contents of the documents.\n
                Question: {{query}}\n
                Documents: {{documents[0].content}}
                """
user_message = ChatMessage.from_user(chat_template)
```

### Step 2: Import RAG Pipeline Components

```python
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack_integrations.components.embedders.mistral.text_embedder import MistralTextEmbedder
from haystack_integrations.components.generators.mistral import MistralChatGenerator
```

### Step 3: Initialize RAG Components

```python
# Create components for the RAG pipeline
text_embedder = MistralTextEmbedder()
retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=1)
prompt_builder = ChatPromptBuilder(template=user_message, variables=["query", "documents"], required_variables=["query", "documents"])
llm = MistralChatGenerator(model='mistral-small', streaming_callback=print_streaming_chunk)
```

Key points about this setup:
- `MistralTextEmbedder` embeds the user's question for similarity search
- `InMemoryEmbeddingRetriever` finds the most relevant document (top_k=1)
- `MistralChatGenerator` uses the `mistral-small` model with streaming enabled

### Step 4: Build the RAG Pipeline

```python
rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)

# Connect the components
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.messages")
```

### Step 5: Query the Pipeline

Now you can ask questions about the indexed content:

```python
question = "What generative endpoints does the Mistral platform have?"

messages = [ChatMessage.from_user(chat_template)]

result = rag_pipeline.run(
    {
        "text_embedder": {"text": question},
        "prompt_builder": {"template": messages, "query": question},
        "llm": {"generation_kwargs": {"max_tokens": 165}},
    },
    include_outputs_from=["text_embedder", "retriever", "llm"],
)
```

The pipeline will:
1. Embed your question using `MistralTextEmbedder`
2. Retrieve the most relevant document from the indexed content
3. Build a prompt combining your question and the retrieved document
4. Generate a response using the `mistral-small` model

**Example Response:**
```
The Mistral platform has three generative endpoints: mistral-tiny, mistral-small, and mistral-medium. Each endpoint serves a different model with varying performance and language support. Mistral-tiny serves Mistral 7B Instruct v0.2, which is the most cost-effective and only supports English. Mistral-small serves Mixtral 8x7B, which supports English, French, Italian, German, Spanish, and code. Mistral-medium serves a prototype model with higher performance, also supporting the same languages and code as Mistral-small. Additionally, the platform offers an embedding endpoint called Mistral-embed, which serves an embedding model with a 1024 embedding dimension designed for retrieval capabilities.
```

## Summary

You've successfully built two pipelines:
1. **Indexing Pipeline**: Fetches web content, generates embeddings, and stores them for retrieval
2. **RAG Pipeline**: Answers questions by retrieving relevant documents and generating responses with Mistral AI models

This setup provides a foundation for building more complex document-based Q&A systems. You can extend it by:
- Adding more documents to the index
- Adjusting the retrieval parameters (e.g., increasing `top_k`)
- Customizing the prompt template for different use cases
- Switching to different Mistral models based on your needs