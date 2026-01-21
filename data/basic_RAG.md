# Building RAG Systems with Mistral AI: A Comprehensive Guide

This guide walks you through implementing a basic Retrieval-Augmented Generation (RAG) system using Mistral AI. You'll learn the core concepts by building from scratch, then see how popular frameworks simplify the process.

## What is RAG?

Retrieval-Augmented Generation (RAG) combines large language models with information retrieval systems. It works in two main steps:
1. **Retrieval**: Find relevant information from a knowledge base using text embeddings stored in a vector database
2. **Generation**: Insert the retrieved context into a prompt for the LLM to generate informed responses

## Prerequisites

Before starting, ensure you have:
- Python installed
- A Mistral AI API key (get one at [console.mistral.ai](https://console.mistral.ai))

## Setup

Install the required packages:

```bash
pip install faiss-cpu==1.7.4 mistralai
```

Import the necessary libraries:

```python
from mistralai import Mistral
import requests
import numpy as np
import faiss
import os
from getpass import getpass

# Initialize the Mistral client
api_key = getpass("Enter your Mistral AI API key: ")
client = Mistral(api_key=api_key)
```

## Part 1: Building RAG from Scratch

This section helps you understand the internal workings of RAG by building it with minimal dependencies.

### Step 1: Load Your Data

We'll use Paul Graham's essay as our knowledge base:

```python
# Download the essay
response = requests.get('https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt')
text = response.text

# Save locally for reference
with open('essay.txt', 'w') as f:
    f.write(text)
```

### Step 2: Split Document into Chunks

Splitting documents into smaller chunks makes retrieval more effective. We'll use a simple character-based approach:

```python
chunk_size = 2048
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
print(f"Created {len(chunks)} chunks")
```

**Considerations for chunking:**
- **Chunk size**: Smaller chunks often improve retrieval accuracy but increase processing overhead
- **Splitting method**: Character splitting is simplest, but consider token-based splitting for API limits, sentence/paragraph splitting for coherence, or AST parsing for code

### Step 3: Create Text Embeddings

Embeddings convert text into numerical representations where similar meanings have closer proximity in vector space:

```python
def get_text_embedding(input_text):
    """Get embeddings for a single text chunk"""
    embeddings_response = client.embeddings.create(
        model="mistral-embed",
        inputs=input_text
    )
    return embeddings_response.data[0].embedding

# Create embeddings for all chunks
text_embeddings = np.array([get_text_embedding(chunk) for chunk in chunks])
print(f"Embeddings shape: {text_embeddings.shape}")
```

### Step 4: Store Embeddings in a Vector Database

FAISS provides efficient similarity search for our embeddings:

```python
# Initialize FAISS index
d = text_embeddings.shape[1]  # Embedding dimension
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)
print("Embeddings added to vector database")
```

**Vector database considerations:** Evaluate options based on speed, scalability, cloud management, filtering capabilities, and open-source vs. proprietary solutions.

### Step 5: Process User Questions

When a user asks a question, we need to embed it using the same model:

```python
question = "What were the two main things the author worked on before college?"
question_embeddings = np.array([get_text_embedding(question)])
print(f"Question embeddings shape: {question_embeddings.shape}")
```

**Advanced consideration:** For some queries, generating a hypothetical answer first (HyDE technique) can improve retrieval relevance.

### Step 6: Retrieve Relevant Context

Search the vector database for chunks similar to the question:

```python
# Find 2 most similar chunks
D, I = index.search(question_embeddings, k=2)
retrieved_chunks = [chunks[i] for i in I.tolist()[0]]

print("Retrieved chunks:")
for i, chunk in enumerate(retrieved_chunks):
    print(f"\nChunk {i+1} (first 200 chars): {chunk[:200]}...")
```

**Retrieval strategies to consider:**
- Metadata filtering before similarity search
- Statistical methods like TF-IDF or BM25
- Context expansion (retrieving parent chunks)
- Time-weighted retrieval for recency
- Document reordering to combat "lost in the middle" issues

### Step 7: Generate the Final Response

Combine the retrieved context with the question in a prompt:

```python
# Create the prompt with context
prompt = f"""
Context information is below.
---------------------
{retrieved_chunks}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {question}
Answer:
"""

def run_mistral(user_message, model="mistral-large-latest"):
    """Call Mistral's chat completion API"""
    messages = [{"role": "user", "content": user_message}]
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    return chat_response.choices[0].message.content

# Generate the answer
answer = run_mistral(prompt)
print(f"\nAnswer: {answer}")
```

**Prompt engineering tips:** Apply few-shot learning, explicit formatting instructions, and other prompting techniques to improve response quality.

## Part 2: Implementing RAG with Popular Frameworks

Now let's see how frameworks simplify RAG implementation.

### LangChain Implementation

First, install LangChain dependencies:

```bash
pip install langchain langchain-mistralai langchain_community
```

Here's the complete LangChain implementation:

```python
from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load and split documents
loader = TextLoader("essay.txt")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

# Setup embeddings and vector store
embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# Configure LLM and prompt
model = ChatMistralAI(mistral_api_key=api_key)
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# Create and run the retrieval chain
document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({
    "input": "What were the two main things the author worked on before college?"
})
print(response["answer"])
```

### LlamaIndex Implementation

Install LlamaIndex dependencies:

```bash
pip install llama-index llama-index-llms-mistralai llama-index-embeddings-mistralai
```

Implement RAG with LlamaIndex:

```python
import os
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding

# Load documents
reader = SimpleDirectoryReader(input_files=["essay.txt"])
documents = reader.load_data()

# Configure LLM and embeddings
Settings.llm = MistralAI(model="mistral-medium", api_key=api_key)
Settings.embed_model = MistralAIEmbedding(model_name='mistral-embed', api_key=api_key)

# Create index and query engine
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=2)

# Query the system
response = query_engine.query(
    "What were the two main things the author worked on before college?"
)
print(str(response))
```

### Haystack Implementation

Install Haystack dependencies:

```bash
pip install mistral-haystack
```

Implement RAG with Haystack's pipeline approach:

```python
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret
from haystack.components.builders import DynamicChatPromptBuilder
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.embedders.mistral import MistralDocumentEmbedder, MistralTextEmbedder
from haystack_integrations.components.generators.mistral import MistralChatGenerator

# Initialize document store
document_store = InMemoryDocumentStore()

# Process documents
docs = TextFileToDocument().run(sources=["essay.txt"])
split_docs = DocumentSplitter(split_by="passage", split_length=2).run(documents=docs["documents"])
embeddings = MistralDocumentEmbedder(api_key=Secret.from_token(api_key)).run(documents=split_docs["documents"])
DocumentWriter(document_store=document_store).run(documents=embeddings["documents"])

# Setup pipeline components
text_embedder = MistralTextEmbedder(api_key=Secret.from_token(api_key))
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
prompt_builder = DynamicChatPromptBuilder(runtime_variables=["documents"])
llm = MistralChatGenerator(api_key=Secret.from_token(api_key), model='mistral-small')

# Define chat template
chat_template = """Answer the following question based on the contents of the documents.
Question: {{query}}
Documents:
{% for document in documents %}
    {{document.content}}
{% endfor%}
"""
messages = [ChatMessage.from_user(chat_template)]

# Build the pipeline
rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)

# Connect components
rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

# Run the pipeline
question = "What were the two main things the author worked on before college?"
result = rag_pipeline.run({
    "text_embedder": {"text": question},
    "prompt_builder": {"template_variables": {"query": question}, "prompt_source": messages},
    "llm": {"generation_kwargs": {"max_tokens": 225}},
})

print(result["llm"]["replies"][0].content)
```

## Key Takeaways

1. **From Scratch**: Building RAG from scratch helps you understand the core components: chunking, embedding, vector storage, retrieval, and generation.

2. **Framework Benefits**: LangChain, LlamaIndex, and Haystack abstract away implementation details, providing:
   - Built-in document processors
   - Pre-configured pipelines
   - Easier experimentation with different components
   - Production-ready patterns

3. **Customization**: Regardless of approach, you can customize:
   - Chunking strategies
   - Retrieval methods
   - Prompt templates
   - LLM parameters

4. **Performance Optimization**: Consider experimenting with:
   - Different embedding models
   - Various retrieval strategies
   - Prompt engineering techniques
   - Chunk sizes and overlaps

This guide provides a foundation for building RAG systems with Mistral AI. Start with the from-scratch approach to understand the fundamentals, then use frameworks to accelerate development and scale your applications.