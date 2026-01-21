# Gemini, LlamaIndex & Chroma: Build a Question-Answering App

This guide walks you through building a Retrieval-Augmented Generation (RAG) application. You'll use Google's Gemini model to answer questions based on content from a website, using LlamaIndex for data orchestration and Chroma as a vector database.

## Overview

The application has two core components:
1.  **Retriever:** Fetches relevant context from your data source (a website) based on a user's question.
2.  **Generator:** Uses the Gemini LLM to synthesize an answer using the retrieved context.

## Prerequisites & Setup

### 1. Install Required Libraries
Run the following commands to install the necessary packages.

```bash
pip install -q -U llama-index
pip install -q -U llama-index-llms-google-genai
pip install -q -U llama-index-embeddings-google-genai
pip install -q -U llama-index-readers-web
pip install -q -U llama-index-vector-stores-chroma
pip install -q -U chromadb
pip install -q -U bs4
```

### 2. Configure Your API Key
Set your `GOOGLE_API_KEY` as an environment variable. If you're running this in Google Colab, you can use a secret.

```python
import os
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
```

### 3. Import Libraries
Import all the modules you'll need for the tutorial.

```python
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate
import chromadb
```

---

## Step 1: Build the Retriever

The retriever is responsible for fetching relevant snippets from your data. This involves loading data, creating embeddings (vector representations), and storing them for efficient search.

### Step 1.1: Load and Parse Website Data
First, use LlamaIndex's `SimpleWebPageReader` to fetch content from a URL. Then, parse the HTML to extract clean text using `BeautifulSoup`.

```python
# Load raw HTML content from a webpage
web_documents = SimpleWebPageReader().load_data(
    ["https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/"]
)
html_content = web_documents[0].text

# Parse HTML to extract text from paragraph tags
soup = BeautifulSoup(html_content, 'html.parser')
p_tags = soup.find_all('p')
text_content = ""
for each in p_tags:
    text_content += each.text + "\n"

# Convert the cleaned text back into a LlamaIndex Document object
documents = [Document(text=text_content)]
```

### Step 1.2: Initialize the LLM and Embedding Model
Configure the Gemini models you'll use for generating text and creating embeddings.

```python
# Initialize the text generation model (Gemini 2.5 Flash)
llm = GoogleGenAI(model_name="models/gemini-2.5-flash")

# Initialize the embedding model
gemini_embedding_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001")
```

### Step 1.3: Store Data in Chroma Vector Database
Now, create a vector store to persist the embeddings of your website data. This involves setting up a Chroma client, collection, and connecting it to LlamaIndex.

```python
# Create a persistent ChromaDB client and collection
client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = client.get_or_create_collection("quickstart")

# Create a LlamaIndex vector store interface for Chroma
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create a storage context for the index
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Set global settings for the LLM and embedding model
Settings.llm = llm
Settings.embed_model = gemini_embedding_model

# Create and persist a vector index from your documents
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
```

### Step 1.4: Test the Retriever
Verify that your retriever works by loading the persisted index and performing a test query.

```python
# Load the index from disk
load_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = load_client.get_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

# Create a simple query engine and test it
test_query_engine = index.as_query_engine()
response = test_query_engine.query("AIME")
display(Markdown(response.response))
```
**Expected Output Snippet:**
> Gemini 2.5 Pro leads in math and science benchmarks like AIME 2025.

---

## Step 2: Build the Generator

The generator uses the Gemini LLM to produce final answers. You'll create a custom prompt template that instructs the model to use the context provided by the retriever.

### Step 2.1: Create a Custom Prompt Template
Define a template that formats the user's question and the retrieved context into an effective prompt for the LLM.

```python
template = (
    """You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.\n
Question: {query_str} \nContext: {context_str} \nAnswer:"""
)
llm_prompt = PromptTemplate(template)
```

### Step 2.2: Query the Model
Create a query engine that uses your custom prompt, then ask a question about your data.

```python
# Create a query engine using the custom prompt template
query_engine = index.as_query_engine(text_qa_template=llm_prompt)

# Ask a question
response = query_engine.query("What is Gemini?")
display(Markdown(response.response))
```
**Expected Output Snippet:**
> Gemini is a natively multimodal AI model developed by Google. It is pre-trained on different modalities and fine-tuned with additional multimodal data. Gemini's capabilities are state of the art in nearly every domain, exceeding current results on many academic benchmarks. It can understand, explain, and generate high-quality code in popular programming languages. Gemini is being rolled out across Google products like Bard, Pixel, and Search, and will be available to developers and enterprise customers via the Gemini API.

---

## Summary

You've successfully built a RAG application that:
1.  Ingests and processes content from a website.
2.  Creates and stores vector embeddings using Gemini's embedding model in ChromaDB.
3.  Retrieves relevant context based on a user's query.
4.  Generates accurate, concise answers using the Gemini LLM with a custom prompt.

This architecture can be adapted to use private documents, databases, or other data sources by swapping the data loader. Explore the [LlamaIndex documentation](https://docs.llamaindex.ai/) and [Google AI for developers](https://ai.google.dev/) site to build more advanced applications.