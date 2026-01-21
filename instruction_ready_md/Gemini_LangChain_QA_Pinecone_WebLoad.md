# Guide: Building a Question-Answering System with Gemini, LangChain, and Pinecone

This guide walks you through building a Retrieval-Augmented Generation (RAG) application. You will create a system that answers questions using custom data from a website by combining Google's Gemini models, LangChain's orchestration framework, and Pinecone's vector database.

## Prerequisites

*   A **Google AI API Key** for Gemini models.
*   A **Pinecone API Key** and account.
*   This tutorial assumes you are running the code in a Google Colab environment, where secrets can be managed via `userdata`.

## Setup and Installation

Begin by installing the necessary Python libraries.

```bash
!pip install --quiet -U langchain
!pip install --quiet -U langchain-google-genai
!pip install --quiet -U langchain-pinecone
!pip install --quiet -U pinecone
!pip install --quiet -U langchain-community
!pip install --quiet -U bs4
```

Next, configure your API keys as environment variables.

```python
import os
from google.colab import userdata

# Set Google AI (Gemini) API Key
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Set Pinecone API Key
PINECONE_API_KEY = userdata.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
```

## 1. Import Required Libraries

Import all necessary modules from LangChain, Pinecone, and Google's integration packages.

```python
from langchain import hub
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as pc
from pinecone import ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
```

## 2. Build the Retriever

The retriever's job is to find relevant information from your custom data based on a user's query. This involves loading data, creating embeddings (vector representations), and storing them for efficient search.

### Step 2.1: Load and Parse Website Data

Use LangChain's `WebBaseLoader` to fetch content from a webpage. For this example, we'll use a blog post about Gemini.

```python
# Load data from a specific URL
loader = WebBaseLoader("https://blog.google/technology/ai/google-gemini-ai/")
docs = loader.load()
```

Often, you only need a specific section of a document. You can extract it using standard Python string operations and convert it back into LangChain's `Document` format.

```python
# Extract the relevant portion of the text
text_content = docs[0].page_content
text_content_1 = text_content.split("code, audio, image and video.",1)[1]
final_text = text_content_1.split("Cloud TPU v5p",1)[0]

# Convert the extracted text back into a Document object
docs = [Document(page_content=final_text, metadata={"source": "local"})]
```

### Step 2.2: Initialize the Embedding Model

We'll use Google's `gemini-embedding-001` model to convert our text data into numerical vectors (embeddings).

```python
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
```

### Step 2.3: Store Embeddings in Pinecone

Pinecone is a vector database that allows for fast, similarity-based search. We need to create an index (a collection of vectors) and populate it with our document embeddings.

First, initialize the Pinecone client and check if our desired index exists. If not, we'll create it.

```python
# Initialize Pinecone client
pine_client = pc(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "langchain-demo"

# Create the index if it doesn't exist
if index_name not in pine_client.list_indexes().names():
    print("Creating index")
    pine_client.create_index(
        name=index_name,
        metric="cosine",       # Similarity metric
        dimension=3072,        # Dimension of Gemini embeddings
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(pine_client.describe_index(index_name))
```

Now, create the vector store by generating embeddings for our documents and inserting them into the Pinecone index.

```python
vectorstore = PineconeVectorStore.from_documents(docs, gemini_embeddings, index_name=index_name)
```

### Step 2.4: Create the Retriever Object

Finally, create a retriever interface from the vector store. This object will be used to fetch relevant document chunks for any given query.

```python
retriever = vectorstore.as_retriever()

# Test the retriever
print(len(retriever.invoke("MMLU")))  # Should return a positive number
```

## 3. Build the Generator (LLM Chain)

The generator takes the user's question and the relevant context retrieved by the retriever, formats them into a prompt, and sends it to a Large Language Model (LLM) to generate an answer.

### Step 3.1: Initialize the LLM

We'll use the `gemini-2.0-flash` model for its speed and summarization capabilities.

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
# You can configure parameters like temperature here:
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
```

### Step 3.2: Create the Prompt Template

Define a template that instructs the LLM on how to use the provided context to answer the question.

```python
llm_prompt_template = """You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:"""

llm_prompt = PromptTemplate.from_template(llm_prompt_template)
```

### Step 3.3: Assemble the RAG Chain

We'll use LangChain Expression Language (LCEL) to create a "stuff" chain. This chain:
1.  Retrieves relevant documents.
2.  Formats them into a single string.
3.  Passes the formatted context and the user's question to the prompt.
4.  Sends the prompt to the LLM.
5.  Parses the LLM's output into a clean string.

```python
# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the chain using LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | llm_prompt
    | llm
    | StrOutputParser()
)
```

## 4. Query the System

Your RAG application is ready! Use the `invoke` method on the chain to ask a question. The system will retrieve relevant context from the website data and generate an answer.

```python
from IPython.display import Markdown

answer = rag_chain.invoke("What is Gemini?")
Markdown(answer)
```

**Example Output:**
> Gemini is a state-of-the-art, natively multimodal AI model developed by Google. It is designed to understand and reason about various inputs, including text, images, audio, and video, from the ground up. Optimized for different sizes like Ultra, Pro, and Nano, Gemini can efficiently run on a wide range of devices, from data centers to mobile phones. It demonstrates sophisticated reasoning capabilities, excelling at complex tasks, extracting insights, and generating high-quality code. Gemini has surpassed current state-of-the-art performance on numerous benchmarks, including outperforming human experts on MMLU.

## Summary and Next Steps

You have successfully built a question-answering system using the RAG architecture with Gemini, LangChain, and Pinecone. This system can now answer questions based on the specific context you provided from a website.

**What can you do next?**
*   **Use Different Data Sources:** Replace the `WebBaseLoader` with other LangChain document loaders for PDFs, Word documents, or databases.
*   **Experiment with Chunking:** Use `RecursiveCharacterTextSplitter` for more sophisticated document splitting before creating embeddings.
*   **Tune the Chain:** Adjust the prompt template, LLM parameters (like `temperature`), or the number of documents the retriever fetches.
*   **Build an Interface:** Package this chain into a web application using frameworks like Streamlit or Gradio.

This pattern provides a robust foundation for building AI applications that leverage private or domain-specific knowledge.