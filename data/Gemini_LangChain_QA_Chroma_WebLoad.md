# Guide: Building a Question-Answering App with Gemini, LangChain, and Chroma

This guide walks you through creating a Retrieval-Augmented Generation (RAG) application. You will build a system that answers questions using context from a website by combining Google's Gemini model, the LangChain framework, and the Chroma vector database.

## Prerequisites

Ensure you have a Google AI API key. Store it securely as you will need it to authenticate with the Gemini models.

## Setup and Installation

Begin by installing the required Python packages.

```bash
pip install langchain-core==0.1.23 langchain==0.1.1 langchain-google-genai==0.0.6
pip install -U langchain-community==0.0.20 chromadb bs4
```

Now, import the necessary modules.

```python
from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
```

## Step 1: Configure Your API Key

Set your Google AI API key as an environment variable. Replace `'YOUR_API_KEY'` with your actual key.

```python
GOOGLE_API_KEY = 'YOUR_API_KEY'  # Replace with your actual key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
```

## Step 2: Understand the RAG Architecture

Our application uses a two-stage RAG pipeline:
1.  **Retriever:** Fetches relevant text snippets from our private data source (a website) based on the user's question.
2.  **Generator:** Uses the Gemini LLM to synthesize an answer, using the retrieved snippets as context.

## Step 3: Build the Retriever

The retriever is responsible for finding relevant information from our data store.

### 3.1 Load and Parse Website Data

We'll use LangChain's `WebBaseLoader` to fetch content from a webpage. For this example, we'll use a Google AI blog post about Gemini.

```python
# Load the webpage content
loader = WebBaseLoader("https://blog.google/technology/ai/google-gemini-ai/")
docs = loader.load()
```

Often, you only need a specific section of a webpage. We'll extract the relevant portion and convert it back into LangChain's `Document` format.

```python
# Extract the text content
text_content = docs[0].page_content

# Select the relevant portion of the text (adjust splits based on the page structure)
text_content_1 = text_content.split("code, audio, image and video.", 1)[1]
final_text = text_content_1.split("Cloud TPU v5p", 1)[0]

# Create a new Document object with the extracted text
docs = [Document(page_content=final_text, metadata={"source": "local"})]
```

### 3.2 Initialize the Embedding Model

We need to convert text into numerical vectors (embeddings) for efficient searching. We'll use Gemini's embedding model.

```python
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
```

### 3.3 Create and Persist the Vector Store

We'll store the embeddings of our website text in a Chroma database. This allows for fast similarity searches later.

```python
# Create a Chroma vector store from the documents and persist it to disk
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=gemini_embeddings,
    persist_directory="./chroma_db"  # Saves the database locally
)
```

### 3.4 Create the Retriever Object

Now, we load the persisted vector store and create a retriever interface from it. This retriever will find the most relevant document chunk for a given query.

```python
# Load the vector store from disk
vectorstore_disk = Chroma(
    persist_directory="./chroma_db",
    embedding_function=gemini_embeddings
)

# Create a retriever. We set k=1 to retrieve only the top match.
retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 1})

# Test the retriever
test_docs = retriever.get_relevant_documents("MMLU")
print(f"Retrieved {len(test_docs)} document(s).")
```

## Step 4: Build the Generator

The generator uses the Gemini LLM to produce answers, enhanced by the context provided by the retriever.

### 4.1 Initialize the Gemini LLM

We'll use the `gemini-2.0-flash` model for its speed and capability.

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
```

### 4.2 Create the Prompt Template

Define a prompt that instructs the model to answer questions concisely using the provided context.

```python
llm_prompt_template = """You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.\n
Question: {question} \nContext: {context} \nAnswer:"""

llm_prompt = PromptTemplate.from_template(llm_prompt_template)
```

### 4.3 Assemble the RAG Chain

We use LangChain Expression Language (LCEL) to create a pipeline that:
1.  Takes a question.
2.  Retrieves relevant context.
3.  Formats the context and question into the prompt.
4.  Sends the prompt to the LLM.
5.  Parses the output.

```python
# Helper function to format multiple documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | llm_prompt
    | llm
    | StrOutputParser()
)
```

## Step 5: Query the Application

Your RAG application is ready. Invoke the chain with any question related to the content of the website you used.

```python
# Ask a question
question = "What is Gemini?"
answer = rag_chain.invoke(question)
print(f"Q: {question}")
print(f"A: {answer}")
```

**Example Output:**
```
Q: What is Gemini?
A: Gemini is a flexible, state-of-the-art AI model designed by Google. It is natively multimodal, trained to seamlessly understand and reason about various inputs like text, images, and audio simultaneously. Gemini possesses sophisticated reasoning capabilities and can understand, explain, and generate high-quality code. It is optimized to efficiently run on everything from data centers to mobile devices. Its first version, Gemini 1.0, is available in three sizes: Ultra, Pro, and Nano, tailored for different tasks.
```

## Conclusion

You have successfully built a question-answering application using the RAG pattern. This application leverages private data (a website) to provide accurate, context-aware answers through the Gemini LLM. You can extend this by:
*   Using different data sources (PDFs, databases).
*   Implementing more sophisticated retrieval strategies.
*   Adding a chat memory for conversational context.