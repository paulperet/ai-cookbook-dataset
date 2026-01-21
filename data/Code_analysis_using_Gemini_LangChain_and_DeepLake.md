# Code Analysis with Gemini API, LangChain, and DeepLake

This guide demonstrates how to build a code analysis system using Google's Gemini API, LangChain, and DeepLake. You will learn how to load a code repository, create a searchable vector database, and set up a Retrieval-Augmented Generation (RAG) chain to answer technical questions about the codebase.

## Prerequisites

Ensure you have the following installed and configured before you begin.

### 1. Install Required Libraries

Run the following command to install the necessary Python packages.

```bash
pip install -q -U langchain-google-genai langchain-deeplake langchain langchain-text-splitters langchain-community
```

### 2. Import Modules

```python
from glob import glob
from IPython.display import Markdown, display

from langchain.document_loaders import TextLoader
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_deeplake.vectorstores import DeeplakeVectorStore
```

### 3. Configure Your Gemini API Key

You need a Gemini API key. Store it in your environment variables. If you are using Google Colab, you can use a secret named `GEMINI_API_KEY`.

```python
import os
from google.colab import userdata

GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
```

## Step 1: Prepare the Code Repository

You will analyze a real-world codebase. For this example, you'll use the `langchain-google` repository, which contains integrations for Gemini and other Google services.

Clone the repository.

```bash
git clone https://github.com/langchain-ai/langchain-google
```

This tutorial focuses specifically on the Gemini API integration files within the repository.

Define a pattern to match the relevant Python files.

```python
repo_match = "langchain-google/libs/genai/langchain_google_genai**/*.py"
```

## Step 2: Load and Split the Code Files

To process the code effectively, you need to load each file and split it into meaningful chunks. Using `RecursiveCharacterTextSplitter` with the `Language.PYTHON` specification ensures the splitter understands Python syntax, preventing it from breaking code in the middle of a function or class.

```python
docs = []
for file in glob(repo_match, recursive=True):
    loader = TextLoader(file, encoding='utf-8')
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=2000,
        chunk_overlap=0
    )
    docs.extend(loader.load_and_split(splitter))
```

**Why this matters:** The `Language` enum provides language-specific separators (like `\nclass` and `\ndef` for Python). This results in cleaner document chunks that preserve logical code structures.

You can inspect the separators used for Python:

```python
RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)
```

## Step 3: Create the Vector Database with DeepLake

Now, you will create a vector database to store the embedded document chunks. This enables semantic search over the codebase.

### 3.1 Define the Database Path

For this example, you'll use an in-memory DeepLake dataset. The `mem://` prefix specifies this.

```python
dataset_path = 'mem://deeplake/langchain_google'
```

### 3.2 Initialize the Embedding Model

Use the Gemini Embeddings model to convert text into vector representations.

```python
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
```

### 3.3 Populate the Database

Create the vector store by passing your documents and embedding function to DeepLake.

```python
db = DeeplakeVectorStore.from_documents(
    dataset_path=dataset_path,
    embedding=embeddings,
    documents=docs,
    overwrite=True
)
```

This process should complete within a few seconds.

## Step 4: Set Up the Question-Answering Chain

With the database ready, you can now configure a retrieval system and a language model to answer questions.

### 4.1 Configure the Retriever

Create a retriever from the vector store. Configure it to use cosine similarity and return the top 20 most relevant document chunks.

```python
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 20
```

### 4.2 Initialize the Chat Model

Use a Gemini model for generating answers. This example uses the `gemini-3-flash-preview` model for its speed and capability.

```python
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
```

### 4.3 Create the QA Chain

Assemble the `RetrievalQA` chain, which combines the retriever and the LLM. This chain will fetch relevant context from the database and use the LLM to synthesize an answer.

```python
qa = RetrievalQA.from_llm(llm, retriever=retriever)
```

## Step 5: Query the Codebase

You can now ask questions about the code. Let's create a helper function to format the answers nicely.

```python
def call_qa_chain(prompt):
    response = qa.invoke(prompt)
    display(Markdown(response["result"]))
```

### Example Queries and Results

**1. Show the class hierarchy for `_BaseGoogleGenerativeAI`.**

```python
call_qa_chain("Show hierarchy for _BaseGoogleGenerativeAI. Do not show content of classes.")
```
*Output:*
```
_BaseGoogleGenerativeAI
    └── BaseModel
```

**2. What is the return type of the embedding models?**

```python
call_qa_chain("What is the return type of embedding models.")
```
*Output Summary:*
The `GoogleGenerativeAIEmbeddings` class has two main methods:
*   `embed_query(text)`: Returns a `List[float]`.
*   `embed_documents(texts)`: Returns a `List[List[float]]`.

**3. What classes are related to Attributed Question and Answering (AQA)?**

```python
call_qa_chain("What classes are related to Attributed Question and Answering.")
```
*Output Summary:*
The system identifies several related classes, including the main `GenAIAqa` class, its input/output dataclasses (`AqaInput`, `AqaOutput`), and supporting internal classes like `_AqaModel` and `GroundedAnswer`.

**4. What are the dependencies of the `GenAIAqa` class?**

```python
call_qa_chain("What are the dependencies of the GenAIAqa class?")
```
*Output Summary:*
The answer lists dependencies such as LangChain's `RunnableSerializable`, the `google.ai.generativelanguage` client library, and internal helper modules.

## Summary

You have successfully built a code analysis system that can answer complex questions about a software repository. The workflow involved:
1.  **Loading and Splitting:** Intelligently chunking source code while preserving its structure.
2.  **Embedding and Indexing:** Creating a searchable vector database using Gemini embeddings and DeepLake.
3.  **Retrieval and Generation:** Setting up a RAG pipeline that retrieves relevant code snippets and uses a Gemini model to generate precise answers.

This integration showcases how Gemini API works seamlessly with LangChain for advanced document processing and question-answering tasks.

## Next Steps

This tutorial covered a foundational RAG use case. You can explore more advanced applications, such as:
*   Using `ConversationalRetrievalChain` for multi-turn chat with your codebase.
*   Persisting the DeepLake dataset to cloud storage for larger projects.
*   Implementing agents that can use this system as a tool for code generation or refactoring tasks.