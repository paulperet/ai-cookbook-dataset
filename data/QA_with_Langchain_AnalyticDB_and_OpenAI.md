# Building a Question Answering System with LangChain, AnalyticDB, and OpenAI

This guide walks you through implementing a production-ready Question Answering (QA) system. You'll use LangChain to orchestrate the workflow, AnalyticDB as a vector database for your knowledge base, and OpenAI's models for generating embeddings and answers.

## Prerequisites

Before you begin, ensure you have the following:

1.  **An AnalyticDB Cloud Instance:** Set up an [AnalyticDB for PostgreSQL](https://www.alibabacloud.com/help/en/analyticdb-for-postgresql/latest/product-introduction-overview) instance.
2.  **An OpenAI API Key:** Obtain a key from your [OpenAI account](https://platform.openai.com/account/api-keys).
3.  **Environment Variables:** You will need to set several environment variables for your database connection and API key.

## Step 1: Environment Setup

First, install the required Python packages and configure your environment.

### 1.1 Install Dependencies

Run the following command to install the necessary libraries:

```bash
pip install openai tiktoken langchain psycopg2cffi
```

*   `openai`: Provides access to the OpenAI API.
*   `tiktoken`: A tokenizer for OpenAI's models.
*   `langchain`: A framework for building LLM-powered applications.
*   `psycopg2cffi`: A PostgreSQL adapter for connecting to AnalyticDB.

### 1.2 Configure Your OpenAI API Key

Your OpenAI API key is essential for generating text embeddings and powering the language model. Set it as an environment variable.

**On Linux/macOS:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Alternatively, set it directly in Python:**
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

Let's verify the key is correctly set:

```python
import os

if os.getenv("OPENAI_API_KEY") is not None:
    print("OPENAI_API_KEY is ready")
else:
    print("OPENAI_API_KEY environment variable not found")
```

### 1.3 Configure Your AnalyticDB Connection

You need your AnalyticDB instance credentials to build a connection string. Set the following environment variables:

**On Linux/macOS:**
```bash
export PG_HOST="your-analyticdb-host-url"
export PG_PORT=5432
export PG_DATABASE=postgres
export PG_USER="your-username"
export PG_PASSWORD="your-password"
```

Now, construct the connection string using LangChain's helper function:

```python
import os
from langchain.vectorstores.analyticdb import AnalyticDB

CONNECTION_STRING = AnalyticDB.connection_string_from_db_params(
    driver=os.environ.get("PG_DRIVER", "psycopg2cffi"),
    host=os.environ.get("PG_HOST", "localhost"),
    port=int(os.environ.get("PG_PORT", "5432")),
    database=os.environ.get("PG_DATABASE", "postgres"),
    user=os.environ.get("PG_USER", "postgres"),
    password=os.environ.get("PG_PASSWORD", "postgres"),
)
```

## Step 2: Load Your Knowledge Data

For this tutorial, we'll use a sample dataset of questions and answers. We'll download it and load it into memory.

```python
import wget
import json

# Download sample Q&A data from the Natural Questions dataset
wget.download("https://storage.googleapis.com/dataset-natural-questions/questions.json")
wget.download("https://storage.googleapis.com/dataset-natural-questions/answers.json")

# Load the data
with open("questions.json", "r") as fp:
    questions = json.load(fp)

with open("answers.json", "r") as fp:
    answers = json.load(fp)

# Let's inspect a sample
print("Sample Question:", questions[0])
print("\nSample Answer (first 500 chars):", answers[0][:500], "...")
```

## Step 3: Build the Knowledge Base with AnalyticDB

Now, you will create your vector-based knowledge base. LangChain's `AnalyticDB` integration handles the process of generating embeddings for your text and storing them in the database.

```python
from langchain.vectorstores import AnalyticDB
from langchain.embeddings import OpenAIEmbeddings

# Initialize the embedding model
embeddings = OpenAIEmbeddings()

# Create the vector store from your list of answers
# This step sends each answer to OpenAI to get its vector embedding,
# then stores all vectors in your AnalyticDB collection.
doc_store = AnalyticDB.from_texts(
    texts=answers,
    embedding=embeddings,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=True, # Clears any existing collection with the same name
)
```

## Step 4: Create the QA Chain

With the knowledge base populated, you can define the complete QA pipeline using LangChain's `RetrievalQA` chain. This chain automates the process of retrieving relevant context and generating an answer.

```python
from langchain.chains import RetrievalQA
from langchain import OpenAI

# Initialize the language model (OpenAI's GPT-3 by default)
llm = OpenAI()

# Create the QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # A simple method that "stuffs" all context into the prompt
    retriever=doc_store.as_retriever(),
    return_source_documents=False,
)
```

**How it works:**
1.  A user asks a question.
2.  The question is converted into a vector embedding using the same `OpenAIEmbeddings` model.
3.  AnalyticDB performs a nearest-neighbor search to find the most relevant answer chunks from your knowledge base.
4.  These chunks are inserted into a prompt template as context.
5.  The final prompt, containing both the context and the original question, is sent to the LLM (OpenAI) to generate a concise answer.

## Step 5: Query the System

Let's test the QA system with some sample questions.

```python
import random

# Select a few random questions from our dataset
random.seed(52)
selected_questions = random.choices(questions, k=3)

# Ask each question and print the answer
for question in selected_questions:
    print("> Question:", question)
    answer = qa.run(question)
    print("Answer:", answer, end="\n\n")
```

## Step 6: Customize the Prompt Template (Advanced)

The default prompt used by the `"stuff"` chain is effective, but you can customize it to change the model's behavior. For example, you can instruct it to give shorter answers or handle unknown questions creatively.

### 6.1 Define a Custom Prompt

Here's a custom template that asks for single-sentence answers and suggests a song if the answer is unknown.

```python
from langchain.prompts import PromptTemplate

custom_prompt_template = PromptTemplate(
    template="""
Use the following pieces of context to answer the question at the end. Please provide
a short single-sentence summary answer only. If you don't know the answer or if it's
not present in given context, don't try to make up an answer, but suggest me a random
unrelated song title I could listen to.

Context: {context}
Question: {question}
Helpful Answer:
""",
    input_variables=["context", "question"]
)
```

### 6.2 Create a QA Chain with the Custom Prompt

Pass your custom template when creating the chain.

```python
custom_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=doc_store.as_retriever(),
    return_source_documents=False,
    chain_type_kwargs={"prompt": custom_prompt_template},
)
```

### 6.3 Test the Customized Chain

```python
random.seed(41)
for question in random.choices(questions, k=3):
    print("> Question:", question)
    answer = custom_qa.run(question)
    print("Answer:", answer, end="\n\n")
```

## Summary

You have successfully built a Question Answering system that:
1.  **Ingests** textual knowledge (answers) and stores them as vectors in **AnalyticDB**.
2.  **Retrieves** relevant context for a user's question using vector similarity search.
3.  **Synthesizes** a natural language answer using **OpenAI's LLM**.
4.  Can be **customized** with specific prompt engineering to control the style and behavior of the answers.

This architecture forms the core of a Retrieval-Augmented Generation (RAG) system, enabling you to build powerful, knowledge-grounded AI applications. You can extend it by adding more documents, implementing more complex retrieval strategies, or fine-tuning the prompts for your specific use case.