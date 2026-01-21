# Building a Question Answering System with LangChain, Tair, and OpenAI

This guide walks you through creating a Retrieval-Augmented Generation (RAG) system. You'll use LangChain to orchestrate the workflow, OpenAI's embeddings for vectorization, and Tair as a high-performance vector database to store and retrieve contextual information.

## Prerequisites

Before you begin, ensure you have the following:
1.  A **Tair instance** (e.g., from [Tair Cloud](https://www.alibabacloud.com/help/en/tair/latest/what-is-tair)).
2.  An **OpenAI API key** from your [OpenAI account](https://platform.openai.com/account/api-keys).
3.  Basic familiarity with Python and LangChain concepts.

## Step 1: Environment Setup

First, install the required Python libraries. These packages provide the core functionality for embeddings, language models, and database interaction.

```bash
pip install openai tiktoken langchain tair
```

Next, securely configure your API keys and database connection.

```python
import getpass

# Securely input your OpenAI API key
openai_api_key = getpass.getpass("Input your OpenAI API key: ")

# Securely input your Tair connection URL
# Format: redis://[[username]:[password]]@host:port/db_number
TAIR_URL = getpass.getpass("Input your Tair URL: ")
```

## Step 2: Load the Sample Dataset

We'll use a sample from the Google Natural Questions dataset to build our knowledge base. This dataset contains real-world questions and their corresponding answers.

```python
import wget
import json

# Download the sample question and answer files
wget.download("https://storage.googleapis.com/dataset-natural-questions/questions.json")
wget.download("https://storage.googleapis.com/dataset-natural-questions/answers.json")

# Load the data into Python lists
with open("questions.json", "r") as fp:
    questions = json.load(fp)

with open("answers.json", "r") as fp:
    answers = json.load(fp)

# Inspect the first entry of each list
print("Sample Question:", questions[0])
print("\nSample Answer:", answers[0])
```

## Step 3: Initialize the Vector Store and QA Chain

Now, you'll create the core components of the RAG pipeline: the vector store for your documents and the question-answering chain.

### 3.1 Create the Tair Vector Store

LangChain's `Tair` class simplifies the process of generating embeddings and storing them. The `from_texts` method handles chunking, embedding generation via OpenAI, and indexing in Tair.

```python
from langchain.vectorstores import Tair
from langchain.embeddings import OpenAIEmbeddings

# Initialize the embedding model
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Create and populate the Tair vector store with our answer documents
doc_store = Tair.from_texts(
    texts=answers,
    embedding=embeddings,
    tair_url=TAIR_URL,
)
```

### 3.2 Define the QA Chain

With the knowledge base ready, you can construct the QA chain. This chain will:
1.  Receive a user's question.
2.  Convert it to an embedding.
3.  Retrieve the most relevant context (answers) from Tair.
4.  Pass both the question and context to an LLM to generate a final answer.

```python
from langchain import VectorDBQA, OpenAI

# Initialize the language model (OpenAI's GPT-3)
llm = OpenAI(openai_api_key=openai_api_key)

# Create the QA chain using the "stuff" method (it stuffs all relevant context into the prompt)
qa = VectorDBQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    vectorstore=doc_store,
    return_source_documents=False, # Set to True if you want to see which documents were retrieved
)
```

## Step 4: Query the System

Let's test the system with a few sample questions. The chain will handle the entire retrieval and generation process.

```python
import random
import time

# Select 5 random questions for demonstration
random.seed(52)
selected_questions = random.choices(questions, k=5)

# Query the system for each question
for question in selected_questions:
    print("> Question:", question)
    answer = qa.run(question)
    print("Answer:", answer, end="\n\n")
    # Pause to respect OpenAI's rate limits for the free tier
    time.sleep(20)
```

## Step 5: Customize the Prompt (Advanced)

The default prompt used by the `"stuff"` chain is effective, but you can customize it to change the LLM's behavior. The key is to preserve the `{context}` and `{question}` placeholders.

### 5.1 Create a Custom Prompt Template

Let's create a prompt that instructs the model to give concise answers and, if the answer isn't in the context, to suggest a song title instead.

```python
from langchain.prompts import PromptTemplate

custom_prompt_text = """
Use the following pieces of context to answer the question at the end. Please provide
a short single-sentence summary answer only. If you don't know the answer or if it's
not present in given context, don't try to make up an answer, but suggest me a random
unrelated song title I could listen to.

Context: {context}
Question: {question}
Helpful Answer:
"""

custom_prompt_template = PromptTemplate(
    template=custom_prompt_text,
    input_variables=["context", "question"]
)
```

### 5.2 Build a QA Chain with the Custom Prompt

Pass your custom template to the QA chain via the `chain_type_kwargs` parameter.

```python
custom_qa = VectorDBQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    vectorstore=doc_store,
    return_source_documents=False,
    chain_type_kwargs={"prompt": custom_prompt_template},
)
```

### 5.3 Test the Customized Chain

Now, test the new chain to see how the model's responses change with your custom instructions.

```python
random.seed(41)
for question in random.choices(questions, k=5):
    print("> Question:", question)
    answer = custom_qa.run(question)
    print("Answer:", answer, end="\n\n")
    time.sleep(20)
```

## Summary

You have successfully built a functional RAG-based question-answering system. You learned how to:
1.  Set up a Tair vector database as a knowledge base.
2.  Use LangChain's `VectorDBQA` chain to seamlessly integrate retrieval and generation.
3.  Customize the LLM's prompt to tailor its response style.

This architecture provides a robust foundation for building applications that can answer questions based on your private or domain-specific data.