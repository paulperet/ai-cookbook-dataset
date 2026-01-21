# Building a Question Answering System with LangChain, Deep Lake, and OpenAI

This guide walks you through creating a question answering system using LangChain, Deep Lake as a vector database, and OpenAI's embeddings and language models. You'll learn how to load a dataset, create a searchable vector store, and query it with natural language.

## Prerequisites

Ensure you have the following Python packages installed:

```bash
pip install deeplake langchain openai tiktoken
```

## Step 1: Set Up Your OpenAI API Key

First, securely set your OpenAI API key as an environment variable:

```python
import getpass
import os

os.environ['OPENAI_API_KEY'] = getpass.getpass('Enter your OpenAI API key: ')
```

## Step 2: Load Your Text Dataset

We'll use a sample from the Cohere Wikipedia dataset hosted on Deep Lake. This dataset contains text passages suitable for building a knowledge base.

```python
import deeplake

# Load a 20,000-sample subset of the Wikipedia dataset
ds = deeplake.load("hub://activeloop/cohere-wikipedia-22-sample")

# View a summary of the dataset structure
ds.summary()
```

To understand the data format, let's examine the first few samples:

```python
# Preview the text content of the first three entries
print(ds[:3].text.data()["value"])
```

## Step 3: Initialize the Deep Lake Vector Store

Now, create a vector store where we'll store text embeddings for efficient similarity search.

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

# Define the path for your local vector store
dataset_path = 'wikipedia-embeddings-deeplake'

# Initialize the OpenAI embedding model
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

# Create the Deep Lake vector store
db = DeepLake(dataset_path, embedding=embedding, overwrite=True)
```

## Step 4: Populate the Vector Store with Text

Add the dataset text to the vector store in batches. This process converts text into embeddings and stores them for retrieval.

```python
from tqdm.auto import tqdm

batch_size = 100
nsamples = 10  # For demonstration. Use len(ds) to process the full dataset

for i in tqdm(range(0, nsamples, batch_size)):
    # Determine the batch boundaries
    i_end = min(nsamples, i + batch_size)
    
    # Extract batch data
    batch = ds[i:i_end]
    id_batch = batch.ids.data()["value"]
    text_batch = batch.text.data()["value"]
    meta_batch = batch.metadata.data()["value"]
    
    # Add texts with their metadata and IDs to the vector store
    db.add_texts(text_batch, metadatas=meta_batch, ids=id_batch)
```

## Step 5: Verify the Vector Store Structure

Inspect the underlying dataset to confirm your data was stored correctly:

```python
# Display the structure of the vector store
db.vectorstore.summary()
```

This shows the tensors in your dataset (typically `ids`, `text`, `metadata`, and `embedding`) with the number of samples you added.

## Step 6: Create the Question Answering Chain

Set up a retrieval-augmented generation (RAG) pipeline using LangChain. This combines the vector store retriever with an LLM to answer questions.

```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Initialize the language model
llm = ChatOpenAI(model='gpt-3.5-turbo')

# Create the QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Simple method for stuffing context into the prompt
    retriever=db.as_retriever()
)
```

## Step 7: Query Your Knowledge Base

Now you can ask questions! The system will retrieve relevant text passages and use the LLM to generate an answer.

```python
query = "Why does the military not say 24:00?"
answer = qa.run(query)
print(answer)
```

The system performs an embedding search to find the most relevant context from your dataset, then passes that context along with your question to the LLM to produce a coherent answer.

## Next Steps

You've successfully built a question answering system. To expand this project:

1.  **Use your own data:** Replace the sample dataset with your own documents (PDFs, text files, etc.).
2.  **Experiment with models:** Try different embedding models or LLMs like GPT-4.
3.  **Refine retrieval:** Adjust the retriever parameters (like `search_kwargs`) to control how many documents are fetched.

This pattern forms the foundation for building sophisticated RAG applications for document analysis, code understanding, or personalized recommendations.