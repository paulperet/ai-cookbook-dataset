

# Gemini API: Question Answering LlamaIndex and Chroma

<a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/examples/llamaindex/Gemini_LlamaIndex_QA_Chroma_WebPageReader.ipynb"></a>

This notebook requires paid tier rate limits to run properly.  
(cf. pricing for more details).

## Overview

Gemini is a family of generative AI models that lets developers generate content and solve problems. These models are designed and trained to handle both text and images as input.

LlamaIndex is a simple, flexible data framework that can be used by Large Language Model(LLM) applications to connect custom data sources to LLMs.

Chroma is an open-source embedding database focused on simplicity and developer productivity. Chroma allows users to store embeddings and their metadata, embed documents and queries, and search the embeddings quickly.

In this notebook, you'll learn how to create an application that answers questions using data from a website with the help of Gemini, LlamaIndex, and Chroma.

## Setup

First, you must install the packages and set the necessary environment variables.

### Installation

Install LlamaIndex's Python library, `llama-index`. Install LlamaIndex's integration package for Gemini, `llama-index-llms-gemini` and the integration package for Gemini embedding model, `llama-index-embeddings-gemini`. Next, install LlamaIndex's web page reader, `llama-index-readers-web`. Finally, install ChromaDB's Python client SDK, `chromadb` and


```
%pip install -q -U llama-index
%pip install -q -U llama-index-llms-google-genai
%pip install -q -U llama-index-embeddings-google-genai
%pip install -q -U llama-index-readers-web
%pip install -q -U llama-index-vector-stores-chroma
%pip install -q -U chromadb
%pip install -q -U bs4
```

## Configure your API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see Authentication for an example.



```
import os
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
```

## Basic steps
LLMs are trained offline on a large corpus of public data. Hence they cannot answer questions based on custom or private data accurately without additional context.

If you want to make use of LLMs to answer questions based on private data, you have to provide the relevant documents as context alongside your prompt. This approach is called Retrieval Augmented Generation (RAG).

You will use this approach to create a question-answering assistant using the Gemini text model integrated through LlamaIndex. The assistant is expected to answer questions about Google's Gemini model. To make this possible you will add more context to the assistant using data from a website.

In this tutorial, you'll implement the two main components in a RAG-based architecture:

1. Retriever

    Based on the user's query, the retriever retrieves relevant snippets that add context from the document. In this tutorial, the document is the website data.
    The relevant snippets are passed as context to the next stage - "Generator".

2. Generator

    The relevant snippets from the website data are passed to the LLM along with the user's query to generate accurate answers.

You'll learn more about these stages in the upcoming sections while implementing the application.

## Import the required libraries


```
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
import re
```

## 1. Retriever

In this stage, you will perform the following steps:

1. Read and parse the website data using LlamaIndex.

2. Create embeddings of the website data.

    Embeddings are numerical representations (vectors) of text. Hence, text with similar meaning will have similar embedding vectors. You'll make use of Gemini's embedding model to create the embedding vectors of the website data.

3. Store the embeddings in Chroma's vector store.
    
    Chroma is a vector database. The Chroma vector store helps in the efficient retrieval of similar vectors. Thus, for adding context to the prompt for the LLM, relevant embeddings of the text matching the user's question can be retrieved easily using Chroma.

4. Create a Retriever from the Chroma vector store.

    The retriever will be used to pass relevant website embeddings to the LLM along with user queries.

### Read and parse the website data

LlamaIndex provides a wide variety of data loaders. To read the website data as a document, you will use the `SimpleWebPageReader` from LlamaIndex.

To know more about how to read and parse input data from different sources using the data loaders of LlamaIndex, read LlamaIndex's loading data guide.


```
web_documents = SimpleWebPageReader().load_data(
    ["https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/"]
)

# Extract the content from the website data document
html_content = web_documents[0].text
```

You can use variety of HTML parsers to extract the required text from the html content.

In this example, you'll use Python's `BeautifulSoup` library to parse the website data. After processing, the extracted text should be converted back to LlamaIndex's `Document` format.


```
# Parse the data.
soup = BeautifulSoup(html_content, 'html.parser')
p_tags = soup.find_all('p')
text_content = ""
for each in p_tags:
    text_content += each.text + "\n"

# Convert back to Document format
documents = [Document(text=text_content)]
```

### Initialize Gemini's embedding model

To create the embeddings from the website data, you'll use Gemini's embedding model, **gemini-embedding-001** which supports creating text embeddings.

To use this embedding model, you have to import `GeminiEmbedding` from LlamaIndex. To know more about the embedding model, read Google AI's language documentation.


```
gemini_embedding_model = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001")
```

### Initialize Gemini

You must import `Gemini` from LlamaIndex to initialize your model.
 In this example, you will use **gemini-2.0-flash**, as it supports text summarization. To know more about the text model, read Google AI's model documentation.

You can configure the model parameters such as ***temperature*** or ***top_p***,  using the  ***generation_config*** parameter when initializing the `Gemini` LLM.  To learn more about the model parameters and their uses, read Google AI's concepts guide.


```
from llama_index.llms.google_genai import GoogleGenAI

# To configure model parameters use the `generation_config` parameter.
# eg. generation_config = {"temperature": 0.7, "topP": 0.8, "topK": 40}
# If you only want to set a custom temperature for the model use the
# "temperature" parameter directly.

llm = GoogleGenAI(model_name="models/gemini-2.5-flash")
```

### Store the data using Chroma

 Next, you'll store the embeddings of the website data in Chroma's vector store using LlamaIndex.

 First, you have to initiate a Python client in `chromadb`. Since the plan is to save the data to the disk, you will use the `PersistentClient`. You can read more about the different clients in Chroma in the client reference guide.

After initializing the client, you have to create a Chroma collection. You'll then initialize the `ChromaVectorStore` class in LlamaIndex using the collection created in the previous step.

Next, you have to set `Settings` and create storage contexts for the vector store.

`Settings` is a collection of commonly used resources that are utilized during the indexing and querying phase in a LlamaIndex pipeline. You can specify the LLM, Embedding model, etc that will be used to create the application in the `Settings`. To know more about `Settings`, read the module guide for Settings.

`StorageContext` is an abstraction offered by LlamaIndex around different types of storage. To know more about storage context, read the storage context API guide.

The final step is to load the documents and build an index over them. LlamaIndex offers several indices that help in retrieving relevant context for a user query. Here you'll use the `VectorStoreIndex` since the website embeddings have to be stored in a vector store.

To create the index you have to pass the storage context along with the documents to the `from_documents` function of `VectorStoreIndex`.
The `VectorStoreIndex` uses the embedding model specified in the `Settings` to create embedding vectors from the documents and stores these vectors in the vector store specified in the storage context. To know more about the
`VectorStoreIndex` you can read the Using VectorStoreIndex guide.


```
# Create a client and a new collection
client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = client.get_or_create_collection("quickstart")

# Create a vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create a storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Set Global settings
Settings.llm = llm
Settings.embed_model = gemini_embedding_model

# Create an index from the documents and save it to the disk.
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
```

### Create a retriever using Chroma

You'll now create a retriever that can retrieve data embeddings from the newly created Chroma vector store.

First, initialize the `PersistentClient` with the same path you specified while creating the Chroma vector store. You'll then retrieve the collection `"quickstart"` you created previously from Chroma. You can use this collection to initialize the `ChromaVectorStore` in which you store the embeddings of the website data. You can then use the `from_vector_store` function of `VectorStoreIndex` to load the index.


```
from IPython.display import Markdown

# Load from disk
load_client = chromadb.PersistentClient(path="./chroma_db")

# Fetch the collection
chroma_collection = load_client.get_collection("quickstart")

# Fetch the vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Get the index from the vector store
index = VectorStoreIndex.from_vector_store(
    vector_store
)

# Check if the retriever is working by trying to fetch the relevant docs related
# to the phrase 'MMLU' (Multimodal Machine Learning Understanding).
# If the length is greater than zero, it means that the retriever is
# functioning well.
# You can ask questions about your data using a generic interface called
# a query engine. You have to use the `as_query_engine` function of the
# index to create a query engine and use the `query` function of query engine
# to inquire the index.
test_query_engine = index.as_query_engine()
response = test_query_engine.query("AIME")
Markdown(response.response)
```

Gemini 2.5 Pro leads in math and science benchmarks like AIME 2025.

## 2. Generator

The Generator prompts the LLM for an answer when the user asks a question. The retriever you created in the previous stage from the Chroma vector store will be used to pass relevant embeddings from the website data to the LLM to provide more context to the user's query.

You'll perform the following steps in this stage:

1. Create a prompt for answering any question using LlamaIndex.
    
2. Use a query engine to ask a question and prompt the model for an answer.

### Create prompt templates

You'll use LlamaIndex's PromptTemplate to generate prompts to the LLM for answering questions.

In the `llm_prompt`, the variable `query_str` will be replaced later by the input question, and the variable `context_str` will be replaced by the relevant text from the website retrieved from the Chroma vector store.


```
from llama_index.core import PromptTemplate

template = (
    """ You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.\n
Question: {query_str} \nContext: {context_str} \nAnswer:"""
)
llm_prompt = PromptTemplate(template)
```

### Prompt the model using Query Engine

You will use the `as_query_engine` function of the `VectorStoreIndex` to create a query engine from the index using the `llm_prompt` passed as the value for the `text_qa_template` argument. You can then use the `query` function of the query engine to prompt the LLM. To know more about custom prompting in LlamaIndex, read LlamaIndex's prompts usage pattern documentation.


```
# Query data from the persisted index
query_engine = index.as_query_engine(text_qa_template=llm_prompt)
response = query_engine.query("What is Gemini?")
Markdown(response.response)
```

Gemini is a natively multimodal AI model developed by Google. It is pre-trained on different modalities and fine-tuned with additional multimodal data. Gemini's capabilities are state of the art in nearly every domain, exceeding current results on many academic benchmarks. It can understand, explain, and generate high-quality code in popular programming languages. Gemini is being rolled out across Google products like Bard, Pixel, and Search, and will be available to developers and enterprise customers via the Gemini API.

## What's next?

This notebook showed only one possible use case for langchain with Gemini API. You can find many more here.