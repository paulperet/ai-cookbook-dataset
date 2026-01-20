##### Copyright 2025 Google LLC.


```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Gemini API: Question Answering using LangChain and Pinecone

This notebook requires paid tier rate limits to run properly.  
(cf. pricing for more details).

## Overview

Gemini is a family of generative AI models that lets developers generate content and solve problems. These models are designed and trained to handle both text and images as input.

LangChain is a data framework designed to make integration of Large Language Models (LLM) like Gemini easier for applications.

Pinecone is a cloud-first vector database that allows users to search across billions of embeddings with ultra-low query latency.

In this notebook, you'll learn how to create an application that answers questions using data from a website with the help of Gemini, LangChain, and Pinecone.

## Setup

First, you must install the packages and set the necessary environment variables.

### Installation

Install LangChain's Python library, `langchain` and LangChain's integration package for Gemini, `langchain-google-genai`. Next, install LangChain's integration package for the new version of Pinecone, `langchain-pinecone` and the `pinecone-client`, which is Pinecone's Python SDK. Finally, install `langchain-community` to access the `WebBaseLoader` module later.


```
%pip install --quiet -U langchain
%pip install --quiet -U langchain-google-genai
%pip install --quiet -U langchain-pinecone
%pip install --quiet -U pinecone
%pip install --quiet -U langchain-community
%pip install --quiet -U bs4
```

## Configure your API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see Authentication for an example.



```
import os
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
```

### Setup Pinecone

To use Pinecone in your application, you must have an API key. To create an API key you have to set up a Pinecone account. Visit Pinecone's app page, and Sign up/Log in to your account. Then navigate to the "API Keys" section and copy your API key.

For more detailed instructions on getting the API key, you can read Pinecone's Quickstart documentation.

Set the environment variable `PINECONE_API_KEY` to configure Pinecone to use your API key.



```
PINECONE_API_KEY=userdata.get('PINECONE_API_KEY')

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
```

## Basic steps
LLMs are trained offline on a large corpus of public data. Hence they cannot answer questions based on custom or private data accurately without additional context.

If you want to make use of LLMs to answer questions based on private data, you have to provide the relevant documents as context alongside your prompt. This approach is called Retrieval Augmented Generation (RAG).

You will use this approach to create a question-answering assistant using the Gemini text model integrated through LangChain. The assistant is expected to answer questions about Gemini model. To make this possible you will add more context to the assistant using data from a website.

In this tutorial, you'll implement the two main components in an RAG-based architecture:

1. Retriever

    Based on the user's query, the retriever retrieves relevant snippets that add context from the document. In this tutorial, the document is the website data.
    The relevant snippets are passed as context to the next stage - "Generator".

2. Generator

    The relevant snippets from the website data are passed to the LLM along with the user's query to generate accurate answers.

You'll learn more about these stages in the upcoming sections while implementing the application.

## Import the required libraries


```
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
```

## Retriever

In this stage, you will perform the following steps:

1. Read and parse the website data using LangChain.

2. Create embeddings of the website data.

    Embeddings are numerical representations (vectors) of text. Hence, text with similar meaning will have similar embedding vectors. You'll make use of Gemini's embedding model to create the embedding vectors of the website data.

3. Store the embeddings in Pinecone's vector store.
    
    Pinecone is a vector database. The Pinecone vector store helps in the efficient retrieval of similar vectors. Thus, for adding context to the prompt for the LLM, relevant embeddings of the text matching the user's question can be retrieved easily using Pinecone.

4. Create a Retriever from the Pinecone vector store.

    The retriever will be used to pass relevant website embeddings to the LLM along with user queries.

### Read and parse the website data

LangChain provides a wide variety of document loaders. To read the website data as a document, you will use the `WebBaseLoader` from LangChain.

To know more about how to read and parse input data from different sources using the document loaders of LangChain, read LangChain's document loaders guide.


```
loader = WebBaseLoader("https://blog.google/technology/ai/google-gemini-ai/")
docs = loader.load()
```

If you only want to select a specific portion of the website data to add context to the prompt, you can use regex, text slicing, or text splitting.

In this example, you'll use Python's `split()` function to extract the required portion of the text. The extracted text should be converted back to LangChain's `Document` format.


```
# Extract the text from the website data document
text_content = docs[0].page_content
# The text content between the substrings "code, audio, image and video." to
# "Cloud TPU v5p" is relevant for this tutorial. You can use Python's `split()`
# to select the required content.
text_content_1 = text_content.split("code, audio, image and video.",1)[1]
final_text = text_content_1.split("Cloud TPU v5p",1)[0]

# Convert the text to LangChain's `Document` format
docs = [Document(page_content=final_text, metadata={"source": "local"})]
```

### Initialize Gemini's embedding model

To create the embeddings from the website data, you'll use Gemini's embedding model, **gemini-embedding-001** which supports creating text embeddings.

To use this embedding model, you have to import `GoogleGenerativeAIEmbeddings` from LangChain. To know more about the embedding model, read Google AI's language documentation.


```
from langchain_google_genai import GoogleGenerativeAIEmbeddings

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
```

### Store the data using Pinecone


To create a Pinecone vector database, first, you have to initialize your Pinecone client connection using the API key you set previously.

In Pinecone, vector embeddings have to be stored in indexes. An index represents the vector data's top-level organizational unit. The vectors in any index must have the same dimensionality and distance metric for calculating similarity. You can read more about indexes in Pinecone's Indexes documentation.

First, you'll create an index using Pinecone's `create_index` function. Pinecone allows you to create two types of indexes, Serverless indexes and Pod-based indexes. Pinecone's free starter plan lets you create only one project and one pod-based starter index with sufficient resources to support 100,000 vectors. For this tutorial, you have to create a pod-based starter index. To know more about different indexes and how they can be created, read Pinecone's create indexes guide.


Next, you'll insert the documents you extracted earlier from the website data into the newly created index using LangChain's `Pinecone.from_documents`. Under the hood, this function creates embeddings from the documents created by the document loader of LangChain using any specified embedding model and inserts them into the specified index in a Pinecone vector database.  

You have to specify the `docs` you created from the website data using LangChain's `WebBasedLoader` and the `gemini_embeddings` as the embedding model when invoking the `from_documents` function to create the vector database from the website data.


```
# Initialize Pinecone client

pine_client= pc(
    api_key = os.getenv("PINECONE_API_KEY"),  # API key from app.pinecone.io
    )
index_name = "langchain-demo"

# First, check if the index already exists. If it doesn't, create a new one.
if index_name not in pine_client.list_indexes().names():
    # Create a new index.
    # https://docs.pinecone.io/docs/new-api#creating-a-starter-index
    print("Creating index")
    pine_client.create_index(name=index_name,
                      # `cosine` distance metric compares different documents
                      # for similarity.
                      # Read more about different distance metrics from
                      # https://docs.pinecone.io/docs/indexes#distance-metrics.
                      metric="cosine",
                      # The Gemini embedding model `gemini-embedding-001` uses
                      # 3072 dimensions.
                      dimension=3072,
                      # The `pod_type` is the type of pod to use.
                      # Read more about different pod types from
                      # https://docs.pinecone.io/docs/pod-types.
                      # Specify the pod details.
                      spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                        ),
    )
    print(pine_client.describe_index(index_name))

vectorstore = PineconeVectorStore.from_documents(docs,
                      gemini_embeddings, index_name=index_name)

```

    [Creating index, ..., {'deletion_protection': 'disabled', 'dimension': 3072, 'host': 'langchain-demo-u4710cy.svc.aped-4627-b74a.pinecone.io', 'metric': 'cosine', 'name': 'langchain-demo', 'spec': {'serverless': {'cloud': 'aws', 'region': 'us-east-1'}}, 'status': {'ready': True, 'state': 'Ready'}, 'tags': None, 'vector_type': 'dense'}]


### Create a retriever using Pinecone

You'll now create a retriever that can retrieve website data embeddings from the newly created Pinecone vector store. This retriever can be later used to pass embeddings that provide more context to the LLM for answering user's queries.

Invoke the `as_retriever` function of the vector store you initialized in the last step, to create a retriever.


```
retriever = vectorstore.as_retriever()
# Check if the retriever is working by trying to fetch the relevant docs related
# to the word 'MMLU'(Massive Multitask Language Understanding). If the length is
# greater than zero, it means that the retriever is functioning well.
print(len(retriever.invoke("MMLU")))
```

    1


## Generator

The Generator prompts the LLM for an answer when the user asks a question. The retriever you created in the previous stage from the Pinecone vector store will be used to pass relevant embeddings from the website data to the LLM to provide more context to the user's query.

You'll perform the following steps in this stage:

1. Chain together the following:
    * A prompt for extracting the relevant embeddings using the retriever.
    * A prompt for answering any question using LangChain.
    * An LLM model from Gemini for prompting.
    
2. Run the created chain with a question as input to prompt the model for an answer.


### Initialize Gemini

You must import `ChatGoogleGenerativeAI` from LangChain to initialize your model.
 In this example, you will use **gemini-2.0-flash**, as it supports text summarization. To know more about the text model, read Google AI's language documentation.

You can configure the model parameters such as ***temperature*** or ***top_p***,  by passing the appropriate values when initializing the `ChatGoogleGenerativeAI` LLM.  To learn more about the parameters and their uses, read Google AI's concepts guide.


```
from langchain_google_genai import ChatGoogleGenerativeAI

# To configure model parameters use the `generation_config` parameter.
# eg. generation_config = {"temperature": 0.7, "topP": 0.8, "topK": 40}
# If you only want to set a custom temperature for the model use the
# "temperature" parameter directly.

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
```

### Create prompt templates

You'll use LangChain's PromptTemplate to generate prompts to the LLM for answering questions.

In the `llm_prompt`, the variable `question` will be replaced later by the input question, and the variable `context` will be replaced by the relevant text from the website retrieved from the Pinecone vector store.


```
# Prompt template to query Gemini
llm_prompt_template = """You are an assistant for question-answering tasks.
Use the following context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:"""

llm_prompt = PromptTemplate.from_template(llm_prompt_template)

print(llm_prompt)
```

    input_variables=['context', 'question'] input_types={} partial_variables={} template="You are an assistant for question-answering tasks.\nUse the following context to answer the question.\nIf you don't know the answer, just say that you don't know.\nUse five sentences maximum and keep the answer concise.\n\nQuestion: {question}\nContext: {context}\nAnswer:"


### Create a stuff documents chain

LangChain provides Chains for chaining together LLMs with each other or other components for complex applications. You will create a **stuff documents chain** for this application. A stuff documents chain lets you combine all the relevant documents, insert them into the prompt, and pass that prompt to the LLM.

You can create a stuff documents chain using the LangChain Expression Language (LCEL).

To learn more about different types of document chains, read LangChain's chains guide.

The stuff documents chain for this application retrieves the relevant website data and passes it as the context to an LLM prompt along with the input question.


```
# Combine data from documents to readable string format.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create stuff documents chain using LCEL.
# This is called a chain because you are chaining
# together different elements with the LLM.
# In the following example, to create a stuff chain,
# you will combine content, prompt, LLM model, and
# output parser together like a chain using LCEL.
#
# The chain implements the following pipeline:
# 1. Extract data from documents and save to the variable `context`.
# 2. Use the `RunnablePassthrough` option to provide question during invoke.
# 3. The `context` and `question` are then passed to the prompt and
#    input variables in the prompt are populated.
# 4. The prompt is then passed to the LLM (`gemini-2.0-flash`).
# 5. Output from the LLM is passed through an output parser
#    to structure the model response.
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | llm_prompt
    | llm
    | StrOutputParser()
)
```

### Prompt the model

You can now query the LLM by passing any question to the `invoke()` function of the stuff documents chain you created previously.


```
from IPython.display import Markdown

Markdown(rag_chain.invoke("What is Gemini?"))
```




Gemini is a state-of-the-art, natively multimodal AI model developed by Google. It is designed to understand and reason about various inputs, including text, images, audio, and video, from the ground up. Optimized for different sizes like Ultra, Pro, and Nano, Gemini can efficiently run on a wide range of devices, from data centers to mobile phones. It demonstrates sophisticated reasoning capabilities, excelling at complex tasks, extracting insights, and generating high-quality code. Gemini has surpassed current state-of-the-art performance on numerous benchmarks, including outperforming human experts on MMLU.



## Summary

Gemini API works great with Langchain. The integration is seamless and provides an easy interface for:
- loading and splitting files
- creating Pinecone database with embeddings
- answering questions based on context from files

## What's next?

This notebook showed only one possible use case for langchain with Gemini API. You can find many more here.