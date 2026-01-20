# RAG Observability with Mistral AI and Langtrace

This Notebook shows the instructions for setting up OpenTelemetry based tracing for Mistral with Langtrace AI.

The Goal for this notebook to showcase a simple RAG app where you can chat with the United states consititution pdf.

```python
%pip install mistralai langtrace-python-sdk chromadb pypdf langchain langchain-community
```

## Imports & Initialize clients

```python
import chromadb
from mistralai import Mistral
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from langtrace_python_sdk import langtrace, with_langtrace_root_span


langtrace.init(api_key='<langtrace_api_key>')
mistral = Mistral(api_key='<mistral_api_key>')
client = chromadb.Client()
```

## Use Langchain to split pdf into chunks

```python
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    return chunks
```

## Setup Chroma & Insert pdf chunks
Create a chroma collection, specifying the default embedding function which will be used in our RAG when inserting pdf chunks

```python
def setup_chroma():

    return client.get_or_create_collection(
        name="mistral-rag",
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    )


def add_documents_to_collection(collection, chunks):
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk.page_content],
            metadatas=[{"source": chunk.metadata["source"]}],
            ids=[str(i)],
        )
```

## Query Collection
1. take query from user, get nearest 3 results from chunked pdf
2. construct a prompt structure
3. Give query and prompt to mistral for the actual response

```python
def query_pdf(collection, query):
    results = collection.query(query_texts=[query], n_results=3)
    # Construct the prompt with context
    context = "\n".join(results["documents"][0])
    prompt = f"""Use the following context to answer the question. If you cannot answer based on the context, say so.

            Context:
            {context}

            Question: {query}

            Answer:"""
    response = mistral.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

## Run everything together and monitor using Langtrace.

```python
@with_langtrace_root_span("main")
def main():
    print("Creating collection")
    collection = setup_chroma()
    print("Loading and splitting pdf")
    chunks = load_and_split_pdf("https://www.govinfo.gov/content/pkg/CDOC-112hdoc129/pdf/CDOC-112hdoc129.pdf")
    print("Adding documents to collection")
    add_documents_to_collection(collection, chunks)
    print("Querying pdf")
    print(query_pdf(collection, "What is the purpose of the constitution?"))


if __name__ == "__main__":
    main()
```

That's it! Now you should be able to see the traces for all your inference calls on Langtrace!

## First Two Screenshots showcase the Trace and span structure of the whole RAG App.

## Second Two Screenshots are details of Mistral's Run.

- You can see what prompt is specfically fetched from chromadb and sent to mistral as well as the response