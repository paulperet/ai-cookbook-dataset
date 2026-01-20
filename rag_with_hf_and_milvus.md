# Build RAG with Hugging Face and Milvus

_Authored by: [Chen Zhang](https://github.com/zc277584121)_

[Milvus](https://milvus.io/) is a popular open-source vector database that powers AI applications with highly performant and scalable vector similarity search. In this tutorial, we will show you how to build a RAG (Retrieval-Augmented Generation) pipeline with Hugging Face and Milvus.

The RAG system combines a retrieval system with an LLM. The system first retrieves relevant documents from a corpus using Milvus vector database, then uses an LLM hosted in Hugging Face to generate answers based on the retrieved documents.

## Preparation
### Dependencies and Environment

```python
! pip install --upgrade pymilvus sentence-transformers huggingface-hub langchain_community langchain_text_splitters pypdf tqdm
```

> If you are using Google Colab, to enable the dependencies, you may need to **restart the runtime** (click on the "Runtime" menu at the top of the screen, and select "Restart session" from the dropdown menu).

In addition, we recommend that you configure your [Hugging Face User Access Token](https://huggingface.co/docs/hub/security-tokens), and set it in your environment variables because we will use a LLM from the Hugging Face Hub. You may get a low limit of requests if you don't set the token environment variable.

```python
import os

os.environ["HF_TOKEN"] = "hf_..."
```

### Prepare the data

We use the [AI Act PDF](https://artificialintelligenceact.eu/wp-content/uploads/2021/08/The-AI-Act.pdf), a regulatory framework for AI with different risk levels corresponding to more or less regulation, as the private knowledge in our RAG.

```bash
%%bash

if [ ! -f "The-AI-Act.pdf" ]; then
    wget -q https://artificialintelligenceact.eu/wp-content/uploads/2021/08/The-AI-Act.pdf
fi
```

We use the [`PyPDFLoader`](https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/pdf/) from LangChain to extract the text from the PDF, and then split the text into smaller chunks. By default, we set the chunk size as 1000 and the overlap as 200, which means each chunk will nearly have 1000 characters and the overlap between two chunks will be 200 characters.

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("The-AI-Act.pdf")
docs = loader.load()
print(len(docs))
```

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
```

```python
text_lines = [chunk.page_content for chunk in chunks]
```

### Prepare the Embedding Model
Define a function to generate text embeddings. We use [BGE embedding model](https://huggingface.co/BAAI/bge-small-en-v1.5) as an example, but you can use any embedding models, such as those found on the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def emb_text(text):
    return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]
```

Generate a test embedding and print its dimension and first few elements.

```python
test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])
```

## Load data into Milvus

### Create the Collection

```python
from pymilvus import MilvusClient

milvus_client = MilvusClient(uri="./hf_milvus_demo.db")

collection_name = "rag_collection"
```

> As for the argument of `MilvusClient`:
> - Setting the `uri` as a local file, e.g.`./hf_milvus_demo.db`, is the most convenient method, as it automatically utilizes [Milvus Lite](https://milvus.io/docs/milvus_lite.md) to store all data in this file.
> - If you have a large amount of data, say more than a million vectors, you can set up a more performant Milvus server on [Docker or Kubernetes](https://milvus.io/docs/quickstart.md). In this setup, please use the server uri, e.g.`http://localhost:19530`, as your `uri`.
> - If you want to use [Zilliz Cloud](https://zilliz.com/cloud), the fully managed cloud service for Milvus, adjust the `uri` and `token`, which correspond to the [Public Endpoint and Api key](https://docs.zilliz.com/docs/on-zilliz-cloud-console#cluster-details) in Zilliz Cloud.

Check if the collection already exists and drop it if it does.

```python
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)
```

Create a new collection with specified parameters. 

If we don't specify any field information, Milvus will automatically create a default `id` field for primary key, and a `vector` field to store the vector data. A reserved JSON field is used to store non-schema-defined fields and their values.

```python
milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Strong consistency level
)
```

### Insert data
Iterate through the text lines, create embeddings, and then insert the data into Milvus.

Here is a new field `text`, which is a non-defined field in the collection schema. It will be automatically added to the reserved JSON dynamic field, which can be treated as a normal field at a high level.

```python
from tqdm import tqdm

data = []

for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    data.append({"id": i, "vector": emb_text(line), "text": line})

insert_res = milvus_client.insert(collection_name=collection_name, data=data)
insert_res["insert_count"]
```

[Creating embeddings: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 429/429 [00:52<00:00,  8.10it/s], 429]

## Build RAG

### Retrieve data for a query

Let's specify a question to ask about the corpus.

```python
question = "What is the legal basis for the proposal?"
```

Search for the question in the collection and retrieve the top 3 semantic matches.

```python
search_res = milvus_client.search(
    collection_name=collection_name,
    data=[
        emb_text(question)
    ],  # Use the `emb_text` function to convert the question to an embedding vector
    limit=3,  # Return top 3 results
    search_params={"metric_type": "IP", "params": {}},  # Inner product distance
    output_fields=["text"],  # Return the text field
)
```

Let's take a look at the search results of the query

```python
import json

retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]
print(json.dumps(retrieved_lines_with_distances, indent=4))
```

### Use LLM to get an RAG response

Before composing the prompt for LLM, let's first flatten the retrieved document list into a plain string.

```python
context = "\n".join(
    [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
)
```

Define prompts for the Language Model. This prompt is assembled with the retrieved documents from Milvus.

```python
PROMPT = """
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""
```

We use the [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) hosted on Hugging Face inference server to generate a response based on the prompt.

```python
from huggingface_hub import InferenceClient

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

llm_client = InferenceClient(model=repo_id, timeout=120)
```

Finally, we can format the prompt and generate the answer.

```python
prompt = PROMPT.format(context=context, question=question)
```

```python
answer = llm_client.text_generation(
    prompt,
    max_new_tokens=1000,
).strip()
print(answer)
```

Congratulations! You have built an RAG pipeline with Hugging Face and Milvus.