# Advanced RAG on Hugging Face Documentation using LangChain

_Authored by: [Aymeric Roucher](https://huggingface.co/m-ric)_

This guide demonstrates how to build an advanced Retrieval Augmented Generation (RAG) system for answering questions about a specific knowledge base—in this case, the Hugging Face documentation—using LangChain.

RAG systems are complex, with many configurable components. This tutorial will explore several enhancement techniques to help you tune your RAG system for optimal performance.

## Prerequisites

First, install the required dependencies.

```bash
pip install -q torch transformers accelerate bitsandbytes langchain sentence-transformers faiss-cpu openpyxl pacmap datasets langchain-community ragatouille
```

Now, import the necessary libraries.

```python
from tqdm.notebook import tqdm
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
import matplotlib.pyplot as plt

pd.set_option("display.max_colwidth", None)  # Helpful for visualizing retriever outputs
```

## 1. Load Your Knowledge Base

We'll use a dataset containing Hugging Face documentation.

```python
import datasets

ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")
```

Convert the dataset into LangChain Document objects for processing.

```python
from langchain.docstore.document import Document as LangchainDocument

RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in tqdm(ds)
]
```

## 2. Building the Retriever with Embeddings

The retriever acts as an internal search engine: given a user query, it returns relevant snippets from your knowledge base. These snippets are then fed to the reader model to generate an answer.

Our objective is to find the most relevant snippets to answer a user's question. This involves two key parameters:
- `top_k`: How many snippets to retrieve
- `chunk_size`: The length of each snippet

There's no one-size-fits-all answer, but here are some guidelines:
- Your `chunk_size` can vary between snippets
- Increasing `top_k` increases the chance of getting relevant elements
- The total length of retrieved documents shouldn't be too high to avoid overwhelming the reader model

### 2.1 Split Documents into Chunks

We need to split documents into semantically relevant chunks. We'll use recursive chunking, which breaks down text using a hierarchical list of separators, preserving document structure.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Hierarchical list of separators tailored for Markdown documents
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Maximum characters per chunk
    chunk_overlap=100,  # Character overlap between chunks
    add_start_index=True,  # Include chunk's start index in metadata
    strip_whitespace=True,  # Strip whitespace from document ends
    separators=MARKDOWN_SEPARATORS,
)

docs_processed = []
for doc in RAW_KNOWLEDGE_BASE:
    docs_processed += text_splitter.split_documents([doc])
```

### 2.2 Check Chunk Lengths Against Embedding Model Limits

Embedding models have maximum sequence lengths. We need to ensure our chunks don't exceed this limit to avoid truncation.

```python
from sentence_transformers import SentenceTransformer

print(f"Model's maximum sequence length: {SentenceTransformer('thenlper/gte-small').max_seq_length}")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]

# Plot the distribution of document lengths
fig = pd.Series(lengths).hist()
plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
plt.show()
```

The output shows that some chunks exceed the 512-token limit. We need to adjust our splitting to count tokens instead of characters.

### 2.3 Token-Aware Document Splitting

Let's create a function that splits documents based on token count rather than character count.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

# Apply token-aware splitting
docs_processed = split_documents(
    512,  # Chunk size adapted to our model
    RAW_KNOWLEDGE_BASE,
    tokenizer_name=EMBEDDING_MODEL_NAME,
)

# Visualize the new chunk sizes
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
fig = pd.Series(lengths).hist()
plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
plt.show()
```

Now the chunk length distribution is properly aligned with our model's limits.

### 2.4 Building the Vector Database

We'll compute embeddings for all knowledge base chunks and store them in a vector database. When a user query arrives, it gets embedded with the same model, and a similarity search returns the closest documents.

We'll use:
- **FAISS** for nearest neighbor search
- **Cosine similarity** as our distance metric (requires normalized embeddings)

```python
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)

KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
)
```

### 2.5 Visualizing the Embedding Space

Let's project our embeddings to 2D using PaCMAP to visualize the relationship between documents and queries.

```python
# Embed a user query
user_query = "How to create a pipeline object?"
query_vector = embedding_model.embed_query(user_query)
```

```python
import pacmap
import numpy as np
import plotly.express as px

embedding_projector = pacmap.PaCMAP(
    n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1
)

embeddings_2d = [
    list(KNOWLEDGE_VECTOR_DATABASE.index.reconstruct_n(idx, 1)[0])
    for idx in range(len(docs_processed))
] + [query_vector]

# Fit the data
documents_projected = embedding_projector.fit_transform(
    np.array(embeddings_2d), init="pca"
)
```

```python
df = pd.DataFrame.from_dict(
    [
        {
            "x": documents_projected[i, 0],
            "y": documents_projected[i, 1],
            "source": docs_processed[i].metadata["source"].split("/")[1],
            "extract": docs_processed[i].page_content[:100] + "...",
            "symbol": "circle",
            "size_col": 4,
        }
        for i in range(len(docs_processed))
    ]
    + [
        {
            "x": documents_projected[-1, 0],
            "y": documents_projected[-1, 1],
            "source": "User query",
            "extract": user_query,
            "size_col": 100,
            "symbol": "star",
        }
    ]
)

# Visualize the embedding
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="source",
    hover_data="extract",
    size="size_col",
    symbol="symbol",
    color_discrete_map={"User query": "black"},
    width=1000,
    height=700,
)
fig.update_traces(
    marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")),
    selector=dict(mode="markers"),
)
fig.update_layout(
    legend_title_text="<b>Chunk source</b>",
    title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>",
)
fig.show()
```

The visualization shows a spatial representation of document embeddings. Documents with similar meanings should have close embeddings. The user query's embedding is also shown—we want to find the `k` documents with the closest embeddings.

### 2.6 Testing the Retriever

Let's test our retriever with a sample query.

```python
print(f"\nStarting retrieval for {user_query=}...")
retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
print(
    "\n==================================Top document=================================="
)
print(retrieved_docs[0].page_content)
print("==================================Metadata==================================")
print(retrieved_docs[0].metadata)
```

This completes the retriever setup. In the next section, we'll configure the reader model to process these retrieved documents and generate answers.