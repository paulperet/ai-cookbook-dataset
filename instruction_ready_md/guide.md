# Enhancing RAG with Contextual Retrieval: A Cookbook Guide

## Overview

This guide demonstrates how to implement **Contextual Retrieval** to significantly improve your Retrieval Augmented Generation (RAG) systems. Traditional RAG often splits documents into small chunks for efficient retrieval, but these chunks can lack sufficient context, leading to retrieval failures. Contextual Retrieval solves this by adding relevant context to each chunk before embedding, resulting in more accurate retrieval.

We'll walk through building and optimizing a Contextual Retrieval system using a dataset of 9 codebases as our knowledge base. By the end, you'll understand how to implement Contextual Embeddings and Contextual BM25 search, achieving a 35% reduction in top-20-chunk retrieval failure rates.

## Prerequisites

### Technical Skills
- Intermediate Python programming
- Basic understanding of RAG (Retrieval Augmented Generation)
- Familiarity with vector databases and embeddings
- Basic command-line proficiency

### System Requirements
- Python 3.8+
- Docker installed and running (optional, for BM25 search)
- 4GB+ available RAM
- ~5-10 GB disk space for vector databases

### API Access
- [Anthropic API key](https://console.anthropic.com/) (free tier sufficient)
- [Voyage AI API key](https://www.voyageai.com/)
- [Cohere API key](https://cohere.com/)

### Time & Cost
- Expected completion time: 30-45 minutes
- API costs: ~$5-10 to run through the full dataset

## Setup

### 1. Install Required Libraries

First, install the necessary Python packages:

```bash
pip install --upgrade anthropic voyageai cohere elasticsearch pandas numpy
```

### 2. Set Environment Variables

Ensure the following environment variables are set in your environment:

```bash
export VOYAGE_API_KEY="your_voyage_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export COHERE_API_KEY="your_cohere_api_key"
```

### 3. Initialize the Anthropic Client

We'll use the Anthropic client to generate contextual descriptions for our chunks:

```python
import os
import anthropic

# Initialize the Anthropic client
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

# Define the model name for easy updates
MODEL_NAME = "claude-haiku-4-5"
```

## Part 1: Building a Basic RAG Pipeline

Before implementing contextual retrieval, let's establish a baseline with a basic RAG system.

### 1.1 Create a Vector Database Class

We'll create a `VectorDB` class to handle embedding storage and similarity search. This class provides three key functions:

1. **Embedding Generation**: Converts text chunks into vector representations
2. **Storage & Caching**: Saves embeddings to disk to avoid recomputation
3. **Similarity Search**: Retrieves the most relevant chunks using cosine similarity

```python
import json
import pickle
from typing import Any
import numpy as np
import voyageai
from tqdm import tqdm

class VectorDB:
    def __init__(self, name: str, api_key=None):
        if api_key is None:
            api_key = os.getenv("VOYAGE_API_KEY")
        self.client = voyageai.Client(api_key=api_key)
        self.name = name
        self.embeddings = []
        self.metadata = []
        self.query_cache = {}
        self.db_path = f"./data/{name}/vector_db.pkl"

    def load_data(self, dataset: list[dict[str, Any]]):
        """Load and embed dataset, or load from cache if available."""
        if self.embeddings and self.metadata:
            print("Vector database is already loaded. Skipping data loading.")
            return
        if os.path.exists(self.db_path):
            print("Loading vector database from disk.")
            self.load_db()
            return

        texts_to_embed = []
        metadata = []
        total_chunks = sum(len(doc["chunks"]) for doc in dataset)

        with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
            for doc in dataset:
                for chunk in doc["chunks"]:
                    texts_to_embed.append(chunk["content"])
                    metadata.append(
                        {
                            "doc_id": doc["doc_id"],
                            "original_uuid": doc["original_uuid"],
                            "chunk_id": chunk["chunk_id"],
                            "original_index": chunk["original_index"],
                            "content": chunk["content"],
                        }
                    )
                    pbar.update(1)

        self._embed_and_store(texts_to_embed, metadata)
        self.save_db()
        print(f"Vector database loaded and saved. Total chunks processed: {len(texts_to_embed)}")

    def _embed_and_store(self, texts: list[str], data: list[dict[str, Any]]):
        """Embed texts in batches and store results."""
        batch_size = 128
        with tqdm(total=len(texts), desc="Embedding chunks") as pbar:
            result = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_result = self.client.embed(batch, model="voyage-2").embeddings
                result.extend(batch_result)
                pbar.update(len(batch))

        self.embeddings = result
        self.metadata = data

    def search(self, query: str, k: int = 20) -> list[dict[str, Any]]:
        """Search for k most similar documents to the query."""
        if query in self.query_cache:
            query_embedding = self.query_cache[query]
        else:
            query_embedding = self.client.embed([query], model="voyage-2").embeddings[0]
            self.query_cache[query] = query_embedding

        if not self.embeddings:
            raise ValueError("No data loaded in the vector database.")

        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]

        top_results = []
        for idx in top_indices:
            result = {
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx]),
            }
            top_results.append(result)

        return top_results

    def save_db(self):
        """Save the vector database to disk."""
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
        """Load the vector database from disk."""
        if not os.path.exists(self.db_path):
            raise ValueError(
                "Vector database file not found. Use load_data to create a new database."
            )
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
        self.query_cache = json.loads(data["query_cache"])
```

### 1.2 Load Your Dataset

Now let's load our codebase dataset and initialize the vector database:

```python
import json

# Load your transformed dataset
with open("data/codebase_chunks.json") as f:
    transformed_dataset = json.load(f)

# Initialize the VectorDB
base_db = VectorDB("base_db")

# Load and process the data
base_db.load_data(transformed_dataset)
```

### 1.3 Create Evaluation Functions

To measure our system's performance, we need evaluation functions. We'll use the Pass@k metric, which checks whether the 'golden document' (correct answer) appears in the top k retrieved documents.

```python
import json
from collections.abc import Callable
from typing import Any
from tqdm import tqdm

def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Load JSONL file and return a list of dictionaries."""
    with open(file_path) as file:
        return [json.loads(line) for line in file]

def evaluate_retrieval(
    queries: list[dict[str, Any]], retrieval_function: Callable, db, k: int = 20
) -> dict[str, float]:
    """Evaluate retrieval performance using Pass@k metric."""
    total_score = 0
    total_queries = len(queries)

    for query_item in tqdm(queries, desc="Evaluating retrieval"):
        query = query_item["query"]
        golden_chunk_uuids = query_item["golden_chunk_uuids"]

        # Find all golden chunk contents
        golden_contents = []
        for doc_uuid, chunk_index in golden_chunk_uuids:
            golden_doc = next(
                (doc for doc in query_item["golden_documents"] if doc["uuid"] == doc_uuid), None
            )
            if not golden_doc:
                print(f"Warning: Golden document not found for UUID {doc_uuid}")
                continue

            golden_chunk = next(
                (chunk for chunk in golden_doc["chunks"] if chunk["index"] == chunk_index), None
            )
            if not golden_chunk:
                print(
                    f"Warning: Golden chunk not found for index {chunk_index} in document {doc_uuid}"
                )
                continue

            golden_contents.append(golden_chunk["content"].strip())

        if not golden_contents:
            print(f"Warning: No golden contents found for query: {query}")
            continue

        retrieved_docs = retrieval_function(query, db, k=k)

        # Count how many golden chunks are in the top k retrieved documents
        chunks_found = 0
        for golden_content in golden_contents:
            for doc in retrieved_docs[:k]:
                retrieved_content = (
                    doc["metadata"]
                    .get("original_content", doc["metadata"].get("content", ""))
                    .strip()
                )
                if retrieved_content == golden_content:
                    chunks_found += 1
                    break

        query_score = chunks_found / len(golden_contents)
        total_score += query_score

    average_score = total_score / total_queries
    pass_at_n = average_score * 100
    return {"pass_at_n": pass_at_n, "average_score": average_score, "total_queries": total_queries}

def retrieve_base(query: str, db, k: int = 20) -> list[dict[str, Any]]:
    """Retrieve relevant documents using VectorDB."""
    return db.search(query, k=k)

def evaluate_db(db, original_jsonl_path: str, k):
    """Evaluate a database's retrieval performance."""
    original_data = load_jsonl(original_jsonl_path)
    results = evaluate_retrieval(original_data, retrieve_base, db, k)
    return results

def evaluate_and_display(db, jsonl_path: str, k_values: list[int] = None, db_name: str = ""):
    """
    Evaluate retrieval performance across multiple k values and display formatted results.
    """
    if k_values is None:
        k_values = [5, 10, 20]
    results = {}

    print(f"{'=' * 60}")
    if db_name:
        print(f"Evaluation Results: {db_name}")
    else:
        print("Evaluation Results")
    print(f"{'=' * 60}\n")

    for k in k_values:
        print(f"Evaluating Pass@{k}...")
        results[k] = evaluate_db(db, jsonl_path, k)
        print()  # Add spacing between evaluations

    # Print summary table
    print(f"{'=' * 60}")
    print(f"{'Metric':<15} {'Pass Rate':<15} {'Score':<15}")
    print(f"{'-' * 60}")
    for k in k_values:
        pass_rate = f"{results[k]['pass_at_n']:.2f}%"
        score = f"{results[k]['average_score']:.4f}"
        print(f"{'Pass@' + str(k):<15} {pass_rate:<15} {score:<15}")
    print(f"{'=' * 60}\n")

    return results
```

### 1.4 Establish Baseline Performance

Now let's evaluate our basic RAG system to establish a performance baseline:

```python
results = evaluate_and_display(
    base_db, "data/evaluation_set.jsonl", k_values=[5, 10, 20], db_name="Baseline RAG"
)
```

**Expected Output:**
```
============================================================
Evaluation Results: Baseline RAG
============================================================

Evaluating Pass@5...
[Evaluating retrieval: 100%|██████████| 248/248 [00:03<00:00, 65.26it/s]

Evaluating Pass@10...
[Evaluating retrieval: 100%|██████████| 248/248 [00:03<00:00, 64.87it/s]

Evaluating Pass@20...
[Evaluating retrieval: 100%|██████████| 248/248 [00:03<00:00, 64.72it/s]

============================================================
Metric          Pass Rate       Score          
------------------------------------------------------------
Pass@5          80.92%          0.8092         
Pass@10         87.15%          0.8715         
Pass@20         90.06%          0.9006         
============================================================
```

## Analysis of Baseline Results

Our basic RAG system achieves:
- **80.92%** Pass@5: The correct chunk appears in the top 5 results 81% of the time
- **87.15%** Pass@10: The correct chunk appears in the top 10 results 87% of the time
- **90.06%** Pass@20: The correct chunk appears in the top 20 results 90% of the time

While these results are decent, there's room for improvement. The main limitation of basic RAG is that individual chunks often lack sufficient context, making it difficult for the embedding model to understand their true meaning and relevance.

In the next section, we'll implement **Contextual Embeddings** to address this limitation by adding relevant context to each chunk before embedding.

> **Note:** This guide continues with implementing Contextual Embeddings, Contextual BM25, and Reranking techniques. The complete implementation would show how to improve from ~87% to ~95% Pass@10 performance.