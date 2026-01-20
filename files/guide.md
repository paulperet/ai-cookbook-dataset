# Enhancing RAG with Contextual Retrieval

> Note: For more background information on Contextual Retrieval, including additional performance evaluations on various datasets, we recommend reading our accompanying  [blog post](https://www.anthropic.com/news/contextual-retrieval).

Retrieval Augmented Generation (RAG) enables Claude to leverage your internal knowledge bases, codebases, or any other corpus of documents when providing a response. Enterprises are increasingly building RAG applications to improve workflows in customer support, Q&A over internal company documents, financial & legal analysis, code generation, and much more.

In a [separate guide](https://github.com/anthropics/anthropic-cookbook/blob/main/capabilities/retrieval_augmented_generation/guide.ipynb), we walked through setting up a basic retrieval system, demonstrated how to evaluate its performance, and then outlined a few techniques to improve performance. In this guide, we present a technique for improving retrieval performance: Contextual Embeddings.

In traditional RAG, documents are typically split into smaller chunks for efficient retrieval. While this approach works well for many applications, it can lead to problems when individual chunks lack sufficient context. Contextual Embeddings solve this problem by adding relevant context to each chunk before embedding. This method improves the quality of each embedded chunk, allowing for more accurate retrieval and thus better overall performance. Averaged across all data sources we tested, Contextual Embeddings reduced the top-20-chunk retrieval failure rate by 35%.

The same chunk-specific context can also be used with BM25 search to further improve retrieval performance. We introduce this technique in the "Contextual BM25" section.

In this guide, we'll demonstrate how to build and optimize a Contextual Retrieval system using a dataset of 9 codebases as our knowledge base. We'll walk through:

1) Setting up a basic retrieval pipeline to establish a baseline for performance.

2) Contextual Embeddings: what it is, why it works, and how prompt caching makes it practical for production use cases.

3) Implementing Contextual Embeddings and demonstrating performance improvements.

4) Contextual BM25: improving performance with *contextual* BM25 hybrid search.

5) Improving performance with reranking,

### Evaluation Metrics & Dataset:

We use a pre-chunked dataset of 9 codebases - all of which have been chunked according to a basic character splitting mechanism. Our evaluation dataset contains 248 queries - each of which contains a 'golden chunk.' We'll use a metric called Pass@k to evaluate performance. Pass@k checks whether or not the 'golden document' was present in the first k documents retrieved for each query. Contextual Embeddings in this case helped us to improve Pass@10 performance from ~87% --> ~95%.

You can find the code files and their chunks in `data/codebase_chunks.json` and the evaluation dataset in `data/evaluation_set.jsonl`

#### Additional Notes:

Prompt caching is helpful in managing costs when using this retrieval method. This feature is currently available on Anthropic's first-party API, and is coming soon to our third-party partner environments in AWS Bedrock and GCP Vertex. We know that many of our customers leverage AWS Knowledge Bases and GCP Vertex AI APIs when building RAG solutions, and this method can be used on either platform with a bit of customization. Consider reaching out to Anthropic or your AWS/GCP account team for guidance on this!

To make it easier to use this method on Bedrock, the AWS team has provided us with code that you can use to implement a Lambda function that adds context to each document. If you deploy this Lambda function, you can select it as a custom chunking option when configuring a [Bedrock Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-create.html). You can find this code in `contextual-rag-lambda-function`. The main lambda function code is in `lambda_function.py`.

## Table of Contents

1) Setup

2) Basic RAG

3) Contextual Embeddings

4) Contextual BM25

5) Reranking

## Setup

Before starting this guide, ensure you have:

**Technical Skills:**
- Intermediate Python programming
- Basic understanding of RAG (Retrieval Augmented Generation)
- Familiarity with vector databases and embeddings
- Basic command-line proficiency

**System Requirements:**
- Python 3.8+
- Docker installed and running (optional, for BM25 search)
- 4GB+ available RAM
- ~5-10 GB disk space for vector databases

**API Access:**
- [Anthropic API key](https://console.anthropic.com/) (free tier sufficient)
- [Voyage AI API key](https://www.voyageai.com/)
- [Cohere API key](https://cohere.com/)

**Time & Cost:**
- Expected completion time: 30-45 minutes
- API costs: ~$5-10 to run through the full dataset

### Libraries 

We'll need a few libraries, including:

1) `anthropic` - to interact with Claude

2) `voyageai` - to generate high quality embeddings

3) `cohere` - for reranking

4) `elasticsearch` for performant BM25 search

3) `pandas`, `numpy`, `matplotlib`, and `scikit-learn` for data manipulation and visualization

### Environment Variables 

Ensure the following environment variables are set:

```
- VOYAGE_API_KEY
- ANTHROPIC_API_KEY
- COHERE_API_KEY
```


```python
%%capture
!pip install --upgrade anthropic voyageai cohere elasticsearch pandas numpy
```

We define our model names up front to make it easier to change models as new models are released


```python
MODEL_NAME = "claude-haiku-4-5"
```

We'll start by initializing the Anthropic client that we'll use for generating contextual descriptions.


```python
import os

import anthropic

client = anthropic.Anthropic(
    # This is the default and can be omitted
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)
```

## Initialize a Vector DB Class

We'll create a VectorDB class to handle embedding storage and similarity search. This class serves three key functions in our RAG pipeline:

1. **Embedding Generation**: Converts text chunks into vector representations using Voyage AI's embedding model
2. **Storage & Caching**: Saves embeddings to disk to avoid re-computing them (which saves time and API costs)
3. **Similarity Search**: Retrieves the most relevant chunks for a given query using cosine similarity


For this guide, we're using a simple in-memory vector database with pickle serialization. This makes the code easy to understand and requires no external dependencies. The class automatically saves embeddings to disk after generation, so you only pay the embedding cost once. 

For production use, consider hosted vector database solutions.

The VectorDB class below follows the same interface patterns you'd use with production solutions, making it easy to swap out later. Key features include batch processing (128 chunks at a time), progress tracking with tqdm, and query caching to speed up repeated searches during evaluation.


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
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata,
            "query_cache": json.dumps(self.query_cache),
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as file:
            pickle.dump(data, file)

    def load_db(self):
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

Now we can use this class to load our dataset


```python
# Load your transformed dataset
with open("data/codebase_chunks.json") as f:
    transformed_dataset = json.load(f)

# Initialize the VectorDB
base_db = VectorDB("base_db")

# Load and process the data
base_db.load_data(transformed_dataset)
```

    [Processing chunks: 100%|██████████| 737/737 [00:00<00:00, 985400.72it/s], Embedding chunks: 100%|██████████| 737/737 [00:42<00:00, 17.28it/s]]

    Vector database loaded and saved. Total chunks processed: 737


    


## Basic RAG

To get started, we'll set up a basic RAG pipeline using a bare bones approach. This is sometimes called 'Naive RAG' by many in the industry. A basic RAG pipeline includes the following 3 steps:

1) Chunk documents by heading - containing only the content from each subheading

2) Embed each document

3) Use Cosine similarity to retrieve documents in order to answer query


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
    """
    Retrieve relevant documents using either VectorDB or ContextualVectorDB.

    :param query: The query string
    :param db: The VectorDB or ContextualVectorDB instance
    :param k: Number of top results to retrieve
    :return: List of retrieved documents
    """
    return db.search(query, k=k)


def evaluate_db(db, original_jsonl_path: str, k):
    # Load the original JSONL data for queries and ground truth
    original_data = load_jsonl(original_jsonl_path)

    # Evaluate retrieval
    results = evaluate_retrieval(original_data, retrieve_base, db, k)
    return results


def evaluate_and_display(db, jsonl_path: str, k_values: list[int] = None, db_name: str = ""):
    """
    Evaluate retrieval performance across multiple k values and display formatted results.

    Args:
        db: Vector database instance (VectorDB or ContextualVectorDB)
        jsonl_path: Path to evaluation dataset
        k_values: List of k values to evaluate (default: [5, 10, 20])
        db_name: Optional name for the database being evaluated

    Returns:
        Dict mapping k values to their results
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

Now let's establish our baseline performance by evaluating the basic RAG system. We'll test at k=5, 10, and 20 to see how many of the golden chunks appear in the top retrieved results. This gives us a benchmark to measure improvement against.



```python
results = evaluate_and_display(
    base_db, "data/evaluation_set.jsonl", k_values=[5, 10, 20], db_name="Baseline RAG"
)
```

    ============================================================
    Evaluation Results: Contextual Embeddings
    ============================================================
    
    Evaluating Pass@5...
    [Evaluating retrieval: 100%|██████████| 248/248 [00:03<00:00, 65.26it/s]]
    
    Evaluating Pass@10...
    [Evaluating retrieval: 100%|██████████| 248/248 [00:03<00:00, 64.87it/s]]
    
    Evaluating Pass@20...
    [Evaluating retrieval: 100%|██████████| 248/248 [00:03<00:00, 64.72it/s]]
    
    ============================================================
    Metric          Pass Rate       Score          
    ------------------------------------------------------------
    Pass@5          80.92%          0.8092         
    Pass@10         87.15%          0.8715         
    Pass@20         90.06%          0.9006         
    ============================================================
    


    


These results show our baseline RAG performance. The system successfully retrieves the correct chunk 81% of the time in the