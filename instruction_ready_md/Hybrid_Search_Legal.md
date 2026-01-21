# Building a Legal AI QA Pipeline with Gemini Embeddings and Qdrant Hybrid Search

## Overview

In the legal domain, **accuracy** and **factual correctness** are paramount. A common approach to securing both in Legal AI applications is to frame problems as retrieval tasks, ensuring all outputs are grounded in valid source documents. This eliminates hallucinations by design.

This guide demonstrates how to build a high-quality retrieval pipeline for legal Question Answering (QA) by combining Google's `gemini-embedding-001` model with Qdrant's vector search engine. You'll implement a hybrid search system that leverages both semantic (dense) and keyword-based (sparse) retrieval, optimized using Matryoshka Representations to balance cost and accuracy.

### What You'll Learn
- How to set up hybrid search (dense + keyword) in Qdrant
- How to use Matryoshka Representations of Gemini embeddings to trade off quality vs. cost
- How to build and evaluate a retrieval pipeline for legal QA

## Prerequisites & Setup

### Install Required Libraries
First, install the necessary Python packages:

```bash
pip install -q -U "google-genai>=1.0.0" qdrant-client[fastembed] datasets
```

### Configure API Keys
You'll need the following API keys:

1. **GOOGLE_API_KEY**: For accessing `gemini-embedding-001` embeddings
   - Generate one at the [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)

2. **QDRANT_API_KEY** and **QDRANT_URL**: From a free Qdrant Cloud cluster
   - Sign up and create a cluster at [Qdrant Cloud](https://cloud.qdrant.io/)

Set these as environment variables or configure them in your environment:

```python
import os

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
```

## Step 1: Load and Prepare the Legal Dataset

You'll use the LegalQAEval dataset from Hugging Face, which contains realistic legal questions and corresponding text chunks that may contain answers.

```python
from datasets import load_dataset, concatenate_datasets

# Load the dataset splits
corpus = concatenate_datasets(
    load_dataset('isaacus/LegalQAEval', split=['val', 'test'])
)
```

### Deduplicate Text Chunks
Since multiple questions can reference the same text chunk, deduplicate to avoid storing identical information multiple times:

```python
import pandas as pd
import datasets

# Convert to pandas DataFrame for deduplication
df = corpus.to_pandas()

# Group by 'text' and aggregate 'id' into a list
grouped_corpus = df.groupby('text')['id'].apply(list).reset_index().rename(columns={'id': 'ids'})

# Convert back to Hugging Face Dataset
corpus_deduplicated = datasets.Dataset.from_pandas(grouped_corpus)
```

## Step 2: Configure the Qdrant Collection for Hybrid Search

In a legal chatbot scenario, you need to store embeddings of text chunks and retrieve the most relevant ones for user questions. You'll configure Qdrant to support:

1. **Matryoshka Representations**: Using truncated embeddings (768-dim) for fast retrieval and full embeddings (3072-dim) for precision reranking
2. **Hybrid Search**: Combining dense vector search with keyword-based sparse retrieval using miniCOIL (Qdrant's optimized BM25 variant)

```python
from qdrant_client import QdrantClient, models

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Configuration constants
COLLECTION_NAME = "legal_AI_QA"
GEMINI_EMBEDDING_RETRIEVAL_SIZE = 768  # Truncated for faster retrieval
GEMINI_EMBEDDING_FULL_SIZE = 3072      # Full size for precision reranking

# Create collection if it doesn't exist
if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "gemini_embedding_retrieve": models.VectorParams(
                size=GEMINI_EMBEDDING_RETRIEVAL_SIZE,
                distance=models.Distance.COSINE,
            ),
            "gemini_embedding_rerank": models.VectorParams(
                size=GEMINI_EMBEDDING_FULL_SIZE,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(
                    m=0  # No vector index needed for reranking-only embeddings
                ),
                on_disk=True,  # Save RAM during retrieval
            ),
        },
        sparse_vectors_config={
            "miniCOIL": models.SparseVectorParams(
                modifier=models.Modifier.IDF  # Inverse Document Frequency
            )
        },
    )
```

## Step 3: Generate Embeddings and Index Data

Now you'll embed the text chunks and upload them to Qdrant. You'll create a streaming function that processes data in batches for efficiency.

### Define Embedding Generation Function

```python
from google import genai
from google.genai import types
from google.api_core import retry
import uuid

# Initialize Google AI client
google_client = genai.Client(api_key=GOOGLE_API_KEY)
GEMINI_MODEL_ID = "gemini-embedding-001"

@retry.Retry(timeout=300)
def get_embeddings_batch(texts, task_type: str = "RETRIEVAL_DOCUMENT"):
    """Generates embeddings for a batch of texts."""
    try:
        res = google_client.models.embed_content(
            model=GEMINI_MODEL_ID,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in res.embeddings]
    except Exception as e:
        print(f"An error occurred while getting embeddings: {e}")
        raise
```

### Create Data Streaming Function

```python
def qdrant_points_stream(corpus, avg_corpus_text_length, gemini_batch_size: int = 8):
    """Streams Qdrant points with embeddings for a given corpus."""
    for start in range(0, len(corpus), gemini_batch_size):
        end = min(start + gemini_batch_size, len(corpus))
        batch = corpus.select(range(start, end))
        
        # Generate embeddings for the current batch
        gemini_embeddings_full = get_embeddings_batch(
            [row["text"] for row in batch], 
            task_type="RETRIQUERY_DOCUMENT"
        )
        
        # Create Qdrant point structures
        for batch_item, gemini_embedding_full in zip(batch, gemini_embeddings_full):
            yield models.PointStruct(
                id=str(uuid.uuid4()),
                payload={
                    "text": batch_item["text"],
                    "ids": batch_item["ids"],
                },
                vector={
                    "gemini_embedding_rerank": gemini_embedding_full,
                    "gemini_embedding_retrieve": gemini_embedding_full[:768],
                    "miniCOIL": models.Document(
                        text=batch_item["text"],
                        model="Qdrant/minicoil-v1",
                        options={
                            "avg_len": avg_corpus_text_length,
                            "k": 0.9,
                            "b": 0.4
                        },
                    ),
                },
            )
```

### Upload Data to Qdrant

```python
import tqdm

# Calculate average text length for BM25 optimization
SUBSET_SIZE = 1000
avg_corpus_text_length = sum(
    len(text.split()) for text in corpus["text"][:SUBSET_SIZE]
) / SUBSET_SIZE

# Upload points to Qdrant
qdrant_client.upload_points(
    collection_name=COLLECTION_NAME,
    points=tqdm.tqdm(
        qdrant_points_stream(
            corpus_deduplicated,
            avg_corpus_text_length=avg_corpus_text_length,
            gemini_batch_size=4
        ),
        desc="Uploading points",
    ),
    batch_size=4,
)
```

## Step 4: Prepare for Evaluation

To evaluate retrieval performance, you'll use questions with known answers and measure whether the correct text chunk appears in the top results.

### Filter Questions with Known Answers

```python
# Only use questions where answers are available
questions = corpus.filter(lambda item: len(item['answers']) > 0)
```

### Pre-compute Question Embeddings

Pre-compute embeddings for all questions to enable efficient experimentation:

```python
import tqdm

question_embeddings = {}
question_texts = [q['question'] for q in questions]
question_ids = [q['id'] for q in questions]
all_embeddings = []

BATCH_SIZE = 32

# Generate embeddings in batches
for i in tqdm.tqdm(range(0, len(question_texts), BATCH_SIZE)):
    batch_texts = question_texts[i:i+BATCH_SIZE]
    batch_embeddings = get_embeddings_batch(
        batch_texts, 
        task_type="RETRIEVAL_QUERY"
    )
    all_embeddings.extend(batch_embeddings)

# Store embeddings with question IDs
for q_id, embedding in zip(question_ids, all_embeddings):
    question_embeddings[q_id] = embedding
```

## Step 5: Define Evaluation Metrics

For RAG applications, you typically want the correct result within the top-N retrieved items. You'll use **hit@1** as your primary metric, which measures whether the top-ranked text chunk contains the answer.

```python
def evaluate_retrieval(question_id, retrieved_text_ids, ground_truth_ids):
    """
    Evaluate if the correct answer is in the retrieved results.
    
    Args:
        question_id: ID of the question being evaluated
        retrieved_text_ids: List of IDs from retrieved text chunks
        ground_truth_ids: List of IDs that contain the correct answer
    
    Returns:
        hit@1 score (1 if correct answer is top result, 0 otherwise)
    """
    # For hit@1, check if the top result contains the answer
    if retrieved_text_ids and ground_truth_ids:
        # Check if any of the ground truth IDs are in the retrieved IDs
        # (accounting for multiple correct answers)
        return 1 if any(gt_id in retrieved_text_ids[0] for gt_id in ground_truth_ids) else 0
    return 0
```

## Step 6: Perform Hybrid Search Queries

Now you can experiment with different search strategies. Here's how to perform a hybrid search query:

```python
def hybrid_search_query(question_text, question_embedding, top_k=5):
    """
    Perform hybrid search combining dense and sparse retrieval.
    
    Args:
        question_text: The question as a string
        question_embedding: Pre-computed embedding for the question
        top_k: Number of results to return
    
    Returns:
        List of retrieved text chunks with scores
    """
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=models.NamedVector(
            name="gemini_embedding_retrieve",
            vector=question_embedding[:768]  # Use truncated embedding for retrieval
        ),
        query_filter=None,
        limit=top_k * 2,  # Retrieve more for reranking
        with_payload=True,
        with_vectors=False,
        search_params=models.SearchParams(
            quantization=models.QuantizationSearchParams(
                ignore=False,
                rescore=True,
                oversampling=2.0,
            )
        ),
        sparse_vectors=[
            models.NamedSparseVector(
                name="miniCOIL",
                vector=models.SparseVector(
                    indices=[],
                    values=[],
                    text=question_text  # Qdrant will compute sparse vector
                )
            )
        ],
        fusion=models.Fusion.RRF,  # Reciprocal Rank Fusion for combining results
    )
    
    # Rerank using full embeddings if needed
    # (Implementation depends on your specific reranking strategy)
    
    return search_result[:top_k]  # Return top-k results
```

## Next Steps

With your retrieval pipeline set up, you can now:

1. **Experiment with different search configurations**: Try varying the balance between dense and sparse retrieval weights
2. **Implement reranking**: Use the full 3072-dimensional embeddings to rerank initial results for higher precision
3. **Integrate with an LLM**: Use the retrieved text chunks as context for a Gemini model to generate grounded answers
4. **Benchmark performance**: Compare hybrid search against pure dense or pure keyword search

Remember that in legal applications, retrieval quality is critical. The hybrid approach demonstrated here combines the strengths of semantic understanding (through embeddings) with precise keyword matching (through miniCOIL), providing a robust foundation for building accurate legal AI systems.