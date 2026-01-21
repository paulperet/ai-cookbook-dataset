# Code Search with Vector Embeddings and Qdrant

*Authored by: [Qdrant Team](https://qdrant.tech/)*

In this tutorial, you will learn how to use vector embeddings to navigate a codebase and find relevant code snippets using natural language queries. We'll demonstrate a dual-model approach: one model for understanding natural language queries and another for understanding code semantics. You can see a [live deployment](https://code-search.qdrant.tech/) of this technique searching the Qdrant codebase.

## Overview

Our approach uses two specialized embedding models:
1.  **NLP Model:** A general-purpose text encoder (`sentence-transformers/all-MiniLM-L6-v2`) to understand natural language queries.
2.  **Code Model:** A specialized code embedding model (`jinaai/jina-embeddings-v2-base-code`) to find semantically similar code snippets.

We will preprocess code into a natural language format for the NLP model while feeding raw code to the code model. The results from both models are then combined to provide comprehensive search results.

## Prerequisites and Setup

First, install the required Python libraries.

```bash
pip install inflection qdrant-client fastembed
```

*   **`inflection`**: For string transformations (e.g., converting CamelCase to words).
*   **`fastembed`**: A lightweight, CPU-first library for generating embeddings (GPU support is available).
*   **`qdrant-client`**: The official Python client for interacting with Qdrant.

## Step 1: Prepare and Load the Codebase Data

We need a parsed representation of a codebase. For this tutorial, we'll use the Qdrant Rust codebase, which has been pre-processed and exported into JSON Lines (JSONL) format. This file contains code structures (like functions and structs) along with their metadata.

### 1.1 Download the Data File

Download the pre-parsed code structures.

```python
!wget https://storage.googleapis.com/tutorial-attachments/code-search/structures.jsonl
```

### 1.2 Load the Data

Load the JSONL file into a list of Python dictionaries.

```python
import json

structures = []
with open("structures.jsonl", "r") as fp:
    for row in fp:
        entry = json.loads(row)
        structures.append(entry)
```

Let's examine the first entry to understand the data structure.

```python
structures[0]
```

```json
{
  "name": "InvertedIndexRam",
  "signature": '# [doc = " Inverted flatten index from dimension id to posting list"] # [derive (Debug , Clone , PartialEq)] pub struct InvertedIndexRam { # [doc = " Posting lists for each dimension flattened (dimension id -> posting list)"] # [doc = " Gaps are filled with empty posting lists"] pub postings : Vec < PostingList > , # [doc = " Number of unique indexed vectors"] # [doc = " pre-computed on build and upsert to avoid having to traverse the posting lists."] pub vector_count : usize , }',
  "code_type": "Struct",
  "docstring": '= " Inverted flatten index from dimension id to posting list"',
  "line": 15,
  "line_from": 13,
  "line_to": 22,
  "context": {
    "module": "inverted_index",
    "file_path": "lib/sparse/src/index/inverted_index/inverted_index_ram.rs",
    "file_name": "inverted_index_ram.rs",
    "struct_name": None,
    "snippet": "/// Inverted flatten index from dimension id to posting list\n#[derive(Debug, Clone, PartialEq)]\npub struct InvertedIndexRam {\n    /// Posting lists for each dimension flattened (dimension id -> posting list)\n    /// Gaps are filled with empty posting lists\n    pub postings: Vec<PostingList>,\n    /// Number of unique indexed vectors\n    /// pre-computed on build and upsert to avoid having to traverse the posting lists.\n    pub vector_count: usize,\n}\n"
  }
}
```

Each entry contains the code's name, signature, type, docstring, line numbers, and contextual information like its module and file path.

## Step 2: Convert Code to Natural Language

General-purpose language models don't understand raw code syntax. We need to convert each code structure into a natural language description. Our `textify` function performs this conversion by:
1.  Extracting the function/struct signature.
2.  Converting camelCase and snake_case names into separate words.
3.  Incorporating docstrings and comments.
4.  Building a descriptive sentence using a template.

```python
import inflection
import re
from typing import Dict, Any

def textify(chunk: Dict[str, Any]) -> str:
    # Convert names to human-readable format
    name = inflection.humanize(inflection.underscore(chunk["name"]))
    signature = inflection.humanize(inflection.underscore(chunk["signature"]))

    # Include docstring if available
    docstring = ""
    if chunk["docstring"]:
        docstring = f"that does {chunk['docstring']} "

    # Build location context
    context = (
        f"module {chunk['context']['module']} " f"file {chunk['context']['file_name']}"
    )
    if chunk["context"]["struct_name"]:
        struct_name = inflection.humanize(
            inflection.underscore(chunk["context"]["struct_name"])
        )
        context = f"defined in struct {struct_name} {context}"

    # Combine all parts into a single text representation
    text_representation = (
        f"{chunk['code_type']} {name} "
        f"{docstring}"
        f"defined as {signature} "
        f"{context}"
    )

    # Clean up special characters and tokenize
    tokens = re.split(r"\W", text_representation)
    tokens = filter(lambda x: x, tokens)
    return " ".join(tokens)
```

Now, apply this conversion to all code structures.

```python
text_representations = list(map(textify, structures))
```

Let's view a sample converted entry.

```python
text_representations[1000]
```

```
'Function Hnsw discover precision that does Checks discovery search precision when using hnsw index this is different from the tests in defined as Fn hnsw discover precision module integration file hnsw_discover_test rs'
```

## Step 3: Generate Embeddings

We will generate two sets of vector embeddings: one for the natural language representations and one for the raw code snippets.

### 3.1 Generate Natural Language Embeddings

Use the `sentence-transformers/all-MiniLM-L6-v2` model via FastEmbed.

```python
from fastembed import TextEmbedding

batch_size = 5
nlp_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2", threads=0)
nlp_embeddings = nlp_model.embed(text_representations, batch_size=batch_size)
```

### 3.2 Generate Code Embeddings

Use the specialized `jinaai/jina-embeddings-v2-base-code` model on the raw code snippets.

```python
code_snippets = [structure["context"]["snippet"] for structure in structures]
code_model = TextEmbedding("jinaai/jina-embeddings-v2-base-code")
code_embeddings = code_model.embed(code_snippets, batch_size=batch_size)
```

## Step 4: Build the Qdrant Vector Search Collection

We'll use Qdrant to store our embeddings and perform efficient similarity searches. For this tutorial, we'll use an in-memory Qdrant client. For production, you would use a persistent server (Docker or Qdrant Cloud).

### 4.1 Create a Multi-Vector Collection

Create a collection with two separate vector fields: one for text embeddings (384-dim) and one for code embeddings (768-dim).

```python
from qdrant_client import QdrantClient, models

COLLECTION_NAME = "qdrant-sources"
client = QdrantClient(":memory:")  # Use in-memory storage for prototyping

client.create_collection(
    COLLECTION_NAME,
    vectors_config={
        "text": models.VectorParams(
            size=384,  # Dimension of the NLP model embeddings
            distance=models.Distance.COSINE,
        ),
        "code": models.VectorParams(
            size=768,  # Dimension of the code model embeddings
            distance=models.Distance.COSINE,
        ),
    },
)
```

### 4.2 Upload Points to the Collection

Now, upload the generated embeddings along with their original payload data (the code structures) to Qdrant.

```python
from tqdm import tqdm

points = []
total = len(structures)
print("Number of points to upload: ", total)

for id, (text_embedding, code_embedding, structure) in tqdm(
    enumerate(zip(nlp_embeddings, code_embeddings, structures)), total=total
):
    points.append(
        models.PointStruct(
            id=id,
            vector={
                "text": text_embedding,
                "code": code_embedding,
            },
            payload=structure,
        )
    )

    # Upload in batches for efficiency
    if len(points) >= batch_size:
        client.upload_points(COLLECTION_NAME, points=points, wait=True)
        points = []

# Upload any remaining points
if points:
    client.upload_points(COLLECTION_NAME, points=points)

print(f"Total points in collection: {client.count(COLLECTION_NAME).count}")
```

## Step 5: Query the Codebase

With the data indexed, you can now search the codebase using natural language. Qdrant's Query API allows you to search using either the text or code embeddings, or to fuse results from both.

### 5.1 Search with the NLP Model

Let's start by searching for "How do I count points in a collection?" using the natural language embeddings.

```python
query = "How do I count points in a collection?"

hits = client.query_points(
    COLLECTION_NAME,
    query=next(nlp_model.query_embed(query)).tolist(),
    using="text",  # Search in the 'text' vector field
    limit=3,
).points
```

The top results will include structures related to counting points, such as the `CountRequestInternal` struct and functions like `get_points_with_value_count`.

### 5.2 Search with the Code Model

Now, let's run the same query using the code embeddings.

```python
hits = client.query_points(
    COLLECTION_NAME,
    query=next(code_model.query_embed(query)).tolist(),
    using="code",  # Search in the 'code' vector field
    limit=3,
).points
```

This search returns different results, such as various `count_indexed_points` functions, highlighting how the code model captures semantic similarities in the implementation logic.

### 5.3 Fuse Results from Both Models (Hybrid Search)

To get the most comprehensive results, you can perform a hybrid search that queries both vector fields and fuses the results using Reciprocal Rank Fusion (RRF).

```python
from qdrant_client import models

hits = client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        models.Prefetch(
            query=next(nlp_model.query_embed(query)).tolist(),
            using="text",
            limit=5,
        ),
        models.Prefetch(
            query=next(code_model.query_embed(query)).tolist(),
            using="code",
            limit=5,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF)
).points

# Display the fused results
for hit in hits:
    print(
        "| ",
        hit.payload["context"]["module"], " | ",
        hit.payload["context"]["file_path"], " | ",
        hit.score, " | `",
        hit.payload["signature"], "` |"
    )
```

This hybrid approach combines the strengths of both models: the NLP model finds structures related to the query's intent, while the code model finds structures with similar implementation patterns.

## Conclusion

You have successfully built a semantic code search system using Qdrant and dual embedding models. This tutorial covered:

1.  Preprocessing a codebase into searchable chunks.
2.  Converting code structures to natural language.
3.  Generating embeddings for both text and code.
4.  Storing and indexing vectors in Qdrant.
5.  Querying with single models and performing hybrid searches.

You can extend this approach to other programming languages by using appropriate Language Server Protocol (LSP) tools to parse the codebase. The combination of general and specialized embeddings provides a powerful way to navigate and understand large codebases using natural language.