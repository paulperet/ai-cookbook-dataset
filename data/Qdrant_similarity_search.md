# Gemini API & Qdrant: Semantic Search Tutorial

This guide demonstrates how to build a semantic search system using Google's Gemini API for generating embeddings and Qdrant as a vector database. You will learn to index content from a website and perform similarity searches against it.

## Prerequisites

This tutorial requires a paid-tier Google API key with access to the Gemini API. Ensure you have your `GEMINI_API_KEY` available.

## Setup

Begin by installing the required Python packages.

```bash
pip install "google-genai>=1.0.0"
pip install protobuf==4.25.1 qdrant-client[fastembed]
```

Next, import the necessary libraries and configure your API key.

```python
from google import genai
from bs4 import BeautifulSoup
from qdrant_client import models, QdrantClient
from urllib.request import urlopen
from google.genai import types

# Configure your Gemini API client
GEMINI_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual key
client = genai.Client(api_key=GEMINI_API_KEY)
```

## Step 1: Create a Search Index

In this stage, you will parse website data, convert it into vector embeddings, and store them in Qdrant.

### 1.1 Parse Website Data

Use `BeautifulSoup` to extract and clean text from a target webpage.

```python
# Define the URL to scrape
url = "https://blog.google/outreach-initiatives/sustainability/report-ai-sustainability-google-cop28/"

# Fetch and parse the HTML
html = urlopen(url).read()
soup = BeautifulSoup(html, features="html.parser")

# Remove scripts and styles for clean text
for script in soup(["script", "style"]):
    script.extract()

# Get the raw text
text_content = soup.get_text()

# Extract a relevant section of the article
text_content_1 = text_content.split("Later this month at COP28", 1)[1]
final_text = text_content_1.split("POSTED IN:", 1)[0]

# Split the text into sentences
texts = final_text.split(".")

# Chunk the text into groups of 3 sentences for manageable documents
documents = []
for i in range(0, len(texts), 3):
    documents.append({"content": " ".join(texts[i:i+3])})
```

### 1.2 Initialize the Embedding Model

You will use the `gemini-embedding-001` model to create vector representations of your text. Define a helper function to generate embeddings.

```python
MODEL_ID = "gemini-embedding-001"

def make_embed_text_fn(text, model=MODEL_ID, task_type="retrieval_document"):
    """Converts text into a vector embedding using the Gemini API."""
    embedding = client.models.embed_content(
        model=model,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    return embedding.embeddings[0].values
```

**Note on `task_type`:** This parameter guides the model to produce embeddings suitable for specific tasks. For indexing documents, use `retrieval_document`. For search queries, you will later use `retrieval_query`.

### 1.3 Store Embeddings in Qdrant

First, initialize a Qdrant client and create a collection to store your vectors.

```python
# Initialize an in-memory Qdrant client (for demonstration)
qdrant = QdrantClient(":memory:")

# Create a collection with vector parameters matching the embedding model
qdrant.create_collection(
    collection_name="GeminiCollection",
    vectors_config=models.VectorParams(
        size=3072,  # Vector size for gemini-embedding-001
        distance=models.Distance.COSINE,
    ),
)
```

Now, insert your document chunks into the collection as indexed points. Each point contains a vector embedding and the original text as a payload.

```python
# Prepare and upsert points (vectors + payloads) into the collection
qdrant.upsert(
    collection_name="GeminiCollection",
    points=[
        models.PointStruct(
            id=idx,
            vector=make_embed_text_fn(doc["content"]),
            payload=doc
        )
        for idx, doc in enumerate(documents)
    ]
)
```

## Step 2: Query the Index

With your data indexed, you can now perform semantic searches. Convert your search query into an embedding and use Qdrant to find the most similar document chunks.

```python
# Define your search query
query = "How can AI address climate challenges?"

# Generate an embedding for the query, specifying the query task type
query_embedding = make_embed_text_fn(query, task_type="retrieval_query")

# Search the collection for the top 3 most similar points
hits = qdrant.search(
    collection_name="GeminiCollection",
    query_vector=query_embedding,
    limit=3,
)

# Display the results
for hit in hits:
    print(f"Score: {hit.score:.4f}")
    print(f"Content: {hit.payload.get('content').replace(chr(10), ' ')}")
    print("-" * 50)
```

**Expected Output:**
The search will return the top 3 document chunks from the article that are semantically closest to your query, along with a similarity score for each.

## Conclusion

You have successfully built a semantic search pipeline. You learned to:
1.  Extract and preprocess text data from a website.
2.  Generate text embeddings using the Gemini API.
3.  Store and index those embeddings in Qdrant's vector database.
4.  Perform a similarity search by querying the index with an embedded question.

This foundation can be extended to build more advanced Retrieval-Augmented Generation (RAG) systems or intelligent search applications.