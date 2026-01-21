# Using Mistral-Embed with ChromaDB: A Step-by-Step Guide

This tutorial demonstrates how to integrate MistralAI's `mistral-embed` model as a custom embedding function within the ChromaDB vector database. You'll learn to create a collection, embed text documents, and perform similarity searches.

## Prerequisites

Ensure you have a MistralAI API key. You can obtain one from the [MistralAI platform](https://console.mistral.ai/).

## Step 1: Install Required Libraries

Begin by installing the necessary Python packages.

```bash
pip install chromadb mistralai -Uq
```

## Step 2: Import Modules and Configure API Key

Import the required libraries and securely set your MistralAI API key.

```python
import os
import getpass
import chromadb
from mistralai import Mistral
from datetime import datetime
from chromadb import Documents, EmbeddingFunction, Embeddings

# Securely set your API key
if os.environ.get('MISTRAL_API_KEY'):
    api_key = os.environ['MISTRAL_API_KEY']
else:
    api_key = getpass.getpass("Please provide your MistralAI API key: ")
```

## Step 3: Initialize the ChromaDB Client

Create a ChromaDB client. For this tutorial, we'll use an ephemeral (in-memory) client. For a persistent database, you can use `PersistentClient`.

```python
# Create an in-memory client for this session
client = chromadb.EphemeralClient()

# For a persistent database, use:
# client = chromadb.PersistentClient(path=os.getcwd())
```

## Step 4: Define a Custom Embedding Function

To use MistralAI's embeddings with ChromaDB, you must create a custom class that inherits from `EmbeddingFunction`. This class will handle API calls to the `mistral-embed` model.

```python
class MistralEmbedFn(EmbeddingFunction):
    """Custom embedding function using MistralAI's mistral-embed model."""

    def __init__(self, api_key: str = None) -> None:
        if api_key:
            self.api_key = api_key
        else:
            try:
                self.api_key = getpass.getpass("Please provide your MistralAI API Key: ")
            except Exception as e:
                print(f'Error getting API key from user: {e}')

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for a list of text documents."""
        client = Mistral(api_key=self.api_key)
        try:
            response = client.embeddings.create(
                model='mistral-embed',
                inputs=input
            )
            # Extract the embedding vectors from the response
            embeddings = [item.embedding for item in response.data]
            return embeddings
        except Exception as e:
            print(f'An error occurred getting embeddings from model: {e}')
```

## Step 5: Create a Collection with the Custom Embedder

Instantiate your embedding function and create a new ChromaDB collection. The collection will use Mistral embeddings for all vector operations.

```python
# Instantiate the embedding function
embed_fn = MistralEmbedFn(api_key=api_key)

# Create a new collection
collection = client.create_collection(
    name="quotes",
    embedding_function=embed_fn,
    metadata={
        "description": "Quotes about Computer Science",
        "created": str(datetime.now())
    }
)
```

## Step 6: Add Documents to the Collection

Populate your collection with sample documents. Each document requires the text, associated metadata, and a unique ID.

```python
# Add two famous quotes to the collection
collection.add(
    documents=[
        "A new, a vast, and a powerful language is developed for the future use of analysis, in which to wield its truths so that these may become of more speedy and accurate practical application for the purposes of mankind than the means hitherto in our possession have rendered possible.",
        "A computer would deserve to be called intelligent if it could deceive a human into believing that it was human."
    ],
    metadatas=[
        {"attribution": "Ada Lovelace"},
        {"attribution": "Alan Turing"}
    ],
    ids=['id0', 'id1']
)
```

## Step 7: Verify the Collection

Use the `peek()` method to inspect the contents of your collection and confirm the documents were added correctly.

```python
# View the first few entries in the collection
print(collection.peek())
```

**Expected Output:**
```
{'ids': ['id0', 'id1'], 'embeddings': None, 'metadatas': [{'attribution': 'Ada Lovelace'}, {'attribution': 'Alan Turing'}], 'documents': ['A new, a vast, and a powerful language is developed...', 'A computer would deserve to be called intelligent...'], 'uris': None, 'data': None}
```

## Next Steps

Your ChromaDB collection is now ready. You can proceed to:
- Perform similarity searches using `collection.query()`.
- Update or delete documents.
- Experiment with different embedding models and metadata schemas.

For more advanced operations, refer to the [ChromaDB Documentation](https://docs.trychroma.com) and [Mistral Embeddings Guide](https://docs.mistral.ai/capabilities/embeddings/).