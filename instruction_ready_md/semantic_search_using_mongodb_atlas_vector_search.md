# Building a Semantic Search Application with OpenAI and MongoDB Atlas

This guide walks you through creating a semantic search application using OpenAI's embedding models and MongoDB Atlas Vector Search. You will embed movie plot descriptions and perform similarity searches to find films based on conceptual meaning rather than just keywords.

## Prerequisites

Before you begin, ensure you have the following:

1.  **A MongoDB Atlas Cluster** (version 6.0.11 or higher):
    *   Create a [free account and cluster](https://www.mongodb.com/atlas/database).
    *   Load the **"sample_mflix"** sample dataset via the Atlas UI.
    *   Note your cluster connection URI.
2.  **An OpenAI API Key**:
    *   Sign up at the [OpenAI Platform](https://platform.openai.com/) and generate an API key.

## Setup

First, install the required Python libraries and configure your environment.

```bash
pip install pymongo openai
```

Now, let's import the necessary modules and securely set up your credentials.

```python
import getpass
import openai
import pymongo

# Securely input your credentials
MONGODB_ATLAS_CLUSTER_URI = getpass.getpass("MongoDB Atlas Cluster URI: ")
OPENAI_API_KEY = getpass.getpass("OpenAI API Key: ")

# Initialize the MongoDB client and OpenAI
client = pymongo.MongoClient(MONGODB_ATLAS_CLUSTER_URI)
db = client.sample_mflix
collection = db.movies

openai.api_key = OPENAI_API_KEY

# Define constants for the vector search index and embedding field
ATLAS_VECTOR_SEARCH_INDEX_NAME = "default"
EMBEDDING_FIELD_NAME = "embedding_openai_nov19_23"
```

## Step 1: Create an Embedding Generation Function

To perform semantic search, we need to convert text into numerical vector representations (embeddings). We'll use OpenAI's `text-embedding-3-small` model for this task.

```python
model = "text-embedding-3-small"

def generate_embedding(text: str) -> list[float]:
    """Generates a vector embedding for the input text using OpenAI."""
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding
```

## Step 2: Generate and Store Embeddings

Next, we will add an embedding field to documents in our `movies` collection. This embedding will represent the semantic meaning of each movie's plot.

```python
from pymongo import ReplaceOne

# Prepare a list of update operations
requests = []

# Limit to 500 documents for speed. Remove `.limit(500)` to process all ~23,000.
for doc in collection.find({'plot': {"$exists": True}}).limit(500):
    # Generate an embedding for the movie plot
    embedding = generate_embedding(doc['plot'])
    # Add the embedding to the document
    doc[EMBEDDING_FIELD_NAME] = embedding
    # Create an operation to replace the document in the database
    requests.append(ReplaceOne({'_id': doc['_id']}, doc))

# Execute all update operations in a single batch
if requests:
    collection.bulk_write(requests)
    print(f"Embeddings generated and stored for {len(requests)} documents.")
```

**Note:** The sample dataset includes a pre-populated `sample_mflix.embedded_movies` collection with embeddings. You can use it directly to skip this generation step.

## Step 3: Create a Vector Search Index

To enable fast similarity searches on the embeddings, we need to create a Vector Search Index in Atlas. You can create it via the Atlas UI or programmatically with PyMongo.

### Option A: Create Index via Atlas UI (Recommended for Beginners)

1.  Navigate to your cluster in the [Atlas UI](https://cloud.mongodb.com).
2.  Go to the "Atlas Search" tab for your cluster and click "Create Search Index".
3.  Select the **JSON Editor** method.
4.  Use the following index definition, ensuring the `path` matches your `EMBEDDING_FIELD_NAME`:

```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding_openai_nov19_23": {
        "dimensions": 1536,
        "similarity": "dotProduct",
        "type": "knnVector"
      }
    }
  }
}
```

### Option B: Create Index Programmatically with PyMongo

If your MongoDB Atlas cluster is version 7.0+ and you have the latest PyMongo driver, you can create the index directly from your Python script.

```python
collection.create_search_index(
    {
        "definition": {
            "mappings": {
                "dynamic": True,
                "fields": {
                    EMBEDDING_FIELD_NAME: {
                        "dimensions": 1536,
                        "similarity": "dotProduct",
                        "type": "knnVector"
                    }
                }
            }
        },
        "name": ATLAS_VECTOR_SEARCH_INDEX_NAME
    }
)
print("Vector search index creation initiated.")
```

## Step 4: Perform Semantic Search Queries

With the index in place, you can now query your data. The function below uses the `$vectorSearch` aggregation stage to find documents whose plot embeddings are most similar to the embedding of your query text.

```python
def query_movies(query: str, result_limit: int = 5):
    """Performs a vector search to find movies with semantically similar plots."""
    # Generate an embedding for the search query
    query_embedding = generate_embedding(query)

    # Define the vector search pipeline
    pipeline = [
        {
            '$vectorSearch': {
                "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
                "path": EMBEDDING_FIELD_NAME,
                "queryVector": query_embedding,
                "numCandidates": 50,  # Number of potential matches considered
                "limit": result_limit, # Number of results to return
            }
        }
    ]

    # Execute the search
    results = collection.aggregate(pipeline)
    return results
```

Let's test the search with an example query.

```python
# Define your search query
search_query = "imaginary characters from outerspace at war with earthlings"

# Execute the search
movies = query_movies(search_query, 5)

# Print the results
print(f"Top results for: '{search_query}'\n")
for movie in movies:
    print(f'Movie Title: {movie["title"]}')
    print(f'Movie Plot: {movie["plot"]}\n{("-"*50)}\n')
```

## Conclusion

You have successfully built a semantic search engine! By leveraging OpenAI's embeddings and MongoDB Atlas Vector Search, your application can now understand the intent behind a query and return conceptually relevant results. You can extend this foundation by:
*   Searching on other fields like `genres` or `cast`.
*   Implementing a user interface for the search.
*   Using the results to power a recommendation system.