# Guide: Implementing Hybrid Search with Weaviate and OpenAI

This guide walks you through setting up a vector database using Weaviate with OpenAI's embedding module. You'll learn how to store and search unstructured text data using hybrid search, which combines vector (semantic) and keyword (BM25) search techniques.

## Prerequisites

Before you begin, ensure you have:

1.  **A Weaviate Instance:** You can use the free Weaviate Cloud Service (WCS) sandbox or run Weaviate locally via Docker.
2.  **An OpenAI API Key:** Required for generating text embeddings. Obtain one from the [OpenAI platform](https://platform.openai.com/api-keys).

## Step 1: Environment Setup

First, install the necessary Python libraries and configure your environment.

### Install Dependencies

Open your terminal or a code cell in your notebook and run the following commands:

```bash
pip install weaviate-client>3.11.0
pip install datasets apache-beam
```

### Configure Your OpenAI API Key

Set your OpenAI API key as an environment variable. This key will be used by Weaviate to generate embeddings for your data and queries.

```python
import os

# Set your OpenAI API key. Replace 'your-key-goes-here' with your actual key.
os.environ['OPENAI_API_KEY'] = 'your-key-goes-here'

# Verify the key is set
if os.getenv("OPENAI_API_KEY") is not None:
    print("OPENAI_API_KEY is ready")
else:
    print("OPENAI_API_KEY environment variable not found")
```

## Step 2: Connect to Your Weaviate Instance

Now, establish a connection to your Weaviate instance. The connection parameters differ slightly between WCS and local deployments.

```python
import weaviate

# For Weaviate Cloud Service (WCS):
client = weaviate.Client(
    url="https://your-wcs-instance-name.weaviate.network/", # Your WCS cluster URL
    auth_client_secret=weaviate.auth.AuthApiKey(api_key="<YOUR-WEAVIATE-API-KEY>"), # Your WCS API key
    additional_headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY") # Pass your OpenAI key
    }
)

# For a local Docker instance (comment out the auth line):
# client = weaviate.Client(
#     url="http://localhost:8080/",
#     additional_headers={
#         "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
#     }
# )

# Test the connection
print("Weaviate instance is ready:", client.is_ready())
```

## Step 3: Define the Data Schema

A schema defines the structure of your data within Weaviate and configures how it should be vectorized. Here, we'll create a schema for `Article` objects.

```python
# Clear any existing schema to start fresh
client.schema.delete_all()

# Define the new schema for the Article class
article_schema = {
    "class": "Article",
    "description": "A collection of articles",
    "vectorizer": "text2vec-openai", # Use the OpenAI module for embeddings
    "moduleConfig": {
        "text2vec-openai": {
          "model": "text-embedding-3-small", # Specify the embedding model
          "type": "text"
        }
    },
    "properties": [
        {
            "name": "title",
            "description": "Title of the article",
            "dataType": ["string"]
        },
        {
            "name": "content",
            "description": "Contents of the article",
            "dataType": ["text"]
        },
        {
            "name": "url",
            "description": "URL to the article",
            "dataType": ["string"],
            "moduleConfig": { "text2vec-openai": { "skip": True } } # Do not vectorize the URL
        }
    ]
}

# Create the class in Weaviate
client.schema.create_class(article_schema)

# Retrieve and print the schema to confirm
print("Current schema:", client.schema.get())
```

**Key Configuration Notes:**
*   `vectorizer`: Tells Weaviate to use the `text2vec-openai` module.
*   `model`: Defines which OpenAI embedding model to use (e.g., `text-embedding-3-small`).
*   `skip`: Setting this to `True` for the `url` property instructs Weaviate not to generate embeddings for it, as it's not useful for semantic search.

## Step 4: Import Your Data

With the schema in place, you can now import data. Weaviate will automatically call the OpenAI API to generate vector embeddings for each object during import.

### Load the Sample Dataset

We'll use a subset of the Simple Wikipedia dataset for this example.

```python
from datasets import load_dataset

# Load the dataset
dataset = list(load_dataset("wikipedia", "20220301.simple")["train"])

# Limit the dataset for demonstration (adjust as needed)
dataset = dataset[:2500]  # Use 2500 articles
```

### Configure Batch Import

Batch importing improves efficiency by sending multiple objects in a single request.

```python
client.batch.configure(
    batch_size=100,      # Number of objects per batch
    dynamic=True,        # Dynamically adjust batch size based on performance
    timeout_retries=3,   # Retry failed requests
)
```

### Perform the Import

Now, iterate through the dataset and add each article to Weaviate.

```python
print("Importing Articles...")

counter = 0
with client.batch as batch:
    for article in dataset:
        # Optional: Print progress every 100 articles
        if counter % 100 == 0:
            print(f"Imported {counter} / {len(dataset)} articles")

        # Define the data object properties
        properties = {
            "title": article["title"],
            "content": article["text"],
            "url": article["url"]
        }
        # Add the object to the current batch
        batch.add_data_object(properties, "Article")
        counter += 1

print("Import complete!")
```

### Verify the Import

Check that your data was successfully loaded.

```python
# Count the total number of Article objects
result = client.query.aggregate("Article").with_fields("meta { count }").do()
print("Object count:", result["data"]["Aggregate"]["Article"][0]["meta"]["count"])

# Fetch and inspect one sample article
sample_article = client.query.get(
    "Article", ["title", "url", "content"]
).with_limit(1).do()

article_data = sample_article["data"]["Get"]["Article"][0]
print("\nSample Article Title:", article_data['title'])
print("Sample Article URL:", article_data['url'])
print("Content preview:", article_data['content'][:200], "...")
```

## Step 5: Perform Hybrid Search Queries

Hybrid search in Weaviate combines vector search (for semantic meaning) and BM25 search (for keyword matching). The `alpha` parameter controls the blend:
*   `alpha=1`: Pure vector search.
*   `alpha=0`: Pure keyword search.
*   `alpha=0.5`: Equal weight to both methods.

### Define a Search Function

Create a helper function to execute hybrid queries.

```python
def hybrid_query_weaviate(query, collection_name, alpha=0.5):
    """
    Executes a hybrid search query.
    
    Args:
        query (str): The search query string.
        collection_name (str): The Weaviate class name to search (e.g., 'Article').
        alpha (float): Weight between vector (alpha) and keyword (1-alpha) search.
    
    Returns:
        list: Search results.
    """
    properties = [
        "title", "content", "url",
        "_additional { score }"  # Include the relevance score
    ]

    result = client.query.get(
        collection_name, properties
    ).with_hybrid(
        query=query,
        alpha=alpha
    ).with_limit(10).do()

    # Handle potential API errors (e.g., rate limits)
    if "errors" in result:
        print("Error:", result["errors"][0]['message'])
        return []
    
    return result["data"]["Get"][collection_name]
```

### Run Example Queries

Test your search with different queries.

```python
print("Query: 'modern art in Europe'\n")
results = hybrid_query_weaviate("modern art in Europe", "Article", 0.5)

for i, article in enumerate(results):
    print(f"{i+1}. {article['title']} (Score: {article['_additional']['score']:.3f})")
```

```python
print("\nQuery: 'Famous battles in Scottish history'\n")
results = hybrid_query_weaviate("Famous battles in Scottish history", "Article", 0.5)

for i, article in enumerate(results):
    print(f"{i+1}. {article['title']} (Score: {article['_additional']['score']:.3f})")
```

## Conclusion

You have successfully set up a Weaviate vector database, configured it to use OpenAI embeddings, imported a dataset, and performed hybrid searches. This foundation enables you to build powerful AI applications like semantic search engines, retrieval-augmented generation (RAG) systems, and intelligent chatbots.

**Next Steps:**
*   Experiment with the `alpha` parameter to fine-tune search results.
*   Explore Weaviate's [GraphQL API](https://weaviate.io/developers/weaviate/api/graphql) for more complex queries and filters.
*   Integrate this search backend with a large language model (LLM) to create a full RAG pipeline.