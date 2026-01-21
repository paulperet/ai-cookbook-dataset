# Building a Semantic Search Engine with Weaviate and OpenAI

This guide walks you through creating a vector search engine using Weaviate with OpenAI's embedding models. You'll learn how to store, index, and semantically search through documents without manually handling vectorization.

## Prerequisites

Before starting, ensure you have:
- A Weaviate instance (cloud or local)
- An OpenAI API key
- Python 3.7+

## Setup

### 1. Install Required Libraries

First, install the necessary Python packages:

```bash
pip install weaviate-client>=3.11.0
pip install datasets apache-beam
```

### 2. Configure Your OpenAI API Key

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Verify the key is properly set:

```python
import os

if os.getenv("OPENAI_API_KEY") is not None:
    print("OPENAI_API_KEY is ready")
else:
    print("OPENAI_API_KEY environment variable not found")
```

## Step 1: Connect to Your Weaviate Instance

Connect to your Weaviate instance using the Python client. Replace the placeholders with your actual instance details:

```python
import weaviate
import os

# Connect to your Weaviate instance
client = weaviate.Client(
    url="https://your-wcs-instance-name.weaviate.network/",  # Your WCS URL
    # url="http://localhost:8080/",  # Use this for local instances
    auth_client_secret=weaviate.auth.AuthApiKey(
        api_key="<YOUR-WEAVIATE-API-KEY>"
    ),  # Comment out if not using authentication
    additional_headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    }
)

# Verify the connection
print("Connection ready:", client.is_ready())
```

## Step 2: Define Your Data Schema

The schema tells Weaviate how to structure and vectorize your data. We'll create a schema for articles that uses OpenAI's `text-embedding-3-small` model:

```python
# Clear any existing schema
client.schema.delete_all()

# Define the article schema
article_schema = {
    "class": "Article",
    "description": "A collection of articles",
    "vectorizer": "text2vec-openai",
    "moduleConfig": {
        "text2vec-openai": {
            "model": "text-embedding-3-small",
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
            "moduleConfig": {
                "text2vec-openai": {"skip": True}  # Don't vectorize URLs
            }
        }
    ]
}

# Create the schema
client.schema.create_class(article_schema)

# Verify the schema was created
print("Schema created:", client.schema.get())
```

**Key Points:**
- The `vectorizer` specifies we're using OpenAI's text-to-vector module
- We configure the specific embedding model (`text-embedding-3-small`)
- The `skip` option on the `url` property tells Weaviate not to vectorize URLs

## Step 3: Load and Import Data

We'll use the Simple Wikipedia dataset as our sample data. Weaviate will automatically vectorize the content using OpenAI's API during import.

### 3.1 Load the Dataset

```python
from datasets import load_dataset

# Load the Simple Wikipedia dataset
dataset = list(load_dataset("wikipedia", "20220301.simple")["train"])

# Limit to 2,500 articles for demonstration
dataset = dataset[:2500]

print(f"Loaded {len(dataset)} articles")
```

### 3.2 Configure Batch Import

Batch importing improves performance by sending multiple objects at once:

```python
client.batch.configure(
    batch_size=100,  # Start with 100 objects per batch
    dynamic=True,    # Automatically adjust batch size
    timeout_retries=3,  # Retry failed imports
)
```

### 3.3 Import the Data

```python
print("Importing articles...")

counter = 0
with client.batch as batch:
    for article in dataset:
        if counter % 100 == 0:
            print(f"Imported {counter}/{len(dataset)} articles")
        
        properties = {
            "title": article["title"],
            "content": article["text"],
            "url": article["url"]
        }
        
        batch.add_data_object(properties, "Article")
        counter += 1

print("Import complete!")
```

### 3.4 Verify the Import

Check that your data was successfully imported:

```python
# Count imported objects
result = client.query.aggregate("Article").with_fields("meta { count }").do()
print("Total articles imported:", result["data"]["Aggregate"]["Article"][0]["meta"]["count"])

# Inspect one sample article
sample_article = client.query.get(
    "Article", ["title", "url", "content"]
).with_limit(1).do()

article_data = sample_article["data"]["Get"]["Article"][0]
print(f"\nSample article:")
print(f"Title: {article_data['title']}")
print(f"URL: {article_data['url']}")
print(f"Content preview: {article_data['content'][:200]}...")
```

## Step 4: Perform Semantic Searches

Now that your data is indexed, you can perform semantic searches. We'll create a helper function for querying:

```python
def query_weaviate(query, collection_name="Article", limit=10):
    """
    Perform a semantic search on the specified collection.
    
    Args:
        query: Search query string
        collection_name: Name of the collection to search
        limit: Maximum number of results to return
    
    Returns:
        List of matching articles with metadata
    """
    nearText = {
        "concepts": [query],
        "distance": 0.7,  # Maximum distance threshold
    }
    
    properties = [
        "title", "content", "url",
        "_additional {certainty distance}"  # Include similarity scores
    ]
    
    result = (
        client.query
        .get(collection_name, properties)
        .with_near_text(nearText)
        .with_limit(limit)
        .do()
    )
    
    # Handle potential errors
    if "errors" in result:
        error_msg = result["errors"][0]['message']
        print("Error:", error_msg)
        if "rate limit" in error_msg.lower():
            print("You may have exceeded OpenAI's API rate limit (60 calls/minute).")
        raise Exception(error_msg)
    
    return result["data"]["Get"][collection_name]
```

### 4.1 Test Your Search

Let's run some example queries:

```python
# Search for articles about modern art
print("Searching for 'modern art in Europe':")
results = query_weaviate("modern art in Europe")

for i, article in enumerate(results):
    score = round(article['_additional']['certainty'], 3)
    print(f"{i+1}. {article['title']} (Score: {score})")
    print(f"   URL: {article['url']}")
    print()
```

```python
# Search for historical battles
print("Searching for 'Famous battles in Scottish history':")
results = query_weaviate("Famous battles in Scottish history")

for i, article in enumerate(results):
    score = round(article['_additional']['certainty'], 3)
    print(f"{i+1}. {article['title']} (Score: {score})")
```

## Step 5: Understanding the Results

The search results include:
- **Title and Content**: The actual article data
- **URL**: The source URL (not vectorized)
- **Certainty Score**: How closely the article matches your query (0-1 scale)
- **Distance**: The vector distance (lower is better)

**Key Insights:**
1. **Automatic Vectorization**: Weaviate handled all embedding generation via OpenAI
2. **Semantic Understanding**: Searches match meaning, not just keywords
3. **Configurable Similarity**: The `distance` parameter controls match strictness

## Next Steps

Now that you have a working vector search engine, consider exploring:

1. **Hybrid Search**: Combine vector search with keyword filtering
2. **Custom Schemas**: Adapt the schema for your specific data types
3. **Production Scaling**: Increase dataset size and optimize performance
4. **Multi-modal Search**: Add image or audio vectorization capabilities

## Troubleshooting

**Common Issues:**

1. **OpenAI Rate Limits**: Free accounts have 60 requests/minute limits
2. **Authentication Errors**: Verify your Weaviate and OpenAI API keys
3. **Import Failures**: Reduce batch size or check network connectivity
4. **Schema Conflicts**: Clear existing schemas before creating new ones

**To Reset Everything:**
```python
# Clear all data and schemas
client.schema.delete_all()
print("All data and schemas cleared")
```

## Conclusion

You've successfully built a semantic search engine using Weaviate and OpenAI embeddings. The key advantage of this approach is the complete abstraction of vectorization—Weaviate automatically handles embedding generation and storage, allowing you to focus on building search applications rather than managing vector pipelines.

This foundation enables powerful applications like intelligent chatbots, content recommendation systems, and knowledge discovery tools.