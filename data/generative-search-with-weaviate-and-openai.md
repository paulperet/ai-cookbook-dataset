# Guide: Generative Search with Weaviate and OpenAI

This guide walks you through using Weaviate's Generative OpenAI module to perform generative search queries on your existing data. You'll learn how to generate responses for individual search results and for groups of results.

## Prerequisites

Before starting, ensure you have:

1. Completed the [Getting Started with Weaviate and OpenAI](getting-started-with-weaviate-and-openai.ipynb) guide
2. A running Weaviate instance with data imported
3. An [OpenAI API key](https://beta.openai.com/account/api-keys)

## Setup

### 1. Configure Your OpenAI API Key

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

Verify the key is properly set:

```python
import os

if os.getenv("OPENAI_API_KEY") is not None:
    print("OPENAI_API_KEY is ready")
else:
    print("OPENAI_API_KEY environment variable not found")
```

### 2. Connect to Your Weaviate Instance

Establish a connection to your Weaviate instance with the necessary authentication headers:

```python
import weaviate
import os

# Connect to your Weaviate instance
client = weaviate.Client(
    url="https://your-wcs-instance-name.weaviate.network/",  # Your Weaviate URL
    # url="http://localhost:8080/",  # Use this for local instances
    auth_client_secret=weaviate.auth.AuthApiKey(
        api_key="<YOUR-WEAVIATE-API-KEY>"
    ),  # Comment out if not using authentication
    additional_headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    }
)

# Verify the connection
print(f"Connection ready: {client.is_ready()}")
```

## Understanding Generative Search

Weaviate's Generative OpenAI module generates responses based on data retrieved from your vector database. The query structure is similar to standard semantic search, with the addition of a `with_generate()` function that accepts either:

- `single_prompt`: Generate a response for each returned object
- `grouped_task`: Generate a single response from all returned objects

## Step 1: Generative Search Per Item

First, let's create a function that generates a response for each search result individually. This is useful when you want to process each matching document separately.

```python
def generative_search_per_item(query, collection_name):
    """
    Perform generative search with individual prompts for each result.
    
    Args:
        query: Search query text
        collection_name: Name of the Weaviate collection to search
    
    Returns:
        List of search results with generated responses
    """
    prompt = "Summarize in a short tweet the following content: {content}"
    
    result = (
        client.query
        .get(collection_name, ["title", "content", "url"])
        .with_near_text({"concepts": [query], "distance": 0.7})
        .with_limit(5)
        .with_generate(single_prompt=prompt)
        .do()
    )
    
    # Check for errors
    if "errors" in result:
        print("You probably have run out of OpenAI API calls for the current minute.")
        raise Exception(result["errors"][0]['message'])
    
    return result["data"]["Get"][collection_name]
```

Now, let's test this function with a sample query:

```python
# Search for articles about football clubs
query_result = generative_search_per_item("football clubs", "Article")

# Display results with generated summaries
for i, article in enumerate(query_result):
    print(f"{i+1}. {article['title']}")
    print(f"Generated summary: {article['_additional']['generate']['singleResult']}")
    print("-" * 40)
```

The `{content}` placeholder in the prompt template is automatically replaced with the actual content from each search result.

## Step 2: Generative Search for Groups

Next, let's create a function that generates a single response based on all search results. This is useful for finding common themes or patterns across multiple documents.

```python
def generative_search_group(query, collection_name):
    """
    Perform generative search with a grouped task across all results.
    
    Args:
        query: Search query text
        collection_name: Name of the Weaviate collection to search
    
    Returns:
        Search results with a grouped generated response
    """
    generate_task = "Explain what these have in common"
    
    result = (
        client.query
        .get(collection_name, ["title", "content", "url"])
        .with_near_text({"concepts": [query], "distance": 0.7})
        .with_generate(grouped_task=generate_task)
        .with_limit(5)
        .do()
    )
    
    # Check for errors
    if "errors" in result:
        print("You probably have run out of OpenAI API calls for the current minute.")
        raise Exception(result["errors"][0]['message'])
    
    return result["data"]["Get"][collection_name]
```

Test the grouped search function:

```python
# Search for articles about football clubs with grouped analysis
query_result = generative_search_group("football clubs", "Article")

# Display the grouped analysis
print("Common themes across all results:")
print(query_result[0]['_additional']['generate']['groupedResult'])
```

## Key Concepts

### Distance Parameter
The `distance` parameter in `with_near_text()` controls the similarity threshold (0-1). Lower values mean stricter similarity matching.

### Error Handling
The code includes error handling for OpenAI API rate limits (60 calls per minute). If you encounter errors, wait a minute before retrying.

### Prompt Engineering
You can customize the prompts to suit your specific use case:
- For `single_prompt`: Use `{property_name}` placeholders to reference document properties
- For `grouped_task`: Frame questions that require synthesis across multiple documents

## Next Steps

You're now equipped to perform generative searches with Weaviate and OpenAI. Experiment with different prompts, adjust the similarity thresholds, and explore combining generative search with other Weaviate features like filters and aggregations.

For more advanced use cases, explore other cookbooks in this repository that cover topics like hybrid search, custom modules, and production deployment strategies.