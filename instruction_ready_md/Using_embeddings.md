# Efficient Text Embedding with OpenAI's API

This guide demonstrates how to generate text embeddings using OpenAI's `text-embedding-3-small` model. You'll learn the basic method and a production-ready best practice that implements exponential backoff to handle API rate limits gracefully.

## Prerequisites

First, ensure you have the required Python packages installed.

```bash
pip install openai tenacity
```

## 1. Basic Embedding Generation

Let's start with the simplest way to create an embedding. This snippet is useful for quick, one-off tasks.

```python
from openai import OpenAI

# Initialize the client
client = OpenAI()

# Generate an embedding for a single piece of text
embedding = client.embeddings.create(
    input="Your text goes here",
    model="text-embedding-3-small"
).data[0].embedding

# Check the dimensionality of the embedding vector
print(f"Embedding dimension: {len(embedding)}")
```

This will output the length of the embedding vector (e.g., `1536` for the `text-embedding-3-small` model), confirming the operation was successful.

## 2. The Problem: Rate Limits

When generating embeddings at scale, you will quickly encounter API rate limits. The naive approach below is inefficient and will fail under load.

```python
from openai import OpenAI
client = OpenAI()

num_embeddings = 10000  # A large batch
for i in range(num_embeddings):
    # This loop will likely trigger rate limit errors
    embedding = client.embeddings.create(
        input="Your text goes here",
        model="text-embedding-3-small"
    ).data[0].embedding
    print(len(embedding))
```

Running this loop will cause the OpenAI API to throttle your requests, resulting in failed calls and incomplete data.

## 3. Best Practice: Embedding with Exponential Backoff

For reliable, production-grade embedding generation, you should implement retry logic with exponential backoff. The `tenacity` library makes this straightforward.

The following function will retry failed API calls up to 6 times. The wait time between retries increases exponentially (with some randomness), starting at 1 second and capping at 20 seconds. This pattern is respectful of the API's rate limits and maximizes throughput under constraints.

```python
from tenacity import retry, wait_random_exponential, stop_after_attempt
from openai import OpenAI

# Initialize the client
client = OpenAI()

# Define a robust embedding function with retry logic
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Generates an embedding for the given text with automatic retries."""
    # Note: The input is passed as a list
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Use the function
embedding = get_embedding("Your text goes here", model="text-embedding-3-small")
print(f"Embedding dimension: {len(embedding)}")
```

**Why this works:**
*   **`@retry` Decorator:** Automatically re-executes the function if it raises an exception.
*   **`wait_random_exponential`:** Increases the wait time between retries, helping to clear temporary API limits.
*   **`stop_after_attempt(6)`:** Prevents infinite retry loops by giving up after 6 attempts.

You can now safely call `get_embedding()` in a loop or as part of a larger data processing pipeline. This function will handle temporary network issues and API rate limits transparently, ensuring your application remains robust.