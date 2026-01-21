# Handling Long Texts with OpenAI Embeddings: A Step-by-Step Guide

OpenAI's embedding models have a maximum context length measured in tokens. When your text exceeds this limit, the API will return an error. This guide shows you how to handle long texts using two practical approaches: truncation and chunking.

We'll use the `text-embedding-3-small` model for demonstration, but these techniques apply to other embedding models as well.

## Prerequisites

First, ensure you have the required libraries installed and your environment configured.

```bash
pip install openai tiktoken numpy tenacity
```

## Setup: Import Libraries and Configure Client

Start by importing the necessary libraries and setting up your OpenAI client.

```python
from openai import OpenAI
import os
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type
import tiktoken
import numpy as np
from itertools import islice

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

# Model configuration
EMBEDDING_MODEL = 'text-embedding-3-small'
EMBEDDING_CTX_LENGTH = 8191  # Maximum tokens for this model
EMBEDDING_ENCODING = 'cl100k_base'  # Tokenizer for this model

# Embedding function with retry logic
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), retry=retry_if_not_exception_type(openai.BadRequestError))
def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
    """Get embeddings from the OpenAI API."""
    return client.embeddings.create(input=text_or_tokens, model=model).data[0].embedding
```

## Understanding the Problem: Context Length Limits

The `text-embedding-3-small` model accepts a maximum of 8191 tokens. Let's see what happens when we exceed this limit.

```python
# Create a text that exceeds the token limit
long_text = 'AGI ' * 5000

try:
    get_embedding(long_text)
except openai.BadRequestError as e:
    print(f"Error: {e}")
```

This will produce an error similar to:
```
Error: This model's maximum context length is 8191 tokens...
```

Now let's explore two solutions to handle this situation.

## Method 1: Truncating Long Texts

The simplest approach is to truncate your text to fit within the model's token limit.

### Step 1: Create a Truncation Function

```python
def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]
```

### Step 2: Apply Truncation

```python
# Truncate the long text to the maximum token length
truncated_tokens = truncate_text_tokens(long_text)

# Get embeddings for the truncated text
embedding = get_embedding(truncated_tokens)
print(f"Successfully created embedding with {len(embedding)} dimensions")
```

**Note:** Truncation is fast and simple, but it discards potentially important information from the end of your text.

## Method 2: Chunking Long Texts

A more sophisticated approach splits your text into manageable chunks and embeds each chunk separately. You can then use the individual chunk embeddings or combine them.

### Step 1: Create Helper Functions

First, we need a function to batch data into chunks:

```python
def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch
```

Next, create a function to chunk text into token sequences:

```python
def chunked_tokens(text, encoding_name, chunk_length):
    """Encode text into tokens and split into chunks of specified length."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator
```

### Step 2: Create the Safe Embedding Function

Now, let's build the main function that handles long texts by chunking:

```python
def len_safe_get_embedding(text, model=EMBEDDING_MODEL, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING, average=True):
    """
    Get embeddings for text that may exceed the model's token limit.
    
    Args:
        text: The input text to embed
        model: The embedding model to use
        max_tokens: Maximum tokens per chunk
        encoding_name: Tokenizer encoding name
        average: If True, return weighted average of chunk embeddings
                 If False, return list of individual chunk embeddings
    
    Returns:
        Either a single averaged embedding vector or a list of chunk embeddings
    """
    chunk_embeddings = []
    chunk_lens = []
    
    # Process text in chunks
    for chunk in chunked_tokens(text, encoding_name=encoding_name, chunk_length=max_tokens):
        chunk_embeddings.append(get_embedding(chunk, model=model))
        chunk_lens.append(len(chunk))
    
    if average:
        # Calculate weighted average of chunk embeddings
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        # Normalize the resulting vector to unit length
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        chunk_embeddings = chunk_embeddings.tolist()
    
    return chunk_embeddings
```

### Step 3: Use the Safe Embedding Function

Let's test both modes of the function:

```python
# Get a single averaged embedding vector
average_embedding = len_safe_get_embedding(long_text, average=True)
print(f"Averaged embedding: Single vector with {len(average_embedding)} dimensions")

# Get individual chunk embeddings
chunk_embeddings = len_safe_get_embedding(long_text, average=False)
print(f"Chunk embeddings: {len(chunk_embeddings)} vectors, one for each chunk")
```

## Choosing Between Methods

### When to Use Truncation:
- Your text is only slightly longer than the limit
- The most important information is at the beginning of the text
- You need the simplest, fastest solution

### When to Use Chunking:
- Your text is significantly longer than the limit
- All parts of the text contain important information
- You need to preserve the complete semantic content

## Advanced Considerations

For better semantic preservation, consider splitting your text on natural boundaries:

1. **Paragraph Boundaries:** Split at double newlines (`\n\n`)
2. **Sentence Boundaries:** Use NLP libraries like spaCy or NLTK
3. **Semantic Chunking:** Use embeddings to find natural break points

Here's a simple example of paragraph-based chunking:

```python
def chunk_by_paragraphs(text, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING):
    """Chunk text by paragraphs, ensuring no chunk exceeds max_tokens."""
    encoding = tiktoken.get_encoding(encoding_name)
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for paragraph in paragraphs:
        paragraph_tokens = encoding.encode(paragraph)
        paragraph_length = len(paragraph_tokens)
        
        if current_length + paragraph_length > max_tokens and current_chunk:
            # Save current chunk and start new one
            chunks.append(encoding.decode(current_chunk))
            current_chunk = paragraph_tokens
            current_length = paragraph_length
        else:
            # Add paragraph to current chunk
            current_chunk.extend(paragraph_tokens)
            current_length += paragraph_length
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(encoding.decode(current_chunk))
    
    return chunks
```

## Summary

You now have two reliable methods to handle texts that exceed OpenAI's embedding model limits:

1. **Truncation:** Simple and fast, but discards information
2. **Chunking:** More complex but preserves all content

The `len_safe_get_embedding()` function provides a complete solution that automatically handles long texts by chunking them and optionally averaging the resulting embeddings. Choose the method that best fits your use case and performance requirements.