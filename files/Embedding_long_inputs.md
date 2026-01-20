# Embedding texts that are longer than the model's maximum context length

OpenAI's embedding models cannot embed text that exceeds a maximum length. The maximum length varies by model, and is measured by _tokens_, not string length. If you are unfamiliar with tokenization, check out [How to count tokens with tiktoken](How_to_count_tokens_with_tiktoken.ipynb).

This notebook shows how to handle texts that are longer than a model's maximum context length. We'll demonstrate using embeddings from `text-embedding-3-small`, but the same ideas can be applied to other models and tasks. To learn more about embeddings, check out the [OpenAI Embeddings Guide](https://beta.openai.com/docs/guides/embeddings).

## 1. Model context length

First, we select the model and define a function to get embeddings from the API.

```python
from openai import OpenAI
import os
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

EMBEDDING_MODEL = 'text-embedding-3-small'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

# let's make sure to not retry on an invalid request, because that is what we want to demonstrate
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), retry=retry_if_not_exception_type(openai.BadRequestError))
def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
    return client.embeddings.create(input=text_or_tokens, model=model).data[0].embedding
```

The `text-embedding-3-small` model has a context length of 8191 tokens with the `cl100k_base` encoding, and we can see that going over that limit causes an error.

```python
long_text = 'AGI ' * 5000
try:
    get_embedding(long_text)
except openai.BadRequestError as e:
    print(e)
```

Clearly we want to avoid these errors, particularly when handling programmatically with a large number of embeddings. Yet, we still might be faced with texts that are longer than the maximum context length. Below we describe and provide recipes for the main approaches to handling these longer texts: (1) simply truncating the text to the maximum allowed length, and (2) chunking the text and embedding each chunk individually.

## 1. Truncating the input text

The simplest solution is to truncate the input text to the maximum allowed length. Because the context length is measured in tokens, we have to first tokenize the text before truncating it. The API accepts inputs both in the form of text or tokens, so as long as you are careful that you are using the appropriate encoding, there is no need to convert the tokens back into string form. Below is an example of such a truncation function.

```python
import tiktoken

def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]
```

Our example from before now works without error.

```python
truncated = truncate_text_tokens(long_text)
len(get_embedding(truncated))
```

## 2. Chunking the input text

Though truncation works, discarding potentially relevant text is a clear drawback. Another approach is to divide the input text into chunks and then embed each chunk individually. Then, we can either use the chunk embeddings separately, or combine them in some way, such as averaging (weighted by the size of each chunk).

We will take a function from [Python's own cookbook](https://docs.python.org/3/library/itertools.html#itertools-recipes) that breaks up a sequence into chunks.

```python
from itertools import islice

def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch
```

Now we define a function that encodes a string into tokens and then breaks it up into chunks.

```python
def chunked_tokens(text, encoding_name, chunk_length):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator
```

Finally, we can write a function that safely handles embedding requests, even when the input text is longer than the maximum context length, by chunking the input tokens and embedding each chunk individually. The `average` flag can be set to `True` to return the weighted average of the chunk embeddings, or `False` to simply return the unmodified list of chunk embeddings.

```python
import numpy as np


def len_safe_get_embedding(text, model=EMBEDDING_MODEL, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING, average=True):
    chunk_embeddings = []
    chunk_lens = []
    for chunk in chunked_tokens(text, encoding_name=encoding_name, chunk_length=max_tokens):
        chunk_embeddings.append(get_embedding(chunk, model=model))
        chunk_lens.append(len(chunk))

    if average:
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings
```

Once again, we can now handle long input texts.

```python
average_embedding_vector = len_safe_get_embedding(long_text, average=True)
chunks_embedding_vectors = len_safe_get_embedding(long_text, average=False)

print(f"Setting average=True gives us a single {len(average_embedding_vector)}-dimensional embedding vector for our long text.")
print(f"Setting average=False gives us {len(chunks_embedding_vectors)} embedding vectors, one for each of the chunks.")
```

In some cases, it may make sense to split chunks on paragraph boundaries or sentence boundaries to help preserve the meaning of the text.