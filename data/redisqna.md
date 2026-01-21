# Redis as a Context Store with OpenAI Chat: A Step-by-Step Guide

This guide demonstrates how to use Redis as a high-speed, vector-based memory store to provide ChatGPT with up-to-date context, enabling it to answer questions about events beyond its built-in knowledge cutoff date.

## Prerequisites

Before you begin, ensure you have:
*   A running Redis instance with the **Redis Search** and **Redis JSON** modules enabled.
*   An **OpenAI API key**.
*   Python 3.7+ installed.

## Step 1: Environment Setup

First, install the required Python libraries.

```bash
pip install redis openai python-dotenv
```

Create a `.env` file in your project directory to securely store your OpenAI API key.

```bash
# .env
OPENAI_API_KEY=your_actual_api_key_here
```

## Step 2: Configure OpenAI Client

Load your API key and create a helper function for interacting with the OpenAI Chat Completions API.

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize the OpenAI client
oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_completion(prompt, model="gpt-3.5-turbo"):
    """A helper function to get a completion from the specified OpenAI model."""
    messages = [{"role": "user", "content": prompt}]
    response = oai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # Set temperature to 0 for deterministic outputs
    )
    return response.choices[0].message.content
```

## Step 3: Identify the Knowledge Gap

To illustrate the problem, let's ask a question about an event that occurred after the model's training data cutoff. `gpt-3.5-turbo` was trained on data up to September 2021. We'll ask about the FTX scandal, which unfolded in late 2022.

```python
prompt = "Is Sam Bankman-Fried's company, FTX, considered a well-managed company?"
response = get_completion(prompt)
print(response)
```

**Observation:** The model will likely provide a confident but incorrect or outdated answer, as it lacks the necessary context.

## Step 4: Mitigate Guessing with Prompt Engineering

One simple mitigation is to instruct the model to admit uncertainty.

```python
prompt = "Is Sam Bankman-Fried's company, FTX, considered a well-managed company? If you don't know for certain, say unknown."
response = get_completion(prompt)
print(response)
```

While this is safer, it doesn't provide the user with the correct information. The better solution is to *give* the model the context it needs.

## Step 5: Set Up Redis as a Context Store

We will use Redis to store relevant documents (e.g., news articles) and their vector embeddings. This allows us to perform a semantic search to find the most relevant context for our question.

### 5.1 Start Redis

Ensure your Redis instance is running. If you're using Docker, you can start a container with the required modules.

```bash
docker run -d -p 6379:6379 redis/redis-stack:latest
```

### 5.2 Connect to Redis

Initialize the Redis client in Python.

```python
from redis import from_url

REDIS_URL = 'redis://localhost:6379'
client = from_url(REDIS_URL)
# Test the connection
print("Connected to Redis:", client.ping())
```

### 5.3 Create a Vector Search Index

We need to create an index on the JSON documents we'll store, specifying a field for vector similarity search (VSS).

```python
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# Define the index schema: a vector field and a text field.
schema = [
    VectorField('$.vector',  # Path to the vector in the JSON
                "FLAT",      # Index type
                {"TYPE": 'FLOAT32',
                 "DIM": 1536, # Dimension of OpenAI's text-embedding-3-small vectors
                 "DISTANCE_METRIC": "COSINE"
                }, as_name='vector'),
    TextField('$.content', as_name='content') # The actual text content
]

# Define that the index applies to JSON documents with the prefix 'doc:'
idx_def = IndexDefinition(index_type=IndexType.JSON, prefix=['doc:'])

# Drop the index if it exists (for a clean start in this tutorial)
try:
    client.ft('idx').dropindex()
except:
    pass

# Create the index
client.ft('idx').create_index(schema, definition=idx_def)
print("Vector search index 'idx' created.")
```

## Step 6: Load Context Data into Redis

Now, we'll load text documents (e.g., from a local `./assets/` folder), generate embeddings for them using OpenAI, and store them in Redis.

```python
import os

directory = './assets/'  # Directory containing your .txt files
model = 'text-embedding-3-small' # OpenAI's efficient embedding model
doc_id = 1

for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

            # Generate an embedding vector for the document's content
            response = oai_client.embeddings.create(
                model=model,
                input=[content]
            )
            vector = response.data[0].embedding

            # Store the document and its vector as a JSON object in Redis
            client.json().set(f'doc:{doc_id}', '$', {'content': content, 'vector': vector})
            print(f"Loaded document: doc:{doc_id}")
            doc_id += 1
```

## Step 7: Retrieve Relevant Context for a Question

When a user asks a question, we follow this process:
1.  Embed the user's question.
2.  Use vector similarity search in Redis to find the most relevant stored document.

```python
from redis.commands.search.query import Query
import numpy as np

# 1. Embed the user's question
user_question = "Is Sam Bankman-Fried's company, FTX, considered a well-managed company?"

response = oai_client.embeddings.create(
    input=[user_question],
    model=model
)
embedding_vector = response.data[0].embedding

# 2. Prepare the vector for the Redis query (convert to bytes)
query_vec = np.array(embedding_vector, dtype=np.float32).tobytes()

# 3. Build the K-Nearest Neighbors (KNN) query
q = (Query('*=>[KNN 1 @vector $query_vec AS vector_score]')
     .sort_by('vector_score')
     .return_fields('content')
     .dialect(2)) # Dialect 2 is required for vector queries

# 4. Execute the search
params = {"query_vec": query_vec}
search_result = client.ft('idx').search(q, query_params=params)

if len(search_result.docs) > 0:
    context = search_result.docs[0].content
    print("Most relevant context retrieved successfully.\n")
    # Optional: Print a snippet of the context
    print(context[:500] + "...")
else:
    context = ""
    print("No relevant context found.")
```

## Step 8: Augment the Prompt and Get an Informed Answer

Finally, we construct a new prompt that includes the retrieved context and ask the model again. This time, it has the information needed to answer accurately.

```python
augmented_prompt = f"""
Using the information delimited by triple backticks, answer this question: {user_question}

Context: ```{context}```
"""

informed_response = get_completion(augmented_prompt)
print("\n=== Informed Response from GPT ===")
print(informed_response)
```

## Conclusion

You have successfully built a system that uses **Redis as a high-speed context store**. By combining vector embeddings with Redis's vector search capabilities, you can dynamically provide large language models with relevant, up-to-date information, overcoming their inherent knowledge limitations. This pattern is the foundation for advanced Retrieval-Augmented Generation (RAG) applications.