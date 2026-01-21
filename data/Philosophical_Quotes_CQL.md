# Building a Philosophy Quote Finder & Generator with Vector Embeddings, OpenAI, and Cassandra/Astra DB

This guide walks you through building a "philosophy quote finder & generator" using OpenAI's vector embeddings and Apache Cassandra® (or DataStax Astra DB through CQL) as the vector store. You'll learn how to store, search, and generate philosophical quotes using semantic similarity.

## How It Works

**Indexing**: Each quote is converted into an embedding vector using OpenAI's `Embedding` model. These vectors, along with metadata like author and tags, are stored in Cassandra/Astra DB.

**Search**: To find similar quotes, your search query is converted into a vector on the fly. The database performs an approximate nearest neighbor (ANN) search to return the most semantically similar quotes. You can optionally filter by author or tags.

**Generation**: Given a topic or tentative quote, the search step retrieves similar quotes, which are then fed into an LLM prompt to generate new, original philosophical text.

## Prerequisites

Before starting, ensure you have:
- An Astra DB instance (or Cassandra cluster with vector capabilities)
- Your Astra DB Secure Connect Bundle
- Your Astra DB Application Token
- An OpenAI API key

## Setup

Install the required dependencies:

```bash
pip install --quiet "cassandra-driver>=0.28.0" "openai>=1.0.0" datasets
```

Import the necessary modules:

```python
import os
from uuid import uuid4
from getpass import getpass
from collections import Counter

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

import openai
from datasets import load_dataset
```

## Step 1: Connect to Your Database

First, gather your connection credentials:

```python
# Detect if running in Google Colab
try:
    from google.colab import files
    IS_COLAB = True
except ModuleNotFoundError:
    IS_COLAB = False

# Get your Secure Connect Bundle path
if IS_COLAB:
    print('Please upload your Secure Connect Bundle zipfile: ')
    uploaded = files.upload()
    if uploaded:
        astraBundleFileTitle = list(uploaded.keys())[0]
        ASTRA_DB_SECURE_BUNDLE_PATH = os.path.join(os.getcwd(), astraBundleFileTitle)
    else:
        raise ValueError('Cannot proceed without Secure Connect Bundle. Please re-run the cell.')
else:
    ASTRA_DB_SECURE_BUNDLE_PATH = input("Please provide the full path to your Secure Connect Bundle zipfile: ")

# Get your database credentials
ASTRA_DB_APPLICATION_TOKEN = getpass("Please provide your Database Token ('AstraCS:...' string): ")
ASTRA_DB_KEYSPACE = input("Please provide the Keyspace name for your Database: ")
```

Now establish the database connection:

```python
cluster = Cluster(
    cloud={
        "secure_connect_bundle": ASTRA_DB_SECURE_BUNDLE_PATH,
    },
    auth_provider=PlainTextAuthProvider(
        "token",
        ASTRA_DB_APPLICATION_TOKEN,
    ),
)

session = cluster.connect()
keyspace = ASTRA_DB_KEYSPACE
```

## Step 2: Create the Vector Table

Create a table to store quotes, their vector embeddings, and metadata:

```python
create_table_statement = f"""CREATE TABLE IF NOT EXISTS {keyspace}.philosophers_cql (
    quote_id UUID PRIMARY KEY,
    body TEXT,
    embedding_vector VECTOR<FLOAT, 1536>,
    author TEXT,
    tags SET<TEXT>
);"""

session.execute(create_table_statement)
```

## Step 3: Create Search Indexes

Add a vector index for ANN search. Since OpenAI embeddings are normalized to unit length, use the dot product similarity function:

```python
create_vector_index_statement = f"""CREATE CUSTOM INDEX IF NOT EXISTS idx_embedding_vector
    ON {keyspace}.philosophers_cql (embedding_vector)
    USING 'org.apache.cassandra.index.sai.StorageAttachedIndex'
    WITH OPTIONS = {{'similarity_function' : 'dot_product'}};
"""

session.execute(create_vector_index_statement)
```

Add additional indexes for filtering by author and tags:

```python
create_author_index_statement = f"""CREATE CUSTOM INDEX IF NOT EXISTS idx_author
    ON {keyspace}.philosophers_cql (author)
    USING 'org.apache.cassandra.index.sai.StorageAttachedIndex';
"""
session.execute(create_author_index_statement)

create_tags_index_statement = f"""CREATE CUSTOM INDEX IF NOT EXISTS idx_tags
    ON {keyspace}.philosophers_cql (VALUES(tags))
    USING 'org.apache.cassandra.index.sai.StorageAttachedIndex';
"""
session.execute(create_tags_index_statement)
```

## Step 4: Connect to OpenAI

Set up your OpenAI client:

```python
OPENAI_API_KEY = getpass("Please enter your OpenAI API Key: ")
client = openai.OpenAI(api_key=OPENAI_API_KEY)
embedding_model_name = "text-embedding-3-small"
```

Test the embedding functionality:

```python
result = client.embeddings.create(
    input=[
        "This is a sentence",
        "A second sentence"
    ],
    model=embedding_model_name,
)

print(f"Number of embeddings: {len(result.data)}")
print(f"Embedding dimension: {len(result.data[1].embedding)}")
```

## Step 5: Load and Prepare the Quotes Dataset

Load the philosopher quotes dataset:

```python
philo_dataset = load_dataset("datastax/philosopher-quotes")["train"]

# Inspect the dataset
print(f"Total quotes: {len(philo_dataset)}")
print("Sample entry:")
print(philo_dataset[16])
```

The dataset contains 450 quotes from 9 philosophers (50 quotes each).

## Step 6: Insert Quotes into the Vector Store

Prepare the insertion statement and process quotes in batches:

```python
prepared_insertion = session.prepare(
    f"INSERT INTO {keyspace}.philosophers_cql (quote_id, author, body, embedding_vector, tags) VALUES (?, ?, ?, ?, ?);"
)

BATCH_SIZE = 20
num_batches = ((len(philo_dataset) + BATCH_SIZE - 1) // BATCH_SIZE)

quotes_list = philo_dataset["quote"]
authors_list = philo_dataset["author"]
tags_list = philo_dataset["tags"]

print("Starting to store entries:")
for batch_i in range(num_batches):
    b_start = batch_i * BATCH_SIZE
    b_end = (batch_i + 1) * BATCH_SIZE
    
    # Compute embeddings for this batch
    b_emb_results = client.embeddings.create(
        input=quotes_list[b_start : b_end],
        model=embedding_model_name,
    )
    
    # Insert each quote with its embedding and metadata
    print("B ", end="")
    for entry_idx, emb_result in zip(range(b_start, b_end), b_emb_results.data):
        if tags_list[entry_idx]:
            tags = {tag for tag in tags_list[entry_idx].split(";")}
        else:
            tags = set()
        
        author = authors_list[entry_idx]
        quote = quotes_list[entry_idx]
        quote_id = uuid4()
        
        session.execute(
            prepared_insertion,
            (quote_id, author, quote, emb_result.embedding, tags),
        )
        print("*", end="")
    print(f" done ({len(b_emb_results.data)})")

print("\nFinished storing entries.")
```

## Step 7: Build the Quote Search Engine

Create a function to search for quotes based on semantic similarity:

```python
def find_quote_and_author(query_quote, n, author=None, tags=None):
    # Convert query to embedding vector
    query_vector = client.embeddings.create(
        input=[query_quote],
        model=embedding_model_name,
    ).data[0].embedding
    
    # Build WHERE clause based on optional filters
    where_clauses = []
    where_values = []
    
    if author:
        where_clauses += ["author = %s"]
        where_values += [author]
    
    if tags:
        for tag in tags:
            where_clauses += ["tags CONTAINS %s"]
            where_values += [tag]
    
    # Construct the search query
    if where_clauses:
        search_statement = f"""SELECT body, author FROM {keyspace}.philosophers_cql
            WHERE {' AND '.join(where_clauses)}
            ORDER BY embedding_vector ANN OF %s
            LIMIT %s;
        """
    else:
        search_statement = f"""SELECT body, author FROM {keyspace}.philosophers_cql
            ORDER BY embedding_vector ANN OF %s
            LIMIT %s;
        """
    
    # Execute the search
    query_values = tuple(where_values + [query_vector] + [n])
    result_rows = session.execute(search_statement, query_values)
    
    return [(result_row.body, result_row.author) for result_row in result_rows]
```

## Step 8: Test the Search Functionality

### Basic Search
Find quotes similar to a given query:

```python
results = find_quote_and_author("We struggle all our life for nothing", 3)
for quote, author in results:
    print(f"{author}: {quote}")
```

### Filtered Search
Search for quotes by a specific author:

```python
results = find_quote_and_author("knowledge is power", 2, author="aristotle")
for quote, author in results:
    print(f"{author}: {quote}")
```

### Tag-Filtered Search
Search for quotes with specific tags:

```python
results = find_quote_and_author("love and relationships", 2, tags=["love"])
for quote, author in results:
    print(f"{author}: {quote}")
```

## Step 9: Build a Quote Generator

Create a function that generates new philosophical quotes based on similar existing ones:

```python
def generate_new_quote(topic, n_examples=3, author=None, tags=None):
    # First, find similar quotes
    similar_quotes = find_quote_and_author(topic, n_examples, author=author, tags=tags)
    
    if not similar_quotes:
        return "No similar quotes found to base generation on."
    
    # Build a prompt for the LLM
    examples_text = "\n".join([f"- {quote} ({author})" for quote, author in similar_quotes])
    
    prompt = f"""Based on the following philosophical quotes, generate a new, original philosophical quote about '{topic}'.
    
Examples:
{examples_text}

New quote about '{topic}':"""
    
    # Generate using OpenAI's chat completion
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a philosophical quote generator. Create insightful, thought-provoking quotes in the style of the examples."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.8
    )
    
    return response.choices[0].message.content.strip()
```

## Step 10: Test Quote Generation

Generate a new philosophical quote:

```python
new_quote = generate_new_quote("the nature of happiness", n_examples=2)
print(f"Generated quote: {new_quote}")
```

Generate a quote in the style of a specific philosopher:

```python
new_quote = generate_new_quote("human consciousness", n_examples=2, author="sartre")
print(f"Generated quote (Sartre style): {new_quote}")
```

## Step 11: (Optional) Exploit Partitioning for Performance

For production applications with large datasets, consider partitioning your data. Here's an example of how to modify the table schema for better performance:

```python
create_partitioned_table_statement = f"""CREATE TABLE IF NOT EXISTS {keyspace}.philosophers_partitioned_cql (
    author TEXT,
    quote_id UUID,
    body TEXT,
    embedding_vector VECTOR<FLOAT, 1536>,
    tags SET<TEXT>,
    PRIMARY KEY (author, quote_id)
);"""

session.execute(create_partitioned_table_statement)
```

This partitions quotes by author, which can improve query performance when searching within a specific philosopher's works.

## Summary

You've successfully built a complete philosophy quote finder and generator system that:

1. **Stores** quotes as vector embeddings in Cassandra/Astra DB
2. **Searches** for semantically similar quotes using ANN search
3. **Filters** results by author and tags
4. **Generates** new philosophical quotes using LLMs

This pattern can be adapted for various applications beyond quotes, such as document retrieval, recommendation systems, or any scenario where semantic similarity search is valuable.

## Next Steps

- Experiment with different embedding models
- Implement caching of prepared statements for better performance
- Add user authentication and personalization
- Create a web interface for the search and generation functionality
- Explore hybrid search combining vector similarity with keyword matching