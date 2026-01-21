# Building an Interactive Q&A Application with SingleStoreDB and ChatGPT

This guide demonstrates how to build an interactive Q&A application using SingleStoreDB for vector storage and OpenAI's ChatGPT. You'll learn to store and query contextual data to provide accurate, up-to-date answers beyond ChatGPT's base knowledge cutoff.

## Prerequisites

Before starting, ensure you have:
- An OpenAI API key
- A running SingleStoreDB instance (host, user, and password)
- Python 3.7 or higher

## Setup

Install the required Python packages:

```bash
pip install openai pandas singlestoredb wget tiktoken tabulate
```

## Step 1: Test ChatGPT Without Context

First, let's see how ChatGPT handles questions about events beyond its training data cutoff.

```python
import openai

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Ask about a post-2021 event
response = openai.ChatCompletion.create(
    model=GPT_MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the gold medal for curling in Olympics 2022?"},
    ]
)

print(response['choices'][0]['message']['content'])
```

**Expected Output:**
```
I'm sorry, I cannot provide information about events that have not occurred yet. The Winter Olympics 2022 will be held in Beijing, China from February 4 to 20, 2022. The curling events will take place during this time and the results will not be known until after the competition has concluded.
```

As expected, ChatGPT cannot answer questions about events beyond its knowledge cutoff. Now let's provide it with the necessary context.

## Step 2: Prepare Contextual Data

We'll use Wikipedia data about the 2022 Winter Olympics to provide ChatGPT with the context it needs.

```python
import pandas as pd
import os
import wget
import ast

# Download the pre-chunked text with embeddings
embeddings_path = "https://cdn.openai.com/API/examples/data/winter_olympics_2022.csv"
file_path = "winter_olympics_2022.csv"

if not os.path.exists(file_path):
    wget.download(embeddings_path, file_path)
    print("File downloaded successfully.")
else:
    print("File already exists in the local file system.")

# Load the data
df = pd.read_csv("winter_olympics_2022.csv")

# Convert embeddings from string representation back to lists
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# Verify the data structure
print(f"Dataset shape: {df.shape}")
print(df.info(show_counts=True))
```

## Step 3: Set Up SingleStoreDB

Now, let's configure SingleStoreDB to store our contextual data with vector embeddings.

```python
import singlestoredb as s2

# Connect to your SingleStoreDB instance
conn = s2.connect("user:password@host:3306/")
cur = conn.cursor()

# Create database
cur.execute("CREATE DATABASE IF NOT EXISTS winter_wikipedia2;")

# Create table for storing text and embeddings
create_table_stmt = """
CREATE TABLE IF NOT EXISTS winter_wikipedia2.winter_olympics_2022 (
    id INT PRIMARY KEY,
    text TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci,
    embedding BLOB
);
"""
cur.execute(create_table_stmt)
```

## Step 4: Populate the Database

Insert the Wikipedia data into SingleStoreDB, using `JSON_ARRAY_PACK_F64` to efficiently store the vector embeddings.

```python
# Prepare the insert statement
insert_stmt = """
    INSERT INTO winter_wikipedia2.winter_olympics_2022 (
        id,
        text,
        embedding
    )
    VALUES (
        %s,
        %s,
        JSON_ARRAY_PACK_F64(%s)
    )
"""

# Convert DataFrame to NumPy record array for batch processing
record_arr = df.to_records(index=True)

# Set batch size for efficient insertion
batch_size = 1000

# Insert data in batches
for i in range(0, len(record_arr), batch_size):
    batch = record_arr[i:i+batch_size]
    values = [(row[0], row[1], str(row[2])) for row in batch]
    cur.executemany(insert_stmt, values)

print("Data successfully loaded into SingleStoreDB")
```

## Step 5: Implement Semantic Search

Create a function to find the most relevant text passages for a given query using vector similarity search.

```python
from utils.embeddings_utils import get_embedding

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    top_n: int = 100
) -> tuple:
    """Returns a list of strings and similarity scores, sorted from most related to least."""
    
    # Get the embedding of the query
    query_embedding_response = get_embedding(query, EMBEDDING_MODEL)
    
    # SQL statement for vector similarity search
    stmt = """
        SELECT
            text,
            DOT_PRODUCT_F64(JSON_ARRAY_PACK_F64(%s), embedding) AS score
        FROM winter_wikipedia2.winter_olympics_2022
        ORDER BY score DESC
        LIMIT %s
    """
    
    # Execute the search
    cur.execute(stmt, [str(query_embedding_response), top_n])
    results = cur.fetchall()
    
    # Extract results
    strings = []
    relatednesses = []
    
    for row in results:
        strings.append(row[0])
        relatednesses.append(row[1])
    
    return strings[:top_n], relatednesses[:top_n]
```

Test the semantic search function:

```python
from tabulate import tabulate

# Find relevant passages about curling gold medals
strings, relatednesses = strings_ranked_by_relatedness(
    "curling gold medal",
    df,
    top_n=5
)

# Display results
for string, relatedness in zip(strings, relatednesses):
    print(f"Similarity score: {relatedness:.3f}")
    print(tabulate([[string]], headers=['Relevant Text'], tablefmt='fancy_grid'))
    print()
```

## Step 6: Build the Context-Aware Query System

Create functions to manage token limits and construct context-aware prompts for ChatGPT.

```python
import tiktoken

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT with relevant source texts from SingleStoreDB."""
    
    # Get relevant passages
    strings, _ = strings_ranked_by_relatedness(query, df)
    
    # Construct the prompt
    introduction = 'Use the below articles on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    
    # Add relevant passages until we reach token limit
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    
    return message + question

def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,  # Reserve tokens for response
    print_message: bool = False,
) -> str:
    """Answer a query using GPT and relevant texts from SingleStoreDB."""
    
    # Construct the context-aware message
    message = query_message(query, df, model=model, token_budget=token_budget)
    
    if print_message:
        print("Generated prompt:")
        print(message)
        print("\n" + "="*50 + "\n")
    
    # Send to ChatGPT
    messages = [
        {"role": "system", "content": "You answer questions about the 2022 Winter Olympics."},
        {"role": "user", "content": message},
    ]
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0  # Deterministic output for factual questions
    )
    
    return response["choices"][0]["message"]["content"]
```

## Step 7: Get Accurate Answers with Context

Now let's ask the same question again, but this time with the contextual data provided.

```python
from pprint import pprint

# Ask about curling gold medals with context
answer = ask('Who won the gold medal for curling in Olympics 2022?')

print("ChatGPT's answer with context:")
print("="*50)
pprint(answer)
```

**Expected Output:**
```
ChatGPT's answer with context:
==================================================
("There were three curling events at the 2022 Winter Olympics: men's, women's, and mixed doubles. The gold medalists for each event are:\n\n- Men's: Sweden (Niklas Edin, Oskar Eriksson, Rasmus Wranå, Christoffer Sundgren, Daniel Magnusson)\n- Women's: Great Britain (Eve Muirhead, Vicky Wright, Jennifer Dodds, Hailey Duff, Mili Smith)\n- Mixed doubles: Italy (Stefania Constantini, Amos Mosaner)")
```

## Summary

You've successfully built a context-aware Q&A system that:

1. **Stores contextual data** in SingleStoreDB with vector embeddings for efficient similarity search
2. **Performs semantic search** to find relevant passages for any query
3. **Constructs intelligent prompts** that stay within token limits
4. **Provides accurate answers** by combining ChatGPT's reasoning with up-to-date contextual information

This pattern can be extended to any domain where you need to provide ChatGPT with specific, up-to-date information beyond its training data. The key components are:

- Vector storage and similarity search in SingleStoreDB
- Intelligent prompt construction with token management
- Context-aware querying that combines semantic search with LLM capabilities

Try experimenting with different queries or extending the system with your own domain-specific data!