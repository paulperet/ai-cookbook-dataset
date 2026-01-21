# Building a RAG Pipeline with Hologres and OpenAI Embeddings

This guide walks you through creating a Retrieval-Augmented Generation (RAG) pipeline using Hologres as a vector database and OpenAI embeddings. You'll learn how to store, search, and retrieve contextual information to power large language model applications.

## Prerequisites

Before starting, ensure you have:

1.  A running Hologres cloud instance.
2.  An OpenAI API key.
3.  The following Python packages installed.

### Setup and Installation

Run the following command to install the required libraries:

```bash
pip install openai psycopg2-binary pandas wget
```

### Configure Your API Keys

Set your OpenAI API key as an environment variable. You can do this in your terminal or within the script.

```python
import os

# Set your OpenAI API key (optional if already set in your environment)
# os.environ["OPENAI_API_KEY"] = "sk-your-api-key-here"

# Verify the key is accessible
if os.getenv("OPENAI_API_KEY") is not None:
    print("OPENAI_API_KEY is ready")
else:
    print("OPENAI_API_KEY environment variable not found")
```

## Step 1: Connect to Your Hologres Database

First, establish a connection to your Hologres instance using `psycopg2`. Configure your connection parameters via environment variables.

```python
import os
import psycopg2

# Configure connection parameters. Use environment variables for security.
connection = psycopg2.connect(
    host=os.environ.get("PGHOST", "localhost"),
    port=os.environ.get("PGPORT", "5432"),
    database=os.environ.get("PGDATABASE", "postgres"),
    user=os.environ.get("PGUSER", "user"),
    password=os.environ.get("PGPASSWORD", "password")
)
connection.set_session(autocommit=True)

# Create a cursor to execute SQL commands
cursor = connection.cursor()

# Test the connection
cursor.execute("SELECT 1;")
result = cursor.fetchone()
if result == (1,):
    print("Connection successful!")
else:
    print("Connection failed.")
```

## Step 2: Download and Prepare the Sample Dataset

We'll use a pre-computed dataset of Wikipedia article embeddings. Download and extract the file.

```python
import wget
import zipfile
import os

# Download the embeddings dataset
embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"
wget.download(embeddings_url)

# Extract the downloaded ZIP file
current_directory = os.getcwd()
zip_file_path = os.path.join(current_directory, "vector_database_wikipedia_articles_embedded.zip")
output_directory = os.path.join(current_directory, "../../data")

with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(output_directory)

# Verify the CSV file exists
file_name = "vector_database_wikipedia_articles_embedded.csv"
data_directory = os.path.join(current_directory, "../../data")
file_path = os.path.join(data_directory, file_name)

if os.path.exists(file_path):
    print(f"Dataset file '{file_name}' is ready.")
else:
    print(f"Error: File '{file_name}' not found.")
```

## Step 3: Inspect the Data Structure

Load the dataset to understand its structure and verify the embedding dimensions.

```python
import pandas as pd
import json

# Load the CSV file
data = pd.read_csv('../../data/vector_database_wikipedia_articles_embedded.csv')
print("Dataset preview:")
print(data.head())

# Check the dimensionality of the title and content embeddings
title_vector_length = len(json.loads(data['title_vector'].iloc[0]))
content_vector_length = len(json.loads(data['content_vector'].iloc[0])
print(f"\nTitle vector dimension: {title_vector_length}")
print(f"Content vector dimension: {content_vector_length}")
```

## Step 4: Create the Hologres Table and Vector Index

Now, create a table in Hologres to store the articles. We'll enable Proxima, Hologres's high-performance vector search engine, to index both the `title_vector` and `content_vector` columns.

```python
# Enable the Proxima extension
cursor.execute('CREATE EXTENSION IF NOT EXISTS proxima;')

# SQL to create the articles table and Proxima vector indexes
create_proxima_table_sql = '''
BEGIN;
DROP TABLE IF EXISTS articles;
CREATE TABLE articles (
    id INT PRIMARY KEY NOT NULL,
    url TEXT,
    title TEXT,
    content TEXT,
    title_vector float4[] check(
        array_ndims(title_vector) = 1 and 
        array_length(title_vector, 1) = 1536
    ),
    content_vector float4[] check(
        array_ndims(content_vector) = 1 and 
        array_length(content_vector, 1) = 1536
    ),
    vector_id INT
);

-- Configure Proxima indexes for vector similarity search
call set_table_property(
    'articles',
    'proxima_vectors', 
    '{
        "title_vector":{"algorithm":"Graph","distance_method":"Euclidean","builder_params":{"min_flush_proxima_row_count" : 10}},
        "content_vector":{"algorithm":"Graph","distance_method":"Euclidean","builder_params":{"min_flush_proxima_row_count" : 10}}
    }'
);  
COMMIT;
'''

# Execute the table creation SQL
cursor.execute(create_proxima_table_sql)
print("Table 'articles' created with Proxima vector indexes.")
```

## Step 5: Upload Data to Hologres

Upload the pre-computed embeddings to the `articles` table using PostgreSQL's efficient `COPY` command. We need to convert the vector format from JSON arrays (`[1,2,3]`) to PostgreSQL array syntax (`{1,2,3}`).

```python
import io

csv_file_path = '../../data/vector_database_wikipedia_articles_embedded.csv'

def process_file(file_path):
    """Generator to convert JSON array syntax to PostgreSQL array syntax."""
    with open(file_path, 'r') as file:
        for line in file:
            yield line.replace('[', '{').replace(']', '}')

# Create an in-memory file with the modified syntax
modified_lines = io.StringIO(''.join(list(process_file(csv_file_path))))

# Define the COPY command
copy_command = '''
COPY public.articles (id, url, title, content, title_vector, content_vector, vector_id)
FROM STDIN WITH (FORMAT CSV, HEADER true, DELIMITER ',');
'''

# Execute the data upload
cursor.copy_expert(copy_command, modified_lines)
print("Data upload started. This may take several minutes...")

# Run VACUUM to trigger background index building
cursor.execute('vacuum articles;')

# Verify the upload was successful
cursor.execute("select count(*) from articles;")
result = cursor.fetchone()
print(f"Upload complete. Total records in 'articles': {result[0]}")
```

## Step 6: Perform Vector Similarity Search

With the data loaded, you can now query it. Define a function that takes a natural language query, converts it to an embedding using OpenAI, and performs a k-Nearest Neighbors (k-NN) search in Hologres.

```python
import openai

def query_knn(query, table_name, vector_name="title_vector", top_k=20):
    """
    Performs a k-NN search on a specified vector column.
    
    Args:
        query: The natural language query string.
        table_name: The name of the Hologres table.
        vector_name: The vector column to search ('title_vector' or 'content_vector').
        top_k: Number of results to return.
    
    Returns:
        A list of tuples (id, url, title, distance).
    """
    # Generate embedding for the query using OpenAI
    embedded_query = openai.Embedding.create(
        input=query,
        model="text-embedding-3-small",
    )["data"][0]["embedding"]

    # Format the embedding for PostgreSQL
    embedded_query_pg = "{" + ",".join(map(str, embedded_query)) + "}"

    # Construct the k-NN SQL query
    query_sql = f"""
    SELECT id, url, title, pm_approx_euclidean_distance({vector_name},'{embedded_query_pg}'::float4[]) AS distance
    FROM {table_name}
    ORDER BY distance
    LIMIT {top_k};
    """
    
    # Execute the query
    cursor.execute(query_sql)
    results = cursor.fetchall()
    return results
```

### Search by Article Title

Let's test the search function by finding articles related to "modern art in Europe" based on their title embeddings.

```python
print("Searching for 'modern art in Europe' (using title_vector):")
query_results = query_knn("modern art in Europe", "articles", "title_vector")

for i, result in enumerate(query_results):
    # Convert distance to a similarity score (higher is more similar)
    similarity_score = round(1 - result[3], 3)
    print(f"{i + 1}. {result[2]} (Similarity: {similarity_score})")
```

### Search by Article Content

You can also perform a more in-depth search using the content embeddings.

```python
print("\nSearching for 'Famous battles in Scottish history' (using content_vector):")
query_results = query_knn("Famous battles in Scottish history", "articles", "content_vector")

for i, result in enumerate(query_results):
    similarity_score = round(1 - result[3], 3)
    print(f"{i + 1}. {result[2]} (Similarity: {similarity_score})")
```

## Next Steps: Building a Full RAG Pipeline

You have successfully set up a vector database and similarity search. To build a complete RAG application:

1.  **Retrieve Context:** Use the `query_knn` function to fetch the top-k most relevant articles for a user's question.
2.  **Construct a Prompt:** Combine the retrieved article content (context) with the original user query.
3.  **Generate an Answer:** Feed the augmented prompt to a Large Language Model (like GPT-4) to produce a grounded, context-aware answer.

This pipeline ensures your LLM's responses are informed by the accurate, up-to-date information stored in your Hologres database.