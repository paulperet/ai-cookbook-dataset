# Building a RAG System with PolarDB-PG and OpenAI Embeddings

This guide provides a step-by-step tutorial for implementing a Retrieval-Augmented Generation (RAG) system using PolarDB-PG as a vector database with OpenAI embeddings. You'll learn how to store, index, and query vector embeddings efficiently.

## Prerequisites

Before starting, ensure you have:

1. **A PolarDB-PG Cloud Instance**: Deploy via [Alibaba Cloud](https://www.alibabacloud.com/product/polardb-for-postgresql)
2. **OpenAI API Key**: Obtain from [OpenAI Platform](https://platform.openai.com/api-keys)
3. **Python Environment**: With necessary libraries installed

## Setup

Install the required Python packages:

```bash
pip install openai psycopg2 pandas wget
```

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Verify your API key is properly configured:

```python
import os

if os.getenv("OPENAI_API_KEY") is not None:
    print("OPENAI_API_KEY is ready")
else:
    print("OPENAI_API_KEY environment variable not found")
```

## Step 1: Connect to PolarDB-PG

Establish a connection to your PolarDB-PG instance using the `psycopg2` library:

```python
import os
import psycopg2

# Configure connection parameters
connection = psycopg2.connect(
    host=os.environ.get("PGHOST", "localhost"),
    port=os.environ.get("PGPORT", "5432"),
    database=os.environ.get("PGDATABASE", "postgres"),
    user=os.environ.get("PGUSER", "user"),
    password=os.environ.get("PGPASSWORD", "password")
)

# Create a cursor for executing SQL commands
cursor = connection.cursor()
```

Test the connection to ensure it's working:

```python
cursor.execute("SELECT 1;")
result = cursor.fetchone()

if result == (1,):
    print("Connection successful!")
else:
    print("Connection failed.")
```

## Step 2: Download Sample Data

We'll use a pre-computed dataset of Wikipedia article embeddings. Download the dataset:

```python
import wget

embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"
wget.download(embeddings_url)
```

Extract the downloaded ZIP file:

```python
import zipfile
import os

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
    print(f"The file {file_name} exists in the data directory.")
else:
    print(f"The file {file_name} does not exist in the data directory.")
```

## Step 3: Create Database Schema

Create a table to store articles with their vector embeddings. PolarDB-PG supports the `vector` data type for efficient similarity searches:

```python
# Create articles table
create_table_sql = '''
CREATE TABLE IF NOT EXISTS public.articles (
    id INTEGER NOT NULL,
    url TEXT,
    title TEXT,
    content TEXT,
    title_vector vector(1536),
    content_vector vector(1536),
    vector_id INTEGER
);

ALTER TABLE public.articles ADD PRIMARY KEY (id);
'''

# Create vector indexes for efficient similarity search
create_indexes_sql = '''
CREATE INDEX ON public.articles USING ivfflat (content_vector) WITH (lists = 1000);
CREATE INDEX ON public.articles USING ivfflat (title_vector) WITH (lists = 1000);
'''

# Execute the SQL statements
cursor.execute(create_table_sql)
cursor.execute(create_indexes_sql)

# Commit the changes
connection.commit()
```

## Step 4: Load Data into PolarDB-PG

Load the pre-computed embeddings from the CSV file into your database:

```python
import io

# Path to your local CSV file
csv_file_path = '../../data/vector_database_wikipedia_articles_embedded.csv'

# Define a generator function to process the file line by line
def process_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line

# Create a StringIO object to store the modified lines
modified_lines = io.StringIO(''.join(list(process_file(csv_file_path))))

# Create the COPY command for efficient bulk loading
copy_command = '''
COPY public.articles (id, url, title, content, title_vector, content_vector, vector_id)
FROM STDIN WITH (FORMAT CSV, HEADER true, DELIMITER ',');
'''

# Execute the COPY command using the copy_expert method
cursor.copy_expert(copy_command, modified_lines)

# Commit the changes
connection.commit()
```

Verify the data was loaded successfully:

```python
count_sql = """SELECT COUNT(*) FROM public.articles;"""
cursor.execute(count_sql)
result = cursor.fetchone()
print(f"Total articles loaded: {result[0]}")
```

## Step 5: Implement Vector Search Function

Create a function that converts natural language queries into embeddings and performs similarity search in PolarDB-PG:

```python
import openai

def query_polardb(query, collection_name, vector_name="title_vector", top_k=20):
    """
    Query PolarDB-PG for similar vectors using OpenAI embeddings.
    
    Args:
        query: Natural language query string
        collection_name: Database table name
        vector_name: Which vector column to search ('title_vector' or 'content_vector')
        top_k: Number of results to return
    
    Returns:
        List of matching articles with similarity scores
    """
    # Generate embedding vector from user query
    embedded_query = openai.Embedding.create(
        input=query,
        model="text-embedding-3-small",
    )["data"][0]["embedding"]
    
    # Convert the embedded_query to PostgreSQL compatible format
    embedded_query_pg = "[" + ",".join(map(str, embedded_query)) + "]"
    
    # Create SQL query for similarity search
    query_sql = f"""
    SELECT id, url, title, l2_distance({vector_name},'{embedded_query_pg}'::VECTOR(1536)) AS similarity
    FROM {collection_name}
    ORDER BY {vector_name} <-> '{embedded_query_pg}'::VECTOR(1536)
    LIMIT {top_k};
    """
    
    # Execute the query
    cursor.execute(query_sql)
    results = cursor.fetchall()
    
    return results
```

## Step 6: Perform Semantic Searches

Now you can query your vector database using natural language. Let's start with a title-based search:

```python
# Search by article titles
query_results = query_polardb("modern art in Europe", "public.articles")

print("Top matching articles for 'modern art in Europe':")
for i, result in enumerate(query_results):
    # Convert L2 distance to similarity score (1 - distance)
    similarity_score = round(1 - result[3], 3)
    print(f"{i + 1}. {result[2]} (Similarity: {similarity_score})")
```

Next, let's perform a content-based search for more detailed matching:

```python
# Search by article content
query_results = query_polardb("Famous battles in Scottish history", "public.articles", "content_vector")

print("\nTop matching articles for 'Famous battles in Scottish history':")
for i, result in enumerate(query_results):
    similarity_score = round(1 - result[3], 3)
    print(f"{i + 1}. {result[2]} (Similarity: {similarity_score})")
```

## Step 7: Clean Up Resources

Always close your database connections when finished:

```python
# Close cursor and connection
cursor.close()
connection.close()
print("Database connections closed.")
```

## Summary

You've successfully implemented a RAG system using PolarDB-PG as a vector database. Key accomplishments include:

1. **Database Setup**: Created a PolarDB-PG table with vector columns and optimized indexes
2. **Data Ingestion**: Loaded pre-computed OpenAI embeddings into the database
3. **Semantic Search**: Implemented a function that converts natural language queries to embeddings and performs similarity searches
4. **Dual Search Capability**: Enabled both title-based and content-based semantic search

This system can now serve as the retrieval component for a larger RAG pipeline, where search results can be fed into an LLM for answer generation.

## Next Steps

To extend this system, consider:

1. **Real-time Embedding Generation**: Implement on-the-fly embedding of new documents
2. **Hybrid Search**: Combine vector similarity with traditional keyword search
3. **LLM Integration**: Feed retrieved documents to an LLM for answer generation
4. **Performance Optimization**: Experiment with different index types and parameters for your specific use case