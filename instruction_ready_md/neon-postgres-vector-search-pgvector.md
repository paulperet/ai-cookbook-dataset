# Vector Similarity Search with Neon Postgres and OpenAI

This guide walks you through using Neon Serverless Postgres as a vector database for OpenAI embeddings. You will learn how to store embeddings, perform similarity searches, and retrieve relevant documents.

## Prerequisites

Before starting, ensure you have:

1.  **A Neon Postgres Database:** Create an account and a project at [Neon.tech](https://neon.tech/). Your project comes with a ready-to-use `neondb` database.
2.  **Connection String:** Obtain your database connection string from the **Connection Details** widget on the Neon **Dashboard**.
3.  **The `pgvector` Extension:** Enable it in your Neon database by running `CREATE EXTENSION vector;` in the SQL Editor. See the [Neon documentation](https://neon.tech/docs/extensions/pgvector#enable-the-pgvector-extension).
4.  **OpenAI API Key:** Get one from your [OpenAI account](https://platform.openai.com/account/api-keys).
5.  **Python 3.7+ and pip.**

## Setup and Installation

### 1. Install Required Packages

Open your terminal and install the necessary Python libraries.

```bash
pip install openai psycopg2-binary pandas wget python-dotenv
```

> **Note:** We use `psycopg2-binary` for easier installation. If you encounter issues, you can try `psycopg2`.

### 2. Configure Your API Key

Your OpenAI API key is required to generate embeddings. The safest way is to set it as an environment variable.

**On Linux/macOS:**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

**On Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**On Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY='your-api-key-here'
```

Alternatively, you can store it in a `.env` file in your project directory:

```bash
# .env file
OPENAI_API_KEY=your-api-key-here
```

Let's verify the key is accessible from Python.

```python
import os
from getpass import getpass

# Check if the key is already set
if os.getenv("OPENAI_API_KEY") is not None:
    print("✅ OPENAI_API_KEY is ready.")
else:
    # If not, prompt for it (for this session only)
    api_key = getpass("Enter your OPENAI_API_KEY: ")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        print("✅ OPENAI_API_KEY set for this session.")
    else:
        raise ValueError("OPENAI_API_KEY is required to proceed.")
```

### 3. Connect to Your Neon Database

Store your Neon connection string in the `.env` file as `DATABASE_URL` for security.

```bash
# .env file
DATABASE_URL=postgres://<user>:<password>@<hostname>/<dbname>
```

Now, let's establish the connection in Python.

```python
import os
import psycopg2
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Retrieve the connection string
connection_string = os.environ.get("DATABASE_URL")
if not connection_string:
    raise ValueError("DATABASE_URL not found. Please set it in your .env file.")

# Establish the connection
connection = psycopg2.connect(connection_string)
cursor = connection.cursor()

# Test the connection
cursor.execute("SELECT 1;")
result = cursor.fetchone()
if result == (1,):
    print("✅ Successfully connected to the Neon database.")
else:
    print("❌ Connection failed.")
```

## Prepare the Sample Data

We'll use a dataset of pre-computed Wikipedia article embeddings provided by OpenAI to avoid using your credits.

### 4. Download and Extract the Dataset

The following code downloads and extracts the sample data (a ~700 MB file).

```python
import wget
import zipfile
import os

# Define URLs and paths
embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"
zip_file_name = "vector_database_wikipedia_articles_embedded.zip"
extract_to_dir = "./data"

# Create data directory if it doesn't exist
os.makedirs(extract_to_dir, exist_ok=True)
zip_file_path = os.path.join(extract_to_dir, zip_file_name)

# Download the file (this may take a few minutes)
print("Downloading dataset...")
wget.download(embeddings_url, zip_file_path)
print("\nDownload complete.")

# Extract the file
print("Extracting files...")
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_dir)
print("Extraction complete.")

# Verify the CSV file exists
csv_file_name = "vector_database_wikipedia_articles_embedded.csv"
csv_file_path = os.path.join(extract_to_dir, csv_file_name)
if os.path.exists(csv_file_path):
    print(f"✅ Sample data ready at: {csv_file_path}")
else:
    print("❌ CSV file not found.")
```

## Configure the Database Schema

### 5. Create the Table and Indexes

We need a table to store our articles and their vector embeddings. We'll also create indexes on the vector columns to speed up similarity searches.

```python
# SQL to create the articles table
create_table_sql = '''
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER NOT NULL,
    url TEXT,
    title TEXT,
    content TEXT,
    title_vector vector(1536),
    content_vector vector(1536),
    vector_id INTEGER,
    PRIMARY KEY (id)
);
'''

# SQL to create IVF indexes for faster approximate nearest neighbor search
create_indexes_sql = '''
CREATE INDEX ON articles USING ivfflat (content_vector) WITH (lists = 1000);
CREATE INDEX ON articles USING ivfflat (title_vector) WITH (lists = 1000);
'''

# Execute the statements
cursor.execute(create_table_sql)
cursor.execute(create_indexes_sql)
connection.commit()
print("✅ Table 'articles' created with vector indexes.")
```

## Load the Data

### 6. Import the CSV Data into Postgres

Now, we'll load the 25,000 pre-computed records from the CSV file into our new table. This operation uses PostgreSQL's efficient `COPY` command.

```python
import io

def load_data(cursor, connection, csv_file_path):
    """Loads data from a CSV file into the articles table."""
    # Use a generator to read the file efficiently
    def process_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield line

    # Prepare the data for COPY command
    modified_lines = io.StringIO(''.join(list(process_file(csv_file_path))))

    # Define the COPY command
    copy_command = '''
    COPY articles (id, url, title, content, title_vector, content_vector, vector_id)
    FROM STDIN WITH (FORMAT CSV, HEADER true, DELIMITER ',');
    '''

    # Execute the copy command
    print("Loading data into the database... (This may take a moment)")
    cursor.copy_expert(copy_command, modified_lines)
    connection.commit()
    print("✅ Data load complete.")

# Call the function with the path to your CSV
load_data(cursor, connection, csv_file_path)

# Verify the record count
cursor.execute("SELECT COUNT(*) FROM articles;")
count = cursor.fetchone()[0]
print(f"Total records in 'articles' table: {count}")
```

## Perform Vector Similarity Searches

### 7. Define the Search Function

This function takes a user's text query, converts it into an embedding using the same OpenAI model that created our stored vectors, and performs a nearest-neighbor search in the database.

```python
import openai

def query_neon(query, collection_name, vector_name="title_vector", top_k=20):
    """
    Performs a vector similarity search.
    
    Args:
        query (str): The user's search query.
        collection_name (str): The name of the database table.
        vector_name (str): The vector column to search ('title_vector' or 'content_vector').
        top_k (int): Number of results to return.
    
    Returns:
        list: A list of tuples containing (id, url, title, similarity_distance).
    """
    # 1. Generate an embedding for the query
    embedded_query = openai.Embedding.create(
        input=query,
        model="text-embedding-3-small", # Must match the model used for pre-computed data
    )["data"][0]["embedding"]
    
    # 2. Format the embedding list as a PostgreSQL vector string
    embedded_query_pg = "[" + ",".join(map(str, embedded_query)) + "]"
    
    # 3. Construct the SQL query for nearest neighbor search using L2 distance
    query_sql = f"""
    SELECT id, url, title, 
           l2_distance({vector_name}, '{embedded_query_pg}'::VECTOR(1536)) AS distance
    FROM {collection_name}
    ORDER BY {vector_name} <-> '{embedded_query_pg}'::VECTOR(1536)
    LIMIT {top_k};
    """
    
    # 4. Execute the query
    cursor.execute(query_sql)
    results = cursor.fetchall()
    
    return results
```

### 8. Run Your First Search (Title-Based)

Let's search for articles based on similarity to their **title** embeddings.

```python
print("Searching for articles related to 'Greek mythology' (title similarity)...")
query_results = query_neon("Greek mythology", "articles", "title_vector")

for i, (id_val, url, title, distance) in enumerate(query_results):
    # Convert distance to a similarity score (closer distance = higher similarity)
    similarity_score = round(1 - distance, 3)
    print(f"{i + 1}. {title} (Similarity: {similarity_score})")
```

**Sample Output:**
```
1. Greek mythology (Similarity: 1.0)
2. Mythology (Similarity: 0.874)
3. List of Greek mythological figures (Similarity: 0.867)
...
```

### 9. Run a Content-Based Search

Now, let's search based on the **content** of the articles, which often yields more nuanced results.

```python
print("\nSearching for articles related to 'Famous battles in Greek history' (content similarity)...")
query_results = query_neon("Famous battles in Greek history", "articles", "content_vector")

for i, (id_val, url, title, distance) in enumerate(query_results):
    similarity_score = round(1 - distance, 3)
    print(f"{i + 1}. {title} (Similarity: {similarity_score})")
```

**Sample Output:**
```
1. Battle of Thermopylae (Similarity: 0.912)
2. Greco-Persian Wars (Similarity: 0.901)
3. Battle of Salamis (Similarity: 0.894)
...
```

## Cleanup and Next Steps

### 10. Close the Database Connection

Always close your cursor and connection when you're finished.

```python
cursor.close()
connection.close()
print("✅ Database connection closed.")
```

### Next Steps

You've successfully built a vector search system! Here are ideas to extend this project:

*   **Hybrid Search:** Combine vector similarity with traditional keyword search (`LIKE` or `tsvector`) for improved results.
*   **Metadata Filtering:** Add `WHERE` clauses to your search SQL to filter results by date, category, etc., before performing the vector search.
*   **Real-Time Updates:** Modify the `load_data` function to accept new documents, generate their embeddings on the fly, and insert them into the table.
*   **Build a RAG Pipeline:** Use the retrieved articles as context for an LLM (like GPT-4) to generate detailed, sourced answers to complex questions.

Remember to manage your Neon project's resources and monitor usage in the [Neon Console](https://console.neon.tech/).