# Using AnalyticDB as a Vector Database for OpenAI Embeddings

This guide provides a step-by-step tutorial on using AnalyticDB as a vector database for OpenAI embeddings. You will learn how to store precomputed embeddings, perform similarity searches, and integrate everything into a Retrieval-Augmented Generation (RAG) workflow.

## What is AnalyticDB?

AnalyticDB is a high-performance, distributed vector database fully compatible with PostgreSQL syntax. It is a managed, cloud-native database with a powerful vector compute engine, supporting features like advanced indexing algorithms, structured/unstructured data handling, real-time updates, distance metrics, scalar filtering, and time-travel searches. It also offers full OLAP database functionality with production-grade SLAs.

## Prerequisites

Before you begin, ensure you have the following:

1.  An active AnalyticDB cloud instance.
2.  An OpenAI API key.
3.  The necessary Python libraries installed.

## Step 1: Environment Setup

First, install the required Python packages.

```bash
pip install openai psycopg2 pandas wget
```

Next, set your OpenAI API key as an environment variable. You can do this in your terminal or within the script.

```python
import os

# Set your OpenAI API key (replace with your actual key)
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Verify the key is set
if os.getenv("OPENAI_API_KEY") is not None:
    print("OPENAI_API_KEY is ready")
else:
    print("OPENAI_API_KEY environment variable not found")
```

## Step 2: Connect to AnalyticDB

Establish a connection to your AnalyticDB instance using `psycopg2`. Configure the connection parameters with your instance's details.

```python
import psycopg2

# Configure your AnalyticDB connection details
connection = psycopg2.connect(
    host=os.environ.get("PGHOST", "your_host"),
    port=os.environ.get("PGPORT", "5432"),
    database=os.environ.get("PGDATABASE", "postgres"),
    user=os.environ.get("PGUSER", "user"),
    password=os.environ.get("PGPASSWORD", "password")
)

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

## Step 3: Download and Prepare Sample Data

We'll use a precomputed dataset of Wikipedia article embeddings. Download and extract the file.

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

# Verify the extracted CSV file exists
file_name = "vector_database_wikipedia_articles_embedded.csv"
data_directory = os.path.join(current_directory, "../../data")
file_path = os.path.join(data_directory, file_name)

if os.path.exists(file_path):
    print(f"The file {file_name} exists in the data directory.")
else:
    print(f"The file {file_name} does not exist in the data directory.")
```

## Step 4: Create the Database Table and Indexes

Create a table named `articles` to store the data. This table will hold the article text and its corresponding vector embeddings. We'll also create Approximate Nearest Neighbor (ANN) indexes on the vector columns for fast similarity searches.

```python
# SQL to create the articles table
create_table_sql = '''
CREATE TABLE IF NOT EXISTS public.articles (
    id INTEGER NOT NULL,
    url TEXT,
    title TEXT,
    content TEXT,
    title_vector REAL[],
    content_vector REAL[],
    vector_id INTEGER
);
ALTER TABLE public.articles ADD PRIMARY KEY (id);
'''

# SQL to create vector indexes for efficient similarity search
create_indexes_sql = '''
CREATE INDEX ON public.articles USING ann (content_vector) WITH (distancemeasure = l2, dim = '1536', pq_segments = '64', hnsw_m = '100', pq_centers = '2048');
CREATE INDEX ON public.articles USING ann (title_vector) WITH (distancemeasure = l2, dim = '1536', pq_segments = '64', hnsw_m = '100', pq_centers = '2048');
'''

# Execute the SQL statements
cursor.execute(create_table_sql)
cursor.execute(create_indexes_sql)
connection.commit()
print("Table and indexes created successfully.")
```

## Step 5: Load Data into AnalyticDB

Now, load the precomputed embeddings from the CSV file into the `articles` table. We'll use PostgreSQL's efficient `COPY` command.

```python
import io

csv_file_path = '../../data/vector_database_wikipedia_articles_embedded.csv'

# Process the CSV file: convert Python list syntax to PostgreSQL array syntax
def process_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.replace('[', '{').replace(']', '}')

# Prepare the data for the COPY command
modified_lines = io.StringIO(''.join(list(process_file(csv_file_path))))

# Define the COPY command
copy_command = '''
COPY public.articles (id, url, title, content, title_vector, content_vector, vector_id)
FROM STDIN WITH (FORMAT CSV, HEADER true, DELIMITER ',');
'''

# Execute the COPY command to bulk insert data
cursor.copy_expert(copy_command, modified_lines)
connection.commit()
print("Data loaded successfully.")

# Verify the number of records inserted
cursor.execute("SELECT COUNT(*) FROM public.articles;")
result = cursor.fetchone()
print(f"Total records in 'articles' table: {result[0]}")
```

## Step 6: Perform Similarity Searches

With the data loaded, you can now query it. Define a function that takes a natural language query, converts it to an embedding using OpenAI, and finds the most similar articles in AnalyticDB.

```python
import openai

def query_analyticdb(query, collection_name, vector_name="title_vector", top_k=20):
    """
    Query AnalyticDB for articles similar to the input query.

    Args:
        query (str): The search query.
        collection_name (str): The table name (e.g., 'public.articles').
        vector_name (str): The vector column to search against ('title_vector' or 'content_vector').
        top_k (int): Number of results to return.

    Returns:
        list: A list of tuples containing (id, url, title, similarity_score).
    """
    # Generate embedding for the query using OpenAI
    embedded_query = openai.Embedding.create(
        input=query,
        model="text-embedding-3-small",
    )["data"][0]["embedding"]

    # Format the embedding for PostgreSQL
    embedded_query_pg = "{" + ",".join(map(str, embedded_query)) + "}"

    # Construct the similarity search SQL query
    query_sql = f"""
    SELECT id, url, title, l2_distance({vector_name}, '{embedded_query_pg}'::real[]) AS similarity
    FROM {collection_name}
    ORDER BY {vector_name} <-> '{embedded_query_pg}'::real[]
    LIMIT {top_k};
    """

    # Execute the query
    cursor.execute(query_sql)
    results = cursor.fetchall()
    return results
```

### Example 1: Search by Article Title

Search for articles related to "modern art in Europe" using the title vectors.

```python
query_results = query_analyticdb("modern art in Europe", "public.articles", "title_vector")
print("Top results for 'modern art in Europe':")
for i, result in enumerate(query_results):
    # Convert L2 distance to a similarity score (higher is more similar)
    similarity_score = round(1 - result[3], 3)
    print(f"{i + 1}. {result[2]} (Similarity: {similarity_score})")
```

### Example 2: Search by Article Content

Search for articles related to "Famous battles in Scottish history" using the more detailed content vectors.

```python
query_results = query_analyticdb("Famous battles in Scottish history", "public.articles", "content_vector")
print("\nTop results for 'Famous battles in Scottish history':")
for i, result in enumerate(query_results):
    similarity_score = round(1 - result[3], 3)
    print(f"{i + 1}. {result[2]} (Similarity: {similarity_score})")
```

## Conclusion

You have successfully set up AnalyticDB as a vector store for OpenAI embeddings. You learned how to:
1.  Connect to an AnalyticDB instance.
2.  Create a table with vector columns and appropriate indexes.
3.  Bulk load precomputed embeddings.
4.  Perform semantic similarity searches using natural language queries.

This foundation enables you to build more advanced AI applications, such as RAG systems, where AnalyticDB serves as the high-performance knowledge base for retrieving relevant context.