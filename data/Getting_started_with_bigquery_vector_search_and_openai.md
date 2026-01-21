# Building a RAG System with Google Cloud BigQuery and Cloud Functions for ChatGPT

This guide provides a step-by-step tutorial for creating a Retrieval-Augmented Generation (RAG) system using Google Cloud Platform services. You'll learn how to store document embeddings in BigQuery with vector search capabilities, deploy a Cloud Function as an API endpoint, and integrate this system with ChatGPT's Custom GPTs.

## Architecture Overview

The solution consists of three main components:
1. **Data Processing Pipeline**: Embed documents using OpenAI's API and store them in BigQuery
2. **Vector Search Backend**: BigQuery table with vector search capabilities
3. **API Layer**: Cloud Function that handles search queries and integrates with ChatGPT

## Prerequisites

Before starting, ensure you have:
- A Google Cloud Platform project with appropriate permissions
- GCP CLI installed and authenticated
- OpenAI API key
- ChatGPT Plus, Teams, or Enterprise subscription
- Basic Python knowledge

## Step 1: Environment Setup

### Install Required Libraries

First, install all necessary Python packages:

```bash
pip install google-auth openai pandas google-cloud-functions python-dotenv pyperclip PyPDF2 tiktoken google-cloud-bigquery pyyaml
```

### Import Libraries

Create a new Python file and import the required libraries:

```python
# Standard Libraries
import json
import os
import csv
import shutil
from itertools import islice
import concurrent.futures
import yaml

# Third-Party Libraries
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
import tiktoken
from dotenv import load_dotenv
import pyperclip

# OpenAI Libraries
from openai import OpenAI

# Google Cloud Libraries
from google.auth import default
from google.cloud import bigquery
from google.cloud import functions_v1
```

### Configure GCP Project

Set up your GCP project and enable necessary services:

```python
# Set your project ID
project_id = "<your-gcp-project-id>"  # Replace with your actual project ID

# Enable required GCP services
! gcloud config set project {project_id}
! gcloud services enable cloudfunctions.googleapis.com
! gcloud services enable cloudbuild.googleapis.com
! gcloud services enable bigquery.googleapis.com
```

### Configure OpenAI Settings

Set up your OpenAI API key:

```python
# Load environment variables
load_dotenv()

# Configure OpenAI client
openai_api_key = os.environ.get("OPENAI_API_KEY", "<your-openai-api-key>")
openai_client = OpenAI(api_key=openai_api_key)
embeddings_model = "text-embedding-3-small"  # Can change to text-embedding-3-large
```

### Initialize GCP Credentials

```python
# Use default GCP credentials
credentials, project_id = default()
region = "us-central1"  # Choose your preferred region
print(f"Using GCP Project: {project_id}")
```

## Step 2: Prepare and Embed Documents

### Define Helper Functions

Create functions to handle text chunking and embedding:

```python
def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch

def chunked_tokens(text, chunk_length, encoding_name='cl100k_base'):
    """Split text into token chunks for embedding."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator

# Constants for embedding
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

def generate_embeddings(text, model):
    """Generate embeddings for a given text using OpenAI's API."""
    embeddings_response = openai_client.embeddings.create(model=model, input=text)
    embedding = embeddings_response.data[0].embedding
    return embedding

def len_safe_get_embedding(text, model=embeddings_model, max_tokens=EMBEDDING_CTX_LENGTH, encoding_name=EMBEDDING_ENCODING):
    """Safely generate embeddings for long texts by chunking."""
    chunk_embeddings = []
    chunk_texts = []
    
    for chunk in chunked_tokens(text, chunk_length=max_tokens, encoding_name=encoding_name):
        chunk_embeddings.append(generate_embeddings(chunk, model=model))
        chunk_texts.append(tiktoken.get_encoding(encoding_name).decode(chunk))
    
    return chunk_embeddings, chunk_texts
```

### Categorize Documents

Define categories and a function to classify documents:

```python
categories = ['authentication', 'models', 'techniques', 'tools', 'setup', 'billing_limits', 'other']

def categorize_text(text, categories):
    """Categorize document text using GPT-4."""
    messages = [
        {"role": "system", "content": f"""You are an expert in LLMs, and you will be given text that corresponds to an article in OpenAI's documentation.
         Categorize the document into one of these categories: {', '.join(categories)}. Only respond with the category name and nothing else."""},
        {"role": "user", "content": text}
    ]
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        category = response.choices[0].message.content
        return category
    except Exception as e:
        print(f"Error categorizing text: {str(e)}")
        return None
```

### Process Files

Create functions to process different file types:

```python
def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF files."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def process_file(file_path, idx, categories, embeddings_model):
    """Process individual files and generate embeddings."""
    file_name = os.path.basename(file_path)
    print(f"Processing file {idx + 1}: {file_name}")
    
    # Read text content
    if file_name.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    elif file_name.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    
    # Generate embeddings
    title = file_name
    title_vectors, title_text = len_safe_get_embedding(title, embeddings_model)
    content_vectors, content_text = len_safe_get_embedding(text, embeddings_model)
    
    # Categorize document
    category = categorize_text(' '.join(content_text), categories)
    print(f"Categorized {file_name} as {category}")
    
    # Prepare data for storage
    data = []
    for i, content_vector in enumerate(content_vectors):
        data.append({
            "id": f"{idx}_{i}",
            "vector_id": f"{idx}_{i}",
            "title": title_text[0],
            "text": content_text[i],
            "title_vector": json.dumps(title_vectors[0]),
            "content_vector": json.dumps(content_vector),
            "category": category
        })
    
    return data
```

### Process All Documents

Now process your document folder and create embeddings:

```python
# Set your document folder path
folder_name = "../../../data/oai_docs"  # Update with your document folder
files = [os.path.join(folder_name, f) for f in os.listdir(folder_name) 
         if f.endswith('.txt') or f.endswith('.pdf')]

data = []

# Process files concurrently for better performance
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_file, file_path, idx, categories, embeddings_model): 
               idx for idx, file_path in enumerate(files)}
    
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            data.extend(result)
        except Exception as e:
            print(f"Error processing file: {str(e)}")

# Save embeddings to CSV
csv_file = "embedded_data.csv"
with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ["id", "vector_id", "title", "text", "title_vector", "content_vector", "category"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for row in data:
        writer.writerow(row)
        print(f"Saved embedding for ID: {row['id']}")

# Load and verify the data
article_df = pd.read_csv("embedded_data.csv")
article_df["title_vector"] = article_df.title_vector.apply(json.loads)
article_df["content_vector"] = article_df.content_vector.apply(json.loads)
article_df["vector_id"] = article_df["vector_id"].apply(str)
article_df["category"] = article_df["category"].apply(str)

print(f"Successfully processed {len(article_df)} document chunks")
article_df.head()
```

## Step 3: Create BigQuery Dataset and Table

### Create BigQuery Dataset

```python
from google.cloud import bigquery
from google.api_core.exceptions import Conflict

# Define dataset parameters
raw_dataset_id = 'oai_docs'
dataset_id = f"{project_id}.{raw_dataset_id}"

# Initialize BigQuery client
client = bigquery.Client(credentials=credentials, project=project_id)

# Create dataset
dataset = bigquery.Dataset(dataset_id)
dataset.location = "US"

try:
    dataset = client.create_dataset(dataset, timeout=30)
    print(f"Created dataset {dataset_id}")
except Conflict:
    print(f"Dataset {raw_dataset_id} already exists")
```

### Prepare Data for BigQuery

```python
# Load the CSV data
csv_file_path = "embedded_data.csv"
df = pd.read_csv(csv_file_path, engine='python', quotechar='"', quoting=1)

# Preprocess vector data for BigQuery
def preprocess_content_vector(row):
    """Convert string representation of vector to list of floats."""
    row['content_vector'] = [float(x) for x in row['content_vector'][1:-1].split(',')]
    return row

df = df.apply(preprocess_content_vector, axis=1)
```

### Create Table Schema and Upload Data

```python
# Define table schema
schema = [
    bigquery.SchemaField("id", "STRING"),
    bigquery.SchemaField("vector_id", "STRING"),
    bigquery.SchemaField("title", "STRING"),
    bigquery.SchemaField("text", "STRING"),
    bigquery.SchemaField("title_vector", "STRING"),
    bigquery.SchemaField("content_vector", "FLOAT", mode="REPEATED"),  # Array of floats for vector
    bigquery.SchemaField("category", "STRING")
]

# Create table reference
table_id = f"{dataset_id}.embedded_data"
table = bigquery.Table(table_id, schema=schema)

# Create the table
try:
    table = client.create_table(table)
    print(f"Created table {table_id}")
except Conflict:
    print(f"Table {table_id} already exists")

# Upload data to BigQuery
job_config = bigquery.LoadJobConfig(
    schema=schema,
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE  # Replace existing data
)

# Convert DataFrame to list of dictionaries for upload
rows_to_insert = df.to_dict('records')

# Insert rows
errors = client.insert_rows_json(table, rows_to_insert)
if errors:
    print(f"Encountered errors while inserting rows: {errors}")
else:
    print(f"Successfully uploaded {len(rows_to_insert)} rows to BigQuery")
```

## Step 4: Create Cloud Function for Search

### Create Function Code

Create a new file called `main.py` for your Cloud Function:

```python
import functions_framework
import json
import os
from google.cloud import bigquery
from openai import OpenAI
import numpy as np

# Initialize clients
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
bq_client = bigquery.Client()

@functions_framework.http
def search_documents(request):
    """HTTP Cloud Function to search documents in BigQuery."""
    
    # Set CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Max-Age': '3600'
    }
    
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return ('', 204, headers)
    
    try:
        # Parse request
        request_json = request.get_json(silent=True)
        query = request_json.get('query', '')
        category_filter = request_json.get('category', None)
        top_k = request_json.get('top_k', 5)
        
        if not query:
            return (json.dumps({"error": "No query provided"}), 400, headers)
        
        # Generate query embedding
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding
        
        # Build SQL query for vector search
        query_embedding_str = str(query_embedding)
        
        sql = f"""
        SELECT 
            title,
            text,
            category,
            (SELECT 
                SUM((CAST(q AS FLOAT64) - CAST(v AS FLOAT64)) * (CAST(q AS FLOAT64) - CAST(v AS FLOAT64)))
             FROM UNNEST(SPLIT('{query_embedding_str}', ',')) AS q WITH OFFSET pos1
             JOIN UNNEST(content_vector) AS v WITH OFFSET pos2
             ON pos1 = pos2
            ) AS distance
        FROM `{os.environ.get('BIGQUERY_TABLE')}`
        """
        
        # Add category filter if specified
        if category_filter:
            sql += f" WHERE category = '{category_filter}'"
        
        sql += " ORDER BY distance ASC LIMIT {top_k}"
        
        # Execute query
        query_job = bq_client.query(sql)
        results = []
        
        for row in query_job:
            results.append({
                "title": row.title,
                "text": row.text,
                "category": row.category,
                "score": float(row.distance)
            })
        
        return (json.dumps({"results": results}), 200, headers)
        
    except Exception as e:
        return (json.dumps({"error": str(e)}), 500, headers)
```

### Create requirements.txt

Create a `requirements.txt` file for dependencies:

```txt
functions-framework==3.*
google-cloud-bigquery==3.12.0
openai==1.3.0
numpy==1.24.3
```

### Deploy Cloud Function

Deploy the function using gcloud CLI:

```bash
# Set environment variables
export OPENAI_API_KEY="your-openai-api-key"
export BIGQUERY_TABLE="your-project.oai_docs.embedded_data"

# Deploy the function
gcloud functions deploy document-search \
  --runtime python310 \
  --trigger-http \
  --allow-unauthenticated \
  --region=us-central1 \
  --set-env-vars="OPENAI_API_KEY=$OPENAI_API_KEY,BIGQUERY_TABLE=$BIGQUERY_TABLE" \
  --source=.
```

Note the function URL returned after deployment - you'll need this for ChatGPT integration.

## Step 5: Test the Search Function

Test your Cloud Function with a sample query:

```python
import requests
import json

# Replace with your Cloud Function URL
function_url = "https://your-region-your-project.cloudfunctions.net/document-search"

# Test query
test_query = {
    "query": "How do I authenticate with OpenAI API?",
    "category": "authentication",
    "top_k": 3
}

response = requests.post(function_url, json=test_query)
results = response.json()

print("Search Results:")
for i, result in enumerate(results.get("results", [])):
    print(f"\n{i+1}. {result['title']}")
    print(f"   Category: {result['category']}")
    print(f"   Score: {result['score']:.4f}")
    print(f"   Text: {result['text'][:200]}...")
```

## Step 6: Integrate with ChatGPT Custom GPT

### Create OpenAPI Specification

Create an `openapi.yaml` file for ChatGPT integration:

```yaml
openapi: 3.0.0
info:
  title: Document Search API
  version: 1.0.0
servers:
  - url: https://your-region-your-project.cloudfunctions.net
paths:
  /document-search:
    post:
      summary: Search documents using vector similarity
      operationId: searchDocuments
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
                  description: The search query
                category:
                  type: string
                  description: Optional category filter
                top_k:
                  type: integer
                  description: Number of results to return
      responses:
        '200':
          description: Successful search
          content:
            application/json:
              schema:
                type: object
                properties:
                  results:
                    type: array
                    items:
                      type: object
                      properties:
                        title:
                          type: string
                        text:
                          type: string
                        category:
                          type: string
                        score:
                          type: number
```

### Configure Custom GPT in ChatGPT

1. Go to ChatGPT and navigate to "Explore GPTs"
2. Click "Create a GPT"
3. In the "Configure" tab:
   - Add a name and description for your GPT
   - In the "Instructions" section, add:
     ```
     You are a document search assistant. When users ask questions, use the document search API to find relevant information before answering.
     
     Instructions:
     1. When a user asks a question, first call the search_documents API with their query
     2. Use the returned documents to provide accurate, context-aware answers
     3. Cite the source documents when providing information
     4. If no relevant documents are found, acknowledge this and answer based on your general knowledge
     ```
   - In the "Actions" section, click "Create new action"
   - Paste your OpenAPI specification
   - Save