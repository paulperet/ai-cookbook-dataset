# Industrial Knowledge Agent: A Smart Agentic Workflow for Equipment Information

## Overview

This guide demonstrates how to build an intelligent agentic workflow that integrates Retrieval-Augmented Generation (RAG) with structured database querying to manage industrial equipment information. You will create a system that can answer complex technical queries by retrieving and synthesizing information from both unstructured documents (PDFs) and structured databases.

## Prerequisites & Setup

Before starting, ensure you have the following:

1.  A **Mistral AI API key** from the [Mistral Console](https://console.mistral.ai/api-keys/).
2.  A **Qdrant Cloud** instance or a local Docker setup. Follow the [Qdrant Cloud documentation](https://qdrant.tech/cloud/) to set this up.
3.  The synthetic dataset used in this tutorial, which will be downloaded automatically.

### 1. Install Required Packages

Begin by installing the necessary Python libraries.

```bash
pip install mistralai==1.5.1     # Mistral AI client
pip install qdrant-client==1.13.2 # Vector database client
pip install gdown==5.2.0        # Google Drive download
```

### 2. Import Libraries

Import all required modules for LLM operations, data processing, and database management.

```python
# Core libraries
import os
import json
import functools
import warnings
from typing import List, Dict, Any, Tuple

# LLM and Data Processing
from mistralai import Mistral
from pydantic import BaseModel
import pandas as pd
import sqlite3
from tqdm import tqdm

# Vector Database
from qdrant_client import QdrantClient
from qdrant_client.models import (
   PointStruct, VectorParams, Distance,
   Filter, FieldCondition, MatchValue
)

# Data Download
import gdown
import zipfile

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
```

### 3. Download and Prepare the Dataset

The dataset contains synthetic CSV files for databases and PDF documents. The following code downloads and extracts it.

```python
# Download the dataset from Google Drive
file_id = "1lwYSN6ry3JOA7pw3WAx72a_IXGqqmR8y"
output_file = "data.zip"
gdrive_url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(gdrive_url, output_file, quiet=False)
print(f"✅ File downloaded: {output_file}")

# Extract the ZIP file
with zipfile.ZipFile(output_file, 'r') as zip_ref:
    zip_ref.extractall(".")
print(f"✅ Files extracted to: {os.getcwd()}")

# Change to the data directory
output_dir = "data"
os.chdir(output_dir)
print(f"📂 Current directory: {os.getcwd()}")

# List the extracted files
print("📜 Extracted files:", os.listdir())
```

### 4. Configure API Keys

Set your Mistral API key as an environment variable. You will also need your Qdrant URL and API key for the next step.

```python
os.environ["MISTRAL_API_KEY"] = "<YOUR MISTRAL API KEY>"
```

### 5. Initialize Clients

Initialize the Mistral LLM client and the Qdrant vector database client. We'll use the `mistral-small-latest` model.

```python
model = "mistral-small-latest"
mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# Replace with your Qdrant Cloud URL and API Key
qdrant_client = QdrantClient(
    url="<YOUR_QDRANT_URL>",
    api_key="<YOUR_QDRANT_API_KEY>",
)
```

### 6. Define System Prompts

These prompts guide the LLM for different tasks: summarizing documents, generating responses from context, and integrating final answers from multiple sources.

```python
# Prompt for generating a response from retrieved context
response_generation_prompt = '''Based on the following context answer the query:\n\n Context: {context}\n\n Query: {query}'''

# Prompt for summarizing PDF text concisely
summarization_prompt = '''Your task is to summarize the following text focusing on the core essence of the text in maximum of 2-3 sentences.'''

# Prompt for synthesizing a final answer from multiple sources (databases and docs)
final_response_summarization_prompt = """You are an expert technical assistant. Your task is to create a comprehensive,
coherent response by combining information from multiple sources: database records and documentation.

Consider the following guidelines:
1. Integrate information from both sources seamlessly
2. Resolve any conflicts between sources, if they exist
3. Present information in a logical, step-by-step manner when applicable
4. Include specific technical details, measurements, and procedures when available
5. Prioritize safety-related information when present
6. Add relevant maintenance intervals or schedules if mentioned
7. Reference specific part numbers or specifications when provided

The user's query is: {query}

Based on the following responses from different sources, create a unified, clear answer:
{responses}

Remember to:
- Focus on accuracy and completeness
- Maintain technical precision
- Use clear, professional language
- Address all aspects of the query
- Highlight any important warnings or precautions"""
```

## Step 1: Build the Data Processor

The `DataProcessor` class is the backbone for handling all data ingestion. It processes PDFs, loads CSV data into SQLite databases, and generates embeddings for the RAG system.

### 1.1 Define the DataProcessor Class

Create the class with methods for document processing, summarization, embedding generation, and database operations.

```python
class DataProcessor:
    """
    Handles all data processing operations including:
    - PDF parsing and text extraction
    - CSV to database ingestion
    - Embedding generation and storage
    - Batch processing of documents and data
    """
    def __init__(self, mistral_client: Mistral, qdrant_client: QdrantClient):
        self.mistral_client = mistral_client
        self.qdrant_client = qdrant_client

    def get_categorized_filepaths(self, root_dir: str) -> List[Dict[str, str]]:
        """
        Walk through the directory structure and get file paths with their categories.
        """
        categorized_files = []
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if not os.path.isdir(category_path):
                continue
            for root, _, files in os.walk(category_path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        filepath = os.path.join(root, file)
                        categorized_files.append({
                            'filepath': filepath,
                            'category': category
                        })
        return categorized_files

    def parse_pdf(self, file_path: str) -> str:
        """Parse a PDF file and extract text from all pages using Mistral OCR."""
        uploaded_pdf = self.mistral_client.files.upload(
            file={
                "file_name": file_path,
                "content": open(file_path, "rb"),
            },
            purpose="ocr"
        )
        signed_url = self.mistral_client.files.get_signed_url(file_id=uploaded_pdf.id)
        ocr_response = self.mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            }
        )
        text = "\n".join([x.markdown for x in (ocr_response.pages)])
        return text

    def summarize(self, text: str, summarization_prompt: str = summarization_prompt) -> str:
        """Summarize the given text using the Mistral model."""
        chat_response = self.mistral_client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": summarization_prompt
                },
                {
                    "role": "user",
                    "content": text
                },
            ],
            temperature=0
        )
        return chat_response.choices[0].message.content

    def get_text_embedding(self, inputs: List[str]) -> List[float]:
        """Get the text embedding for the given inputs."""
        embeddings_batch_response = self.mistral_client.embeddings.create(
            model="mistral-embed",
            inputs=inputs
        )
        return embeddings_batch_response.data[0].embedding

    def qdrant_insert_embeddings(self, summaries: List[str], texts: List[str],
                               filepaths: List[str], categories: List[str]):
        """Insert embeddings into Qdrant with metadata."""
        embeddings = [self.get_text_embedding([t]) for t in summaries]
        if not self.qdrant_client.collection_exists("embeddings"):
            self.qdrant_client.create_collection(
                collection_name="embeddings",
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
        self.qdrant_client.upsert(
            collection_name="embeddings",
            points=[
                PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        "filepath": filepaths[idx],
                        "category": categories[idx],
                        "text": texts[idx]
                    }
                ) for idx, embedding in enumerate(embeddings)
            ]
        )

    def process_single_pdf(self, file_info: Dict[str, str]) -> Dict[str, any]:
        """Process a single PDF file through the pipeline."""
        filepath = file_info['filepath']
        category = file_info['category']
        pdf_text = self.parse_pdf(filepath)
        summary = self.summarize(pdf_text)
        return {
            'filepath': filepath,
            'category': category,
            'full_text': pdf_text,
            'summary': summary
        }

    def process_documents(self, file_list: List[Dict[str, str]]) -> List[Dict[str, any]]:
        """Process documents sequentially."""
        processed_docs = []
        for file_info in tqdm(file_list, desc="Processing PDFs"):
            try:
                processed_doc = self.process_single_pdf(file_info)
                processed_docs.append(processed_doc)
            except Exception as e:
                print(f"Error processing {file_info['filepath']}: {str(e)}")
                continue
        return processed_docs

    def insert_csv_to_table(self, file_path: str, db_path: str, table_name: str):
        """
        Insert CSV data into a table of SQLite database.
        """
        df = pd.read_csv(file_path)
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()

    def insert_data_database(self, db_path: str, file_mappings: Dict[str, str]):
        """
        Bulk insert multiple CSV files into their respective database tables.
        """
        for table_name, file_path in file_mappings.items():
            try:
                self.insert_csv_to_table(file_path, db_path, table_name)
                print(f"Successfully inserted data into {table_name}")
            except Exception as e:
                print(f"Error inserting data into {table_name}: {str(e)}")

    def process_and_store_embeddings(self, docs: List[Dict[str, any]], batch_size: int = 10):
        """Generate embeddings and store them in Qdrant in batches."""
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            texts = [doc['full_text'] for doc in batch]
            summaries = [doc['summary'] for doc in batch]
            filepaths = [doc['filepath'] for doc in batch]
            categories = [doc['category'] for doc in batch]
            try:
                self.qdrant_insert_embeddings(summaries, texts, filepaths, categories)
                print(f"Processed batch {i//batch_size + 1}/{(len(docs) + batch_size - 1)//batch_size}")
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {str(e)}")
                continue
```

### 1.2 Process PDF Documents and Generate Embeddings

Now, instantiate the `DataProcessor` and use it to parse all PDFs, create summaries, and store their embeddings in Qdrant for later retrieval.

```python
# Initialize the DataProcessor
data_processor = DataProcessor(mistral_client, qdrant_client)

# Get all PDF file paths categorized by folder (e.g., technical_manuals, safety_protocols)
pdf_root_dir = "pdf_data"
categorized_files = data_processor.get_categorized_filepaths(pdf_root_dir)
print(f"Found {len(categorized_files)} PDF files.")

# Process all PDF documents (OCR, summarization)
processed_docs = data_processor.process_documents(categorized_files)
print(f"Successfully processed {len(processed_docs)} documents.")

# Generate and store embeddings for the processed documents in Qdrant
data_processor.process_and_store_embeddings(processed_docs, batch_size=10)
print("✅ PDF embeddings stored in Qdrant.")
```

### 1.3 Load Structured Data into SQLite Databases

Next, load the CSV files containing structured data (maintenance logs, parts inventory, etc.) into SQLite databases. This data will be queried by the `DatabaseQueryAgent`.

```python
# Define the mapping between CSV files and their corresponding database tables
csv_data_dir = "csv_data"
file_mappings = {
    'compliance': os.path.join(csv_data_dir, 'compliance.csv'),
    'maintenance': os.path.join(csv_data_dir, 'maintenance.csv'),
    'technical_specifications': os.path.join(csv_data_dir, 'technical_specifications.csv'),
    'parts_inventory': os.path.join(csv_data_dir, 'parts_inventory.csv')
}

# Create a SQLite database and load all CSV data
db_path = "industrial_data.db"
data_processor.insert_data_database(db_path, file_mappings)
print("✅ Structured data loaded into SQLite database.")
```

## Step 2: Create the RAG Agent

The `RAGAgent` handles querying the vector store (Qdrant) and generating context-aware answers from the retrieved document chunks.

### 2.1 Define the Query Category Model

First, define a Pydantic model to structure the output of the query categorization step. This helps filter searches by document type.

```python
class Category(BaseModel):
   category: str
```

### 2.2 Define the RAGAgent Class

This class contains the logic for categorizing a user's query, searching the vector database, and generating a final answer.

```python
class RAGAgent:
    """
    Agent responsible for Retrieval-Augmented Generation (RAG) operations.
    """
    def __init__(self, mistral_client: Mistral, qdrant_client: QdrantClient):
        self.mistral_client = mistral_client
        self.qdrant_client = qdrant_client

    def generate_response(self, context: str, query: str) -> str:
        """Generate a response based on the given context and query."""
        chat_response = self.mistral_client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": response_generation_prompt.format(context=context, query=query)
                },
            ]
        )
        return chat_response.choices[0].message.content

    def query_categorization(self, query: str) -> str:
        """Classify the query into a predefined document category."""
        chat_response = self.mistral_client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Categorize the following query into one of these categories: technical_manual, safety_protocol, maintenance_guide, troubleshooting_guide. Query: {query}. Return only the category name."
                },
            ],
            response_format=Category
        )
        return chat_response.choices[0].message.content

    def qdrant_search(self, query: str, category: str = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search Qdrant for documents relevant to the query.
        Optionally filter by document category.
        """
        # Generate an embedding for the query
        query_embedding = self.mistral_client.embeddings.create(
            model="mistral-embed",
            inputs=[query]
        ).data[0].embedding

        # Build a filter if a category is specified
        search_filter = None
        if category:
            search_filter = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            )

        # Perform the search
        search_result = self.qdrant_client.search(
            collection_name="embeddings",
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=top_k
        )

        # Format the results
        results = []
        for point in search_result:
            results.append({
                'text': point.payload['text'],
                'filepath': point.payload['filepath'],
                'category': point.payload['category'],
                'score': point.score
            })
        return results

    def query(self, user_query: str) -> str:
        """
        Main method to handle a user query end-to-end.
        1. Categorize the query.
        2. Search for relevant documents.
        3. Generate a response using the retrieved context.
        """
        print(f"🔍 Processing query: {user_query}")

        # Step 1: Categorize the query
        category = self.query_categorization(user_query)
        print(f"   Identified category: {category}")

        # Step 2: Search Qdrant for relevant document chunks
        search_results = self.qdrant_search(user_query, category=category, top_k=3)
        print(f"   Retrieved {len(search_results)} relevant document chunks.")

        if not search_results:
            return "I couldn't find any relevant documentation to answer your query."

        # Step 3: Combine the retrieved text into a single context
        context = "\n\n---\n\n".join([res['text'] for res in search_results])

        # Step 4: Generate the final answer
        answer = self.generate_response(context, user_query)

        # Step 5: Add citations
        citation_text = "\n\n**Sources:**\n"
        for i, res in enumerate(search_results, 1):
            citation_text += f"{i}. {os.path.basename(res['filepath'])} (Category: {res['category']})\n"

        return answer + citation_text
```

### 2.3 Test the R