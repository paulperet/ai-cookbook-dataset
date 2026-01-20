# IndustrialKnowledgeAgent: The Smart Industrial Equipment Knowledge Agent

## Problem Statement

In industrial settings, engineers and technicians often struggle to manage and retrieve comprehensive information about various equipment. This information is scattered across technical manuals, maintenance logs, safety protocols, troubleshooting guides, and parts inventories. The fragmented nature of this data makes it difficult to access and utilize effectively, leading to inefficiencies and potential safety risks. This problem requires an intelligent, adaptive solution to provide real-time, context-aware responses to queries.

## Proposed Solution

To address these challenges, we propose an agentic workflow that integrates a Retrieval-Augmented Generation (RAG) system with a database querying system (FunctionCalling). This solution leverages LLMs (including structured output mechanism), embedding models and structured data retrieval to provide contextually relevant and precise information. The workflow is orchestrated by multiple agents, each with a specific role:

1. **RAGAgent**: Utilizes LLMs and Embedding models to retrieve and generate contextually relevant information from technical documents.
2. **DatabaseQueryAgent**: Handles precise and structured data retrieval from databases containing maintenance logs, technical specifications, parts inventories, and compliance records.
3. **WorkflowOrchestrator**: Orchestrates interactions between the RAGSearchAgent and DatabaseAgent, ensuring seamless and efficient query resolution.

## Dataset Details

### PDF Documents

The PDF documents contain detailed information about various industrial equipment, categorized into:
1. **Technical Manuals**: Operation and maintenance guides.
2. **Maintenance Guides**: Routine and preventive maintenance tasks.
3. **Troubleshooting Guides**: Solutions to common issues.
4. **Safety Protocols**: Safety procedures and guidelines.

### Databases

The databases contain structured information that complements the PDF documents:
1. **Compliance Database (`compliance_db`)**: Safety certifications and compliance statuses.
2. **Maintenance Database (`maintenance_db`)**: Logs of maintenance activities.
3. **Technical Specifications Database (`technical_specifications_db`)**: Detailed technical specifications.
4. **Parts Inventory and Compatibility Database (`parts_inventory_compatibility_db`)**: Information on parts, compatibility, and inventory status.

By integrating these datasets, the proposed agentic workflow aims to provide a comprehensive and efficient system for managing and retrieving industrial equipment information, ensuring that engineers and technicians have access to the most relevant and up-to-date information.

*NOTE*: Please note that all data used in this demonstration has been synthetically generated.

### Technical Architecture:

### Installation

Installs the necessary Python packages for the IndusAgent system.


```python
!pip install mistralai==1.5.1     # Mistral AI client
!pip install qdrant-client==1.13.2 # Vector database client
!pip install gdown==5.2.0        # Google Drive download
```

    [Collecting mistralai==1.5.1, ..., Successfully installed eval-type-backport-0.2.2 jsonpath-python-1.0.6 mistralai-1.5.1 mypy-extensions-1.0.0 typing-inspect-0.9.0]

### Imports

Imports required libraries for LLM operations, data processing, vector database management, and utility functions.


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

### Download Data

Downloads the dataset from Google Drive, extracts it to a data directory, and sets up the working environment. The dataset contains CSV files for database operations and PDFs for document processing.

By the end of the process, you should be able to see the downloaded data, as shown in the image below.

#### Download data from Google Drive


```python
file_id = "1lwYSN6ry3JOA7pw3WAx72a_IXGqqmR8y"
output_file = "data.zip"  # Change this if your file is not a ZIP file

# Google Drive direct download URL
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

# Download the file
gdown.download(gdrive_url, output_file, quiet=False)

print(f"✅ File downloaded: {output_file}")
```

    ✅ File downloaded: data.zip

#### Extract and setup data directory


```python
# Unzip the file into the current directory
with zipfile.ZipFile(output_file, 'r') as zip_ref:
    zip_ref.extractall(".")  # Extracts directly to the current directory

print(f"✅ Files extracted to: {os.getcwd()}")  # Confirm extraction path


output_dir = "data"

# Change working directory to the extracted folder
os.chdir(output_dir)

# Verify the new working directory
print(f"📂 Current directory: {os.getcwd()}")
```

    ✅ Files extracted to: /content
    📂 Current directory: /content/data


```python
# List files in the extracted folder
print("📜 Extracted files:", os.listdir())
```

    📜 Extracted files: ['csv_data', 'pdf_data']

### Set up environment variables

Sets up the Mistral API key as an environment variable for authentication.


```python
os.environ["MISTRAL_API_KEY"] = "<YOUR MISTRAL API KEY>" # Get your Mistral API key from https://console.mistral.ai/api-keys/
```

### Initialize Mistral LLM and Qdrant Vector Database

Initializes the Mistral LLM client for text generation and Qdrant vector database client for similarity search operations.

*Note*:

1. We will use our latest model, `Mistral Small 3` for demonstration.
2. You need to set up Qdrant Cloud or a Docker setup before proceeding. You can refer to the [documentation](https://qdrant.tech/cloud/) for the setup instructions.


```python
model = "mistral-small-latest"
mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
qdrant_client = QdrantClient(
    url= "<URL>",
    api_key= "<API KEY>",
) # Replace with your Qdrant API key and URL if you are using Qdrant Cloud - https://cloud.qdrant.io/
```

### System Prompts

The system uses three different types of prompts to guide the LLMs for response generation:

1. *PDF Summarization Prompt*: `summarization_prompt` is used to create concise summaries of PDF documents.
2. *Response Generation Prompt*: `response_generation_prompt` is used to generate responses based on retrieved context.
3. *Final Response Integration Prompt*: `final_response_generation_prompt` is used to summarize responses from multiple sources - PDFs and different databases.



```python
# Define the prompt for generating a response
response_generation_prompt = '''Based on the following context answer the query:\n\n Context: {context}\n\n Query: {query}'''

# Prompt for summarizing the PDF text
summarization_prompt = '''Your task is to summarize the following text focusing on the core essence of the text in maximum of 2-3 sentences.'''

# Prompt for final response summarization
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

## DataProcessor

The `DataProcessor` class is a comprehensive component that handles all data processing operations in the system. It manages both unstructured (PDFs) and structured (CSV) data, along with embedding generation and storage.

- PDF document processing and text extraction using Mistral OCR.
- CSV to database ingestion
- Embedding generation and vector storage
- Batch processing of documents and data

### Main Components

#### 1. Document Processing
- `get_categorized_filepaths`: Walks through the directory structure to get categorized PDF file paths
- `parse_pdf`: Extracts text from all pages of a PDF file using Mistral OCR.
- `process_single_pdf`: Processes individual PDFs through the complete pipeline
- `process_documents`: Handles sequential processing of multiple documents

#### 2. Summarization and Embeddings
- `summarize`: Generates concise summaries of text using the Mistral model
- `get_text_embedding`: Creates text embeddings using Mistral's embedding model
- `qdrant_insert_embeddings`: Stores embeddings with metadata in Qdrant vector database
- `process_and_store_embeddings`: Handles batch processing of embeddings

#### 3. Database Operations
- `insert_csv_to_table`: Loads a single CSV file into a specified database table
- `insert_data_database`: Handles multiple CSV files insertion into their respective tables


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

        # Upload a file

        uploaded_pdf = self.mistral_client.files.upload(
            file={
                "file_name": file_path,
                "content": open(file_path, "rb"),
            },
            purpose="ocr"
        )

        # Get a signed URL for the uploaded file

        signed_url = self.mistral_client.files.get_signed_url(file_id=uploaded_pdf.id)

        # Get OCR results

        ocr_response = self.mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            }
        )

        # Extract text from the OCR response

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

        Args:
            file_path (str): Path to the CSV file
            db_path (str): Path to the SQLite database
            table_name (str): Name of the table to create/update
        """
        df = pd.read_csv(file_path)
        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        conn.close()

    def insert_data_database(self, db_path: str, file_mappings: Dict[str, str]):
        """
        Bulk insert multiple CSV files into their respective database tables.

        Args:
            db_path (str): Path to the SQLite database
            file_mappings (Dict[str, str]): Dictionary mapping table names to CSV file paths
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

## RAGAgent

The `RAGAgent` class implements Retrieval-Augmented Generation (RAG) to provide intelligent search and response generation. It combines vector search capabilities with the LLM to give contextually relevant answers.

- Query categorization and classification
- Vector similarity search in Qdrant
- Context-aware response generation
- Document citation handling

### Main Components

#### 1. Query Processing
- `query_categorization`: Classifies queries into predefined categories (technical manual, safety protocol, etc.)
- `query`: Orchestrates the complete RAG pipeline from query to final response

#### 2. Search and Retrieval  
- `qdrant_search`: Performs semantic search using query embeddings, filters results by document category and returns top-k most relevant documents.

#### 3. Response Generation
- `generate_response`: Creates natural language responses using retrieved context, uses LLM with specialized prompts, Provides citations to source documents.

#### Query Category Model

A Pydantic model that defines the structure for query categorization, used by RAGAgent to classify queries into relevant categories (technical_manual, safety_protocol, etc.).


```python
# Define category model for query classification
class Category(BaseModel):
   category: str
```


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

    def query_categorization(self, query: str) ->