# Guide: Implementing File Search with OpenAI's Responses API

## Overview
This guide demonstrates how to use OpenAI's file search tool within the Responses API to build a simplified RAG (Retrieval-Augmented Generation) system. You'll learn to create a vector store from PDF documents, perform semantic searches, and evaluate retrieval performance—all through a unified API interface.

## Prerequisites

Ensure you have the following Python packages installed:

```bash
pip install PyPDF2 pandas tqdm openai
```

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Step 1: Initialize the OpenAI Client

```python
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import concurrent
import PyPDF2
import os
import pandas as pd
import base64

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
```

## Step 2: Prepare Your PDF Directory

Organize your PDF files in a local directory. This example uses PDFs from OpenAI's blog:

```python
dir_pdfs = 'openai_blog_pdfs'  # Directory containing your PDFs
pdf_files = [os.path.join(dir_pdfs, f) for f in os.listdir(dir_pdfs)]
```

## Step 3: Create and Populate a Vector Store

### 3.1 Create the Vector Store

First, define functions to create a vector store and upload PDFs:

```python
def create_vector_store(store_name: str) -> dict:
    """Create a new vector store on OpenAI."""
    try:
        vector_store = client.vector_stores.create(name=store_name)
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed
        }
        print("Vector store created:", details)
        return details
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}

def upload_single_pdf(file_path: str, vector_store_id: str):
    """Upload a single PDF to the vector store."""
    file_name = os.path.basename(file_path)
    try:
        file_response = client.files.create(
            file=open(file_path, 'rb'), 
            purpose="assistants"
        )
        attach_response = client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_response.id
        )
        return {"file": file_name, "status": "success"}
    except Exception as e:
        print(f"Error with {file_name}: {str(e)}")
        return {"file": file_name, "status": "failed", "error": str(e)}

def upload_pdf_files_to_vector_store(vector_store_id: str):
    """Upload all PDFs in parallel to the vector store."""
    pdf_files = [os.path.join(dir_pdfs, f) for f in os.listdir(dir_pdfs)]
    stats = {
        "total_files": len(pdf_files), 
        "successful_uploads": 0, 
        "failed_uploads": 0, 
        "errors": []
    }
    
    print(f"{len(pdf_files)} PDF files to process. Uploading in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(upload_single_pdf, file_path, vector_store_id): file_path 
            for file_path in pdf_files
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures), 
            total=len(pdf_files)
        ):
            result = future.result()
            if result["status"] == "success":
                stats["successful_uploads"] += 1
            else:
                stats["failed_uploads"] += 1
                stats["errors"].append(result)

    return stats
```

### 3.2 Execute the Upload

Now create your vector store and upload all PDFs:

```python
store_name = "openai_blog_store"
vector_store_details = create_vector_store(store_name)
upload_stats = upload_pdf_files_to_vector_store(vector_store_details["id"])
```

OpenAI automatically processes each PDF: splitting content into chunks, generating embeddings, and storing them in the vector store for efficient retrieval.

## Step 4: Perform Standalone Vector Search

You can query the vector store directly using the vector search API:

```python
query = "What's Deep Research?"
search_results = client.vector_stores.search(
    vector_store_id=vector_store_details['id'],
    query=query
)

# Examine the results
for result in search_results.data:
    print(f"{len(result.content[0].text)} characters of content from {result.filename} with relevance score: {result.score}")
```

Each result includes the filename, content snippet, and a relevance score calculated through hybrid search.

## Step 5: Integrate File Search with LLM Responses

The Responses API allows you to combine file search and LLM generation in a single call:

```python
query = "What's Deep Research?"
response = client.responses.create(
    input=query,
    model="gpt-4o-mini",
    tools=[{
        "type": "file_search",
        "vector_store_ids": [vector_store_details['id']],
    }]
)

# Extract the retrieved files
annotations = response.output[1].content[0].annotations
retrieved_files = set([result.filename for result in annotations])

print(f'Files used: {retrieved_files}')
print('Response:')
print(response.output[1].content[0].text)
```

The model automatically retrieves relevant content from your vector store and generates an answer based on that context.

## Step 6: Evaluate Retrieval Performance

To measure how well your file search system performs, you'll need an evaluation dataset.

### 6.1 Generate Evaluation Questions

Create questions that can only be answered by specific documents:

```python
def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def generate_questions(pdf_path):
    """Generate a question specific to a PDF's content."""
    text = extract_text_from_pdf(pdf_path)

    prompt = (
        "Can you generate a question that can only be answered from this document?:\n"
        f"{text}\n\n"
    )

    response = client.responses.create(
        input=prompt,
        model="gpt-4o",
    )

    question = response.output[0].content[0].text
    return question

# Generate questions for all PDFs
questions_dict = {}
for pdf_path in pdf_files:
    question = generate_questions(pdf_path)
    questions_dict[os.path.basename(pdf_path)] = question
```

### 6.2 Evaluate Retrieval Metrics

Define a function to process queries and calculate retrieval metrics:

```python
def process_query(row, vector_store_id, k=5):
    """Process a single query and evaluate retrieval performance."""
    query = row['query']
    expected_filename = row['_id'] + '.pdf'
    
    # Call file_search via Responses API
    response = client.responses.create(
        input=query,
        model="gpt-4o-mini",
        tools=[{
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
            "max_num_results": k,
        }],
        tool_choice="required"  # Force file_search for evaluation
    )
    
    # Extract annotations
    annotations = None
    if hasattr(response.output[1], 'content') and response.output[1].content:
        annotations = response.output[1].content[0].annotations
    elif hasattr(response.output[1], 'annotations'):
        annotations = response.output[1].annotations

    if annotations is None:
        print(f"No annotations for query: {query}")
        return False, 0, 0

    # Get top-k retrieved filenames
    retrieved_files = [result.filename for result in annotations[:k]]
    
    # Calculate metrics
    if expected_filename in retrieved_files:
        rank = retrieved_files.index(expected_filename) + 1
        reciprocal_rank = 1 / rank
        correct = True
    else:
        reciprocal_rank = 0
        correct = False

    # Calculate Average Precision
    precisions = []
    num_relevant = 0
    for i, fname in enumerate(retrieved_files):
        if fname == expected_filename:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    
    return correct, reciprocal_rank, avg_precision
```

### 6.3 Run the Evaluation

Convert your questions to a structured format and evaluate:

```python
# Prepare evaluation data
rows = []
for filename, query in questions_dict.items():
    rows.append({"query": query, "_id": filename.replace(".pdf", "")})

# Set evaluation parameters
k = 5
total_queries = len(rows)

# Process all queries in parallel
with ThreadPoolExecutor() as executor:
    results = list(tqdm(
        executor.map(
            lambda row: process_query(row, vector_store_details['id'], k), 
            rows
        ), 
        total=total_queries
    ))

# Calculate aggregate metrics
correct_retrievals_at_k = 0
reciprocal_ranks = []
average_precisions = []

for correct, rr, avg_precision in results:
    if correct:
        correct_retrievals_at_k += 1
    reciprocal_ranks.append(rr)
    average_precisions.append(avg_precision)

recall_at_k = correct_retrievals_at_k / total_queries
precision_at_k = recall_at_k  # In this context, same as recall
mrr = sum(reciprocal_ranks) / total_queries
map_score = sum(average_precisions) / total_queries

# Print results
print(f"Metrics at k={k}:")
print(f"Recall@{k}: {recall_at_k:.4f}")
print(f"Precision@{k}: {precision_at_k:.4f}")
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
print(f"Mean Average Precision (MAP): {map_score:.4f}")
```

## Key Takeaways

Through this guide, you've learned to:

1. **Create evaluation datasets** by generating document-specific questions using LLMs
2. **Build a vector store** and populate it with PDF documents through OpenAI's API
3. **Retrieve answers** using the `file_search` tool in the Responses API, which handles storage, embeddings, and retrieval in a single call
4. **Analyze retrieval performance** using standard metrics like Recall@k, MRR, and MAP

The file search tool simplifies RAG implementation by handling document processing, embedding generation, and semantic search internally. This reduces the complexity of building and maintaining separate vector databases and embedding pipelines.

## Next Steps

- Experiment with different `k` values to balance recall and precision
- Implement metadata filtering to refine search results
- Test with your own domain-specific documents
- Compare performance between different OpenAI models
- Add human evaluation to supplement automated metrics

Remember that while automated evaluation provides useful benchmarks, human verification remains crucial for production applications.