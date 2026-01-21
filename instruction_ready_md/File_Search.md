# File Search Quickstart Guide

This guide demonstrates how to use the Gemini File Search tool to build a Retrieval-Augmented Generation (RAG) application. You will learn to create a managed document store, upload files, and have Gemini answer questions using your specific data with accurate citations.

## Prerequisites

### 1. Install the SDK
Ensure you have the latest Google Gen AI SDK installed. Version 1.49.0 or higher is required for File Search functionality.

```bash
pip install -U 'google-genai>=1.49.0'
```

### 2. Set Up Authentication
The File Search API uses API keys for authentication. **Important:** Your API key grants access to all data you upload, so keep it secure.

Set your API key as an environment variable or load it from a secure location.

```python
from google import genai
from google.genai import types

# Replace with your actual API key
GEMINI_API_KEY = "YOUR_API_KEY"
client = genai.Client(api_key=GEMINI_API_KEY)
```

## Step 1: Create a File Search Store

First, create a managed store to hold your documents.

```python
file_search_store = client.file_search_stores.create(
    config=types.CreateFileSearchStoreConfig(
        display_name='My File Search Store'
    )
)

print(f"Created store: {file_search_store.name}")
```

## Step 2: Prepare and Upload a Sample Document

### Download a Sample File
For this tutorial, download a sample text file from Project Gutenberg.

```bash
wget -q https://www.gutenberg.org/cache/epub/76401/pg76401.txt -O sample_poetry.txt
```

### Upload the File to Your Store
Upload the file directly to your newly created store. The ingestion process requires a moment to complete.

```python
import time

upload_op = client.file_search_stores.upload_to_file_search_store(
    file_search_store_name=file_search_store.name,
    file='sample_poetry.txt',
    config=types.UploadToFileSearchStoreConfig(
        display_name='A Survey of Modernist Poetry',
    )
)

print(f"Upload started: {upload_op.name}")

# Wait for processing to complete
while not (upload_op := client.operations.get(upload_op)).done:
    time.sleep(1)
    print(".", end="")

print("\nProcessing complete.")
```

### Alternative: Import from the File API
If you have already uploaded a file using the Gemini File API, you can import it directly into a store.

```python
# First, upload via the File API
file_ref = client.files.upload(
    file='sample_poetry.txt',
    config=types.UploadFileConfig(
        display_name='A Survey of Modernist Poetry',
        mime_type='text/plain',
    )
)
print(f"Uploaded via File API: {file_ref.name}")

# Then, import it into your File Search store
import_op = client.file_search_stores.import_file(
    file_search_store_name=file_search_store.name,
    file_name=file_ref.name,
)

print(f"File import started: {import_op.name}")

while not (import_op := client.operations.get(import_op)).done:
    time.sleep(1)
    print(".", end="")

print("\nProcessing complete.")
```

## Step 3: Generate Content Using File Search

Now, use the `file_search` tool in a generation request. Gemini will retrieve relevant information from your uploaded document to answer the question.

```python
MODEL_ID = "gemini-2.5-flash-lite"  # Choose a suitable model

response = client.models.generate_content(
    model=MODEL_ID,
    contents='What does the text say about E.E. Cummings?',
    config=types.GenerateContentConfig(
        tools=[types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=[file_search_store.name],
            )
        )]
    )
)

print(response.text)
```

### Configure Search Parameters
You can control the search behavior using parameters like `top_k`, which limits the number of document chunks passed to the model.

```python
top_K = 1  # Use only the most relevant chunk

response = client.models.generate_content(
    model=MODEL_ID,
    contents='What does the text say about E.E. Cummings?',
    config=types.GenerateContentConfig(
        tools=[types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=[file_search_store.name],
                top_k=top_K,
            )
        )]
    )
)

print(response.text)
```

## Step 4: Inspect Grounding Metadata and Citations

The response includes `grounding_metadata`, which contains citations and references to the source documents.

### View Retrieved Chunks
Examine the specific text chunks that were retrieved to inform the model's answer.

```python
import textwrap

grounding = response.candidates[0].grounding_metadata

if grounding and grounding.grounding_chunks:
    print(f"Found {len(grounding.grounding_chunks)} grounding chunks.")
    for i, chunk in enumerate(grounding.grounding_chunks, start=1):
        print(f"\nChunk {i} source: {chunk.retrieved_context.title}")
        print("Chunk text:")
        print(textwrap.indent(chunk.retrieved_context.text[:150] + "...", "  "))
else:
    print("No grounding metadata found.")
```

### Annotate the Response with Citations
The `grounding_supports` field maps spans of the generated text back to the source chunks, allowing you to create annotated outputs.

```python
# Accumulate the response as it is annotated.
annotated_response_parts = []

if not grounding or not grounding.grounding_supports:
    print("No grounding metadata or supports found for annotation.")
else:
    cursor = 0
    for support in grounding.grounding_supports:
        # Add the text before the current support
        annotated_response_parts.append(response.text[cursor:support.segment.start_index])

        # Construct the superscript citation from chunk IDs
        chunk_ids = ', '.join(map(str, support.grounding_chunk_indices))
        citation = f"<sup>{chunk_ids}</sup>"

        # Append the formatted, cited, supported text
        annotated_response_parts.append(f"**{support.segment.text}**{citation}")

        cursor = support.segment.end_index

    # Append any remaining text after the last support
    annotated_response_parts.append(response.text[cursor:])

    final_annotated_response = "".join(annotated_response_parts)
    print(final_annotated_response)
```

## Step 5: Use Custom Metadata for Filtering

You can attach custom metadata to files and use it to filter search results.

### Upload a File with Metadata
Add another document with metadata describing its genre and author.

```bash
wget -q https://www.gutenberg.org/files/11/11-0.txt -O alice_in_wonderland.txt
```

```python
upload_op = client.file_search_stores.upload_to_file_search_store(
    file_search_store_name=file_search_store.name,
    file='alice_in_wonderland.txt',
    config=types.UploadToFileSearchStoreConfig(
        display_name='Alice in Wonderland',
        custom_metadata=[
            types.CustomMetadata(key='genre', string_value='fiction'),
            types.CustomMetadata(key='author', string_value='Lewis Carroll'),
        ]
    )
)

while not (upload_op := client.operations.get(upload_op)).done:
    time.sleep(1)
    print(".", end="")

print("\nUpload complete.")
```

### Query with a Metadata Filter
When generating content, use a `metadata_filter` to restrict the search to documents matching specific criteria.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents='Who is the Queen?',
    config=types.GenerateContentConfig(
        tools=[types.Tool(
            file_search=types.FileSearch(
                file_search_store_names=[file_search_store.name],
                metadata_filter='genre = "fiction"'
            )
        )]
    )
)

print(response.text)
print('-' * 80)

# Verify only the filtered document was used
if grounding := response.candidates[0].grounding_metadata:
    unique_titles = {c.retrieved_context.title for c in grounding.grounding_chunks}
    print(f"Sources used: {unique_titles}")
```

## Summary

You have successfully built a basic RAG pipeline using Gemini's File Search tool. You learned to:
1.  Create a managed document store.
2.  Upload and ingest documents.
3.  Generate answers grounded in your specific data.
4.  Inspect citations and source material.
5.  Use custom metadata to filter search results.

This provides a foundation for building more complex applications that leverage your private or domain-specific data.