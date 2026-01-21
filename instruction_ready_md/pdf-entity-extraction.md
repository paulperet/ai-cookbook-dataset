# PDF Entity Extraction with Indexify and Mistral

This guide demonstrates how to build a robust entity extraction pipeline for PDF documents using Indexify and Mistral's large language models. You will learn how to efficiently extract named entities from PDF files for applications like information retrieval, content analysis, and data mining.

## Introduction

Entity extraction, or Named Entity Recognition (NER), involves identifying and classifying key information (entities) in text into predefined categories such as persons, organizations, locations, and dates. Applying this technique to PDF documents allows you to automatically convert unstructured text into structured, analyzable data.

## Prerequisites

Before you begin, ensure you have:

- Python 3.9 or later installed.
- A virtual environment (recommended).
- A Mistral API key.
- Basic familiarity with Python and the command line.

## Step 1: Set Up Your Environment

First, create and activate a Python virtual environment:

```bash
python3.9 -m venv ve
source ve/bin/activate
```

## Step 2: Install and Start Indexify

Indexify is the orchestration layer for our extraction pipeline. Install it using the official script:

```bash
curl https://getindexify.ai | sh
```

Start the Indexify server in the background:

```bash
./indexify server -d
```

This command starts a long-running server that exposes the ingestion and retrieval APIs.

## Step 3: Install Required Extractors

In a new terminal window (with your virtual environment still active), install the Indexify Extractor SDK and the specific extractors needed for this tutorial:

```bash
pip install indexify-extractor-sdk
indexify-extractor download tensorlake/pdfextractor
indexify-extractor download tensorlake/mistral
```

Once downloaded, start the extractor server:

```bash
indexify-extractor join-server
```

Keep this terminal running—it hosts the extractor services.

## Step 4: Create the Extraction Graph

The extraction graph defines the data flow for our pipeline. It specifies that text is first extracted from a PDF and then processed by Mistral to identify entities.

Create a new Python script (e.g., `create_graph.py`) and add the following code:

```python
from indexify import IndexifyClient, ExtractionGraph

# Initialize the client
client = IndexifyClient()

# Define the extraction graph using YAML syntax
extraction_graph_spec = """
name: 'pdf_entity_extractor'
extraction_policies:
  - extractor: 'tensorlake/pdfextractor'
    name: 'pdf_to_text'
  - extractor: 'tensorlake/mistral'
    name: 'text_to_entities'
    input_params:
      model_name: 'mistral-large-latest'
      key: 'YOUR_MISTRAL_API_KEY'  # Replace with your actual key
      system_prompt: 'Extract and categorize all named entities from the following text. Provide the results in a JSON format with categories: persons, organizations, locations, dates, and miscellaneous.'
    content_source: 'pdf_to_text'
"""

# Create the graph from the specification
extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
client.create_extraction_graph(extraction_graph)

print("Extraction graph created successfully.")
```

**Important:** Replace `'YOUR_MISTRAL_API_KEY'` with your actual Mistral API key.

Run the script to create the graph:

```bash
python create_graph.py
```

## Step 5: Implement the Entity Extraction Pipeline

Now, let's write the main logic to upload a PDF and extract its entities. Create a new script (e.g., `extract_entities.py`).

First, import the necessary libraries:

```python
import json
import os
import requests
from indexify import IndexifyClient
```

### Helper Function to Download a PDF

Add a helper function to download a sample PDF from a URL:

```python
def download_pdf(url, save_path):
    """Downloads a PDF from a given URL."""
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"PDF downloaded and saved to {save_path}")
```

### Main Extraction Function

Next, implement the core function that uses the Indexify client to upload a file and retrieve the extracted entities:

```python
def extract_entities_from_pdf(pdf_path):
    """Uploads a PDF to Indexify and returns the extracted entities."""
    client = IndexifyClient()
    
    # 1. Upload the PDF file to the extraction graph
    content_id = client.upload_file("pdf_entity_extractor", pdf_path)
    print(f"File uploaded. Content ID: {content_id}")
    
    # 2. Wait for the extraction pipeline to complete
    client.wait_for_extraction(content_id)
    print("Extraction completed.")
    
    # 3. Retrieve the extracted content from the 'text_to_entities' policy
    entities_content = client.get_extracted_content(
        content_id=content_id,
        graph_name="pdf_entity_extractor",
        policy_name="text_to_entities"
    )
    
    # 4. Parse the JSON response
    # The content is returned as bytes, so we decode it to a string first
    entities_json_str = entities_content[0]['content'].decode('utf-8')
    entities = json.loads(entities_json_str)
    
    return entities
```

### Execute the Pipeline

Finally, add the code to run the pipeline on a sample PDF:

```python
if __name__ == "__main__":
    # URL of a sample PDF (an arXiv paper in this case)
    pdf_url = "https://arxiv.org/pdf/2310.06825.pdf"
    pdf_path = "sample_document.pdf"
    
    # Download the PDF
    download_pdf(pdf_url, pdf_path)
    
    # Extract entities
    extracted_entities = extract_entities_from_pdf(pdf_path)
    
    # Print the results in a readable format
    print("\n" + "="*50)
    print("EXTRACTED ENTITIES")
    print("="*50)
    
    for category, entity_list in extracted_entities.items():
        print(f"\n{category.upper()}:")
        for entity in entity_list:
            print(f"  - {entity}")
```

Run the script:

```bash
python extract_entities.py
```

You should see output listing the extracted entities grouped by category (Persons, Organizations, etc.).

## Step 6: Customize the Extraction

You can easily customize the entity extraction by modifying the `system_prompt` in the extraction graph YAML. For example:

- **To extract only specific entity types:**
  ```yaml
  system_prompt: 'Extract only person names and organizations from the following text. Provide the results in a JSON format with categories: persons and organizations.'
  ```

- **To include relationships between entities:**
  ```yaml
  system_prompt: 'Extract named entities and their relationships from the following text. Provide the results in a JSON format with categories: entities (including type and name) and relationships (including type and involved entities).'
  ```

You can also experiment with different Mistral models by changing the `model_name` parameter (e.g., `'mistral-medium-latest'`) to balance speed and accuracy for your use case.

## Advantages of Using Indexify

While the code is straightforward, using Indexify provides significant operational benefits:

1.  **Scalability & High Availability:** The Indexify server can be deployed in the cloud to process thousands of PDFs concurrently. The pipeline automatically retries failed steps on another machine.
2.  **Flexibility:** You can swap the PDF extractor or the LLM for different models available in the [Indexify ecosystem](https://docs.getindexify.ai/usecases/pdf_extraction/) if the ones used here don't suit your specific documents.

## Next Steps

- Explore the [Indexify documentation](https://docs.getindexify.ai) to learn more about its capabilities.
- Try another tutorial, like [building a PDF summarization pipeline at scale](../pdf-summarization/).