# Structured Data Extraction from Documents using Mistral AI Annotations

This guide demonstrates how to use Mistral AI's OCR capabilities with Annotations to extract structured data from documents. You'll learn to enhance vision models with OCR-powered structured outputs for better data extraction.

## Prerequisites

First, install the required library and download a sample PDF document.

```bash
pip install mistralai
```

```python
import base64
import json
from enum import Enum
from mistralai import Mistral
from mistralai.models import OCRResponse
from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel, Field
from IPython.display import Markdown, display
```

## Setup

### Download Sample PDF

```python
import requests

# Download sample PDF
pdf_url = "https://raw.githubusercontent.com/mistralai/cookbook/refs/heads/main/mistral/ocr/mistral7b.pdf"
pdf_response = requests.get(pdf_url)

with open("mistral7b.pdf", "wb") as f:
    f.write(pdf_response.content)
```

### Initialize Mistral Client

Create an API key from the [Mistral AI Platform](https://console.mistral.ai/api-keys/) and initialize the client.

```python
# Initialize Mistral client with your API key
api_key = "YOUR_API_KEY_HERE"  # Replace with your actual API key
client = Mistral(api_key=api_key)
```

## Step 1: Basic OCR Processing

First, let's understand how to process a document with Mistral's OCR without annotations.

### Encode PDF to Base64

```python
def encode_pdf(pdf_path):
    """Encode a PDF file to base64 string."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Encode the PDF
pdf_path = "mistral7b.pdf"
base64_pdf = encode_pdf(pdf_path)
```

### Process Document with OCR

```python
# Call the OCR API
pdf_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{base64_pdf}"
    },
    include_image_base64=True
)

# Convert response to JSON for inspection
response_dict = json.loads(pdf_response.model_dump_json())
print(json.dumps(response_dict, indent=4)[:1000])  # Show first 1000 characters
```

### Display OCR Results

Now, let's create helper functions to display the OCR results in a readable format.

```python
def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """
    Replace image placeholders in markdown with base64-encoded images.
    
    Args:
        markdown_str: Markdown text containing image placeholders
        images_dict: Dictionary mapping image IDs to base64 strings
    
    Returns:
        Markdown text with images replaced by base64 data
    """
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", 
            f"![{img_name}](data:image/png;base64,{base64_str})"
        )
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """
    Combine OCR text and images into a single markdown document.
    
    Args:
        ocr_response: Response from OCR processing containing text and images
    
    Returns:
        Combined markdown string with embedded images
    """
    markdowns = []
    
    # Extract images from each page
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        
        # Replace image placeholders with actual images
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))
    
    return "\n\n".join(markdowns)

# Display the OCR results
display(Markdown(get_combined_markdown(pdf_response)))
```

## Step 2: Define Annotation Schemas with Pydantic

Annotations allow you to extract structured data from documents. Let's define schemas for both document-level and bounding box (image) annotations.

```python
# Define image types as an Enum
class ImageType(str, Enum):
    GRAPH = "graph"
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"

# Schema for bounding box/image annotations
class Image(BaseModel):
    image_type: ImageType = Field(
        ..., 
        description="The type of the image. Must be one of 'graph', 'text', 'table' or 'image'."
    )
    description: str = Field(
        ..., 
        description="A description of the image."
    )

# Schema for document-level annotations
class Document(BaseModel):
    language: str = Field(
        ..., 
        description="The language of the document in ISO 639-1 code format (e.g., 'en', 'fr')."
    )
    summary: str = Field(
        ..., 
        description="A summary of the document."
    )
    authors: list[str] = Field(
        ..., 
        description="A list of authors who contributed to the document."
    )
```

## Step 3: Process Document with Annotations

Now, let's use these schemas to extract structured data from our document.

```python
# Process document with annotations
annotations_response = client.ocr.process(
    model="mistral-ocr-latest",
    pages=list(range(8)),  # Document annotations have an 8-page limit
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{base64_pdf}"
    },
    bbox_annotation_format=response_format_from_pydantic_model(Image),
    document_annotation_format=response_format_from_pydantic_model(Document),
    include_image_base64=False  # We don't need the actual images for annotation extraction
)

# Convert response to JSON for inspection
response_dict = json.loads(annotations_response.model_dump_json())
print(json.dumps(response_dict, indent=4))
```

### Display Extracted Annotations

Let's examine the structured data we extracted.

```python
print("Document Annotation:")
print(annotations_response.document_annotation)
print("\n" + "="*50 + "\n")

print("Bounding Box/Image Annotations:")
for page in annotations_response.pages:
    for image in page.images:
        print(f"\nImage ID: {image.id}")
        print(f"Location:")
        print(f"  - Top Left: ({image.top_left_x}, {image.top_left_y})")
        print(f"  - Bottom Right: ({image.bottom_right_x}, {image.bottom_right_y})")
        print(f"Annotation: {image.image_annotation}")
```

## Step 4: Display Full Document with Annotations

Now, let's create a comprehensive view that shows the document content alongside its annotations.

```python
# Process document with annotations and include images
annotations_response_with_images = client.ocr.process(
    model="mistral-ocr-latest",
    pages=list(range(8)),
    document={
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{base64_pdf}"
    },
    bbox_annotation_format=response_format_from_pydantic_model(Image),
    document_annotation_format=response_format_from_pydantic_model(Document),
    include_image_base64=True  # Include images for display
)

def replace_images_in_markdown_annotated(markdown_str: str, images_dict: dict) -> str:
    """
    Replace image placeholders in markdown with base64-encoded images and their annotations.
    
    Args:
        markdown_str: Markdown text containing image placeholders
        images_dict: Dictionary mapping image IDs to base64 strings and annotations
    
    Returns:
        Markdown text with images replaced by base64 data and their annotations
    """
    for img_name, data in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", 
            f"![{img_name}](data:image/png;base64,{data['image']})\n\n**Annotation:** {data['annotation']}"
        )
    return markdown_str

def get_combined_markdown_annotated(ocr_response: OCRResponse) -> str:
    """
    Combine OCR text, annotations, and images into a single markdown document.
    
    Args:
        ocr_response: Response from OCR processing containing text, images, and annotations
    
    Returns:
        Combined markdown string with embedded images and annotations
    """
    # Start with document annotation
    markdowns = [f"**Document Annotation:** {ocr_response.document_annotation}"]
    
    # Add each page's content with image annotations
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = {
                "image": img.image_base64,
                "annotation": img.image_annotation
            }
        
        # Replace image placeholders with actual images and annotations
        markdowns.append(replace_images_in_markdown_annotated(page.markdown, image_data))
    
    return "\n\n" + "="*50 + "\n\n".join(markdowns)

# Display the complete annotated document
display(Markdown(get_combined_markdown_annotated(annotations_response_with_images)))
```

## Step 5: Advanced Example - Financial Document Analysis

Let's apply the same technique to a different type of document with a modified schema.

```python
# Define custom schemas for financial document analysis
class FinancialImageType(str, Enum):
    GRAPH = "graph"
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"

class FinancialImage(BaseModel):
    image_type: FinancialImageType = Field(
        ..., 
        description="The type of the image in a financial context."
    )
    description: str = Field(
        ..., 
        description="A description of the financial image content."
    )

class FinancialDocument(BaseModel):
    languages: list[str] = Field(
        ..., 
        description="The list of languages present in the document in ISO 639-1 code format."
    )
    summary: str = Field(
        ..., 
        description="A summary of the financial document."
    )

# Process a financial document from a URL
financial_response = client.ocr.process(
    model="mistral-ocr-latest",
    pages=list(range(8)),
    document={
        "type": "document_url",
        "document_url": "https://upload.wikimedia.org/wikipedia/foundation/f/f6/WMF_Mid-Year-Financials_08-09-FINAL.pdf"
    },
    bbox_annotation_format=response_format_from_pydantic_model(FinancialImage),
    document_annotation_format=response_format_from_pydantic_model(FinancialDocument),
    include_image_base64=True
)

# Display the financial document with annotations
display(Markdown(get_combined_markdown_annotated(financial_response)))
```

## Key Takeaways

1. **Annotations Enhance OCR**: Mistral AI's annotation capabilities allow you to extract structured data beyond simple text recognition.

2. **Two Annotation Types**:
   - `document_annotation`: Extracts structured information from the entire document
   - `bbox_annotation`: Annotates specific bounding boxes (images, charts, tables) within the document

3. **Schema Flexibility**: Use Pydantic models to define exactly what data you want to extract, tailoring the extraction to your specific use case.

4. **Page Limitations**: Document annotations are limited to 8 pages. For longer documents, split them into chunks.

5. **Practical Applications**: This technique is particularly useful for:
   - Extracting metadata from documents
   - Analyzing financial reports
   - Processing invoices and receipts
   - Extracting structured data from research papers

By combining OCR with structured annotations, you can build powerful document processing pipelines that extract meaningful, structured data from unstructured documents.