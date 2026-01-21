# Enterprise Document AI with Mistral: Healthcare Document Processing

Healthcare generates 30% of the world's data, yet much of it remains locked in PDFs, scanned faxes, and unstructured documents. As regulations like the CMS prior authorization mandate push toward digital-first operations and hospital staffing shortages intensify, automated document processing has become critical infrastructure, not just for patient intake, but for back office operations like invoice management, medical billing and coding, and clinical documentation at scale.

**Key challenges driving Document AI adoption:**
*   30% of global data originates in healthcare, mostly unstructured
*   Legacy systems rely on paper, fax, and non-digital formats
*   Regulatory pressure (CMS mandates, interoperability requirements)
*   Severe staffing shortages across clinical and administrative roles

**Mistral OCR 3** handles intricate healthcare documents—handwritten notes, nested lab tables, checkboxes, and multi-page forms—with accuracy comparable to commercial solutions at a fraction of the cost. This guide demonstrates how to get started.

> You can also interactively explore Document AI in [AI Studio](https://console.mistral.ai/build/document-ai/ocr-playground)

## 1. Setup

First, let's install the required library and download the sample document.

### 1.1 Install the Mistral Client

```bash
pip install mistralai
```

### 1.2 Download the Sample Document

This guide uses `patient-packet-completed.pdf` - a synthetic multi-page patient packet containing demographics, vitals, and clinical notes.

```bash
wget https://raw.githubusercontent.com/mistralai/cookbook/refs/heads/main/mistral/ocr/hcls/patient-packet-completed.pdf
```

### 1.3 Verify the Document

Let's verify the file exists in our working directory.

```python
import os

# Path to your pdf (using local file)
pdf_path = "patient-packet-completed.pdf"

if os.path.exists(pdf_path):
    print(f"✅ Found: {pdf_path}")
    print(f"   Size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
else:
    print(f"❌ File not found: {pdf_path}")
    print("   Please ensure patient-packet-completed.pdf is in the working directory")
```

### 1.4 Create the Mistral Client

You will need an API key, which you can create on [AI Studio](https://console.mistral.ai/api-keys/).

```python
# Initialize Mistral client with API key
import os
from mistralai import Mistral

api_key = os.getenv("MISTRAL_API_KEY") # Replace with your way to retrieve API key

if not api_key:
    print("⚠️  WARNING: No API key found!")
else:
    client = Mistral(api_key=api_key)
    print("✅ Mistral client initialized")
```

---
## 2. Use Case: Patient Medical Record Packet OCR Processing

This section showcases **Mistral OCR 3** capabilities using a 3-page patient packet. We will use each page to highlight various features:

| Page | Document Type | OCR Feature |
|------|--------------|-------------|
| 1 | Patient Admission Form | **Form elements** - checkboxes, handwriting, unified unicode representation |
| 2 | Vital Signs Flowsheet | **HTML table output** - complex tables with rowspan/colspan |
| 3 | Foot X-ray | **Image annotations** - embedded images with descriptions |

> **Note**: Sample data is synthetic/anonymized.

### 2.1 Load and Process the Document

First, we need to encode the PDF to base64 and send it to the OCR API.

#### 2.1.1 Encode the PDF

```python
import base64

def encode_pdf(pdf_path):
    """Encode the pdf to base64."""
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {pdf_path} was not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
```

#### 2.1.2 Process the Full Document

Now, let's call the OCR API to process the entire PDF packet. We'll specify `table_format="html"` to get structured table outputs.

```python
import json

# Getting the base64 string
base64_pdf = encode_pdf(pdf_path)

# Call the OCR API
pdf_response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "document_url",
        "document_url": f""
    },
    include_image_base64=True,
    table_format="html" #Specify HTML format to render complex table formats
)

# Convert response to JSON format
response_dict = json.loads(pdf_response.model_dump_json())
```

Let's check the first part of the response to confirm it was successful.

```python
print(json.dumps(response_dict, indent=4)[0:1000]) # check the first 1000 characters
```

### 2.2 Create Helper Functions for Display

To better visualize the results, we'll create helper functions to display pages with styled tables and embedded images.

```python
from IPython.display import display, HTML, Markdown
from mistralai.models import OCRResponse
from bs4 import BeautifulSoup

# CSS styling for tables (reusable constant)
TABLE_STYLE = """
<style>
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 10px 0;
    }
    th, td {
        border: 1px solid black;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
    }
</style>
"""

def display_page_with_tables(page_index: int, ocr_data: dict, pdf_response: OCRResponse):
    """
    Display a page with styled HTML tables and embedded images.
    Tables are inserted inline at their original positions.
    Uses REST API response for tables, SDK response for images.

    Args:
        page_index: Index of the page to display (0-based)
        ocr_data: JSON data from REST API response
        pdf_response: OCRResponse object from SDK
    """
    if page_index >= len(ocr_data["pages"]):
        print(f"Page {page_index} not found")
        return

    page = ocr_data["pages"][page_index]
    markdown = page["markdown"]

    # Replace table placeholders with styled HTML tables (preserves order)
    # This specifically handles the format [tbl-X.html](tbl-X.html) where X is the table index
    if "tables" in page and page["tables"]:
        for table in page["tables"]:
            table_id = table.get("id", "")
            if table_id:
                # Replace the exact placeholder format from the OCR output
                placeholder = f"[{table_id}]({table_id})"
                styled_table = TABLE_STYLE + table["content"]
                markdown = markdown.replace(placeholder, styled_table)

    # Replace image placeholders with base64 from pdf_response
    if page_index < len(pdf_response.pages):
        for img in pdf_response.pages[page_index].images:
            markdown = markdown.replace(
                f"![{img.id}]({img.id})",
                f"<img src='{img.image_base64}' style='max-width:100%;'/>"
            )

    # Display as HTML with whitespace preservation
    display(HTML(f"<div style='white-space: pre-wrap;'>{markdown}</div>"))
```

### 2.3 Form Elements: Checkboxes & Structured Fields (Page 1)

Page 1 contains a **Patient Admission Form** with checkboxes, handwriting, and fill-in lines. Mistral OCR 3 uses **unified Unicode checkbox representation** (☐ unchecked, ☑ checked) for consistent parsing.

Let's display the first page.

```python
# Display Page 1 - Form Elements
print("📄 PAGE 1: Patient Admission Form")
print("Notice: Checkboxes rendered as ☐ (unchecked) and ☑ (checked)\n")
display_page_with_tables(0, response_dict, pdf_response)
```

### 2.4 HTML Table Output: Vital Signs Flowsheet (Page 2)

Page 2 contains a **Vital Signs Flowsheet** with complex table structures. Mistral OCR 3 gives the option to output tables as **HTML** with proper `rowspan` and `colspan` attributes, preserving the original structure for accurate data extraction.

Now, let's display the second page.

```python
# Display Page 2 - Vital Signs Flowsheet with HTML table
print("📄 PAGE 2: Vital Signs Flowsheet")
print("Notice: Tables output as HTML with rowspan/colspan preserved\n")
display_page_with_tables(1, response_dict, pdf_response)
```