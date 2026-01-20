# Enterprise Document AI with Mistral: Healthcare Document Processing

Healthcare generates 30% of the world's data, yet much of it remains locked in PDFs, scanned faxes, and unstructured documents. As regulations like the CMS prior authorization mandate push toward digital-first operations and hospital staffing shortages intensify, automated document processing has become critical infrastructure, not just for patient intake, but for back office operations like invoice management, medical billing and coding, and clinical documentation at scale.

**Key challenges driving Document AI adoption:**

*   30% of global data originates in healthcare, mostly unstructured
*   Legacy systems rely on paper, fax, and non-digital formats
*   Regulatory pressure (CMS mandates, interoperability requirements)
*   Severe staffing shortages across clinical and administrative roles

**Mistral OCR 3** handles intricate healthcare documents—handwritten notes, nested lab tables, checkboxes, and multi-page forms—with accuracy comparable to commercial solutions at a fraction of the cost. This cookbook demonstrates how to get started.


> You can also interactively explore Document AI in [AI Studio](https://console.mistral.ai/build/document-ai/ocr-playground)



## 1. Setup

First, let's install `mistralai` and download the document.


```python
%%capture
!pip install mistralai
```

### Sample Document

This cookbook uses `patient-packet-completed.pdf` - a synthetic multi-page patient packet containing demographics, vitals, and clinical notes.


```python
%%capture
!wget https://raw.githubusercontent.com/mistralai/cookbook/refs/heads/main/mistral/ocr/hcls/patient-packet-completed.pdf
```


```python
# Verify sample document exists
import os

# Path to your pdf (using local file)
pdf_path = "patient-packet-completed.pdf"

if os.path.exists(pdf_path):
    print(f"✅ Found: {pdf_path}")
    print(f"   Size: {os.path.getsize(pdf_path) / 1024:.1f} KB")
else:
    print(f"❌ File not found: {pdf_path}")
    print("   Please ensure patient-packet-completed.pdf is in the working directory")

# List available sample files in the workspace
!ls -la *.pdf 2>/dev/null || echo "No PDF files found in current directory"
```

    ✅ Found: patient-packet-completed.pdf
       Size: 4517.9 KB
    -rw-r--r-- 1 root root 4626348 Dec 18 15:15 patient-packet-completed.pdf


### Create Client

We will need to set up our client. You can create an API key on [AI Studio](https://console.mistral.ai/api-keys/).


```python
# Initialize Mistral client with API key
import os
from mistralai import Mistral
from google.colab import userdata
import requests

api_key = userdata.get('MISTRAL_API_KEY') # Replace with your way to retrieve API key

if not api_key:
    print("⚠️  WARNING: No API key found!")
else:
    client = Mistral(api_key=api_key)
    print("✅ Mistral client initialized")
```

    ✅ Mistral client initialized


---
## 2. Use Case: Patient Medical Record Packet OCR Processing

This section showcases **Mistral OCR 3** capabilities using a 3-page patient packet. We will use each page to highlight various features:

| Page | Document Type | OCR Feature |
|------|--------------|-------------|
| 1 | Patient Admission Form | **Form elements** - checkboxes, handwriting, unified unicode representation |
| 2 | Vital Signs Flowsheet | **HTML table output** - complex tables with rowspan/colspan |
| 3 | Foot X-ray | **Image annotations** - embedded images with descriptions |

> **Note**: Sample data is synthetic/anonymized.

### 2.1 Setup: Load and Process Document

First, let's encode the PDF and run OCR on the full packet. We'll then explore each page's output.


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

Process the full document and get the OCR output:


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


```python
print(json.dumps(response_dict, indent=4)[0:1000]) # check the first 1000 characters
```

    {
        "pages": [
            {
                "index": 0,
                "markdown": "Southern Cross Healthcare\n\nPatient Admission Form\n\nIMPORTANT: Please send this completed form to the hospital where you will have your procedure/surgery.\n\nPERSONAL AND ADMINISTRATION DETAILS\n\n[tbl-0.html](tbl-0.html)\n\nPAYMENT DETAILS\n\nHow will your procedure be paid for? Tick and complete as many as applies:\n\n\u2610 Health insurance\n\u2610 ACC\n\u2610 DHB\n\u2611 Paid personally\n\u2610 Other \u2026\u2026\u2026\u2026\u2026\u2026\u2026\u2026\u2026\u2026\n\nDetails of health insurance\n\u2610 Southern Cross Affiliated Provider contract\n\nName of Insurer: ___________________________\n\nInsurance Plan Name: ___________________________\n\nMembership No: ___________________________\n\nHave you obtained \u201cprior approval\u201d for payment? Yes \u2610 No \u2610\n\nApproval No: ___________________________\n\n(Provide your prior approval letter in advance)\n\nAdditional charges\n\nDepending on your hea


Let's stylize the output for easier understanding


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

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """Replace image placeholders in markdown with base64-encoded images."""
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"<img src='{base64_str}' style='max-width:100%;'/>"
        )
    return markdown_str

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

def display_all_pages(ocr_data: dict, pdf_response: OCRResponse, pages=None):
    """
    Display all pages with styled HTML tables and images.

    Args:
        ocr_data: JSON data from REST API response
        pdf_response: OCRResponse object from SDK
        pages: List of page indices to display (None for all pages)
    """
    if pages is None:
        pages = range(len(ocr_data["pages"]))

    for i in pages:
        # Print page separator with proper newline handling
        print(f"\n{'='*60}")
        print(f"📄 PAGE {ocr_data['pages'][i]['index'] + 1}")
        print('='*60)

        # Display the page content with proper whitespace preservation
        display_page_with_tables(i, ocr_data, pdf_response)

        # Add spacing between pages for better readability
        print(f"\n{'\n'}")
```

### 2.2 Form Elements: Checkboxes & Structured Fields (Page 1)

Page 1 contains a **Patient Admission Form** with checkboxes, handwriting, and fill-in lines. Mistral OCR 3 uses **unified Unicode checkbox representation** (☐ unchecked, ☑ checked) for consistent parsing.


```python
# Display Page 1 - Form Elements
print("📄 PAGE 1: Patient Admission Form")
print("Notice: Checkboxes rendered as ☐ (unchecked) and ☑ (checked)\n")
display_page_with_tables(0, response_dict, pdf_response)
```

    📄 PAGE 1: Patient Admission Form
    Notice: Checkboxes rendered as ☐ (unchecked) and ☑ (checked)
    



<div style='white-space: pre-wrap;'>Southern Cross Healthcare

Patient Admission Form

IMPORTANT: Please send this completed form to the hospital where you will have your procedure/surgery.

PERSONAL AND ADMINISTRATION DETAILS


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
<table><tr><td>Surname (family name): Thompson</td><td>Mr ☑ Mrs ☐ Ms ☐ Miss ☐ Mstr ☐ Dr ☐</td></tr><tr><td>First name(s): Michael</td><td>Preferred name: Mike</td></tr><tr><td>Date of birth: 05, 12, 1998</td><td>NHI: ZAA0067</td></tr><tr><td>Gender: ☑ Male ☐ Female ☐ I identify my gender as</td><td></td></tr><tr><td>Residential address: 124 Mapleview Dr, Springfield, IL 62629</td><td></td></tr><tr><td>Postal address: Same as above</td><td></td></tr><tr><td>Email address: mrthompson12345@gmail.com</td><td></td></tr><tr><td>Telephone: (Home) 555-361-1492 (Business) — (Mobile) —</td><td></td></tr><tr><td colspan="2">New Zealand resident: Yes ☐ No ☐ If No, complete the ‘Acknowledgement Form: Non-NZ resident’ (on our website).</td></tr><tr><td colspan="2">Which ethnic group do you belong to? Tick the box or boxes which apply to you.<br/>☐ New Zealand European ☐ Māori ☐ Samoan ☐ Cook Island Māori ☐ Tongan ☐ Niuean ☐ Chinese ☐ Indian<br/>☑ Other (such as Dutch, Japanese, Tokelauan) Please state: _________________</td></tr><tr><td>General Practitioner (Name): Daniel Park</td><td>Telephone: 555-246-8239</td></tr><tr><td>Medical Centre: Springfield Hospital</td><td></td></tr><tr><td colspan="2">NEXT OF KIN/CONTACT PERSON</td></tr><tr><td>Name: Sarah Thompson</td><td>Relationship to patient: Spouse</td></tr><tr><td>Address: —</td><td></td></tr><tr><td>Telephone: (Home) 555-246-1234 (Business) — (Mobile) —</td><td></td></tr></table>

PAYMENT DETAILS

How will your procedure be paid for? Tick and complete as many as applies:

☐ Health insurance
☐ ACC
☐ DHB
☑ Paid personally
☐ Other …………………………

Details of health insurance
☐ Southern Cross Affiliated Provider contract

Name of Insurer: ___________________________

Insurance Plan Name: ___________________________

Membership No: ___________________________

Have you obtained “prior approval” for payment? Yes ☐ No ☐

Approval No: ___________________________

(Provide your prior approval letter in advance)

Additional charges

Depending on your health insurance policy or plan you may be required to pay an excess (co-payment).

You may also be required to pay for some charges such as visitor meals that are not covered by insurance, ACC or DHB.

Payment prior to surgery

You may be asked to pay a deposit 3-5 days before admission. The amount is based on the estimated cost of the procedure payable by you not otherwise covered by your insurance, ACC or DHB. The deposit will be refunded to you if the procedure is cancelled.

Methods of payment

We accept payment by EFTPOS, VISA, Mastercard, internet banking or online at our website

www.southerncrosshealthcare.co.nz (search “payment information”). Personal cheques are not accepted. We prefer not to receive payment by cash.

I will pay my account by: EFTPOS ☑ Credit Card ☐ Debit Card ☐ Internet Banking ☐

Internet banking details

Payee: Southern Cross Healthcare Ltd
Bank a/c: 12-3113-0126623-00
Particulars: Patient Name
Code: Date of Surgery e.g. 12 Sep 2020
Reference: Hospital e.g. Hamilton

Would you like to receive your invoice via email? ☑ YES ☐ NO

We will send the invoice to the email address you have provided above.

SCHL040 12/2020 Southern Cross Healthcare

Please complete the agreement section on the reverse of this page.</div>


### 2.3 HTML Table Output: Vital Signs Flowsheet (Page 2)

Page 2 contains a **Vital Signs Flowsheet** with complex table structures. Mistral OCR 3 gives the option to output tables as **HTML** with proper `rowspan` and `colspan` attributes, preserving the original structure for accurate data extraction.


```python
# Display Page 2 - Vital Signs Flowsheet with HTML table
print("📄 PAGE 2: Vital Signs Flowsheet")
print("Notice: Tables output as HTML with rowspan/colspan preserved\n")
display_page_with_tables(1, response_dict, pdf_response)
```

    📄 PAGE 2: Vital Signs Flowsheet
    Notice: Tables output as HTML with rowspan/colspan preserved
    



<div style='white-space: pre-wrap;'>Kitty Wilde RN Case Manager

# Vital Signs Flow Sheet


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
<table><tr><td colspan="3">Patient: Michael Thompson</td><td colspan="5" rowspan="4">Notes: Patient presents left foot pain after ladder fall. Suspected metatarsal fracture.</td></tr><tr><td colspan="3">DOB: 05/12/1978</td></tr><tr><td colspan="3">M/F: M</td></tr><tr><td colspan="3">Physician: Dr. Emily Carter</td></tr><tr><td colspan="8"></td></tr><tr><td>Date</td><td>Weight</td><td>Temp.</td><td>BP</td><td>Pulse</td><td>Pulse OX</td><td>Pain</td><td>Initials</td></tr><tr><td>12/01/25</td><td>185</td><td>98.6</td><td>132/84</td><td>78</td><td>98</td><td>8</td><td>EC</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr