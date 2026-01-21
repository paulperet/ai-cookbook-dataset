# Extracting Structured Data with Mistral OCR

This guide demonstrates how to use Mistral's OCR capabilities to extract text and structure from PDFs and images. While this method is functional, note that the newer [Annotations feature](https://github.com/mistralai/cookbook/blob/main/mistral/ocr/data_extraction.ipynb) is recommended for more robust structured data extraction.

**Models Used:** `mistral-ocr-latest`, `pixtral-12b-2409`, `ministral-8b-latest`

## Prerequisites

First, install the required library and download the sample files.

```bash
pip install mistralai
```

```bash
wget https://raw.githubusercontent.com/mistralai/cookbook/refs/heads/main/mistral/ocr/mistral7b.pdf
wget https://raw.githubusercontent.com/mistralai/cookbook/refs/heads/main/mistral/ocr/receipt.png
```

## Step 1: Initialize the Mistral Client

You will need an API key from the [Mistral Platform](https://console.mistral.ai/api-keys/). Initialize the client with your key.

```python
from mistralai import Mistral

api_key = "YOUR_API_KEY"  # Replace with your actual API key
client = Mistral(api_key=api_key)
```

## Step 2: Process a PDF File with OCR

Mistral OCR can process both PDFs and image files. Let's start by uploading and processing a PDF.

```python
from pathlib import Path
from mistralai import DocumentURLChunk
import json

# Verify the downloaded PDF exists
pdf_file = Path("mistral7b.pdf")
assert pdf_file.is_file()

# 1. Upload the PDF file to Mistral's servers
uploaded_file = client.files.upload(
    file={
        "file_name": pdf_file.stem,
        "content": pdf_file.read_bytes(),
    },
    purpose="ocr",
)

# 2. Get a signed URL to access the uploaded file
signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

# 3. Process the PDF with the OCR model
pdf_response = client.ocr.process(
    document=DocumentURLChunk(document_url=signed_url.url),
    model="mistral-ocr-latest",
    include_image_base64=True  # Include any embedded images
)

# Inspect the first part of the raw JSON response
response_dict = json.loads(pdf_response.model_dump_json())
print(json.dumps(response_dict, indent=4)[:1000])
```

## Step 3: Render the Extracted Markdown

The OCR response contains the extracted text and images formatted as Markdown. The following helper functions combine this data into a single, viewable document.

```python
from mistralai.models import OCRResponse
from IPython.display import Markdown, display

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    """
    Replace image placeholders in markdown with base64-encoded images.
    """
    for img_name, base64_str in images_dict.items():
        markdown_str = markdown_str.replace(
            f"![{img_name}]({img_name})", f"![{img_name}]({base64_str})"
        )
    return markdown_str

def get_combined_markdown(ocr_response: OCRResponse) -> str:
    """
    Combine OCR text and images from all pages into a single markdown document.
    """
    markdowns: list[str] = []
    for page in ocr_response.pages:
        image_data = {}
        for img in page.images:
            image_data[img.id] = img.image_base64
        markdowns.append(replace_images_in_markdown(page.markdown, image_data))
    return "\n\n".join(markdowns)

# Display the fully rendered document
display(Markdown(get_combined_markdown(pdf_response)))
```

**Expected Output:**
The displayed output will be the full text of the "Mistral 7B" research paper, including formatted headers, the abstract, and any figures that were embedded as base64 images.

## Step 4: Process an Image File with OCR

The process for an image file is very similar. Instead of a `DocumentURLChunk`, you use an `ImageURLChunk`.

```python
from mistralai import ImageURLChunk
from pathlib import Path

# Verify the downloaded image exists
image_file = Path("receipt.png")
assert image_file.is_file()

# 1. Upload the image
uploaded_image = client.files.upload(
    file={
        "file_name": image_file.stem,
        "content": image_file.read_bytes(),
    },
    purpose="ocr",
)

# 2. Get a signed URL
signed_url = client.files.get_signed_url(file_id=uploaded_image.id, expiry=1)

# 3. Process the image with OCR
image_response = client.ocr.process(
    document=ImageURLChunk(image_url=signed_url.url),
    model="mistral-ocr-latest",
)

# Display the extracted text from the receipt
print(get_combined_markdown(image_response))
```

## Step 5: Feed OCR Output to a Language Model for Structuring

The raw OCR text can be messy. You can use a vision-capable language model (like Pixtral or Ministral) to parse it into a clean, structured format (e.g., JSON). This creates a powerful pipeline: OCR for accurate text extraction, followed by an LLM for understanding and structuring.

In this example, we'll structure the receipt data.

```python
from mistralai import UserMessage

# Use the markdown text extracted from the receipt in the previous step
receipt_text = get_combined_markdown(image_response)

# Create a prompt that asks the model to structure the receipt data
prompt = f"""
Please analyze the following receipt text and extract the key information into a JSON object.
Use the following structure:
{{
    "vendor": "<name of the store>",
    "date": "<date of transaction>",
    "total_amount": <total amount as a float>,
    "items": [
        {{"name": "<item name>", "quantity": <quantity>, "price": <price as float>}}
    ]
}}

Receipt Text:
{receipt_text}
"""

# Call a vision model (Pixtral) to process the prompt
chat_response = client.chat.complete(
    model="pixtral-12b-2409",
    messages=[UserMessage(content=prompt)],
)

# Print the structured output
print(chat_response.choices[0].message.content)
```

**Expected Output (Example):**
```json
{
    "vendor": "Coffee Corner",
    "date": "2024-01-15",
    "total_amount": 12.75,
    "items": [
        {"name": "Latte", "quantity": 2, "price": 5.50},
        {"name": "Croissant", "quantity": 1, "price": 1.75}
    ]
}
```

## Summary

You have successfully built a pipeline to:
1.  Extract raw text and images from PDFs and images using Mistral OCR.
2.  Render the extracted content for verification.
3.  Use a powerful language model to transform the raw OCR text into clean, structured JSON data.

This combination is particularly useful when you need high-accuracy text extraction from documents followed by sophisticated interpretation—such as parsing invoices, receipts, or forms. For more advanced and streamlined structured extraction, consider using the dedicated [Annotations API](https://github.com/mistralai/cookbook/blob/main/mistral/ocr/data_extraction.ipynb).