# Guide: Reading PDFs with the Gemini API

This guide demonstrates how to upload a PDF file to the Gemini API and use a model to analyze its content, including extracting summaries and explaining images.

## Prerequisites

Before you begin, ensure you have the following:

1.  A Google AI API key. If you don't have one, follow the [authentication guide](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb).
2.  The `google-genai` Python library installed.

## Step 1: Install the Required Library

Start by installing the latest version of the `google-genai` client library.

```bash
pip install -Uq 'google-genai>=1.0.0'
```

## Step 2: Configure the API Client

Import the necessary modules and initialize the Gemini client using your API key. This example assumes your key is stored in an environment variable named `GOOGLE_API_KEY`.

```python
from google import genai
import os

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 3: Download a Sample PDF

For this tutorial, we'll use a sample PDF from the Google Research Blog. The following code downloads the file if it doesn't already exist locally.

```python
import pathlib

if not pathlib.Path('test.pdf').exists():
  !curl -o test.pdf https://storage.googleapis.com/generativeai-downloads/data/Smoothly%20editing%20material%20properties%20of%20objects%20with%20text-to-image%20models%20and%20synthetic%20data.pdf
```

## Step 4: Upload the PDF to the Gemini API

To process the PDF with a model, you must first upload it using the Gemini File API. This returns a file reference object.

```python
file_ref = client.files.upload(file='test.pdf')
```

## Step 5: Select a Model

Choose a Gemini model for analysis. The example below uses `gemini-3-flash-preview`, but you can select any model that supports multimodal input.

```python
MODEL_ID = "gemini-3-flash-preview"
```

## Step 6: Analyze the PDF Content

Now you can prompt the model with the uploaded PDF file. Let's start by asking for a summary.

First, check the token count for your request to ensure it fits within the model's context window.

```python
token_count = client.models.count_tokens(
    model=MODEL_ID,
    contents=[file_ref, 'Can you summarize this file as a bulleted list?']
)
print(f"Total tokens: {token_count.total_tokens}")
```

Next, generate the summary.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[file_ref, 'Can you summarize this file as a bulleted list?']
)
print(response.text)
```

**Example Output:**
> *   **Problem Addressed:** The document introduces a new method to overcome the challenges of smoothly and photorealistically editing material properties (like color, shininess, or transparency) of objects in existing images...
> *   **Proposed Solution:** Augmenting T2I models with parametric editing capabilities, leveraging synthetic data for fine-tuning...
> *   **Methodology:** A novel synthetic dataset was created using physically-based rendering of 100 3D household objects...

## Step 7: Query Specific Elements

The Gemini model can also interpret visual elements within the PDF. Ask a question about the images on the first page.

```python
response_2 = client.models.generate_content(
    model=MODEL_ID,
    contents=[file_ref, 'Can you explain the images on the first page of the document?']
)
print(response_2.text)
```

**Example Output:**
> The images on the first page of the document serve as the primary visual demonstration of the paper's core contribution...
> 1.  **Teapot Example (Top Left):** Shows a smooth, brown teapot. Prompt: "change the roughness of the teapot." Output: The teapot now appears visibly rougher...
> 2.  **Cupid Statue Example (Top Right):** Shows a solid, stone-like Cupid statue. Prompt: "change the transparency of the cupid statue." Output: The Cupid statue becomes transparent...

## Key Takeaways and Next Steps

You have successfully uploaded a PDF to the Gemini API and used a model to extract a structured summary and analyze its visual content. The File API supports various multimodal MIME types, including images, audio, and video (files under 2GB).

*   **Learn More:** Explore advanced prompting strategies with media files in the [official documentation](https://ai.google.dev/gemini-api/docs/file-prompting-strategies).
*   **Structured Data Extraction:** For a more advanced use case, see the example on [extracting structured outputs from invoices and forms](https://github.com/google-gemini/cookbook/blob/main/examples/Pdf_structured_outputs_on_invoices_and_forms.ipynb).

Remember, uploaded files are stored for 2 days and cannot be downloaded via the API.