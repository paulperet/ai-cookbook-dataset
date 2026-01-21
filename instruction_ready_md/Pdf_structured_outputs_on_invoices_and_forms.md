# Automated Invoice and Form Data Extraction with Gemini API & Pydantic

This guide demonstrates how to extract structured data from PDF documents using the Gemini API. You will learn to upload files, define output schemas with Pydantic, and reliably parse information like invoice details and form fields into structured JSON.

## Prerequisites

Before you begin, ensure you have:
1.  A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Python installed on your system.

## Step 1: Install the SDK and Set Up the Client

First, install the official `google-genai` Python SDK.

```bash
pip install "google-genai>=1"
```

Next, import the necessary modules and initialize the client with your API key. We'll use the `gemini-2.5-flash` model for its speed and free tier availability.

```python
from google import genai

# Replace with your actual API key
api_key = "YOUR_API_KEY"

# Create a client
client = genai.Client(api_key=api_key)

# Define the model
model_id = "gemini-2.5-flash"  # Options: "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-3-pro-preview"
```

## Step 2: Prepare and Upload PDF Files

Gemini models can process various file types, including PDFs, via the File API. Let's download two sample PDFs: an invoice and a handwritten form.

```bash
# Download sample files (run in terminal or use !wget in a notebook)
wget -q -O handwriting_form.pdf https://storage.googleapis.com/generativeai-downloads/data/pdf_structured_outputs/handwriting_form.pdf
wget -q -O invoice.pdf https://storage.googleapis.com/generativeai-downloads/data/pdf_structured_outputs/invoice.pdf
```

Now, upload a file using the client. The File API stores files for 48 hours and is free to use.

```python
# Upload the invoice PDF
invoice_pdf = client.files.upload(file="invoice.pdf", config={'display_name': 'invoice'})
```

You can check the token count for the uploaded file to understand context usage and estimate costs.

```python
file_size = client.models.count_tokens(model=model_id, contents=invoice_pdf)
print(f'File: {invoice_pdf.display_name} equals to {file_size.total_tokens} tokens')
# Output: File: invoice equals to 821 tokens
```

## Step 3: Understand Structured Outputs with Pydantic

Structured Outputs force the Gemini model to return data in a predefined JSON schema, ensuring consistency for your applications. You can define this schema using Python type hints, a Pydantic `BaseModel`, or a raw dictionary schema.

Let's create a simple example using Pydantic to extract person details from a text prompt.

```python
from pydantic import BaseModel, Field

# Define a Pydantic model with descriptive fields
class Topic(BaseModel):
    name: str = Field(description="The name of the topic")

class Person(BaseModel):
    first_name: str = Field(description="The first name of the person")
    last_name: str = Field(description="The last name of the person")
    age: int = Field(description="The age of the person, if not provided please return 0")
    work_topics: list[Topic] = Field(description="The fields of interest of the person, if not provided please return an empty list")

# Define the prompt
prompt = "Philipp Schmid is a Senior AI Developer Relations Engineer at Google DeepMind working on Gemini, Gemma with the mission to help every developer to build and benefit from AI in a responsible way."

# Generate a structured response
response = client.models.generate_content(
    model=model_id,
    contents=prompt,
    config={
        'response_mime_type': 'application/json',
        'response_schema': Person
    }
)

# Print the raw JSON response
print(response.text)

# The SDK automatically parses the response into the Pydantic model
philipp: Person = response.parsed
print(f"First name is {philipp.first_name}")
```

The model will return a perfectly formatted JSON object matching our `Person` schema.

## Step 4: Extract Structured Data from PDFs

Now, let's combine file uploads and structured outputs. We'll create a helper function that takes a file path and a Pydantic model, then returns the extracted data.

```python
def extract_structured_data(file_path: str, model: BaseModel):
    # 1. Upload the file
    file = client.files.upload(
        file=file_path,
        config={'display_name': file_path.split('/')[-1].split('.')[0]}
    )
    # 2. Generate a structured response
    prompt = f"Extract the structured data from the following PDF file"
    response = client.models.generate_content(
        model=model_id,
        contents=[prompt, file],
        config={
            'response_mime_type': 'application/json',
            'response_schema': model
        }
    )
    # 3. Return the parsed Pydantic model
    return response.parsed
```

### Example 1: Extracting Invoice Data

Define a Pydantic model tailored to the invoice structure.

```python
from pydantic import BaseModel, Field

class Item(BaseModel):
    description: str = Field(description="The description of the item")
    quantity: float = Field(description="The Qty of the item")
    gross_worth: float = Field(description="The gross worth of the item")

class Invoice(BaseModel):
    """Extract the invoice number, date and all list items with description, quantity and gross worth and the total gross worth."""
    invoice_number: str = Field(description="The invoice number e.g. 1234567890")
    date: str = Field(description="The date of the invoice e.g. 2024-01-01")
    items: list[Item] = Field(description="The list of items with description, quantity and gross worth")
    total_gross_worth: float = Field(description="The total gross worth of the invoice")

# Run the extraction
result = extract_structured_data("invoice.pdf", Invoice)

print(f"Extracted Invoice: {result.invoice_number} on {result.date} with total gross worth {result.total_gross_worth}")
for item in result.items:
    print(f"Item: {item.description} with quantity {item.quantity} and gross worth {item.gross_worth}")
```

### Example 2: Extracting Data from a Handwritten Form

Define a different model for the form's structure.

```python
class Form(BaseModel):
    """Extract the form number, fiscal start date, fiscal end date, and the plan liabilities beginning of the year and end of the year."""
    form_number: str = Field(description="The Form Number")
    start_date: str = Field(description="Effective Date")
    beginning_of_year: float = Field(description="The plan liabilities beginning of the year")
    end_of_year: float = Field(description="The plan liabilities end of the year")

result = extract_structured_data("handwriting_form.pdf", Form)

print(f'Extracted Form Number: {result.form_number} with start date {result.start_date}.')
print(f'Plan liabilities beginning of the year {result.beginning_of_year} and end of the year {result.end_of_year}')
```

## Important Note on Gemini 2.0 Models

If you are using a **Gemini 2.0 model** (e.g., `gemini-2.0-flash`), you must explicitly define the property order in your JSON schema. This is not required for Gemini 2.5 or newer models.

```python
from google import genai
from google.genai import types

# Define a schema with explicit property ordering for Gemini 2.0
invoice_schema = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "invoice_number": types.Schema(type=types.Type.STRING),
        "date": types.Schema(type=types.Type.STRING),
        "vendor": types.Schema(type=types.Type.STRING),
        "total_amount": types.Schema(type=types.Type.NUMBER),
    },
    # REQUIRED for Gemini 2.0 only
    property_ordering=["invoice_number", "date", "vendor", "total_amount"],
)

response = client.models.generate_content(
    model="gemini-2.0-flash",  # Note the model version
    contents="Extract invoice details: Invoice #12345 dated 2024-01-15 from Acme Corp for $1,250.00",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=invoice_schema,
    ),
)
print(response.text)
```

## Next Steps

You've successfully built a pipeline to extract structured data from PDFs. To dive deeper, explore these resources:
*   [File API Quickstart](../quickstarts/File_API.ipynb)
*   [Prompting with Media Files](https://ai.google.dev/gemini-api/docs/file-prompting-strategies)
*   [Structured Outputs Documentation](https://ai.google.dev/gemini-api/docs/structured-output?lang=python)

For production use, consider adding validation and retry logic using libraries like [instructor](https://python.useinstructor.com/) to handle potential extraction errors gracefully.