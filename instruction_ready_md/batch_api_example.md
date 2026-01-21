# Batch Processing Receipts with Mistral AI: A Step-by-Step Guide

This guide demonstrates how to use the Mistral Batch API to extract structured information from multiple receipt images. You will learn to process images in bulk, parse the results, and compile them into a clean pandas DataFrame.

## Prerequisites & Setup

First, install the required libraries and set up your environment.

```bash
pip install mistralai datasets
```

Now, import the necessary modules.

```python
import os
import json
import base64
import pandas as pd
from io import BytesIO
from PIL.Image import Image
from datasets import load_dataset
from mistralai import Mistral
```

Set your Mistral API key as an environment variable. Replace `"your-api-key-here"` with your actual key.

```python
# Set your API key (replace with your actual key)
os.environ["MISTRAL_API_KEY"] = "your-api-key-here"

# Initialize the Mistral client
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)
```

## Step 1: Load the Receipt Image Dataset

We'll use a public dataset of supermarket receipts from Hugging Face.

```python
# Load the dataset and convert it to a pandas DataFrame
dataset_name = 'shirastromer/supermarket-receipts'
dataset = load_dataset(dataset_name)
df = pd.DataFrame(dataset['train'])

# Display the first few rows to understand the structure
print(df.head())
```

The DataFrame contains columns for the image, a receipt ID, and pre-extracted text. We will focus on the `image` column.

## Step 2: Prepare a Helper Function for Image Processing

The Mistral API requires images to be base64-encoded. Let's create a utility function to handle this conversion.

```python
def format_image(image: Image) -> str:
    """
    Converts a PIL Image to a base64-encoded string with a data URI prefix.

    Args:
        image (Image): The PIL Image object.

    Returns:
        str: The formatted base64 string.
    """
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    # Add the prefix for base64 format
    formatted_base64 = f"data:image/jpeg;base64,{image_base64}"
    return formatted_base64
```

## Step 3: Test with a Single Image

Before processing in bulk, let's verify our setup works by extracting data from one receipt.

```python
# Define the prompt for the model
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": format_image(df.image[1])
            },
            {
                "type": "text",
                "text": "Extract the name and price of each item on the receipt, categorize each item into one of the following categories: 'Medical', 'Food', 'Beverage', 'Travel', or 'Other', and return the results as a well-structured JSON object. The JSON object should include only the fields: name, price, and classification for each item."
            }
        ]
    },
    {"role": "assistant", "content": "{", "prefix": True},
]

# Call the Mistral Chat API
chat_response = client.chat.complete(
    model="pixtral-large-latest",
    messages=messages,
    response_format={"type": "json_object"}
)

# Print the structured response
print(chat_response.choices[0].message.content)
```

The output should be a JSON object listing items with their names, prices, and classifications.

## Step 4: Prepare Batch Requests

Now, we'll scale up. We'll create a batch of requests for multiple images. Let's process the first 10 receipts as an example.

```python
num_samples = 10
list_of_json = []

for idx in range(num_samples):
    request = {
        "custom_id": str(idx),
        "body": {
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": format_image(df.image[idx])
                        },
                        {
                            "type": "text",
                            "text": "Identify the name and price of each item on the receipt, categorize each item into one of the following categories: 'Medical', 'Food', 'Beverage', 'Travel', or 'Other', and return the results as a well-structured JSON object. The JSON object should include only the fields: name, price, and classification for each item."
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": "{",
                    "prefix": True
                }
            ],
            "response_format": {"type": "json_object"}
        }
    }
    # Convert the request to a JSON string and encode it
    list_of_json.append(json.dumps(request).encode("utf-8"))
```

## Step 5: Upload the Batch File

The Batch API requires the requests to be uploaded as a `.jsonl` file.

```python
# Upload the batch file
batch_data = client.files.upload(
    file={
        "file_name": "file.jsonl",
        "content": b"\n".join(list_of_json)
    },
    purpose="batch"
)

print(f"Uploaded file ID: {batch_data.id}")
```

## Step 6: Create and Monitor a Batch Job

Create a job to process the uploaded file.

```python
# Create a batch job
created_job = client.batch.jobs.create(
    input_files=[batch_data.id],
    model="pixtral-large-latest",
    endpoint="/v1/chat/completions",
    metadata={"job_type": "testing"}
)

print(f"Created job ID: {created_job.id}")
print(f"Initial job status: {created_job.status}")
```

Monitor the job's progress. The status will change from `QUEUED` to `SUCCESS` (or `FAILED`).

```python
# Retrieve the job details
retrieved_job = client.batch.jobs.get(job_id=created_job.id)

print(f"Job Status: {retrieved_job.status}")
print(f"Total requests: {retrieved_job.total_requests}")
print(f"Successful requests: {retrieved_job.succeeded_requests}")
print(f"Failed requests: {retrieved_job.failed_requests}")
```

## Step 7: Download and Parse the Results

Once the job is complete, download the output file.

```python
# Download the results file
output = client.files.download(file_id=retrieved_job.output_file).read().decode("utf-8").strip()
```

The output is a JSON Lines format. Let's parse it and extract the relevant data.

```python
# Parse the JSON lines output
lines = output.strip().split('\n')
extracted_data = []

for line in lines:
    parsed_line = json.loads(line)
    custom_id = parsed_line.get("custom_id")
    response = parsed_line.get("response", {})
    body = response.get("body", {})
    choices = body.get("choices", [])

    for choice in choices:
        message_content = choice.get("message", {}).get("content", "")
        # The content is a JSON string; parse it to get the items
        try:
            # Clean the string (remove potential markdown code fences)
            clean_content = message_content.strip('`')
            items_data = json.loads(clean_content)
            # Handle different possible structures
            items = items_data if isinstance(items_data, list) else items_data.get("items", [])
            for item in items:
                extracted_data.append({
                    "receipt_id": custom_id,
                    "item_name": item.get("name"),
                    "price": item.get("price"),
                    "category": item.get("classification")
                })
        except json.JSONDecodeError:
            # Skip lines that cannot be parsed
            continue

# Create the final DataFrame
df_output = pd.DataFrame(extracted_data)
```

## Step 8: Review the Final Data

Display the cleaned and structured results.

```python
print(df_output)
```

Your `df_output` DataFrame will contain columns for `receipt_id`, `item_name`, `price`, and `category`, providing a clean, queryable dataset extracted from the batch of receipt images.

## Summary

You have successfully:
1.  Loaded a dataset of receipt images.
2.  Processed a single image to test the API.
3.  Prepared and uploaded a batch of requests.
4.  Executed a batch job using the Mistral Batch API.
5.  Downloaded, parsed, and structured the results into a pandas DataFrame.

This workflow is ideal for efficiently processing large volumes of documents or images where structured data extraction is required.