# Guide: OCR at Scale with Mistral's Batch API

This guide demonstrates how to perform Optical Character Recognition (OCR) on a large set of images using Mistral AI. You will learn two methods: a standard loop and a more efficient, cost-effective batch processing approach.

## Prerequisites

Ensure you have the necessary libraries installed and your API key ready.

### 1. Install Required Packages

```bash
pip install mistralai datasets
```

### 2. Import Libraries and Initialize Client

```python
from mistralai import Mistral
import base64
from io import BytesIO
from PIL import Image
from datasets import load_dataset
from tqdm import tqdm
import json
import time

# Initialize the Mistral client
api_key = "YOUR_API_KEY"  # Replace with your key from https://console.mistral.ai/api-keys/
client = Mistral(api_key=api_key)
ocr_model = "mistral-ocr-latest"
```

## Method 1: Standard OCR Processing (Without Batch)

This method processes images one by one in a loop. It's straightforward but less efficient for large volumes.

### Step 1: Prepare an Image Encoding Function

Mistral's Vision API requires images to be base64-encoded. This function handles the conversion.

```python
def encode_image_data(image_data):
    """
    Encode image data (bytes or PIL Image) to a base64 string.
    """
    try:
        if isinstance(image_data, bytes):
            return base64.b64encode(image_data).decode('utf-8')
        else:
            buffered = BytesIO()
            image_data.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None
```

### Step 2: Load a Sample Dataset

We'll use the `HuggingFaceM4/DocumentVQA` dataset for demonstration, loading 100 samples.

```python
n_samples = 100
dataset = load_dataset("HuggingFaceM4/DocumentVQA", split="train", streaming=True)
subset = list(dataset.take(n_samples))
```

### Step 3: Process Each Image and Extract Text

Loop through each image, call the OCR API, and collect the results.

```python
ocr_dataset = []

for sample in tqdm(subset, desc="Processing images"):
    image_data = sample['image']
    base64_image = encode_image_data(image_data)

    # Construct the data URL for the API
    image_url = f"data:image/jpeg;base64,{base64_image}"

    # Call the Mistral OCR API
    response = client.ocr.process(
        model=ocr_model,
        document={
            "type": "image_url",
            "image_url": image_url,
        }
    )

    # Store the result
    ocr_dataset.append({
        'image': base64_image,
        'ocr_content': response.pages[0].markdown  # Single-page image
    })
```

### Step 4: Save the Results

Export the extracted text and image data to a JSON file.

```python
with open('ocr_dataset.json', 'w') as f:
    json.dump(ocr_dataset, f, indent=4)

print("OCR processing complete. Results saved to 'ocr_dataset.json'.")
```

## Method 2: Batch OCR Processing

Batch Inference allows you to process hundreds of images in a single job at a reduced cost. This method is ideal for production-scale workloads.

### Step 1: Prepare Image URLs for Batch Processing

First, encode all images and prepare their data URLs.

```python
image_urls = []

for sample in tqdm(subset, desc="Preparing image URLs"):
    image_data = sample['image']
    base64_image = encode_image_data(image_data)
    image_url = f"data:image/jpeg;base64,{base64_image}"
    image_urls.append(image_url)
```

### Step 2: Create a Batch Request File

The Batch API requires a JSONL file with a specific format. This function creates it.

```python
def create_batch_file(image_urls, output_file):
    """
    Create a JSONL file for batch OCR processing.
    """
    with open(output_file, 'w') as file:
        for index, url in enumerate(image_urls):
            entry = {
                "custom_id": str(index),
                "body": {
                    "document": {
                        "type": "image_url",
                        "image_url": url
                    },
                    "include_image_base64": True
                }
            }
            file.write(json.dumps(entry) + '\n')

batch_file = "batch_file.jsonl"
create_batch_file(image_urls, batch_file)
print(f"Batch file created: {batch_file}")
```

### Step 3: Upload the Batch File

Upload the JSONL file to Mistral's files endpoint.

```python
batch_data = client.files.upload(
    file={
        "file_name": batch_file,
        "content": open(batch_file, "rb")
    },
    purpose="batch"
)
print(f"File uploaded with ID: {batch_data.id}")
```

### Step 4: Create and Monitor a Batch Job

Initiate the batch job and monitor its progress until completion.

```python
# Create the batch job
created_job = client.batch.jobs.create(
    input_files=[batch_data.id],
    model=ocr_model,
    endpoint="/v1/ocr",
    metadata={"job_type": "testing"}
)
print(f"Batch job created with ID: {created_job.id}")

# Monitor job status
while True:
    retrieved_job = client.batch.jobs.get(job_id=created_job.id)

    print(f"Status: {retrieved_job.status}")
    print(f"Progress: {retrieved_job.succeeded_requests + retrieved_job.failed_requests} / {retrieved_job.total_requests}")

    if retrieved_job.status in ["SUCCESS", "FAILED"]:
        break
    time.sleep(5)  # Poll every 5 seconds
```

### Step 5: Download the Results

Once the job succeeds, download the output file containing all OCR results.

```python
if retrieved_job.status == "SUCCESS":
    result = client.files.download(file_id=retrieved_job.output_file)
    # Process the result content as needed
    print("Batch job completed successfully. Results are ready for download.")
else:
    print("Batch job failed.")
```

## Summary

You have successfully performed OCR on a dataset using two methods:
1. **Standard Processing:** Simple loop suitable for small, immediate tasks.
2. **Batch Inference:** Efficient, cost-effective method for processing large volumes of images asynchronously.

Batch Inference reduces costs by approximately 50% and is the recommended approach for production-scale OCR workloads. For more details, refer to the [Mistral Vision Documentation](https://docs.mistral.ai/capabilities/vision/).