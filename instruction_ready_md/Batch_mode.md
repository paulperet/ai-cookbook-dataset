# Guide: Using the Gemini Batch API for High-Throughput AI Tasks

The [Gemini Batch API](https://ai.google.dev/gemini-api/docs/batch-mode) is designed for processing large volumes of non-latency-critical requests asynchronously. It's ideal for workloads that require high throughput, such as pre-processing datasets, running large-scale evaluations, or generating content in bulk.

**Key Benefits:**
*   **High throughput:** Process millions of requests in a single job.
*   **Cost savings:** Batches are priced at a 50% discount compared to the standard API.
*   **Asynchronous:** Submit your job and retrieve the results later, within a 24-hour SLO.

**In this guide, you will learn how to:**
1.  Set up your environment for Batch Mode.
2.  Create a batch job by uploading a JSONL file (recommended for large jobs).
3.  Create a batch job using inline requests (convenient for smaller jobs).
4.  Monitor the status of your job.
5.  Retrieve and parse the results for both job types.
6.  Manage your jobs (list and cancel).
7.  Use batch embeddings.
8.  Generate with multimodal inputs.
9.  Generate images.

---

## Prerequisites & Setup

### 1. Install the SDK
First, install the Google Generative AI SDK from PyPI.

```bash
pip install -q -U "google-genai>=1.34.0"
```

### 2. Set Up Your API Key
Your API key must be available in your environment. This example shows how to retrieve it from a Colab secret, but you can use any secure method (e.g., environment variables).

```python
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

### 3. Initialize the SDK Client
Create a client instance with your API key.

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)
```

### 4. Choose a Model
Most Gemini models are compatible with Batch mode. The queue size varies per model, so refer to the [documentation](https://ai.google.dev/gemini-api/docs/batch-mode#technical-details) for details.

```python
MODEL_ID = "gemini-3-flash-preview" # Example model
```

> **Note:** Media generation models (Imagen, Lyria, Veo) are not currently compatible with the Batch API. However, you can batch-create images using the Nano-Banana model (see example later in this guide).

---

## Creating Batch Jobs: Two Methods

You can create batch jobs in two ways:

1.  **File-based (`src`)**: Upload a JSONL file containing all your requests. This is the recommended method for large datasets.
2.  **Inline (`inlined_requests`)**: Pass a list of request objects directly in your code. This is convenient for smaller, dynamically generated jobs.

---

## Method 1: Create a Job from a File

This is the most common workflow for large-scale tasks. You will prepare an input file, upload it, create the job, monitor it, and retrieve the results.

### Step 1: Prepare and Upload the Input File
The input file must be a **JSONL** file, where each line is a JSON object. Each object must contain a unique `key` to help you correlate inputs with outputs, and a `request` object matching the `GenerateContentRequest` schema.

```python
import json

# Create sample request data
requests_data = [
    {"key": "request_1", "request": {"contents": [{"parts": [{"text": "Explain how AI works in a few words"}]}]}},
    {"key": "request_2", "request": {"contents": [{"parts": [{"text": "Explain how quantum computing works in a few words"}]}]}}
]

# Write data to a JSONL file
json_file_path = 'batch_requests.json'
with open(json_file_path, 'w') as f:
    for req in requests_data:
        f.write(json.dumps(req) + '\n')

# Upload the JSONL file to the File API
print(f"Uploading file: {json_file_path}")
uploaded_batch_requests = client.files.upload(
    file=json_file_path,
    config=types.UploadFileConfig(display_name='batch-input-file')
)
print(f"Uploaded file: {uploaded_batch_requests.name}")
```

### Step 2: Create the Batch Job
Now, pass the uploaded file's resource name to the `client.batches.create` function to create the batch job.

```python
batch_job_from_file = client.batches.create(
    model=MODEL_ID,
    src=uploaded_batch_requests.name,
    config={
        'display_name': 'my-batch-job-from-file',
    }
)
print(f"Created batch job from file: {batch_job_from_file.name}")
```

### Step 3: Monitor Job Status
Jobs can take time to complete (up to 24 hours). You can poll the API to check the status.

```python
import time

job_name = batch_job_from_file.name
print(f"Polling status for job: {job_name}")

# Poll the job status until it's completed.
while True:
    batch_job = client.batches.get(name=job_name)
    if batch_job.state.name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'):
        break
    print(f"Job not finished. Current state: {batch_job.state.name}. Waiting 30 seconds...")
    time.sleep(30)

print(f"Job finished with state: {batch_job.state.name}")
if batch_job.state.name == 'JOB_STATE_FAILED':
    print(f"Error: {batch_job.error}")
```

### Step 4: Retrieve and Parse Results
Once a file-based job succeeds, the results are written to an output file in the Files API.

```python
if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
    # The output is in another file.
    result_file_name = batch_job.dest.file_name
    print(f"Results are in file: {result_file_name}")

    print("\nDownloading and parsing result file content...")
    file_content_bytes = client.files.download(file=result_file_name)
    file_content = file_content_bytes.decode('utf-8')

    # The result file is also a JSONL file. Parse and print each line.
    for line in file_content.splitlines():
      if line:
        parsed_response = json.loads(line)
        # Pretty-print the JSON for readability
        print(json.dumps(parsed_response, indent=2))
        print("-" * 20)
else:
    print(f"Job did not succeed. Final state: {batch_job.state.name}")
```

---

## Method 2: Create a Job with Inline Requests

For smaller tasks, you can pass requests directly without creating a file. The results will be returned directly in the job object itself.

### Step 1: Create and Monitor the Inline Job

```python
# Define your list of requests.
# Note: Unlike the file-based method, a 'key' is not required for inline requests,
# as the order of responses will match the order of requests.
inline_requests_list = [
    {'contents': [{'parts': [{'text': 'Write a short poem about a cloud.'}]}]},
    {'contents': [{'parts': [{'text': 'Write a short poem about a cat.'}]}]}
]

# Create the batch job with the inline requests.
print("Creating inline batch job...")
batch_job_inline = client.batches.create(
    model=MODEL_ID,
    src=inline_requests_list,
    config={'display_name': 'my-batch-job-inline-example'}
)
print(f"Created inline batch job: {batch_job_inline.name}")
print("-" * 20)

# Monitor the job until completion.
job_name = batch_job_inline.name
print(f"Polling status for job: {job_name}")

while True:
    batch_job_inline = client.batches.get(name=job_name)
    if batch_job_inline.state.name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'):
        break
    print(f"Job not finished. Current state: {batch_job_inline.state.name}. Waiting 30 seconds...")
    time.sleep(30)

print(f"Job finished with state: {batch_job_inline.state.name}")
if batch_job_inline.state.name == 'JOB_STATE_FAILED':
    print(f"Error: {batch_job_inline.error}")
```

### Step 2: Retrieve and Print Inline Results
Once the job has succeeded, the results are available in the `inlined_responses` field of the job object.

```python
if batch_job_inline.state.name == 'JOB_STATE_SUCCEEDED':
    print("\nResults are inline:")
    # The results are in the `inlined_responses` field.
    for i, inline_response in enumerate(batch_job_inline.dest.inlined_responses):
        print(f"\n--- Response {i+1} ---")

        # Check for a successful response
        if inline_response.response:
            # The .text property is a shortcut to the generated text.
            try:
                print(inline_response.response.text)
            except AttributeError:
                # Fallback to printing the full response if .text isn't available
                print(inline_response.response)

        # Check for an error in this specific request
        elif inline_response.error:
            print(f"Error: {inline_response.error}")

else:
    print(f"Job did not succeed. Final state: {batch_job_inline.state.name}")
    if batch_job_inline.error:
        print(f"Error: {batch_job_inline.error}")
```

---

## Managing Jobs

Here are some common operations for managing your batch jobs.

### List Your Batch Jobs

```python
print("Listing recent batch jobs:\n")

# Note: The list API currently doesn't return inlined_responses.
# As a workaround, you can make a `get` call for inline jobs to see their results.
batches = client.batches.list(config={'page_size': 10})

for b in batches.page:
    print(f"Job Name: {b.name}")
    print(f"  - Display Name: {b.display_name}")
    print(f"  - State: {b.state.name}")
    print(f"  - Create Time: {b.create_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if it was an inline job (no destination file)
    if b.dest is not None:
      if not b.dest.file_name:
        full_job = client.batches.get(name=b.name)
        if full_job.inlined_responses:
            print("  - Type: Inline ({} responses)".format(len(full_job.inlined_responses)))
      else:
          print(f"  - Type: File-based (Output: {b.dest.file_name})")

    print("-" * 20)
```

### Cancel a Batch Job
You can cancel a job that is still pending or running.

```python
job_to_cancel = 'batches/YOUR_JOB_NAME_HERE'
try:
    client.batches.cancel(name=job_to_cancel)
    print(f"Cancel request sent for job: {job_to_cancel}")
except Exception as e:
    print(f"Failed to cancel job: {e}")
```

---

## Advanced Use Cases

### Batch Embeddings
You can use the Batch API to generate embeddings for multiple text inputs efficiently.

```python
# Prepare embedding requests in a JSONL file
embedding_requests = [
    {"key": "doc_1", "request": {"contents": [{"parts": [{"text": "The theory of relativity"}]}]}},
    {"key": "doc_2", "request": {"contents": [{"parts": [{"text": "Machine learning algorithms"}]}]}}
]

# Write to file, upload, and create batch job (similar to text generation example)
# Use an embedding model like `text-embedding-004`
```

### Generate with Multimodal Inputs
Batch jobs support multimodal inputs (text and images). Ensure your JSONL `request` objects include the appropriate `parts`.

```python
# Example structure for a request with an image (requires a File API URI for the image)
multimodal_request = {
    "key": "multimodal_1",
    "request": {
        "contents": [{
            "parts": [
                {"text": "Describe this image."},
                {"file_data": {"file_uri": "files/YOUR_IMAGE_FILE_ID", "mime_type": "image/jpeg"}}
            ]
        }]
    }
}
```

### Generate Images in Batch
While media gen models aren't supported, you can use the `nano-banana` model for batch image generation.

```python
image_gen_requests = [
    {"key": "img_1", "request": {"contents": [{"parts": [{"text": "A serene mountain landscape at sunrise"}]}]}},
    {"key": "img_2", "request": {"contents": [{"parts": [{"text": "A futuristic city with flying cars"}]}]}}
]

# Create batch job with the nano-banana model
image_batch_job = client.batches.create(
    model="nano-banana",
    src=uploaded_image_requests_file.name,  # Assuming file upload
    config={'display_name': 'my-batch-image-gen-job'}
)
```

---

## Summary

You've learned how to leverage the Gemini Batch API for high-throughput, cost-effective asynchronous processing. Key takeaways:

1.  **File-based jobs** are ideal for large, static datasets.
2.  **Inline jobs** offer convenience for smaller, dynamic tasks.
3.  Always **monitor job status** and handle potential errors.
4.  The API supports **advanced workflows** like embeddings, multimodal inputs, and batch image generation.

For more details, refer to the official [Gemini Batch API documentation](https://ai.google.dev/gemini-api/docs/batch-mode).