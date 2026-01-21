# Gemini API File API: A Step-by-Step Guide

## Overview
The Gemini API supports multimodal prompting with text, images, and audio. For larger files (up to 2GB each, 20GB total per project), you must first upload them using the **File API**. This guide walks you through uploading files, referencing them in prompts, and managing your uploaded files.

Files are stored for 48 hours at no cost in all regions where the Gemini API is available.

## Prerequisites

### 1. Install the SDK
```bash
pip install -q -U "google-genai>=1.57.0"
```

### 2. Set Up Authentication
The File API uses API keys for authentication. Uploaded files are associated with your API key's cloud project.

**Important:** Your API key grants access to your uploaded data. Keep it secure. Follow Google's [API key security best practices](https://support.google.com/googleapi/answer/6310037).

```python
from google import genai
from google.colab import userdata  # For Colab environments

GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")  # Store your key as a Colab Secret
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 1: Upload a File
First, let's upload a sample image. The File API accepts various MIME types (images, audio, text files).

```python
# Download a sample image
!curl -o image.jpg "https://storage.googleapis.com/generativeai-downloads/images/jetpack.jpg"
```

Now, upload the file to the File API:

```python
sample_file = client.files.upload(file="image.jpg")
print(f"Uploaded file '{sample_file.name}' as: {sample_file.uri}")
```

The response contains a unique `name` (like `files/5djqlsrlsmw7`) and a `uri` you'll use to reference the file in API calls.

## Step 2: Verify File Upload
Retrieve the file metadata to confirm successful upload:

```python
file = client.files.get(name=sample_file.name)
print(f"Retrieved file '{file.name}' as: {sample_file.uri}")
```

## Step 3: Generate Content with the Uploaded File
Now, use the uploaded file in a prompt. Pass the file object directly to the `generate_content` method.

```python
MODEL_ID = "gemini-2.5-flash"  # Choose your model

response = client.models.generate_content(
    model=MODEL_ID,
    contents=["Describe the image with a creative description.", sample_file]
)

print(response.text)
```

## Step 4: Upload Multiple Files
You can upload multiple files in a loop. Here's an example uploading Python files from a repository:

```python
# Clone a sample repository
!git clone -q --depth 1 https://github.com/googleapis/python-genai

import pathlib

files = []
for p in pathlib.Path("python-genai").rglob('*.py'):
    if 'test' in str(p):
        continue
    f = client.files.upload(file=p, config={'display_name': str(p)})
    # Add a text separator for clarity in the prompt
    files.append(f"<<<File: {str(p)}>>>")
    files.append(f)
    print('.', end='')

# Use all uploaded files in a single prompt
response = client.models.generate_content(
    model=MODEL_ID,
    contents=["Hi, could you give me a summary of this code base?"] + files
)
print(response.text)
```

## Step 5: Use Files from Google Cloud Storage
The Gemini API can also access files stored in Google Cloud Storage (GCS). You must first register GCS objects with the File API.

### 5.1 Set Up GCS Authentication
You need credentials with **Storage Object Viewer** permissions. In Colab, configure `gcloud`:

```python
PROJECT_ID = "your-project-id"  # Replace with your GCP project ID

!gcloud config set project {PROJECT_ID}
!gcloud auth application-default login --no-launch-browser --scopes="https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/devstorage.read_only"
```

### 5.2 Register GCS Files
```python
import google.auth

credentials, project_id = google.auth.default()

# Register files from your GCS bucket
registered_gcs_files = client.files.register_files(
    auth=credentials,
    uris=["gs://your-bucket/some-file.pdf"]  # Replace with your GCS URIs
)
```

### 5.3 Use Registered Files in Prompts
```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=["What are these documents about?"] + registered_gcs_files.files
)
print(response.text)
```

## Step 6: Upload and Use Text Files
The File API also works with text files like Markdown or code. Here's an example with a Markdown file:

```python
# Download a sample Markdown file
!curl -so contrib.md https://raw.githubusercontent.com/google-gemini/cookbook/main/CONTRIBUTING.md

# Upload with explicit MIME type
md_file = client.files.upload(
    file="contrib.md",
    config={
        "display_name": "CONTRIBUTING.md",
        "mime_type": "text/markdown"
    }
)

# Use in a prompt
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        "What should I do before I start writing, when following these guidelines?",
        md_file,
    ]
)
print(response.text)
```

## Step 7: Delete Files (Optional)
Files auto-delete after 48 hours, but you can manually delete them sooner:

```python
client.files.delete(name=sample_file.name)
print(f"Deleted {sample_file.name}.")
```

## Summary
You've learned how to:
1. Upload individual and multiple files to the File API
2. Verify uploads and retrieve file metadata
3. Reference uploaded files in Gemini API prompts
4. Register and use files from Google Cloud Storage
5. Work with text files (Markdown, code)
6. Manage file lifecycle with manual deletion

The File API enables handling larger multimedia and text files in your Gemini workflows, expanding beyond the direct file upload limits of standard multimodal prompts.