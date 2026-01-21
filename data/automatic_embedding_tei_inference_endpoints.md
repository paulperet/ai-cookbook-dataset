# Guide: Embedding Documents with Hugging Face Inference Endpoints

## Overview
This guide demonstrates how to embed a text dataset for semantic search, QA, or RAG applications using Hugging Face Inference Endpoints. You'll deploy a high-performance embedding model, process documents in parallel, and store the results as a new dataset—all through a fully API-driven workflow.

## Prerequisites
- A Hugging Face account with a **payment method added** to your account or organization.
- Basic familiarity with Python and asynchronous programming.

## Setup

Install the required packages:

```bash
pip install aiohttp==3.8.3 datasets==2.14.6 pandas==1.5.3 requests==2.31.0 tqdm==4.66.1 huggingface-hub>=0.20
```

Import the necessary modules:

```python
import asyncio
from getpass import getpass
import json
from pathlib import Path
import time
from typing import Optional

from aiohttp import ClientSession, ClientTimeout
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import notebook_login, create_inference_endpoint, list_inference_endpoints, whoami
import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm
```

## Step 1: Configuration

Define your source dataset, target dataset, and endpoint parameters. Adjust `MAX_WORKERS` based on your model's memory requirements and `ROW_COUNT` to limit processing for testing.

```python
DATASET_IN = 'derek-thomas/dataset-creator-reddit-bestofredditorupdates'
DATASET_OUT = "processed-subset-bestofredditorupdates"
ENDPOINT_NAME = "boru-jina-embeddings-demo-ie"

MAX_WORKERS = 5  # Number of concurrent async workers
ROW_COUNT = 100  # Set to None to process all rows
```

Select your GPU instance. This example uses an AWS A10G for its balance of memory and cost.

```python
VENDOR = "aws"
REGION = "us-east-1"
INSTANCE_SIZE = "x1"
INSTANCE_TYPE = "nvidia-a10g"
```

## Step 2: Authentication and Namespace

Log in to Hugging Face and specify the namespace (your username or organization) that has the payment method.

```python
notebook_login()

who = whoami()
organization = getpass(prompt="What is your Hugging Face 🤗 username or organization? (with an added payment method)")

namespace = organization or who['name']
```

## Step 3: Load Your Dataset

Load the source dataset and convert a subset to a list of dictionaries for processing.

```python
dataset = load_dataset(DATASET_IN)
documents = dataset['train'].to_pandas().to_dict('records')[:ROW_COUNT]
print(f"Loaded {len(documents)} documents")
```

## Step 4: Create the Inference Endpoint

Deploy the `jinaai/jina-embeddings-v2-base-en` model using the Text Embeddings Inference image for optimized performance. The code attempts to create a new endpoint or reuse an existing one with the same name.

```python
try:
    endpoint = create_inference_endpoint(
        ENDPOINT_NAME,
        repository="jinaai/jina-embeddings-v2-base-en",
        revision="7302ac470bed880590f9344bfeee32ff8722d0e5",
        task="sentence-embeddings",
        framework="pytorch",
        accelerator="gpu",
        instance_size=INSTANCE_SIZE,
        instance_type=INSTANCE_TYPE,
        region=REGION,
        vendor=VENDOR,
        namespace=namespace,
        custom_image={
            "health_route": "/health",
            "env": {
                "MAX_BATCH_TOKENS": str(MAX_WORKERS * 2048),
                "MAX_CONCURRENT_REQUESTS": "512",
                "MODEL_ID": "/repository"
            },
            "url": "ghcr.io/huggingface/text-embeddings-inference:0.5.0",
        },
        type="protected",
    )
except:
    endpoint = [ie for ie in list_inference_endpoints(namespace=namespace) if ie.name == ENDPOINT_NAME][0]
    print('Loaded existing endpoint')
```

Wait for the endpoint to become active:

```python
endpoint.wait()
print("Endpoint is ready")
```

## Step 5: Test the Endpoint

Verify the endpoint works by embedding a sample sentence. The response is a byte string that needs conversion to a NumPy array.

```python
response = endpoint.client.post(
    json={
        "inputs": 'This sound track was beautiful! It paints the senery in your mind so well I would recomend it even to people who hate vid. game music!',
        'truncate': True
    },
    task="feature-extraction"
)
embedding = np.array(json.loads(response.decode()))
print(f"Sample embedding (first 20 dims): {embedding[0][:20]}")
```

Test truncation for long inputs:

```python
embedding_input = 'This input will get multiplied' * 10000
print(f'Input length: {len(embedding_input)}')
response = endpoint.client.post(
    json={"inputs": embedding_input, 'truncate': True},
    task="feature-extraction"
)
embedding = np.array(json.loads(response.decode()))
print(f"Truncated embedding (first 20 dims): {embedding[0][:20]}")
```

## Step 6: Embed All Documents Asynchronously

Define asynchronous functions to send documents to the endpoint in parallel, using a semaphore to control concurrency.

```python
async def request(document, semaphore):
    async with semaphore:
        result = await endpoint.async_client.post(
            json={"inputs": document['content'], 'truncate': True},
            task="feature-extraction"
        )
        result = np.array(json.loads(result.decode()))
        document['embedding'] = result[0]
        return document

async def main(documents):
    semaphore = asyncio.BoundedSemaphore(MAX_WORKERS)
    tasks = [request(document, semaphore) for document in documents]
    
    for f in tqdm(asyncio.as_completed(tasks), total=len(documents)):
        await f
```

Execute the embedding process and measure the time:

```python
start = time.perf_counter()
await main(documents)

# Verify embeddings
count = 0
for document in documents:
    if 'embedding' in document.keys() and len(document['embedding']) == 768:
        count += 1
print(f'Successfully embedded {count} out of {len(documents)} documents')

elapsed_time = time.perf_counter() - start
minutes, seconds = divmod(elapsed_time, 60)
print(f"Total time: {int(minutes)} min {seconds:.2f} sec")
```

## Step 7: Pause the Endpoint

Pause the endpoint to stop incurring charges while you prepare the dataset.

```python
endpoint = endpoint.pause()
print(f"Endpoint Status: {endpoint.status}")
```

## Step 8: Create and Upload the New Dataset

Convert the list of documents with embeddings into a Hugging Face Dataset and push it to the Hub.

```python
df = pd.DataFrame(documents)
dd = DatasetDict({'train': Dataset.from_pandas(df)})
dd.push_to_hub(repo_id=DATASET_OUT)

print(f'Dataset available at: https://huggingface.co/datasets/{who["name"]}/{DATASET_OUT}')
```

## Step 9: Analyze Usage and Cost

Check your endpoint dashboard to review the cost of this operation.

```python
dashboard_url = f'https://ui.endpoints.huggingface.co/{namespace}/endpoints/{ENDPOINT_NAME}'
print(f"View usage and cost at: {dashboard_url}")
```

For this example, the total cost is approximately $0.04.

## Step 10: Clean Up (Optional)

If you no longer need the endpoint, you can delete it programmatically.

```python
endpoint = endpoint.delete()
if not endpoint:
    print('Endpoint deleted successfully')
else:
    print('Delete the endpoint manually from the dashboard')
```

## Summary

You have successfully:
1. Deployed a high-performance embedding model via Inference Endpoints.
2. Processed a dataset with parallel asynchronous requests.
3. Stored the embeddings as a new dataset on the Hugging Face Hub.
4. Managed costs by pausing and optionally deleting the endpoint.

This workflow provides a scalable, repeatable method for embedding large text collections for downstream AI applications.