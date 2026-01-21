# Fine-Tuning Mistral Models: A Step-by-Step Guide

This guide walks you through the process of fine-tuning a Mistral model using their official API. You'll learn how to prepare a dataset, upload it, create a fine-tuning job, and monitor its progress.

## Prerequisites

Before you begin, ensure you have the following:

1.  **A Mistral API Key:** Sign up at [Mistral AI](https://mistral.ai/) to obtain your API key. Set it as an environment variable named `MISTRAL_API_KEY`.
2.  **Python Environment:** This tutorial uses Python. We'll install the necessary packages in the first step.

## Step 1: Install Required Libraries

Start by installing the `mistralai` client library and `pandas` for data handling.

```bash
pip install mistralai pandas
```

## Step 2: Prepare Your Dataset

We'll use a subset of the `ultrachat_200k` dataset for this example. The process involves loading the data, splitting it into training and validation sets, and saving it in the required JSONL format.

```python
import pandas as pd

# Load a sample of the dataset
df = pd.read_parquet('https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/resolve/main/data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet')

# Split the data (99.5% for training, 0.5% for evaluation)
df_train = df.sample(frac=0.995, random_state=200)
df_eval = df.drop(df_train.index)

# Save the splits as JSONL files
df_train.to_json("ultrachat_chunk_train.jsonl", orient="records", lines=True)
df_eval.to_json("ultrachat_chunk_eval.jsonl", orient="records", lines=True)
```

## Step 3: Reformat the Dataset for Mistral API

The raw JSONL files may not be in the exact format required by the Mistral API. To fix this, we'll use a validation and reformatting script provided by Mistral.

First, download the script:

```bash
wget https://raw.githubusercontent.com/mistralai/mistral-finetune/main/utils/reformat_data.py
```

Next, run the script on both your training and evaluation files. This will validate the structure and ensure compatibility.

```bash
# Reformat the training data
python reformat_data.py ultrachat_chunk_train.jsonl

# Reformat the evaluation data
python reformat_data.py ultrachat_chunk_eval.jsonl
```

The script may skip some samples that don't meet formatting criteria (e.g., missing an assistant message). You can inspect a skipped sample like this:

```python
# Example: Check the structure of a skipped training sample
print(df_train.iloc[3674]['messages'])
```

## Step 4: Upload the Dataset to Mistral

Now, initialize the Mistral client and upload your prepared files. The API will return a file object containing a unique ID for each uploaded dataset, which you'll need for the next step.

```python
from mistralai import Mistral
import os

# Initialize the client with your API key
api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

# Upload the training file
ultrachat_chunk_train = client.files.upload(
    file={
        "file_name": "ultrachat_chunk_train.jsonl",
        "content": open("ultrachat_chunk_train.jsonl", "rb"),
    }
)

# Upload the evaluation file
ultrachat_chunk_eval = client.files.upload(
    file={
        "file_name": "ultrachat_chunk_eval.jsonl",
        "content": open("ultrachat_chunk_eval.jsonl", "rb"),
    }
)

# Helper function to print API responses neatly
import json
def pprint(obj):
    print(json.dumps(obj.dict(), indent=4))

# View the uploaded file details
print("Training File Info:")
pprint(ultrachat_chunk_train)

print("\nEvaluation File Info:")
pprint(ultrachat_chunk_eval)
```

Your output will show the unique IDs for each file, along with metadata like file size and line count.

## Step 5: Create a Fine-Tuning Job

With your datasets uploaded, you can now create a fine-tuning job. Specify the base model, the uploaded file IDs, and your desired hyperparameters.

```python
# Create the fine-tuning job
created_job = client.fine_tuning.jobs.create(
    model="open-mistral-7b",  # The base model to fine-tune
    training_files=[{"file_id": ultrachat_chunk_train.id, "weight": 1}],
    validation_files=[ultrachat_chunk_eval.id],
    hyperparameters={
        "training_steps": 10,     # A small number for this example
        "learning_rate": 0.0001
    },
    auto_start=True  # Start the job immediately after creation
)

print("Job Created Successfully:")
pprint(created_job)
```

The response will include a job ID and show an initial status of `QUEUED`.

## Step 6: Monitor the Job Status

You can check the status of your job in several ways: list all jobs, retrieve a specific job, or poll until completion.

**List all fine-tuning jobs:**
```python
jobs = client.fine_tuning.jobs.list()
print(jobs)
```

**Retrieve your specific job by ID:**
```python
retrieved_job = client.fine_tuning.jobs.get(job_id=created_job.id)
print(retrieved_job.status)  # e.g., 'QUEUED', 'RUNNING', 'SUCCESS'
```

**Poll until the job is complete:**
For a long-running job, you can set up a simple loop to wait for completion.

```python
import time

retrieved_job = client.fine_tuning.jobs.get(job_id=created_job.id)
while retrieved_job.status in ["RUNNING", "QUEUED"]:
    retrieved_job = client.fine_tuning.jobs.get(job_id=created_job.id)
    print(f"Job is {retrieved_job.status}, waiting 10 seconds...")
    time.sleep(10)

print(f"Job finished with status: {retrieved_job.status}")
if retrieved_job.status == "SUCCESS":
    print(f"Your fine-tuned model is: {retrieved_job.fine_tuned_model}")
```

Once the status changes to `SUCCESS`, the response will contain the name of your new fine-tuned model (e.g., `ft:open-mistral-7b:...`). You can now use this model name with the standard Mistral chat completions API.

## Summary

You have successfully:
1.  Prepared and formatted a conversation dataset.
2.  Uploaded it to the Mistral platform.
3.  Created and launched a fine-tuning job.
4.  Monitored the job to completion and obtained your custom model ID.

Your fine-tuned model is now ready for inference. Refer to the [Mistral API documentation](https://docs.mistral.ai/api/#operation/createChatCompletion) for details on how to call your new model.