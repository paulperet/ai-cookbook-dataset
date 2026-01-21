# Guide: Processing Datasets with the Gemini Batch API

This guide walks you through using the Gemini Batch API to evaluate a dataset against multiple models. You'll learn how to prepare a dataset, submit batch jobs, and analyze the results.

## Prerequisites

Before you begin, ensure you have:
- A Gemini API key.
- A dataset exported in JSONL format from the [AI Studio logs dashboard](https://aistudio.google.com/logs).

## Setup

First, install the required SDK and set up authentication.

```bash
pip install -q "google-genai>=1.34.0"
```

```python
from google import genai
from google.genai import types

# Replace with your actual API key
GEMINI_API_KEY = "YOUR_API_KEY"
client = genai.Client(api_key=GEMINI_API_KEY)
```

## Step 1: Upload Your Dataset

Upload your JSONL dataset file to your working environment. If you don't have a dataset, you can use a sample.

```python
import pathlib

dataset_path = pathlib.Path("dataset.jsonl")

# Upload your file or download a sample
from google.colab import files
files.upload_file(filename=str(dataset_path))

# Alternatively, use a sample dataset:
# !wget https://storage.googleapis.com/generativeai-downloads/data/spam_dataset.jsonl -O {dataset_path}
```

## Step 2: Prepare the Dataset for Batch Processing

The Batch API requires all entries in a batch to use the same model. You must clear any existing model specifications and responses from your dataset.

```python
import json
from typing import Iterator

def with_no_model_no_response(ds_path: pathlib.Path) -> Iterator[str]:
    """Generator that yields dataset rows with model and response fields removed."""
    for row in ds_path.open():
        data = json.loads(row)
        # Remove model field to ensure homogeneity
        data['request'].pop('model', None)
        # Remove existing responses for fresh evaluation
        data.pop('response', None)
        yield json.dumps(data)

# Create a cleaned dataset file
clean_dataset = pathlib.Path('clear_dataset.jsonl')
with clean_dataset.open("w") as f:
    for line in with_no_model_no_response(dataset_path):
        print(line, file=f)
```

## Step 3: Upload the Cleaned Dataset via the File API

Upload the prepared dataset so it can be referenced by the Batch API.

```python
clean_dataset_ref = client.files.upload(
    file=clean_dataset,
    config=types.UploadFileConfig(
        display_name='eval dataset',
        mime_type='application/json'
    )
)

print(f"Uploaded file: {clean_dataset_ref.name}")
```

## Step 4: Create Batch Jobs for Each Model

Now, define the models you want to evaluate and submit a batch job for each.

```python
models_to_eval = ['gemini-2.5-flash', 'gemini-2.5-pro']
batches = []

for model in models_to_eval:
    batch_ref = client.batches.create(
        model=model,
        src=clean_dataset_ref.name,  # Use the same dataset for all batches
        config=types.CreateBatchJobConfig(
            display_name=f'Resubmit of dataset with {model}'
        )
    )
    print(f"Created batch job from file: {batch_ref.name}")
    batches.append(batch_ref)
```

## Step 5: Monitor Batch Job Completion

Batch processing can take up to 24 hours. Poll the jobs until they complete.

```python
import time

remaining = {b.name for b in batches}
while remaining:
    for batch in batches:
        if batch.name not in remaining:
            continue

        # Poll for the batch status
        batch_job = client.batches.get(name=batch.name)

        # Check job state
        if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
            print(f'COMPLETE: {batch.name}')
            remaining.remove(batch.name)
        elif batch_job.state.name in {'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'}:
            print(f'ERROR: {batch.name}')
            remaining.remove(batch.name)
        else:
            # Job is still pending
            time.sleep(30)
```

## Step 6: Download and Save the Results

Once the batches succeed, download the result files.

```python
result_files = []
for i, batch in enumerate(batches, start=1):
    batch_job = client.batches.get(name=batch.name)
    result_file_name = batch_job.dest.file_name

    # Download the result file content
    file_content_bytes = client.files.download(file=result_file_name)
    file_content = file_content_bytes.decode('utf-8')

    # Save locally
    result_file = pathlib.Path(f'results_{i}.jsonl')
    result_file.write_text(file_content)
    result_files.append(result_file)
```

## Step 7: Evaluate the Results

Define an evaluation function that matches your criteria. This example checks if the model's response contains a function call.

```python
def is_correct(row) -> bool:
    """Evaluate if a response is correct based on containing a function call."""
    try:
        return 'functionCall' in row['response']['candidates'][0]['content']['parts'][-1]
    except (KeyError, IndexError):
        return False

# Calculate scores for each model
for i, file in enumerate(result_files):
    with file.open() as f:
        file_score = 0
        total_count = 0
        for row in f:
            data = json.loads(row)
            file_score += int(is_correct(data))
            total_count += 1

        model = models_to_eval[i]
        if total_count:
            print(f'{model}: {file_score / total_count:.2%}')
        else:
            print(f'No results found for {model}')
```

**Example Output:**
```
gemini-2.5-flash: 53.33%
gemini-2.5-pro: 93.33%
```

## Next Steps

- Explore the [Logging and Datasets documentation](https://ai.google.dev/gemini-api/docs/logs-datasets) for more details.
- Check out the [Batch API cookbook](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Batch_mode.ipynb) for advanced examples.