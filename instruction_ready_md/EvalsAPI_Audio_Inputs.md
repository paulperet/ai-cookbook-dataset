# Evaluating Audio Models with the Evals API: A Step-by-Step Guide

This guide demonstrates how to use OpenAI's Evals framework to evaluate model performance on audio-based tasks. You will learn to:
1. Prepare an audio dataset for evaluation.
2. Configure an evaluation (Eval) with a custom grader.
3. Run the evaluation (Run) to sample model responses and grade them.
4. Retrieve and analyze the results.

The workflow uses **sampling** to generate model responses from audio inputs and **model grading** to score those responses against a reference answer—all while keeping the data in its native audio format.

## Prerequisites

Ensure you have the following installed and configured:

```bash
pip install openai datasets pandas soundfile torch torchcodec pydub jiwer --quiet
```

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Step 1: Import Required Libraries

```python
from datasets import load_dataset, Audio
from openai import OpenAI
import base64
import os
import json
import time
import io
import soundfile as sf
import numpy as np
import pandas as pd
```

## Step 2: Load and Prepare the Audio Dataset

We'll use the `big_bench_audio` dataset from Hugging Face, which contains audio clips describing logic problems along with their correct answers.

```python
# Load the dataset and decode the audio column
dataset = load_dataset("ArtificialAnalysis/big_bench_audio")
dataset = dataset.cast_column("audio", Audio(decode=True))
```

### Helper: Convert Audio to Base64

Audio inputs must be base64-encoded strings. The following helper function handles various audio representations (file paths, arrays, etc.) and converts them to the required format.

```python
def audio_to_base64(audio_val) -> str:
    """
    Converts an audio input to a base64-encoded WAV string.
    Supports:
        - File paths (dict with 'path' key)
        - Decoded audio dicts ('array' and 'sampling_rate')
        - Raw bytes
    """
    # Try to extract a file path first
    try:
        path = None
        if isinstance(audio_val, dict) and "path" in audio_val:
            path = audio_val["path"]
        else:
            try:
                path = audio_val["path"]
            except Exception:
                path = getattr(audio_val, "path", None)
        if isinstance(path, str) and os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        pass

    # Fallback: Use array and sampling rate to create an in-memory WAV
    try:
        array = None
        sampling_rate = None
        try:
            array = audio_val["array"]
            sampling_rate = audio_val["sampling_rate"]
        except Exception:
            array = getattr(audio_val, "array", None)
            sampling_rate = getattr(audio_val, "sampling_rate", None)
        if array is not None and sampling_rate is not None:
            audio_np = np.array(array)
            buf = io.BytesIO()
            sf.write(buf, audio_np, int(samplingpling_rate), format="WAV")
            return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        pass

    if isinstance(audio_val, (bytes, bytearray)):
        return base64.b64encode(audio_val).decode("ascii")

    raise ValueError("Unsupported audio value; could not convert to base64")
```

### Prepare the Data Source

Now, process the first few examples from the dataset into the format required by the Evals API.

```python
evals_data_source = []

# Use the first 3 examples for a quick test
for example in dataset["train"].select(range(3)):
    audio_val = example["audio"]
    try:
        audio_base64 = audio_to_base64(audio_val)
    except Exception as e:
        print(f"Warning: could not encode audio for id={example['id']}: {e}")
        audio_base64 = None
    evals_data_source.append({
        "item": {
            "id": example["id"],
            "category": example["category"],
            "official_answer": example["official_answer"],
            "audio_base64": audio_base64
        }
    })
```

Each item in `evals_data_source` should look like this:

```json
{
  "item": {
    "id": 0,
    "category": "formal_fallacies",
    "official_answer": "invalid",
    "audio_base64": "UklGRjrODwBXQVZFZm10IBAAAAABAAEAIlYAAESsA..."
  }
}
```

## Step 3: Initialize the OpenAI Client

```python
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

## Step 4: Upload the Data Source File

Because audio data is large, we must save it to a file and upload it via the API.

```python
# Save examples to a JSONL file
file_name = "evals_data_source.json"
with open(file_name, "w", encoding="utf-8") as f:
    for obj in evals_data_source:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# Upload the file
file = client.files.create(
    file=open(file_name, "rb"),
    purpose="evals"
)
```

## Step 5: Configure the Evaluation (Eval)

An Eval defines the data structure and the grading criteria.

### Data Source Configuration

This schema describes the structure of each data item.

```python
data_source_config = {
    "type": "custom",
    "item_schema": {
        "type": "object",
        "properties": {
            "id": { "type": "integer" },
            "category": { "type": "string" },
            "official_answer": { "type": "string" },
            "audio_base64": { "type": "string" }
        },
        "required": ["id", "category", "official_answer", "audio_base64"]
    },
    "include_sample_schema": True,  # Enables sampling
}
```

### Grader Configuration

We'll use a `score_model` grader that evaluates the model's audio response against the official answer. The grader uses `gpt-audio` to assess whether the audio clip reaches the same conclusion.

```python
grader_config = {
    "type": "score_model",
    "name": "Reference answer audio model grader",
    "model": "gpt-audio",
    "input": [
        {
            "role": "system",
            "content": 'You are a helpful assistant that evaluates audio clips to judge whether they match a provided reference answer. The audio clip is the model\'s response to the question. Respond ONLY with a single JSON object matching: {"steps":[{"description":"string","conclusion":"string"}],"result":number}. Do not include any extra text. result must be a float in [0.0, 1.0].'
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Evaluate this audio clip to see if it reaches the same conclusion as the reference answer. Reference answer: {{item.official_answer}}",
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "{{ sample.output_audio.data }}",
                        "format": "wav",
                    },
                },
            ],
        },
    ],
    "range": [0, 1],
    "pass_threshold": 0.6,
}
```

> **Alternative Grader**: You could also use a `string_check` grader to compare the text transcript of the model's audio response against the reference answer. This is useful if you want to evaluate the textual content directly.

### Create the Eval Object

```python
eval_object = client.evals.create(
    name="Audio Grading Cookbook",
    data_source_config=data_source_config,
    testing_criteria=[grader_config],
)
```

## Step 6: Define the Sampling Prompt

This prompt instructs the model on how to respond to the audio input.

```python
sampling_messages = [
    {
        "role": "system",
        "content": "You are a helpful and obedient assistant that can answer questions with audio input. You will be given an audio input containing a question to answer."
    },
    {
        "role": "user",
        "type": "message",
        "content": {
            "type": "input_text",
            "text": "Answer the following question by replying with brief reasoning statements and a conclusion with a single word answer: 'valid' or 'invalid'."
        }
    },
    {
        "role": "user",
        "type": "message",
        "content": {
            "type": "input_audio",
            "input_audio": {
                "data": "{{ item.audio_base64 }}",
                "format": "wav"
            }
        }
    }
]
```

## Step 7: Run the Evaluation

Now, create an Eval Run to sample model responses and grade them.

```python
eval_run = client.evals.runs.create(
    name="Audio Input Eval Run",
    eval_id=eval_object.id,
    data_source={
        "type": "completions",  # Use the Completions API for audio inputs
        "source": {
            "type": "file_id",
            "id": file.id
        },
        "model": "gpt-audio",  # Ensure your model supports audio inputs
        "sampling_params": {
            "temperature": 0.0,
        },
        "input_messages": {
            "type": "template",
            "template": sampling_messages
        },
        "modalities": ["audio", "text"],  # Request both audio and text outputs
    },
)
```

## Step 8: Poll for Results and Display

The run may take some time. Poll its status until it completes, then retrieve and display the results.

```python
while True:
    run = client.evals.runs.retrieve(run_id=eval_run.id, eval_id=eval_object.id)
    if run.status == "completed":
        output_items = list(client.evals.runs.output_items.list(
            run_id=run.id, eval_id=eval_object.id
        ))
        # Create a summary DataFrame
        df = pd.DataFrame({
            "id": [item.datasource_item["id"] for item in output_items],
            "category": [item.datasource_item["category"] for item in output_items],
            "official_answer": [item.datasource_item["official_answer"] for item in output_items],
            "model_response": [item.sample.output[0].content for item in output_items],
            "grading_results": ["passed" if item.results[0]["passed"] else "failed"
                                for item in output_items]
        })
        print(df)
        break
    if run.status == "failed":
        print(run.error)
        break
    time.sleep(5)
```

### Inspect a Full Output Item

To see the complete structure of a single output item (including the sampled audio and grading details), you can examine the first item:

```python
first_item = output_items[0]
print(json.dumps(dict(first_item), indent=2, default=str))
```

## Conclusion

You have successfully run an evaluation of an audio model using the OpenAI Evals API. This workflow allows you to assess model performance directly on audio inputs and outputs, providing a more accurate evaluation for voice-based applications.

### Next Steps
- **Adapt to Your Use Case**: Modify the dataset, prompts, and grader to fit your specific audio evaluation needs.
- **Handle Large Audio Files**: For audio clips up to 8 GB, consider using the [Uploads API](https://platform.openai.com/docs/api-reference/uploads/create).
- **Visualize Results**: Visit the [Evals dashboard](https://platform.openai.com/evaluations) to explore detailed insights and performance metrics from your runs.