# Fine-tune ChatGPT-3.5 and GPT-4 with Weights & Biases

This guide walks you through fine-tuning an OpenAI model (ChatGPT-3.5 or GPT-4) and using the Weights & Biases (W&B) integration to track your experiments, models, and datasets. You'll learn how to prepare a dataset, create a fine-tuned model, log the training run to W&B, and evaluate the results.

**Prerequisites:**
*   An [OpenAI API key](https://platform.openai.com/account/api-keys).
*   A free [Weights & Biases account](https://wandb.ai).

## 1. Setup and Installation

Begin by installing the required Python libraries.

```bash
!pip install -Uq openai tiktoken datasets tenacity wandb
```

> **Note:** At the time of writing, a specific update to the `openai` Python library is required for full W&B integration support. Run the following command if you encounter any issues with the `openai wandb sync` command later in the tutorial.
> ```bash
> !pip uninstall -y openai -qq && pip install git+https://github.com/morganmcg1/openai-python.git@update_wandb_logger -qqq
> ```

## 2. Prepare Your Dataset

In this tutorial, we'll fine-tune a model on a legal reasoning task. We'll use the **Contract NLI Explicit Identification** dataset from [LegalBench](https://hazyresearch.stanford.edu/legalbench/).

### 2.1. Import Libraries and Initialize W&B

```python
import openai
import wandb
import os
import json
import random
import tiktoken
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_fixed

# Set your W&B project name
WANDB_PROJECT = "OpenAI-Fine-Tune"
```

### 2.2. Configure Your OpenAI API Key

```python
# Replace with your actual API key
openai_key = "YOUR_API_KEY"
openai.api_key = openai_key
```

### 2.3. Load and Format the Data

We'll download the dataset, merge its splits, and shuffle it.

```python
from datasets import load_dataset

# Download and merge the dataset
dataset = load_dataset("nguha/legalbench", "contract_nli_explicit_identification")
data = []
for d in dataset["train"]:
    data.append(d)
for d in dataset["test"]:
    data.append(d)

random.shuffle(data)
for idx, d in enumerate(data):
    d["new_index"] = idx

print(f"Total samples: {len(data)}")
print("Sample data:", data[0])
```

Next, we format the data for OpenAI's chat completion models. We adapt the original task prompt into a zero-shot instruction.

```python
base_prompt_zero_shot = "Identify if the clause provides that all Confidential Information shall be expressly identified by the Disclosing Party. Answer with only `Yes` or `No`"

# Split data into training and test sets
n_train = 30
n_test = len(data) - n_train

train_messages = []
test_messages = []

for d in data:
    prompts = [
        {"role": "system", "content": base_prompt_zero_shot},
        {"role": "user", "content": d["text"]},
        {"role": "assistant", "content": d["answer"]}
    ]
    if int(d["new_index"]) < n_train:
        train_messages.append({'messages': prompts})
    else:
        test_messages.append({'messages': prompts})

print(f"Training samples: {len(train_messages)}")
print(f"Test samples: {len(test_messages)}")
```

### 2.4. Save and Validate Data Files

Save the formatted messages to JSONL files, which is the required format for OpenAI fine-tuning.

```python
# Save training data
train_file_path = 'encoded_train_data.jsonl'
with open(train_file_path, 'w') as file:
    for item in train_messages:
        file.write(json.dumps(item) + '\n')

# Save test data
test_file_path = 'encoded_test_data.jsonl'
with open(test_file_path, 'w') as file:
    for item in test_messages:
        file.write(json.dumps(item) + '\n')
```

It's crucial to validate your data's format. The following function, adapted from OpenAI's documentation, checks for common errors and calculates token counts.

```python
def openai_validate_data(dataset_path):
    # Load dataset
    with open(dataset_path) as f:
        dataset = [json.loads(line) for line in f]

    print("Num examples:", len(dataset))
    print("First example:")
    for message in dataset[0]["messages"]:
        print(message)

    # Format error checks
    format_errors = defaultdict(int)
    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            if any(k not in ("role", "content", "name") for k in message):
                format_errors["message_unrecognized_key"] += 1
            if message.get("role", None) not in ("system", "user", "assistant"):
                format_errors["unrecognized_role"] += 1
            content = message.get("content", None)
            if not content or not isinstance(content, str):
                format_errors["missing_content"] += 1
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")

    # Token counting and analysis
    encoding = tiktoken.get_encoding("cl100k_base")
    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens
    def num_assistant_tokens_from_messages(messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
        return num_tokens

    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []
    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

    print("\nDataset Analysis:")
    print("Num examples missing system message:", n_missing_system)
    print("Num examples missing user message:", n_missing_user)
    n_too_long = sum(l > 4096 for l in convo_lens)
    print(f"\n{n_too_long} examples may be over the 4096 token limit and will be truncated.")

# Validate the training data
openai_validate_data(train_file_path)
```

### 2.5. Log Data to Weights & Biases Artifacts

W&B Artifacts provide versioned, centralized storage for your datasets.

```python
# Start a W&B run to log the data
wandb.init(
    project=WANDB_PROJECT,
    job_type="log-data",
    config={'n_train': n_train, 'n_valid': n_test}
)

# Log the training and test files as artifacts
wandb.log_artifact(
    train_file_path,
    name="legalbench-contract_nli_explicit_identification-train",
    type="train-data"
)
wandb.log_artifact(
    test_file_path,
    name="legalbench-contract_nli_explicit_identification-test",
    type="test-data"
)

# Save your W&B entity (username/team name) for later reference
entity = wandb.run.entity
wandb.finish()
```

## 3. Create a Fine-Tuned Model

Now, we'll use the OpenAI API to fine-tune a `gpt-3.5-turbo` model.

### 3.1. Retrieve Training Data from W&B

```python
# Start a new run for the fine-tuning job
wandb.init(project=WANDB_PROJECT, job_type="finetune")

# Retrieve the latest version of the training artifact
artifact_train = wandb.use_artifact(
    f'{entity}/{WANDB_PROJECT}/legalbench-contract_nli_explicit_identification-train:latest',
    type='train-data'
)
train_file = artifact_train.get_path(train_file_path).download("my_data")
print(f"Training file downloaded to: {train_file}")
```

### 3.2. Upload Data to OpenAI

OpenAI needs to process your file before training can begin.

```python
openai_train_file_info = openai.File.create(
    file=open(train_file, "rb"),
    purpose='fine-tune'
)
print("File uploaded. Info:", openai_train_file_info)
# Note: You may need to wait a few minutes for processing.
```

### 3.3. Launch the Fine-Tuning Job

Define your hyperparameters and start the job.

```python
model = 'gpt-3.5-turbo'
n_epochs = 3

openai_ft_job_info = openai.FineTuningJob.create(
    training_file=openai_train_file_info["id"],
    model=model,
    hyperparameters={"n_epochs": n_epochs}
)

ft_job_id = openai_ft_job_info["id"]
print("Fine-tuning job created:", openai_ft_job_info)
```

> **Training Time:** This job will take several minutes to complete. OpenAI will send an email notification when it finishes.

You can check the job's status at any time:

```python
state = openai.FineTuningJob.retrieve(ft_job_id)
print("Status:", state["status"])
print("Trained Tokens:", state.get("trained_tokens"))
print("Finished At:", state.get("finished_at"))
print("Fine-tuned Model ID:", state.get("fine_tuned_model"))
```

## 4. Log the Fine-Tune to Weights & Biases

The `openai wandb sync` command seamlessly logs your fine-tuning job's metrics and configuration to your W&B project.

```bash
# This command logs the specified job to your W&B project.
# Ensure your OpenAI API key is set as an environment variable.
!OPENAI_API_KEY={openai_key} openai wandb sync --id {ft_job_id} --project {WANDB_PROJECT}
```

After syncing, finish the current W&B run.

```python
wandb.finish()
```

Your fine-tuning experiment—including hyperparameters, training loss curves, and the resulting model name—is now visible in your W&B dashboard.

## 5. Evaluate the Fine-Tuned Model

Let's evaluate the model's performance on the held-out test set and compare it to the base model.

### 5.1. Load Test Data and Set Up Evaluation

```python
# Start a new run for evaluation
wandb.init(project=WANDB_PROJECT, job_type='eval')

# Retrieve the test dataset artifact
artifact_valid = wandb.use_artifact(
    f'{entity}/{WANDB_PROJECT}/legalbench-contract_nli_explicit_identification-test:latest',
    type='test-data'
)
test_file = artifact_valid.get_path(test_file_path).download("my_data")

with open(test_file) as f:
    test_dataset = [json.loads(line) for line in f]

print(f"Loaded {len(test_dataset)} test examples.")
wandb.config.update({"num_test_samples": len(test_dataset)})
```

### 5.2. Define a Helper Function for API Calls

We'll add retry logic to handle potential API rate limits or transient errors.

```python
@retry(stop=stop_after_attempt(3), wait=wait_fixed(60))
def call_openai(messages="", model="gpt-3.5-turbo"):
    return openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=10
    )
```

### 5.3. Get the Fine-Tuned Model ID

```python
state = openai.FineTuningJob.retrieve(ft_job_id)
ft_model_id = state["fine_tuned_model"]
print(f"Fine-tuned model ID: {ft_model_id}")
```

### 5.4. Run Inference and Log Predictions

We'll generate predictions for each test example and log them to a W&B Table for easy inspection.

```python
# Create a W&B Table to store predictions
prediction_table = wandb.Table(columns=['messages', 'completion', 'target'])
eval_data = []

for row in tqdm(test_dataset):
    # Use only the system and user messages for the prompt
    messages = row['messages'][:2]
    target = row["messages"][2]  # The assistant's answer

    # Call the fine-tuned model
    res = call_openai(model=ft_model_id, messages=messages)
    completion = res.choices[0].message.content

    eval_data.append([messages, completion, target])
    # Log the user's query, model's completion, and the true target
    prediction_table.add_data(messages[1]['content'], completion, target["content"])

# Log the table to W&B
wandb.log({'predictions': prediction_table})
```

### 5.5. Calculate and Log Accuracy

```python
correct = 0
for e in eval_data:
    if e[1].lower() == e[2]["content"].lower():
        correct += 1

accuracy = correct / len(eval_data)
print(f"Fine-tuned Model Accuracy: {accuracy:.2%}")

# Log the accuracy metric
wandb.log({"eval/accuracy": accuracy})
wandb.summary["eval/accuracy"] = accuracy
```

### 5.6. (Optional) Compare with the Baseline Model

To understand the impact of fine-tuning, run the same evaluation on the base `gpt-3.5-turbo` model.

```python
baseline_prediction_table = wandb.Table(columns=['messages', 'completion', 'target'])
baseline_eval_data = []

for row in tqdm(test_dataset):
    messages = row['messages'][:2]
    target = row["messages"][2]

    res = call_openai(model="gpt-3.5-turbo", messages=messages)
    completion = res.choices[0].message.content

    baseline_eval_data.append([messages, completion, target])
    baseline_prediction_table.add_data(messages[1]['content'], completion, target["content"])

wandb.log({'baseline_predictions': baseline_prediction_table})

# Calculate baseline accuracy
baseline_correct = 0
for e in baseline_eval_data:
    if e[1].lower() == e[2]["content"].lower():
        baseline_correct += 1

baseline_accuracy = baseline_correct / len(baseline_eval_data)
print(f"Baseline Model Accuracy: {baseline_accuracy:.2%}")
wandb.log({"eval/baseline_accuracy": baseline_accuracy})
wandb.summary["eval/baseline_accuracy"] = baseline_accuracy
```

Finally, end the evaluation run.

```python
wandb.finish()
```

## Summary

You have successfully:
1.  Prepared a dataset for OpenAI fine-tuning.
2.  Logged versioned datasets to W&B Artifacts.
3.  Created a fine-tuned `gpt-3.5-turbo` model via the OpenAI API.
4.  Synced the training metrics and configuration to W&B with a single command.
5.  Evaluated the fine-tuned model, logged its predictions, and compared its performance to a baseline.

You can now use `openai wandb sync` to track any future fine-tuning jobs automatically. Explore the results in your W&B dashboard to analyze model performance and iterate on your training data and hyperparameters.