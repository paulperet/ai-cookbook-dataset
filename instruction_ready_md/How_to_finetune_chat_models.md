# Fine-Tuning GPT-4o Mini for Entity Extraction: A Step-by-Step Guide

Fine-tuning allows you to train a model on a large, domain-specific dataset, enabling it to perform specialized tasks more effectively than few-shot prompting. This guide walks you through fine-tuning the `gpt-4o-mini-2024-07-18` model to extract generic ingredients from recipes using the RecipeNLG dataset.

**Note:** GPT-4o mini fine-tuning is available to developers in [Tier 4 and 5 usage tiers](https://platform.openai.com/docs/guides/rate-limits/usage-tiers).

## Prerequisites

Before you begin, ensure you have the OpenAI Python package installed and your API key configured.

```bash
pip install --upgrade --quiet openai
```

```python
import json
import openai
import os
import pandas as pd
from pprint import pprint

# Initialize the OpenAI client
client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    organization="<org id>",  # Replace with your organization ID
    project="<project id>",   # Replace with your project ID
)
```

## Step 1: Load and Explore the Dataset

Fine-tuning performs best on a focused domain. We'll use a curated subset of the RecipeNLG dataset containing recipes from `cookbooks.com`.

```python
# Load the dataset
recipe_df = pd.read_csv("data/cookbook_recipes_nlg_10k.csv")
recipe_df.head()
```

## Step 2: Prepare Training Data in Chat Format

Fine-tuning for chat models requires data in a conversational format. Each training example is a list of messages with `system`, `user`, and `assistant` roles.

First, define helper functions to structure the data.

```python
system_message = "You are a helpful recipe assistant. You are to extract the generic ingredients from each of the recipes provided."

def create_user_message(row):
    return f"Title: {row['title']}\n\nIngredients: {row['ingredients']}\n\nGeneric ingredients: "

def prepare_example_conversation(row):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": create_user_message(row)},
            {"role": "assistant", "content": row["NER"]},
        ]
    }

# Preview a single example
pprint(prepare_example_conversation(recipe_df.iloc[0]))
```

**Important:** Each training example must be under 4096 tokens. Longer examples will be truncated.

## Step 3: Create Training and Validation Sets

Start with a small, high-quality dataset. Performance typically scales linearly with more data.

```python
# Use the first 100 rows for training
training_df = recipe_df.loc[0:100]
training_data = training_df.apply(prepare_example_conversation, axis=1).tolist()

# Optionally, create a validation set to monitor overfitting
validation_df = recipe_df.loc[101:200]
validation_data = validation_df.apply(prepare_example_conversation, axis=1).tolist()

# Inspect the first few training examples
for example in training_data[:5]:
    print(example)
```

## Step 4: Save Data as JSONL Files

The fine-tuning API requires data in JSONL format (one JSON object per line).

```python
def write_jsonl(data_list: list, filename: str) -> None:
    with open(filename, "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)

# Save the datasets
training_file_name = "tmp_recipe_finetune_training.jsonl"
validation_file_name = "tmp_recipe_finetune_validation.jsonl"

write_jsonl(training_data, training_file_name)
write_jsonl(validation_data, validation_file_name)

# Verify the file structure
!head -n 5 tmp_recipe_finetune_training.jsonl
```

## Step 5: Upload Files to the OpenAI API

Upload your training and validation files to make them accessible for fine-tuning.

```python
def upload_file(file_name: str, purpose: str) -> str:
    with open(file_name, "rb") as file_fd:
        response = client.files.create(file=file_fd, purpose=purpose)
    return response.id

training_file_id = upload_file(training_file_name, "fine-tune")
validation_file_id = upload_file(validation_file_name, "fine-tune")

print("Training file ID:", training_file_id)
print("Validation file ID:", validation_file_id)
```

## Step 6: Create a Fine-Tuning Job

Initiate the fine-tuning job using the uploaded files. You can add a custom suffix to identify your model.

```python
MODEL = "gpt-4o-mini-2024-07-18"

response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model=MODEL,
    suffix="recipe-ner",
)

job_id = response.id
print("Job ID:", job_id)
print("Status:", response.status)
```

**Note:** Files must be processed by OpenAI's system first. If you encounter a "File not ready" error, wait a few minutes and retry.

## Step 7: Monitor the Fine-Tuning Job

Check the job status and monitor progress through training events.

```python
# Retrieve job details
response = client.fine_tuning.jobs.retrieve(job_id)
print("Job ID:", response.id)
print("Status:", response.status)
print("Trained Tokens:", response.trained_tokens)

# List job events to track progress
response = client.fine_tuning.jobs.list_events(job_id)
events = response.data
events.reverse()

for event in events:
    print(event.message)
```

Once the status changes to `succeeded`, retrieve your fine-tuned model ID.

```python
response = client.fine_tuning.jobs.retrieve(job_id)
fine_tuned_model_id = response.fine_tuned_model

if fine_tuned_model_id is None:
    raise RuntimeError(
        "Fine-tuned model ID not found. Your job has likely not been completed yet."
    )

print("Fine-tuned model ID:", fine_tuned_model_id)
```

## Step 8: Perform Inference with Your Fine-Tuned Model

Use your new model just like any other chat model via the `ChatCompletions` endpoint.

```python
# Prepare a test example
test_df = recipe_df.loc[201:300]
test_row = test_df.iloc[0]

test_messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": create_user_message(test_row)}
]

pprint(test_messages)

# Generate a completion
response = client.chat.completions.create(
    model=fine_tuned_model_id,
    messages=test_messages,
    temperature=0,
    max_tokens=500
)

print(response.choices[0].message.content)
```

## Conclusion

You have successfully fine-tuned a GPT-4o mini model for a specialized entity extraction task. You can now apply this process to your own datasets and use cases. For more information, refer to the [Fine-tuning documentation](https://platform.openai.com/docs/guides/fine-tuning) and [API reference](https://platform.openai.com/docs/api-reference/fine-tuning).