# Fine-Tuning Mistral 7B as an LLM Hallucination Judge with Weights & Biases

This guide walks you through fine-tuning a Mistral 7B model to act as a specialized "judge" for detecting factual inconsistencies and hallucinations in text. You'll learn how to trace API calls with W&B Weave, evaluate model performance, and improve results using MistralAI's fine-tuning capabilities.

## Prerequisites

Ensure you have the required libraries installed and your API keys set as environment variables.

```bash
pip install "mistralai==0.4.2" "weave==0.50.7"
```

```python
import os
import json
from pathlib import Path

import weave
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Initialize clients
client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
weave.init("llm-judge-webinar")
```

## 1. Load and Prepare the Dataset

We'll use the Factual Inconsistency Benchmark (FIB) dataset, which contains document-summary pairs labeled for consistency.

```python
DATA_PATH = Path("./data")
NUM_SAMPLES = 100  # Use None for all samples

def read_jsonl(path):
    """Read a JSONL file into a list of dictionaries."""
    with open(path, 'r') as file:
        return [json.loads(line) for line in file]

# Load training and validation splits
train_ds = read_jsonl(DATA_PATH / "fib-train.jsonl")
val_ds = read_jsonl(DATA_PATH / "fib-val.jsonl")[0:NUM_SAMPLES]
```

## 2. Create a Traceable Mistral API Call

Wrap your Mistral API calls with `@weave.op()` to automatically log inputs, outputs, and performance.

```python
@weave.op()
def call_mistral(model: str, messages: list, **kwargs) -> str:
    """Call the Mistral API and return parsed JSON."""
    chat_response = client.chat(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        **kwargs,
    )
    return json.loads(chat_response.choices[0].message.content)
```

## 3. Define the Prompt and Model Interface

Create a prompt that instructs the model to detect inconsistencies and a `MistralModel` class to standardize interactions.

```python
prompt = """You are an expert to detect factual inconsistencies and hallucinations. You will be given a document and a summary.
- Carefully read the full document and the provided summary.
- Identify Factual Inconsistencies: any statements in the summary that are not supported by or contradict the information in the document.
Factually Inconsistent: If any statement in the summary is not supported by or contradicts the document, label it as 0
Factually Consistent: If all statements in the summary are supported by the document, label it as 1

Highlight or list the specific statements in the summary that are inconsistent.
Provide a brief explanation of why each highlighted statement is inconsistent with the document.

Return in JSON format with `consistency` and a `reason` for the given choice.

Document: 
{premise}
Summary: 
{hypothesis}
"""

def format_prompt(prompt, premise: str, hypothesis: str, cls=ChatMessage):
    """Format the prompt into a chat message."""
    messages = [
        cls(
            role="user", 
            content=prompt.format(premise=premise, hypothesis=hypothesis)
        )
    ]
    return messages

class MistralModel(weave.Model):
    """A Weave-wrapped model for consistent evaluation."""
    model: str
    prompt: str
    temperature: float = 0.7
    
    @weave.op
    def create_messages(self, premise:str, hypothesis:str):
        return format_prompt(self.prompt, premise, hypothesis)

    @weave.op
    def predict(self, premise:str, hypothesis:str):
        messages = self.create_messages(premise, hypothesis)
        return call_mistral(model=self.model, messages=messages, temperature=self.temperature)
```

## 4. Evaluate the Base Model

First, test the base `open-mistral-7b` model on a single example.

```python
# Test on a single example
premise, hypothesis, target = train_ds[1]['premise'], train_ds[1]['hypothesis'], train_ds[1]['target']
model_7b = MistralModel(model="open-mistral-7b", prompt=prompt, temperature=0.7)
output = model_7b.predict(premise, hypothesis)
print(f"Model output: {output}")
print(f"Target label: {target}")
```

### 4.1 Define Evaluation Metrics

Create scorers to measure accuracy and F1 score across the validation set.

```python
def accuracy(model_output, target):
    """Simple accuracy scorer."""
    class_model_output = model_output.get('consistency') if model_output else None
    return {"accuracy": class_model_output == target}

class BinaryMetrics(weave.Scorer):
    """Compute F1, precision, and recall for binary classification."""
    class_name: str
    eps: float = 1e-8

    @weave.op()
    def summarize(self, score_rows) -> dict:
        # Filter out None rows (model errors)
        score_rows = [score for score in score_rows if score["correct"] is not None]
        # Compute metrics
        tp = sum([not score["negative"] and score["correct"] for score in score_rows])
        fp = sum([not score["negative"] and not score["correct"] for score in score_rows])
        fn = sum([score["negative"] and not score["correct"] for score in score_rows])
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        return {"f1": f1, "precision": precision, "recall": recall}

    @weave.op()
    def score(self, target: dict, model_output: dict) -> dict:
        class_model_output = model_output.get(self.class_name) if model_output else None
        return {
            "correct": class_model_output == target,
            "negative": not class_model_output,
        }

F1 = BinaryMetrics(class_name="consistency")
evaluation = weave.Evaluation(dataset=val_ds, scorers=[accuracy, F1])
```

### 4.2 Run the Evaluation

```python
await evaluation.evaluate(model_7b)
```

## 5. Improve the Prompt with an Example

Incorporate an example from the source blog post to provide clearer context.

```python
prompt_example = """You are an expert to detect factual inconsistencies and hallucinations. You will be given a document and a summary.
- Carefully read the full document and the provided summary.
- Identify Factual Inconsistencies: any statements in the summary that are not supported by or contradict the information in the document.
Factually Inconsistent: If any statement in the summary is not supported by or contradicts the document, label it as 0
Factually Consistent: If all statements in the summary are supported by the document, label it as 1

Here you have an example:

Document: 
Vehicles and pedestrians will now embark and disembark the Cowes ferry separately following Maritime and Coastguard Agency (MCA) guidance. 
Isle of Wight Council said its new procedures were in response to a resident’s complaint. Councillor Shirley Smart said it would 
“initially result in a slower service”. Originally passengers and vehicles boarded or disembarked the so-called “floating bridge” at the same time. 
Ms Smart, who is the executive member for economy and tourism, said the council already had measures in place to control how passengers 
and vehicles left or embarked the chain ferry “in a safe manner”. However, it was “responding” to the MCA’s recommendations “following this 
complaint”. She added: “This may initially result in a slower service while the measures are introduced and our customers get used to 
the changes.” The service has been in operation since 1859.

Inconsistent summary: A new service on the Isle of Wight’s chain ferry has been launched following a complaint from a resident.

Consistent summary: Passengers using a chain ferry have been warned crossing times will be longer because of new safety measures.

Highlight or list the specific statements in the summary that are inconsistent.
Provide a brief explanation of why each highlighted statement is inconsistent with the document.

Return in JSON format with `consistency` and a `reason` for the given choice.

Document: 
{premise}
Summary: 
{hypothesis}
"""

model_7b_ex = MistralModel(model="open-mistral-7b", prompt=prompt_example, temperature=0.7)
await evaluation.evaluate(model_7b_ex)
```

### 5.1 Compare with a Larger Model

Evaluate `mistral-large-latest` to establish a performance ceiling.

```python
model_large = MistralModel(model="mistral-large-latest", prompt=prompt_example, temperature=0.7)
await evaluation.evaluate(model_large)
```

## 6. Fine-Tune Mistral 7B

Fine-tuning can significantly improve performance on this specialized task.

### 6.1 Prepare the Fine-Tuning Dataset

Reformat the dataset into the structure required by MistralAI's fine-tuning API.

```python
ft_prompt = """You are an expert to detect factual inconsistencies and hallucinations. You will be given a document and a summary.
- Carefully read the full document and the provided summary.
- Identify Factual Inconsistencies: any statements in the summary that are not supported by or contradict the information in the document.
Factually Inconsistent: If any statement in the summary is not supported by or contradicts the document, label it as 0
Factually Consistent: If all statements in the summary are supported by the document, label it as 1

Return in JSON format with `consistency` for the given choice.

Document: 
{premise}
Summary: 
{hypothesis}
"""

answer = '{{"consistency": {label}}}'  # JSON schema for assistant response

def format_prompt_ft(row, cls=dict, with_answer=True):
    """Format a row for fine-tuning."""
    premise = row['premise']
    hypothesis = row['hypothesis']
    messages = [
        cls(
            role="user", 
            content=ft_prompt.format(premise=premise, hypothesis=hypothesis)
        )
    ]
    if with_answer:
        label = row['target']
        messages.append(
            cls(
                role="assistant",
                content=answer.format(label=label)
            )
        )
    return messages

# Format datasets
formatted_train_ds = [format_prompt_ft(row) for row in train_ds]
formatted_val_ds = [format_prompt_ft(row) for row in val_ds]

def save_jsonl(ds, path):
    """Save a list of dictionaries as a JSONL file."""
    with open(path, "w") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")

save_jsonl(formatted_train_ds, DATA_PATH/"formatted_fib_train.jsonl")
save_jsonl(formatted_val_ds, DATA_PATH/"formatted_fib_val.jsonl")
```

### 6.2 Upload Datasets to MistralAI

```python
with open(DATA_PATH/"formatted_fib_train.jsonl", "rb") as f:
    ds_train = client.files.create(file=("formatted_df_train.jsonl", f))
with open(DATA_PATH/"formatted_fib_val.jsonl", "rb") as f:
    ds_eval = client.files.create(file=("eval.jsonl", f))
```

### 6.3 Launch the Fine-Tuning Job

Create a fine-tuning job with Weights & Biases integration for tracking.

```python
from mistralai.models.jobs import TrainingParameters, WandbIntegrationIn

created_jobs = client.jobs.create(
    model="open-mistral-7b",
    training_files=[ds_train.id],
    validation_files=[ds_eval.id],
    hyperparameters=TrainingParameters(
        training_steps=35,      # Approximately 10 epochs for this dataset
        learning_rate=0.0001,
    ),
    integrations=[
        WandbIntegrationIn(
            project="llm-judge-webinar",
            run_name="mistral_7b_fib",
            api_key=os.environ.get("WANDB_API_KEY"),
        ).dict()
    ],
)

# Monitor job status
import time
retrieved_job = client.jobs.retrieve(created_jobs.id)
while retrieved_job.status in ["RUNNING", "QUEUED"]:
    retrieved_job = client.jobs.retrieve(created_jobs.id)
    print(f"Job is {retrieved_job.status}, waiting 10 seconds")
    time.sleep(10)
```

### 6.4 Evaluate the Fine-Tuned Model

```python
# Retrieve the fine-tuned model name
jobs = client.jobs.list()
retrieved_job = jobs.data[0]
fine_tuned_model_name = retrieved_job.fine_tuned_model

# Evaluate
mistral_7b_ft = MistralModel(prompt=ft_prompt, model=fine_tuned_model_name)
await evaluation.evaluate(mistral_7b_ft)
```

## 7. Further Improvement with Additional Data

Incorporate the Unified Summarization Benchmark (USB) dataset to improve generalization.

```python
# Load and format USB dataset
train_ds_usb = read_jsonl(DATA_PATH / "usb-train.jsonl")
formatted_train_usb_ds = [format_prompt_ft(row) for row in train_ds_usb]
save_jsonl(formatted_train_usb_ds, DATA_PATH/"formatted_train_usb.jsonl")

# Upload
with open(DATA_PATH/"formatted_train_usb.jsonl", "rb") as f:
    ds_train_usb = client.files.create(file=("formatted_df_train_usb.jsonl", f))

# Create a new fine-tuning job with combined data
created_jobs = client.jobs.create(
    model="open-mistral-7b",
    training_files=[ds_train.id, ds_train_usb.id],
    validation_files=[ds_eval.id],
    hyperparameters=TrainingParameters(
        training_steps=200,
        learning_rate=0.0001,
    ),
    integrations=[
        WandbIntegrationIn(
            project="llm-judge-webinar",
            run_name="mistral_7b_fib_usb",
            api_key=os.environ.get("WANDB_API_KEY"),
        ).dict()
    ],
)

# Evaluate the new model
jobs = client.jobs.list()
created_jobs = jobs.data[1]
fine_tuned_model_name = created_jobs.fine_tuned_model
mistral_7b_usb_ft = MistralModel(prompt=ft_prompt, model=fine_tuned_model_name)
await evaluation.evaluate(mistral_7b_usb_ft)
```

## Summary

You've successfully:
1. Set up tracing for Mistral API calls with W&B Weave
2. Evaluated base and large models on a factual consistency task
3. Fine-tuned Mistral 7B using the FIB dataset
4. Further improved performance by incorporating additional training data from USB

The fine-tuned 7B model now provides a cost-effective alternative to larger models while maintaining strong performance on hallucination detection.