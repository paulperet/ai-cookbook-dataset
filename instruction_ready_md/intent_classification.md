# Intent Detection: Build a Custom Classifier with Mistral AI

In this tutorial, you will learn how to build a custom intent classifier using Mistral AI's Classifier Factory. We'll walk through the entire workflow: preparing a dataset, training a classifier via the Mistral API, and running inference on new examples.

## Prerequisites

Ensure you have the following installed and configured:

```bash
pip install datasets pandas scikit-learn matplotlib tqdm mistralai
```

You will also need:
- A Mistral AI API key from the [API Keys console](https://console.mistral.ai/api-keys/).
- (Optional) A Weights & Biases API key for experiment tracking.

---

## Step 1: Prepare Your Dataset

We'll use a subset of the `mteb/amazon_massive_intent` dataset, focusing on English samples.

### 1.1 Load and Filter the Data

```python
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = load_dataset("mteb/amazon_massive_intent")

# Filter for English samples and select relevant columns
train_samples = dataset["train"].filter(lambda x: x["lang"] == "en")
train_samples = train_samples.select_columns(["text", "label_text"])

# Convert to pandas DataFrame
train_df = pd.DataFrame(train_samples)
```

### 1.2 Balance the Dataset

We'll keep only labels with at least 200 samples and limit each to 600 samples for balance.

```python
def process_labels(df, min_samples=200, max_samples=600):
    label_counts = df["label_text"].value_counts()
    labels_to_keep = label_counts[label_counts >= min_samples].index
    df = df[df["label_text"].isin(labels_to_keep)]

    # Limit each label to max_samples
    balanced_df = pd.DataFrame()
    for label in labels_to_keep:
        label_samples = df[df["label_text"] == label].sample(
            n=min(len(df[df["label_text"] == label]), max_samples),
            random_state=42
        )
        balanced_df = pd.concat([balanced_df, label_samples])
    return balanced_df

train_df = process_labels(train_df)
```

### 1.3 Split into Train, Validation, and Test Sets

```python
train_df, temp_df = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df["label_text"]
)
validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label_text"]
)

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(validation_df)}")
print(f"Test samples: {len(test_df)}")
```

### 1.4 Visualize the Data Distribution

```python
import matplotlib.pyplot as plt
from collections import Counter

def count_labels(samples):
    labels = [sample["label_text"] for sample in samples.to_dict("records")]
    return Counter(labels)

train_label_counts = count_labels(train_df)
validation_label_counts = count_labels(validation_df)
test_label_counts = count_labels(test_df)

fig, axes = plt.subplots(3, 1, figsize=(12, 18))

def plot_label_distribution(ax, label_counts, title):
    sorted_labels, sorted_counts = zip(
        *sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    )
    ax.bar(sorted_labels, sorted_counts, color="skyblue")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)

plot_label_distribution(axes[0], train_label_counts, "Train Samples (Label Distribution)")
plot_label_distribution(axes[1], validation_label_counts, "Validation Samples (Label Distribution)")
plot_label_distribution(axes[2], test_label_counts, "Test Samples (Label Distribution)")

plt.tight_layout()
plt.show()
```

---

## Step 2: Format Data for Training

Mistral's Classifier Factory expects data in JSONL format, where each line is a JSON object with `text` and `labels` fields.

### 2.1 Convert to JSONL

```python
from tqdm import tqdm
import json

def dataset_to_jsonl(dataset, output_file):
    with open(output_file, "w") as f:
        for _, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            text = row["text"]
            intent = row["label_text"]
            json_object = {"text": text, "labels": {"intent": intent}}
            f.write(json.dumps(json_object) + "\n")

# Save the splits
dataset_to_jsonl(train_df, "train.jsonl")
dataset_to_jsonl(validation_df, "validation.jsonl")
dataset_to_jsonl(test_df, "test.jsonl")
```

Each JSONL file contains entries like:

```json
{"text": "place a birthday party with ale ross and amy in my calendar", "labels": {"intent": "calendar_set"}}
{"text": "new music tracks", "labels": {"intent": "play_music"}}
```

---

## Step 3: Train the Classifier via Mistral API

### 3.1 Initialize the Mistral Client

```python
from mistralai import Mistral
import os

api_key = "YOUR_MISTRAL_API_KEY"  # Replace with your key
wandb_key = "YOUR_WANDB_KEY"      # Optional, for experiment tracking

client = Mistral(api_key=api_key)
```

### 3.2 Upload Training and Validation Files

```python
# Upload the training data
training_data = client.files.upload(
    file={
        "file_name": "train.jsonl",
        "content": open("train.jsonl", "rb"),
    }
)

# Upload the validation data
validation_data = client.files.upload(
    file={
        "file_name": "validation.jsonl",
        "content": open("validation.jsonl", "rb"),
    }
)

print(f"Training file ID: {training_data.id}")
print(f"Validation file ID: {validation_data.id}")
```

### 3.3 Create a Fine-Tuning Job

We'll create a classifier job using the `ministral-3b-latest` model.

```python
created_job = client.fine_tuning.jobs.create(
    model="ministral-3b-latest",
    job_type="classifier",
    training_files=[{"file_id": training_data.id, "weight": 1}],
    validation_files=[validation_data.id],
    hyperparameters={
        "training_steps": 100,
        "learning_rate": 0.00004
    },
    auto_start=False,
    integrations=[
        {
            "project": "intent-classifier",
            "api_key": wandb_key,
        }
    ] if wandb_key else []
)

print(json.dumps(created_job.model_dump(), indent=4))
```

### 3.4 Validate and Start the Job

Before starting, we wait for the job to be validated.

```python
import time

retrieved_job = client.fine_tuning.jobs.get(job_id=created_job.id)

# Wait for validation
while retrieved_job.status not in ["VALIDATED"]:
    retrieved_job = client.fine_tuning.jobs.get(job_id=created_job.id)
    print(f"Current status: {retrieved_job.status}")
    time.sleep(1)

print("Job validated. Starting training...")
```

Now, start the training job:

```python
client.fine_tuning.jobs.start(job_id=created_job.id)
retrieved_job = client.fine_tuning.jobs.get(job_id=created_job.id)
print(json.dumps(retrieved_job.model_dump(), indent=4))
```

### 3.5 Monitor Training Progress

Poll the job status until it completes.

```python
while retrieved_job.status in ["QUEUED", "RUNNING"]:
    retrieved_job = client.fine_tuning.jobs.get(job_id=created_job.id)
    print(f"Status: {retrieved_job.status}")
    time.sleep(5)

print("Training completed!")
```

**Optional:** Use the Weights & Biases integration to monitor metrics like training loss and validation accuracy in real-time.

---

## Step 4: Run Inference with Your Trained Model

Once training finishes, you can use the fine-tuned model for classification.

### 4.1 Load a Test Sample

```python
with open("test.jsonl", "r") as f:
    test_samples = [json.loads(l) for l in f.readlines()]

test_text = test_samples[0]["text"]
print(f"Test text: {test_text}")
```

### 4.2 Classify the Text

```python
classifier_response = client.classifiers.classify(
    model=retrieved_job.fine_tuned_model,
    inputs=[test_text],
)

print(json.dumps(classifier_response.model_dump(), indent=4))
```

Example output:

```json
{
    "model": "ft:ministral-3b-latest:your-model-id",
    "classifications": [
        {
            "input": "what's the weather like today?",
            "labels": [
                {
                    "name": "weather_query",
                    "score": 0.998
                },
                {
                    "name": "calendar_query",
                    "score": 0.001
                }
            ]
        }
    ]
}
```

The classifier correctly identifies `weather_query` with high confidence.

---

## Conclusion

You've successfully built a custom intent classifier using Mistral AI's Classifier Factory. The process involved:

1. **Data Preparation:** Loading, balancing, and splitting a dataset.
2. **Format Conversion:** Converting data to JSONL for training.
3. **Model Training:** Creating and monitoring a fine-tuning job via the Mistral API.
4. **Inference:** Using the trained model to classify new text.

### Next Steps

- **Multi-Label Classification:** For scenarios where a single input can belong to multiple categories, see the [Moderation Classifier cookbook](https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/classifier_factory/moderation_classifier.ipynb).
- **Advanced Evaluation:** Compare classifier performance against LLMs in the [Product Classification cookbook](https://colab.research.google.com/github/mistralai/cookbook/blob/main/mistral/classifier_factory/product_classification.ipynb).

Experiment with different hyperparameters, dataset sizes, or use cases to tailor the classifier to your specific needs.