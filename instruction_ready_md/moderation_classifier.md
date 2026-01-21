# Train Your Own Moderation Service with Classifier Factory

In this guide, you will learn how to build a custom content moderation classifier using Mistral's Classifier Factory. We'll focus on a practical example: multi-label classification for detecting toxic content. By the end, you'll have a trained model ready to flag comments for categories like toxicity, insults, and threats.

## Prerequisites

Ensure you have the necessary Python libraries installed.

```bash
pip install datasets pandas matplotlib tqdm mistralai
```

## Step 1: Load and Prepare the Dataset

We'll use a subset of the `google/civil_comments` dataset from Hugging Face. This dataset includes labels for various types of toxic content, which we'll adapt for multi-label classification.

First, import the required modules and load the dataset.

```python
from datasets import load_dataset
import pandas as pd
import random

# Load the civil_comments dataset with streaming
dataset = load_dataset("google/civil_comments", streaming=True)

# Define subset sizes
n_train = 1_000_000
n_validation = 50_000
n_test = 50_000
```

### Convert Scores to Boolean Labels

The original dataset provides continuous scores for each label. We'll convert these to boolean flags using a threshold of 0.5.

```python
def convert_scores_to_booleans(example, threshold=0.5):
    return {
        "text": example["text"],
        "toxicity": example["toxicity"] > threshold,
        "severe_toxicity": example["severe_toxicity"] > threshold,
        "obscene": example["obscene"] > threshold,
        "threat": example["threat"] > threshold,
        "insult": example["insult"] > threshold,
        "identity_attack": example["identity_attack"] > threshold,
        "sexual_explicit": example["sexual_explicit"] > threshold,
    }

# Shuffle, take, and convert the dataset splits
train_samples = [
    convert_scores_to_booleans(example)
    for example in dataset["train"].shuffle(seed=42, buffer_size=n_train).take(n_train)
]
validation_samples = [
    convert_scores_to_booleans(example)
    for example in dataset["validation"]
    .shuffle(seed=42, buffer_size=n_validation)
    .take(n_validation)
]
test_samples = [
    convert_scores_to_booleans(example)
    for example in dataset["test"].shuffle(seed=42, buffer_size=n_test).take(n_test)
]
```

### Balance the Dataset

To avoid class imbalance, we'll filter the samples. This step ensures that:
- Only 10% of the dataset consists of samples with zero flags (completely clean).
- No single label represents more than 20% of the flagged samples.

```python
def filter_dataset(samples, zero_flags_percentage=0.1, max_percentage=0.2):
    zero_flags_samples = []
    non_zero_flags_samples = []
    label_counts = {key: 0 for key in samples[0] if key != "text"}

    for example in samples:
        if not any(
            example[key] for key in example if key != "text"
        ):  # Check if all flags are False
            zero_flags_samples.append(example)
        else:
            non_zero_flags_samples.append(example)

    # Calculate the total number of samples needed
    total_samples = len(non_zero_flags_samples) / (1 - zero_flags_percentage)

    # Calculate the number of zero-flag samples needed
    desired_zero_flags = int(total_samples * zero_flags_percentage)
    desired_non_zero_flags = int(total_samples * (1 - zero_flags_percentage))

    # Keep only the desired number of zero-flag and non-zero-flag samples
    zero_flags_samples = zero_flags_samples[:desired_zero_flags]
    filtered_samples = []

    for example in non_zero_flags_samples[:desired_non_zero_flags]:
        # Check if adding this example exceeds the max percentage for any label
        add_sample = True
        for key in label_counts:
            if example[key]:
                if (label_counts[key] + 1) / desired_non_zero_flags > max_percentage:
                    add_sample = False
                    break

        if add_sample:
            filtered_samples.append(example)
            for key in label_counts:
                if example[key]:
                    label_counts[key] += 1

    # Combine the filtered zero-flag samples with the non-zero-flag samples
    filtered_samples += zero_flags_samples

    # Shuffle the filtered samples
    random.shuffle(filtered_samples)

    return filtered_samples

# Apply the filter to all splits
train_samples = filter_dataset(train_samples)
validation_samples = filter_dataset(validation_samples)
test_samples = filter_dataset(test_samples)
```

### Remove Rare Labels

Some labels may appear in less than 1% of samples. We'll identify and remove them to improve model training.

```python
# Combine all samples to calculate label percentages
all_samples = train_samples + validation_samples + test_samples
all_df = pd.DataFrame(all_samples)

# Calculate the percentage of samples for each label
label_percentages = all_df.drop(columns=["text"]).mean()

# Identify labels with less than 1% samples
labels_to_remove = label_percentages[label_percentages < 0.01].index.tolist()

# Remove the identified labels from all splits
train_df = pd.DataFrame(train_samples).drop(columns=labels_to_remove)
validation_df = pd.DataFrame(validation_samples).drop(columns=labels_to_remove)
test_df = pd.DataFrame(test_samples).drop(columns=labels_to_remove)
```

Let's check the size of our processed datasets.

```python
print("Train set length:", len(train_df))
print("Validation set length:", len(validation_df))
print("Test set length:", len(test_df))
```

**Output:**
```
Train set length: 20607
Validation set length: 1010
Test set length: 1013
```

## Step 2: Analyze the Data Distribution

It's important to understand the distribution of labels in your dataset. The following code creates visualizations to show the proportion of each label and the balance between flagged and non-flagged samples.

```python
import matplotlib.pyplot as plt
from collections import defaultdict

# Create a list of labels for the pie charts from the samples
labels_to_keep = [
    key for key in train_samples[0] if key != "text" and key not in labels_to_remove
]
labels = ["No Flags"] + labels_to_keep

# Function to count the number of samples flagged for each category
def count_flags(samples, labels_to_keep):
    no_flags = sum(
        all(
            value == False
            for key, value in sample.items()
            if key != "text" and key in labels_to_keep
        )
        for sample in samples
    )
    counts = {key: sum(sample[key] for sample in samples) for key in labels_to_keep}
    return [no_flags] + list(counts.values())

# Function to count total flagged and non-flagged samples
def count_total_flagged_vs_non_flagged(samples, labels_to_keep):
    total_flagged = 0
    total_non_flagged = 0
    for sample in samples:
        flagged = any(
            value
            for key, value in sample.items()
            if key != "text" and key in labels_to_keep
        )
        if flagged:
            total_flagged += 1
        else:
            total_non_flagged += 1
    return [total_non_flagged, total_flagged]

# Count flags for each dataset
train_counts = count_flags(train_samples, labels_to_keep)
validation_counts = count_flags(validation_samples, labels_to_keep)
test_counts = count_flags(test_samples, labels_to_keep)

# Count total flagged vs non-flagged for each dataset
total_train_counts = count_total_flagged_vs_non_flagged(train_samples, labels_to_keep)
total_validation_counts = count_total_flagged_vs_non_flagged(
    validation_samples, labels_to_keep
)
total_test_counts = count_total_flagged_vs_non_flagged(test_samples, labels_to_keep)

# Sort the labels and counts for pie charts
sorted_train_labels, sorted_train_counts = zip(
    *sorted(zip(labels, train_counts), key=lambda x: x[1], reverse=True)
)
sorted_validation_labels, sorted_validation_counts = zip(
    *sorted(zip(labels, validation_counts), key=lambda x: x[1], reverse=True)
)
sorted_test_labels, sorted_test_counts = zip(
    *sorted(zip(labels, test_counts), key=lambda x: x[1], reverse=True)
)

# Create a single figure with subplots for pie charts and stacked bar plots
fig, axes = plt.subplots(3, 3, figsize=(30, 18))

# Plot the pie charts for category distribution
axes[0, 0].pie(
    sorted_train_counts, labels=sorted_train_labels, autopct="%1.1f%%", startangle=140
)
axes[0, 0].set_title("Train Samples (Category Distribution)")

axes[1, 0].pie(
    sorted_validation_counts,
    labels=sorted_validation_labels,
    autopct="%1.1f%%",
    startangle=140,
)
axes[1, 0].set_title("Validation Samples (Category Distribution)")

axes[2, 0].pie(
    sorted_test_counts, labels=sorted_test_labels, autopct="%1.1f%%", startangle=140
)
axes[2, 0].set_title("Test Samples (Category Distribution)")

# Plot the pie charts for flagged vs non-flagged distribution
axes[0, 1].pie(
    total_train_counts,
    labels=["None-Flagged", "Flagged"],
    autopct="%1.1f%%",
    startangle=140,
    colors=["blue", "red"],
)
axes[0, 1].set_title("Train Samples (Flagged vs None-Flagged)")

axes[1, 1].pie(
    total_validation_counts,
    labels=["None-Flagged", "Flagged"],
    autopct="%1.1f%%",
    startangle=140,
    colors=["blue", "red"],
)
axes[1, 1].set_title("Validation Samples (Flagged vs None-Flagged)")

axes[2, 1].pie(
    total_test_counts,
    labels=["None-Flagged", "Flagged"],
    autopct="%1.1f%%",
    startangle=140,
    colors=["blue", "red"],
)
axes[2, 1].set_title("Test Samples (Flagged vs None-Flagged)")

# Function to create stacked bar plots for flagged vs. non-flagged samples
def count_flagged_vs_non_flagged(samples, labels_to_keep):
    counts = defaultdict(lambda: {"flagged": 0, "none-flagged": 0})
    for sample in samples:
        for key, value in sample.items():
            if key != "text" and key in labels_to_keep:
                if value:
                    counts[key]["flagged"] += 1
                else:
                    counts[key]["none-flagged"] += 1
    return counts

def plot_stacked_bar(ax, samples, title, labels_to_keep):
    counts = count_flagged_vs_non_flagged(samples, labels_to_keep)
    labels = counts.keys()
    flagged_counts = [counts[label]["flagged"] for label in labels]
    non_flagged_counts = [counts[label]["none-flagged"] for label in labels]

    # Sort the labels and counts for stacked bar plots by flagged counts
    sorted_labels, sorted_non_flagged_counts, sorted_flagged_counts = zip(
        *sorted(
            zip(labels, non_flagged_counts, flagged_counts),
            key=lambda x: x[2],
            reverse=True,
        )
    )

    ax.bar(sorted_labels, sorted_non_flagged_counts, label="None-Flagged", color="blue")
    ax.bar(
        sorted_labels,
        sorted_flagged_counts,
        bottom=sorted_non_flagged_counts,
        label="Flagged",
        color="red",
    )

    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

# Plot stacked bar plots for each dataset
plot_stacked_bar(
    axes[0, 2], train_samples, "Train Samples (Stacked Bar)", labels_to_keep
)
plot_stacked_bar(
    axes[1, 2], validation_samples, "Validation Samples (Stacked Bar)", labels_to_keep
)
plot_stacked_bar(axes[2, 2], test_samples, "Test Samples (Stacked Bar)", labels_to_keep)

plt.tight_layout()
plt.show()
```

The visualizations will show that our dataset is fairly balanced for `toxicity` and `insult`, but less balanced for other labels. In a production scenario, you might need further curation for optimal performance.

## Step 3: Format Data for Training

The Classifier Factory requires data in JSONL format. Each line should be a JSON object with a `text` field and a `labels` field containing an array of moderation categories. For samples with no flags, we'll assign a special label: `"safe"`.

```python
from tqdm import tqdm
import json

def dataset_to_jsonl(dataset, output_file):
    # Extract the possible categories from the dataset columns, excluding the 'text' column
    possible_categories = [col for col in dataset.columns if col != "text"]

    # Open the output file in write mode
    with open(output_file, "w") as f:
        # Iterate over each row in the dataset
        for _, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            # Extract the text and labels from the row
            text = row["text"]
            labels = [
                category
                for category in possible_categories
                if row[category]
            ]
            if len(labels) == 0:
                labels = ["safe"]

            # Create the JSON object
            json_object = {"text": text, "labels": {"moderation": labels}}

            # Write the JSON object to the file as a JSON line
            f.write(json.dumps(json_object) + "\n")

# Save files
dataset_to_jsonl(train_df, "training_file.jsonl")
dataset_to_jsonl(validation_df, "validation_file.jsonl")
dataset_to_jsonl(test_df, "test_file.jsonl")
```

Each JSONL file will contain entries like this:

```json
{"text": "I believe the Trump administration made a big mistake...", "labels": {"moderation": ["safe"]}}
{"text": "Uh huh. Then why don't you behave that way...", "labels": {"moderation": ["toxicity", "insult"]}}
```

## Step 4: Train the Model via the Mistral API

Now, we'll use the Mistral API to upload our data and start a fine-tuning job.

### Initialize the Client

First, set up your Mistral client with an API key. You can create one [here](https://console.mistral.ai/api-keys/).

```python
from mistralai import Mistral

# Set the API key for Mistral
api_key = "YOUR_API_KEY_HERE"

# Initialize the Mistral client
client = Mistral(api_key=api_key)
```

### Upload Training and Validation Files

```python
# Upload the training data
training_data = client.files.upload(
    file={
        "file_name": "training_file.jsonl",
        "content": open("training_file.jsonl", "rb"),
    }
)

# Upload the validation data
validation_data = client.files.upload(
    file={
        "file_name": "validation_file.jsonl",
        "content": open("validation_file.jsonl", "rb"),
    }
)
```

### Create a Fine-Tuning Job

We'll create a classifier job using the `ministral-3b-latest` model. You can adjust hyperparameters like `training_steps` and `learning_rate` based on your needs.

```python
# Create a fine-tuning job
created_job = client.fine_tuning.jobs.create(
    model="ministral-3b-latest",
    job_type="classifier",
    training_files=[{"file_id": training_data.id, "weight": 1}],
    validation_files=[validation_data.id],
    hyperparameters={"training_steps": 200, "learning_rate": 0.00005},
    auto_start=False,
)
print(created_job)
```

**Note:** The `auto_start=False` parameter allows you to review the job before starting it. Remove it or set to `True` to start training immediately.

Optionally, you can integrate with Weights & Biases for experiment tracking by uncommenting the `integrations` section and providing your W&B API key.

## Next Steps

Once the job is created and started, you can monitor its progress via the [Mistral console](https://console.mistral.ai/build/finetuned-models) or the API. After training completes, you'll receive a model ID that you can use for inference.

To use your trained model:

```python
from mistralai import Mistral

client = Mistral(api_key="YOUR_API_KEY")
response = client.classifiers.classify(
    model="YOUR_FINETUNED_MODEL_ID",
    inputs=["Your text to moderate here."]
)
print(response)
```

Congratulations! You've successfully prepared a dataset and trained a custom moderation classifier. You can now deploy this model to automatically flag toxic content in your applications.