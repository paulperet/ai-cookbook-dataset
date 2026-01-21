# Building a Custom Food Classifier with Mistral's Classifier Factory

In this guide, you'll learn how to build a custom multi-target classifier using Mistral's Classifier Factory. We'll create a model that classifies food products by both their country of origin and multiple food categories.

## Prerequisites

First, install the required packages:

```bash
pip install datasets mistralai pandas matplotlib tqdm
```

## 1. Understanding the Problem

We're building a classifier for food products that needs to predict:
- **Country** (single target): One of 8 possible countries
- **Categories** (multi-target): One or more of 8 possible food categories

The dataset contains food names with their corresponding country and category labels.

## 2. Loading and Exploring the Dataset

We'll use a curated subset of the Open Food Facts database:

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('pandora-s/openfood-classification')
print(f"Dataset splits: {list(dataset.keys())}")
print(f"Train samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")
print(f"Test samples: {len(dataset['test'])}")
```

Let's examine a sample from the test set:

```python
import pandas as pd

# Convert test set to pandas DataFrame for easy viewing
df = dataset["test"].to_pandas()
print(df.head())
```

## 3. Preparing Data for Training

The Classifier Factory requires data in JSONL format. Each line should contain:
- `text`: The food product name
- `labels`: A dictionary with classification targets

Here's the conversion function:

```python
from tqdm import tqdm
import json

def dataset_to_jsonl(split):
    """Convert dataset split to JSONL format for classifier training."""
    jsonl_data = []
    
    # Collect all unique labels across the dataset
    all_category_labels = set()
    all_countries = set()
    
    for example in dataset[split]:
        all_category_labels.update(example['category_labels'].keys())
        all_countries.add(example['country_label'])
    
    # Sort for consistent formatting
    all_category_labels = sorted(all_category_labels)
    all_countries = sorted(all_countries)
    
    # Process each example
    for example in tqdm(dataset[split]):
        # Extract active category labels
        active_categories = [
            tag for tag in all_category_labels
            if example['category_labels'][tag] == "true"
        ]
        
        # Create labels dictionary
        labels = {
            "food": active_categories,
            "country_label": example['country_label']
        }
        
        jsonl_data.append({
            "text": example['name'],
            "labels": labels
        })
    
    return jsonl_data, all_category_labels, all_countries

# Convert all splits
train_jsonl, _, _ = dataset_to_jsonl('train')
validation_jsonl, _, _ = dataset_to_jsonl('validation')
test_jsonl, all_category_labels, all_country_labels = dataset_to_jsonl('test')

# Save to files
for split_name, split_data in [
    ('train', train_jsonl),
    ('validation', validation_jsonl),
    ('test', test_jsonl)
]:
    with open(f'{split_name}_openfood_classification.jsonl', 'w') as f:
        for entry in split_data:
            f.write(json.dumps(entry) + '\n')

print("JSONL files saved successfully!")
print(f"Categories: {all_category_labels}")
print(f"Countries: {all_country_labels}")
```

## 4. Setting Up the Mistral Client

Initialize the Mistral client with your API key:

```python
from mistralai import Mistral

# Replace with your actual API keys
api_key = "your_mistral_api_key_here"
wandb_key = "your_wandb_key_here"  # Optional, for tracking metrics

# Initialize client
client = Mistral(api_key=api_key)
```

## 5. Uploading Training Data

Upload your prepared JSONL files to Mistral's platform:

```python
# Upload training data
training_data = client.files.upload(
    file={
        "file_name": "train_openfood_classification.jsonl",
        "content": open("train_openfood_classification.jsonl", "rb"),
    }
)

# Upload validation data (optional but recommended)
validation_data = client.files.upload(
    file={
        "file_name": "validation_openfood_classification.jsonl",
        "content": open("validation_openfood_classification.jsonl", "rb"),
    }
)

print(f"Training file ID: {training_data.id}")
print(f"Validation file ID: {validation_data.id}")
```

## 6. Creating a Fine-tuning Job

Configure and create your classifier training job:

```python
# Create fine-tuning job
created_job = client.fine_tuning.jobs.create(
    model="ministral-3b-latest",
    job_type="classifier",
    training_files=[{"file_id": training_data.id, "weight": 1}],
    validation_files=[validation_data.id],
    hyperparameters={
        "training_steps": 250,
        "learning_rate": 0.00007
    },
    auto_start=False,
    integrations=[
        {
            "project": "product-classifier",
            "api_key": wandb_key,
        }
    ] if wandb_key else []
)

print("Job created successfully!")
print(f"Job ID: {created_job.id}")
```

## 7. Validating and Starting the Job

Before starting training, verify the job configuration:

```python
import time
from IPython.display import clear_output

# Retrieve job details
retrieved_job = client.fine_tuning.jobs.get(job_id=created_job.id)

# Wait for validation
print("Waiting for job validation...")
while retrieved_job.status not in ["VALIDATED"]:
    retrieved_job = client.fine_tuning.jobs.get(job_id=created_job.id)
    clear_output(wait=True)
    print(f"Current status: {retrieved_job.status}")
    time.sleep(1)

print("Job validated successfully!")

# Start the training
client.fine_tuning.jobs.start(job_id=created_job.id)
print("Training started!")
```

## 8. Monitoring Training Progress

Track loss metrics during training:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Initialize data structures for metrics
train_metrics_df = pd.DataFrame(columns=["Step Number", "Train Loss"])
valid_metrics_df = pd.DataFrame(columns=["Step Number", "Valid Loss"])

print("Monitoring training progress...")
while retrieved_job.status in ["QUEUED", "RUNNING"]:
    retrieved_job = client.fine_tuning.jobs.get(job_id=created_job.id)
    
    if retrieved_job.status == "QUEUED":
        time.sleep(5)
        continue
    
    clear_output(wait=True)
    print(f"Status: {retrieved_job.status}")
    
    # Process checkpoints for metrics
    for checkpoint in retrieved_job.checkpoints[::-1]:
        step_number = checkpoint.step_number
        
        if step_number not in train_metrics_df["Step Number"].values:
            # Add training loss
            train_row = {
                "Step Number": step_number,
                "Train Loss": checkpoint.metrics.train_loss,
            }
            train_metrics_df = pd.concat(
                [train_metrics_df, pd.DataFrame([train_row])], 
                ignore_index=True
            )
            
            # Add validation loss if available
            if checkpoint.metrics.valid_loss != 0:
                valid_row = {
                    "Step Number": step_number,
                    "Valid Loss": checkpoint.metrics.valid_loss,
                }
                valid_metrics_df = pd.concat(
                    [valid_metrics_df, pd.DataFrame([valid_row])], 
                    ignore_index=True
                )
    
    # Plot progress if we have data
    if not train_metrics_df.empty:
        train_metrics_df = train_metrics_df.sort_values(by="Step Number")
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            train_metrics_df["Step Number"],
            train_metrics_df["Train Loss"],
            label="Train Loss",
            linestyle="-",
            color="blue"
        )
        
        if not valid_metrics_df.empty:
            valid_metrics_df = valid_metrics_df.sort_values(by="Step Number")
            plt.plot(
                valid_metrics_df["Step Number"],
                valid_metrics_df["Valid Loss"],
                label="Valid Loss",
                linestyle="--",
                color="orange"
            )
        
        plt.xlabel("Step Number")
        plt.ylabel("Loss")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    time.sleep(1)

print(f"Training completed with status: {retrieved_job.status}")
print(f"Fine-tuned model: {retrieved_job.fine_tuned_model}")
```

## 9. Testing the Classifier

Test your trained model on sample data:

```python
# Load test samples
with open("test_openfood_classification.jsonl", "r") as f:
    test_samples = [json.loads(line) for line in f.readlines()]

# Test on first sample
sample_text = test_samples[0]["text"]
true_labels = test_samples[0]["labels"]

print(f"Sample text: {sample_text}")
print(f"True labels: {true_labels}")

# Get prediction from classifier
classifier_response = client.classifiers.classify(
    model=retrieved_job.fine_tuned_model,
    inputs=[sample_text],
)

print("\nClassifier prediction:")
print(json.dumps(classifier_response.model_dump(), indent=2))
```

## 10. Comparing with Alternative Approaches

Let's compare our classifier against other methods:

```python
from pydantic import BaseModel
from enum import Enum
from typing import List
import random

# Define data models for structured output
Category = Enum('Category', {category.replace('-', '_'): category for category in all_category_labels})
Country = Enum('Country', {country.replace('-', '_'): country for country in all_country_labels})

class Food(BaseModel):
    categories: List[Category]
    country: Country

def classify(text: str, model_config: dict):
    """Classify text using different model types."""
    try:
        if model_config["type"] == "random":
            # Random baseline
            predicted_categories = random.sample(
                list(all_category_labels), 
                random.randint(1, len(all_category_labels))
            )
            predicted_country = random.choice(list(all_country_labels))
            return predicted_categories, predicted_country
            
        elif model_config["type"] == "classifier":
            # Our fine-tuned classifier
            response = client.classifiers.classify(
                model=model_config["model_id"],
                inputs=[text],
            )
            results = response.results[0]
            
            # Get categories with scores
            category_scores = results['food'].scores
            
            # Get country with highest score
            country_scores = results['country_label'].scores
            predicted_country = max(country_scores, key=country_scores.get)
            
            return category_scores, predicted_country
            
        else:
            # LLM with structured output
            instruction = f"""Classify the following food product. 
            Provide the country of origin and relevant food categories.
            
            Product: {text}"""
            
            chat_response = client.chat.parse(
                model=model_config["model_id"],
                messages=[{"role": "user", "content": instruction}],
                response_format=Food,
                max_tokens=512,
                temperature=0
            )
            
            parsed = chat_response.choices[0].message.parsed
            return [c.value for c in parsed.categories], parsed.country.value
            
    except Exception as e:
        print(f"Error with {model_config['type']}: {e}")
        return {}, None

# Evaluation functions
def calculate_category_score(actual, predicted):
    """Calculate accuracy for multi-label categories."""
    correct = 0
    total = 0
    
    for act, pred in zip(actual, predicted):
        if act or pred:
            total += 1
            if act and pred:
                correct += 1
    
    return correct / total if total > 0 else 0

def calculate_country_score(actual, predicted):
    """Calculate accuracy for country predictions."""
    correct = sum(a == p for a, p in zip(actual, predicted))
    return correct / len(actual) if actual else 0

# Run evaluation
def evaluate_model(dataset, model_config, n_samples=100):
    """Evaluate a model on the test dataset."""
    actual_categories = []
    actual_countries = []
    predicted_categories = []
    predicted_countries = []
    
    for sample in tqdm(dataset[:n_samples]):
        text = sample["text"]
        actual_labels = sample["labels"]
        
        actual_categories.append(actual_labels["food"])
        actual_countries.append(actual_labels["country_label"])
        
        pred_cats, pred_country = classify(text, model_config)
        predicted_categories.append(pred_cats)
        predicted_countries.append(pred_country)
    
    cat_score = calculate_category_score(actual_categories, predicted_categories)
    country_score = calculate_country_score(actual_countries, predicted_countries)
    
    return {
        "category_accuracy": cat_score,
        "country_accuracy": country_score,
        "overall_accuracy": (cat_score + country_score) / 2
    }

# Compare different approaches
print("Evaluating different classification methods...")

models_to_evaluate = [
    {"type": "random", "name": "Random Baseline"},
    {"type": "classifier", "model_id": retrieved_job.fine_tuned_model, "name": "Fine-tuned Classifier"},
    {"type": "llm", "model_id": "mistral-large-latest", "name": "Mistral Large (zero-shot)"}
]

results = {}
for model_config in models_to_evaluate:
    print(f"\nEvaluating {model_config['name']}...")
    scores = evaluate_model(test_samples, model_config, n_samples=50)
    results[model_config['name']] = scores
    print(f"Results: {scores}")

print("\n=== Final Comparison ===")
for model_name, scores in results.items():
    print(f"\n{model_name}:")
    print(f"  Category Accuracy: {scores['category_accuracy']:.2%}")
    print(f"  Country Accuracy: {scores['country_accuracy']:.2%}")
    print(f"  Overall Accuracy: {scores['overall_accuracy']:.2%}")
```

## 11. Using Your Classifier in Production

Once trained, you can use your classifier for new predictions:

```python
def classify_food_product(product_name: str, model_id: str):
    """Classify a food product using your trained model."""
    response = client.classifiers.classify(
        model=model_id,
        inputs=[product_name],
    )
    
    result = response.results[0]
    
    # Extract top categories (threshold can be adjusted)
    category_scores = result['food'].scores
    top_categories = [
        cat for cat, score in category_scores.items()
        if score > 0.5  # Confidence threshold
    ]
    
    # Get predicted country
    country_scores = result['country_label'].scores
    predicted_country = max(country_scores, key=country_scores.get)
    
    return {
        "product": product_name,
        "predicted_categories": top_categories,
        "predicted_country": predicted_country,
        "confidence_scores": {
            "categories": category_scores,
            "country": country_scores
        }
    }

# Example usage
sample_products = [
    "Dark Chocolate Hazelnut Oatmeal",
    "Tomato Pieces in Olive Oil",
    "Fresh Orange Juice"
]

for product in sample_products:
    prediction = classify_food_product(product, retrieved_job.fine_tuned_model)
    print(f"\nProduct: {prediction['product']}")
    print(f"Predicted Country: {prediction['predicted_country']}")
    print(f"Predicted Categories: {prediction['predicted_categories']}")
```

## Summary

You've successfully built and deployed a custom food classifier that can:
1. Predict both single-target (country) and multi-target (categories) labels
2. Handle overlapping categories for food products
3. Outperform zero-shot LLM approaches for this specific task
4. Be integrated into production pipelines for automated food classification

The key advantages of using Mistral's Classifier Factory include:
- **Specialized performance**: Fine-tuned for your specific classification task
- **Cost efficiency**: Lower inference costs compared to general-purpose LLMs
- **Consistency**: Structured outputs with confidence scores
- **Scalability**: Can handle batch classification efficiently

You can extend this approach to other classification tasks by preparing your dataset in the required JSONL format and following the same training workflow.