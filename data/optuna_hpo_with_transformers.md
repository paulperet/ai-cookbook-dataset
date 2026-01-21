# Hyperparameter Optimization with Optuna and Transformers

_Authored by: [Parag Ekbote](https://github.com/ParagEkbote)_

## Overview
This guide demonstrates how to systematically optimize hyperparameters for transformer-based text classification models using automated search techniques. You'll implement Hyperparameter Optimization (HPO) using Optuna to find optimal learning rates and weight decay values for fine-tuning BERT on sentiment analysis tasks.

## When to Use This Recipe
* You need to fine-tune pre-trained language models for classification tasks
* Your model performance is plateauing and requires parameter refinement
* You want to implement systematic, reproducible hyperparameter optimization

**Note:** For detailed guidance on hyperparameter search with Transformers, refer to the [Hugging Face HPO documentation](https://huggingface.co/docs/transformers/en/hpo_train).

## Prerequisites

First, install the required packages:

```bash
pip install datasets evaluate transformers optuna wandb scikit-learn nbformat matplotlib
```

## Step 1: Prepare Dataset and Initialize Model

Before you can train and evaluate a sentiment analysis model, you need to prepare your dataset and initialize the model architecture.

### 1.1 Load and Prepare the IMDB Dataset

```python
from datasets import load_dataset
import evaluate
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import set_seed

# Set seed for reproducibility
set_seed(42)

# Load and sample the IMDB dataset
train_dataset = load_dataset("imdb", split="train").shuffle(seed=42).select(range(2500))
valid_dataset = load_dataset("imdb", split="test").shuffle(seed=42).select(range(1000))

# Initialize tokenizer
model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

# Apply tokenization
tokenized_train = train_dataset.map(tokenize, batched=True).select_columns(
    ["input_ids", "attention_mask", "label"]
)
tokenized_valid = valid_dataset.map(tokenize, batched=True).select_columns(
    ["input_ids", "attention_mask", "label"]
)

# Load evaluation metric
metric = evaluate.load("accuracy")

# Model initialization function
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
```

## Step 2: Configure Persistent Storage with Optuna

To ensure your hyperparameter optimization experiments are trackable and reproducible, set up persistent storage using Optuna's RDBStorage mechanism.

```python
import optuna
from optuna.storages import RDBStorage

# Define persistent storage
storage = RDBStorage("sqlite:///optuna_trials.db")

# Create or load study
study = optuna.create_study(
    study_name="transformers_optuna_study",
    direction="maximize",
    storage=storage,
    load_if_exists=True
)
```

## Step 3: Set Up Trainer and Observability

Configure the training pipeline with metrics computation and observability using Weights & Biases.

```python
import wandb
from transformers import Trainer, TrainingArguments

# Metrics computation function
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(axis=-1)
    labels = eval_pred.label_ids
    return metric.compute(predictions=predictions, references=labels)

# Objective function for optimization
def compute_objective(metrics):
    return metrics["eval_accuracy"]

# Initialize Weights & Biases
wandb.init(project="hf-optuna", name="transformers_optuna_study")

# Configure training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_strategy="epoch",
    num_train_epochs=3,
    report_to="wandb",
    logging_dir="./logs",
    run_name="transformers_optuna_study",
)

# Initialize Trainer
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)
```

## Step 4: Define Search Space and Run Hyperparameter Optimization

Define the hyperparameter search space and run the optimization trials.

```python
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32, 64, 128]
        ),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.3),
    }

# Run hyperparameter search
best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=5,
    compute_objective=compute_objective,
    study_name="transformers_optuna_study",
    storage="sqlite:///optuna_trials.db",
    load_if_exists=True
)

print(f"Best hyperparameters: {best_run}")
```

After running the optimization, you'll see output similar to:

```
Best hyperparameters: {
    'learning_rate': 3.2e-05,
    'per_device_train_batch_size': 32,
    'weight_decay': 0.1,
    'objective': 0.764
}
```

## Step 5: Visualize Optimization Results

Analyze the optimization process using Optuna's visualization tools.

```python
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_intermediate_values,
    plot_param_importances
)
import matplotlib.pyplot as plt

# Load the study from storage
storage = optuna.storages.RDBStorage("sqlite:///optuna_trials.db")
study = optuna.load_study(
    study_name="transformers_optuna_study",
    storage=storage
)

# Plot optimization history
ax1 = plot_optimization_history(study)
plt.show()

# Plot intermediate values
ax2 = plot_intermediate_values(study)
plt.show()

# Plot parameter importances
ax3 = plot_param_importances(study)
plt.show()
```

## Step 6: Train Final Model with Best Hyperparameters

Now train your final model using the optimized hyperparameters.

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Extract best hyperparameters
best_hparams = best_run.hyperparameters

# Configure final training arguments
training_args = TrainingArguments(
    output_dir="./final_model",
    learning_rate=best_hparams["learning_rate"],
    per_device_train_batch_size=best_hparams["per_device_train_batch_size"],
    weight_decay=best_hparams["weight_decay"],
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_strategy="epoch",
    num_train_epochs=3,
    report_to="wandb",
    run_name="final_run_with_best_hparams"
)

# Create final trainer
trainer = Trainer(
    model=model_init(),
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the final model
trainer.save_model("./final_model")
```

## Step 7: Upload Model to Hugging Face Hub (Optional)

After training your optimized model, you can share it with the community by uploading it to the Hugging Face Hub.

```python
from huggingface_hub import HfApi, HfFolder

# Login to Hugging Face
api = HfApi()
api.set_access_token("your_hf_token_here")

# Upload model
api.upload_folder(
    folder_path="./final_model",
    repo_id="your-username/bert-tiny-imdb-optimized",
    repo_type="model"
)
```

## Summary

In this tutorial, you've learned how to:

1. Prepare a text classification dataset for transformer models
2. Set up persistent storage for reproducible hyperparameter optimization
3. Configure Optuna for systematic hyperparameter search
4. Visualize optimization results to understand parameter importance
5. Train a final model using the best-found hyperparameters
6. Share your optimized model with the community

The systematic approach demonstrated here ensures you get the most out of your model training while maintaining reproducibility and transparency in your optimization process.