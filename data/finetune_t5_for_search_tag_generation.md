# GitHub Tag Generator with T5 and PEFT (LoRA)

## Overview

This guide walks you through building a lightweight, fast, and open-source GitHub tag generator. You'll fine-tune a **T5-small** model using **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA** on a custom dataset of GitHub repository descriptions and their tags. The result is a model that can automatically generate relevant tags for any GitHub project, improving discoverability and organization.

### Use Case
Imagine you're building a tool to help users explore GitHub repositories. Instead of relying on manually written or missing tags, you can train a model to automatically generate descriptive tags for any project. This can:
- Improve search functionality.
- Automatically tag new repositories.
- Build better filters for discovery.

### What You'll Build
By the end of this tutorial, you will have:
1. A fully trained GitHub tag generator model.
2. A model hosted on the Hugging Face Hub for easy sharing and deployment.
3. An inference function to generate tags with just a few lines of code.

---

## Prerequisites

Before you begin, ensure you have the following:

1. **A Hugging Face Account and Token:** You'll need an account to push your model to the Hub. Create a token with write access [here](https://huggingface.co/settings/tokens).
2. **Python Environment:** This tutorial uses Google Colab, but you can adapt it for any environment with Python 3.8+.
3. **Required Libraries:** Install the necessary packages.

### Setup and Installation

Run the following commands to install the required libraries:

```bash
!pip install transformers datasets peft accelerate wandb
```

Now, let's import the necessary modules and set up your Hugging Face token securely.

```python
import os
from google.colab import userdata

# Securely retrieve your Hugging Face token from Colab secrets
os.environ['HUGGINGFACE_TOKEN'] = userdata.get('HUGGINGFACE_TOKEN')
```

```python
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer
)
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model
```

---

## Step 1: Load and Prepare the Dataset

You'll use a dataset containing GitHub repository descriptions and their corresponding tags. The dataset is available on the Hugging Face Hub.

### Load the Dataset

Load the dataset directly from the Hub. It initially has only a "train" split, so you'll split it into training and validation sets.

```python
# Load the dataset from the Hugging Face Hub
dataset = load_dataset("zamal/github-meta-data")

# Split the train set into train and validation (90/10 split)
split = dataset["train"].train_test_split(test_size=0.1, seed=42)

# Wrap into a new DatasetDict
dataset_dict = DatasetDict({
    "train": split["train"],
    "validation": split["test"]
})

# Check the sizes
print(f"Training samples: {len(dataset_dict['train'])}")
print(f"Validation samples: {len(dataset_dict['validation'])}")
```

**Output:**
```
Training samples: 552
Validation samples: 62
```

Each example in the dataset has two fields:
- `input`: A short repository description.
- `target`: A comma-separated list of relevant tags.

---

## Step 2: Initialize the Tokenizer

Load the tokenizer for the `t5-small` model. This tokenizer will prepare your text data for the model.

```python
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

**Note:** You might see a warning about legacy behavior. This is safe to ignore for this tutorial.

---

## Step 3: Preprocess the Dataset

Define a preprocessing function to tokenize both the inputs and the targets. This function will:
1. Tokenize the input descriptions.
2. Tokenize the target tags.
3. Add the tokenized labels to the model inputs.

```python
def preprocess(batch):
    inputs = batch["input"]
    targets = batch["target"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    
    # Tokenize targets (these become our labels)
    labels = tokenizer(
        targets,
        max_length=64,
        truncation=True,
        padding="max_length"
    ).input_ids
    
    model_inputs["labels"] = labels
    return model_inputs
```

Now, apply this function to your entire dataset.

```python
# Apply preprocessing
tokenized = dataset_dict.map(
    preprocess,
    batched=True,
    remove_columns=dataset_dict["train"].column_names
)

# Set format for PyTorch
tokenized.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'labels']
)
```

Your dataset is now ready for training.

---

## Step 4: Load the Base Model

Load the `t5-small` model, which serves as the backbone for your tag generation task.

```python
model = T5ForConditionalGeneration.from_pretrained(model_name)
```

---

## Step 5: Configure LoRA for Parameter-Efficient Fine-Tuning

Instead of fine-tuning the entire model, you'll use **LoRA (Low-Rank Adaptation)**. This technique injects small, trainable matrices into the model's attention layers, drastically reducing the number of parameters you need to update.

Define the LoRA configuration:

```python
lora_config = LoraConfig(
    r=16,                     # Rank of the update matrices
    lora_alpha=32,           # Scaling factor
    target_modules=["q", "v"], # Apply LoRA to query and value projection layers
    lora_dropout=0.05,       # Dropout for LoRA layers
    bias="none",             # Don't train bias parameters
    task_type="SEQ_2_SEQ_LM" # Task type for sequence-to-sequence
)
```

Now, wrap your base model with the LoRA adapters:

```python
model = get_peft_model(model, lora_config)
```

This creates a `PeftModel` where only the LoRA parameters are trainable.

---

## Step 6: Configure Training Arguments

Set up the training hyperparameters and logging options using `TrainingArguments`.

```python
training_args = TrainingArguments(
    output_dir="./t5_tag_generator",      # Directory to save checkpoints
    per_device_train_batch_size=8,        # Batch size for training
    per_device_eval_batch_size=8,         # Batch size for evaluation
    learning_rate=1e-4,                   # Learning rate
    num_train_epochs=25,                  # Number of training epochs
    logging_steps=10,                     # Log metrics every 10 steps
    eval_strategy="steps",                # Evaluate at specific steps
    eval_steps=50,                        # Evaluate every 50 steps
    save_steps=50,                        # Save checkpoint every 50 steps
    save_total_limit=2,                   # Keep only the last 2 checkpoints
    fp16=True,                            # Use mixed precision training
    push_to_hub=True,                     # Push the model to the Hub
    hub_model_id="zamal/github-tag-generatorr", # Your model ID on the Hub
    hub_token=os.environ['HUGGINGFACE_TOKEN'] # Authentication token
)
```

**Important:** Replace `"zamal/github-tag-generatorr"` with your own Hugging Face username and desired model name.

---

## Step 7: Set Up the Trainer

The `Trainer` class handles the training loop, evaluation, and logging. You'll also need a data collator to properly batch your sequences.

```python
# Data collator for sequence-to-sequence tasks
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)
```

**Note:** You might see warnings about deprecation or label names. These are safe to ignore for this tutorial.

---

## Step 8: Train the Model

Start the fine-tuning process. This will train the model using LoRA, saving checkpoints and logging metrics as defined.

```python
trainer.train()
```

Training will take some time depending on your hardware. You'll see logs showing training loss, evaluation loss, and other metrics. The model will automatically be pushed to the Hugging Face Hub upon completion.

---

## Step 9: Save and Push the Model

After training, save the final model and push it to the Hub.

```python
# Save the model locally
trainer.save_model()

# Push to the Hub (if not already done via TrainingArguments)
trainer.push_to_hub()
```

Your model is now available on the Hugging Face Hub and ready for inference.

---

## Step 10: Perform Inference

Now, let's use your trained model to generate tags for new repository descriptions.

### Load the Fine-Tuned Model

First, load your model from the Hub.

```python
from peft import PeftModel, PeftConfig

# Load the base model and LoRA configuration
config = PeftConfig.from_pretrained("zamal/github-tag-generatorr")
base_model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, "zamal/github-tag-generatorr")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
```

**Remember:** Replace `"zamal/github-tag-generatorr"` with your model's ID.

### Define an Inference Function

Create a function that takes a repository description and returns a list of clean, deduplicated tags.

```python
def generate_tags(description, max_length=64):
    # Tokenize the input
    inputs = tokenizer(
        description,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    
    # Generate tags
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Post-process: split by comma, strip whitespace, remove duplicates
    tags = [tag.strip() for tag in generated_text.split(",")]
    tags = list(dict.fromkeys(tags))  # Remove duplicates while preserving order
    
    return tags
```

### Test Your Model

Let's test the model with a few example descriptions.

```python
# Example 1
description1 = "A Python library for natural language processing with transformer models."
tags1 = generate_tags(description1)
print(f"Description: {description1}")
print(f"Generated Tags: {tags1}")
print()

# Example 2
description2 = "A web framework for building fast and scalable APIs in Go."
tags2 = generate_tags(description2)
print(f"Description: {description2}")
print(f"Generated Tags: {tags2}")
```

**Example Output:**
```
Description: A Python library for natural language processing with transformer models.
Generated Tags: ['python', 'nlp', 'transformer', 'library']

Description: A web framework for building fast and scalable APIs in Go.
Generated Tags: ['go', 'web', 'framework', 'api']
```

---

## Conclusion

Congratulations! You've successfully built and deployed a GitHub tag generator using T5 and PEFT with LoRA. You've learned how to:

1. Load and preprocess a custom dataset.
2. Configure LoRA for parameter-efficient fine-tuning.
3. Train a sequence-to-sequence model with the Hugging Face `Trainer`.
4. Save and push your model to the Hugging Face Hub.
5. Perform inference to generate clean, relevant tags.

### Next Steps
- **Experiment:** Try different base models (e.g., `t5-base`) or adjust LoRA parameters (`r`, `alpha`).
- **Improve Data:** Collect more diverse repository descriptions and tags to improve model performance.
- **Deploy:** Create a simple web API using FastAPI or Gradio to serve your model.

Your model is now ready to help automate tag generation for GitHub repositories, making them more discoverable and organized. Happy coding!