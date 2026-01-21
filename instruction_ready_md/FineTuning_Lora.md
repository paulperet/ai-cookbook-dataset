# Fine-tuning Phi-3 with LoRA: A Practical Guide

This guide walks you through fine-tuning Microsoft's Phi-3 Mini language model using **LoRA (Low-Rank Adaptation)**. LoRA is an efficient fine-tuning technique that significantly reduces the number of trainable parameters, making it ideal for adapting large models to custom tasks—like improving conversational understanding—without the computational cost of full fine-tuning.

## Prerequisites

Before you begin, ensure you have the necessary libraries installed.

```bash
pip install loralib transformers datasets peft trl torch
```

## Step 1: Import Required Libraries and Set Up Logging

Start by importing the core libraries and configuring logging to monitor the training process.

```python
import logging
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
import loralib as lora

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## Step 2: Configure Hyperparameters

Define two configuration dictionaries: one for general training settings and one specifically for LoRA.

```python
# General training configuration
training_config = {
    "output_dir": "./phi3-lora-finetuned",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "warmup_steps": 100,
    "learning_rate": 2e-4,
    "logging_dir": "./logs",
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 500,
    "save_total_limit": 2,
    "fp16": True,  # Use mixed precision if supported
}

# LoRA-specific configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Causal language modeling
    r=16,                          # Rank of the low-rank matrices
    lora_alpha=32,                 # Scaling factor
    lora_dropout=0.1,              # Dropout probability
    target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA to
)
```

## Step 3: Load the Pre-trained Model and Tokenizer

Load the Phi-3 Mini model and its tokenizer. We'll configure the model to use efficient attention and mixed precision.

```python
model_name = "microsoft/Phi-3-mini-4k-instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Load model with appropriate settings
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    device_map="auto"
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

# Mark only LoRA parameters as trainable
lora.mark_only_lora_as_trainable(model)
model.print_trainable_parameters()  # Log the number of trainable parameters
```

## Step 4: Prepare Your Dataset

Load and preprocess your custom chat instruction dataset. The dataset should be formatted for conversational fine-tuning.

```python
# Load your dataset (replace with your dataset path)
dataset = load_dataset("json", data_files="path/to/your/chat_dataset.json")

# Tokenization function
def tokenize_function(examples):
    # Combine instruction and response into a single text
    texts = [f"Instruction: {inst}\nResponse: {resp}" for inst, resp in zip(examples["instruction"], examples["response"])]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split into train and evaluation sets
train_test_split = tokenized_dataset["train"].train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
```

## Step 5: Set Up the Data Collator and Training Arguments

Create a data collator for dynamic padding and define the training arguments.

```python
# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling, not masked LM
)

# Training arguments
training_args = TrainingArguments(
    **training_config
)
```

## Step 6: Initialize and Run the Trainer

Create a `Trainer` instance and start the fine-tuning process.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training
trainer.train()
```

## Step 7: Save the Fine-tuned Model

After training, save the LoRA adapters separately. This is efficient, as you only save the small set of trained LoRA weights, not the entire model.

```python
# Save only the LoRA state dict
lora_state_dict = lora.lora_state_dict(model)
torch.save(lora_state_dict, "./phi3-lora-adapters.pt")

# Optionally, save the entire PEFT model for easy reloading
model.save_pretrained("./phi3-lora-peft-model")
tokenizer.save_pretrained("./phi3-lora-peft-model")
```

## Step 8: Load and Use the Fine-tuned Model

To use your fine-tuned model, load the base model and then apply the saved LoRA adapters.

```python
# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load the LoRA adapters
base_model.load_state_dict(torch.load("./phi3-lora-adapters.pt"), strict=False)

# Or, load the saved PEFT model directly
from peft import PeftModel
fine_tuned_model = PeftModel.from_pretrained(base_model, "./phi3-lora-peft-model")

# Example inference
input_text = "Instruction: Explain how LoRA works.\nResponse:"
inputs = tokenizer(input_text, return_tensors="pt").to(fine_tuned_model.device)
outputs = fine_tuned_model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Summary

You have successfully fine-tuned the Phi-3 Mini model using LoRA. This approach allows you to adapt a powerful language model to your specific chat instruction dataset efficiently, leveraging a fraction of the parameters required for full fine-tuning. The saved LoRA adapters can be easily applied to the base model for inference, making deployment straightforward and resource-efficient.