# Fine-Tuning Phi-3-Vision-128K-Instruct: A Step-by-Step Guide

## Overview

This guide walks you through fine-tuning the **Phi-3-Vision-128K-Instruct** model—a lightweight, state-of-the-art multimodal model from Microsoft's Phi-3 family. It supports a 128K token context window and processes both text and images. We'll use a sample dataset of Burberry products to demonstrate how to train the model to generate descriptive text from product images.

### Why Fine-Tune on Sample Data?
Working with sample data is essential for:
*   **Testing & Prototyping:** Validate your pipeline without risking sensitive production data.
*   **Performance Tuning:** Identify bottlenecks using data that mimics real-world scale.
*   **Learning & Exploration:** Safely experiment with model capabilities and training configurations.

## Prerequisites

Ensure you have the necessary Python libraries installed.

```bash
pip install torch transformers datasets peft accelerate bitsandbytes wandb
```

## Step 1: Prepare Your Dataset

We'll use the `DBQ/Burberry.Product.prices.United.States` dataset from Hugging Face, which contains 3,040 product entries with images, titles, categories, and prices. You can substitute any image-text dataset.

```python
from datasets import load_dataset

# Load the Burberry product dataset
dataset = load_dataset("DBQ/Burberry.Product.prices.United.States")
print(f"Dataset structure: {dataset}")
```

## Step 2: Understand the Model Architecture

Phi-3-Vision is a multimodal model. It processes images and text as a single, unified sequence:
1.  **Text** is tokenized into embeddings.
2.  **Images** are encoded using a CLIP vision encoder, and the resulting features are projected to match the text embedding dimension.
3.  A special `<|image_1|>` token in the text prompt is replaced by the image embeddings, allowing the model to understand the combined context.

Our training prompt will follow this structure:

```python
def format_prompt(row):
    text = f"<|user|>\n<|image_1|>What is shown in this image?<|end|><|assistant|>\nProduct: {row['title']}, Category: {row['category3_code']}, Full Price: {row['full_price']}<|end|>"
    return text
```

## Step 3: Configure the Training Setup

We'll use Parameter-Efficient Fine-Tuning (PEFT) via LoRA to adapt the model efficiently.

```python
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Define model ID
model_id = "microsoft/Phi-3-vision-128k-instruct"

# Load processor and model
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Configure LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply PEFT to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

## Step 4: Preprocess the Dataset

Create a preprocessing function that formats the data correctly for the model.

```python
def preprocess_function(examples):
    # Format the prompts
    prompts = [format_prompt(row) for row in examples]
    
    # Tokenize the text prompts
    model_inputs = processor(
        text=prompts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Process and encode the images
    images = [image.convert("RGB") for image in examples["image"]]
    image_inputs = processor(
        images=images,
        return_tensors="pt"
    )
    
    # Combine text and image inputs
    model_inputs["pixel_values"] = image_inputs["pixel_values"]
    
    # Create labels (same as input_ids for causal language modeling)
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    return model_inputs

# Apply preprocessing
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)
```

## Step 5: Set Up Training Arguments

Configure the training parameters. Adjust these based on your hardware and dataset size.

```python
training_args = TrainingArguments(
    output_dir="./phi3-vision-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    learning_rate=2e-4,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="wandb",  # Enable Weights & Biases logging
    run_name="phi3-vision-finetune-burberry"
)
```

## Step 6: Initialize and Run the Trainer

Create the SFTTrainer and start the fine-tuning process.

```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=processor,
    max_seq_length=512,
)

# Start training
trainer.train()
```

## Step 7: Save and Test the Fine-Tuned Model

After training completes, save your model and run a quick inference test.

```python
# Save the model
trainer.save_model("./phi3-vision-finetuned-final")

# Load the saved model for inference
model = AutoModelForCausalLM.from_pretrained(
    "./phi3-vision-finetuned-final",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Prepare a test sample
test_image = dataset["test"][0]["image"].convert("RGB")
prompt = "<|user|>\n<|image_1|>What is shown in this image?<|end|><|assistant|>\n"

# Generate a response
inputs = processor(
    text=prompt,
    images=test_image,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(outputs[0], skip_special_tokens=True)
print(f"Model response: {response}")
```

## Next Steps

1.  **Monitor Training:** Use Weights & Biases to track loss curves and evaluate model performance.
2.  **Experiment:** Adjust hyperparameters like learning rate, batch size, or LoRA rank to optimize results.
3.  **Deploy:** Integrate your fine-tuned model into an application using Hugging Face's `pipeline` API or ONNX runtime for production.

For a complete training script and advanced configuration examples, refer to the [Phi-3-Vision Training Script](../../code/03.Finetuning/Phi-3-vision-Trainingscript.py) and the [Weights & Biases example walkthrough](https://wandb.ai/byyoung3/mlnews3/reports/How-to-fine-tune-Phi-3-vision-on-a-custom-dataset--Vmlldzo4MTEzMTg7).