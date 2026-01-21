# Fine-tuning Microsoft Phi-3 with QLoRA: A Practical Guide

This guide walks you through fine-tuning Microsoft's compact **Phi-3 Mini** language model using **QLoRA (Quantized Low-Rank Adaptation)**. QLoRA enables efficient adaptation of large models by fine-tuning quantized weights with low-rank adapters, dramatically reducing memory requirements while maintaining performance. You'll learn how to set up the environment, prepare your model, and run a fine-tuning session to improve the model's conversational understanding and response generation.

## Prerequisites

Before you begin, ensure you have the necessary libraries installed. You'll need the latest versions of `accelerate`, `transformers`, and `bitsandbytes` to support 4-bit loading and QLoRA training.

```bash
pip install -q -U accelerate transformers
pip install -q -U bitsandbytes
pip install -q -U datasets peft trl
```

## Step 1: Import Required Libraries

Begin by importing all necessary modules. This includes components from Hugging Face for model handling, dataset management, and the PEFT (Parameter-Efficient Fine-Tuning) library for QLoRA.

```python
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTTrainer
```

## Step 2: Configure Model and Training Arguments

Define a configuration class to centralize your model, data, and training parameters. This makes your code modular and easy to adjust.

```python
@dataclass
class ScriptArguments:
    # Model arguments
    model_name: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct")
    # Data arguments
    dataset_name: Optional[str] = field(default="timdettmers/openassistant-guanaco")
    # QLoRA arguments
    use_4bit: bool = field(default=True, metadata={"help": "Activate 4-bit precision base model loading"})
    bnb_4bit_compute_dtype: str = field(default="float16", metadata={"help": "Compute dtype for 4-bit base models"})
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "Quantization type (fp4 or nf4)"})
    use_nested_quant: bool = field(default=False, metadata={"help": "Activate nested quantization for 4-bit base models"})
    # LoRA configuration
    lora_r: int = field(default=64, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=16, metadata={"help": "Alpha parameter for LoRA scaling"})
    lora_dropout: float = field(default=0.1, metadata={"help": "Dropout probability for LoRA layers"})
    # SFT parameters
    max_seq_length: int = field(default=512, metadata={"help": "Maximum sequence length to use"})
    packing: bool = field(default=False, metadata={"help": "Pack multiple short examples in the same input sequence"})
```

## Step 3: Load the Tokenizer and Configure 4-bit Quantization

Load the tokenizer for your chosen model and set up the BitsAndBytes configuration. This step prepares the model to be loaded in 4-bit precision, which is essential for QLoRA's memory efficiency.

```python
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=script_args.use_4bit,
    bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=getattr(torch, script_args.bnb_4bit_compute_dtype),
    bnb_4bit_use_double_quant=script_args.use_nested_quant,
)
```

## Step 4: Load the Base Model in 4-bit Precision

Load the Phi-3 model with the 4-bit configuration you just defined. This significantly reduces the model's memory footprint, enabling fine-tuning on consumer-grade hardware.

```python
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1
```

## Step 5: Configure LoRA Adapters

Define the LoRA configuration. LoRA adds a small number of trainable parameters (adapters) to the model while keeping the base model weights frozen. The `r` value controls the rank of the adapter matrices.

```python
peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
```

## Step 6: Load and Prepare the Dataset

Load your fine-tuning dataset. This example uses the `openassistant-guanaco` dataset, but you can replace it with your own instruction or conversation data.

```python
dataset = load_dataset(script_args.dataset_name, split="train")
```

## Step 7: Define Training Arguments

Set the hyperparameters for the training process. These control aspects like learning rate, batch size, and checkpointing.

```python
training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=1,
    max_steps=100,
    fp16=True,
    push_to_hub=False,
    report_to="none",
)
```

## Step 8: Initialize the SFTTrainer

Create the `SFTTrainer` object, which orchestrates the supervised fine-tuning process. It handles dataset formatting, model training with LoRA, and logging.

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
)
```

## Step 9: Train the Model

Start the fine-tuning process. The trainer will update only the LoRA adapter weights, keeping the original 4-bit quantized model parameters frozen.

```python
trainer.train()
```

## Step 10: Save the Fine-Tuned Model

Once training is complete, save the adapter weights. This creates a small, efficient checkpoint containing only the fine-tuned LoRA parameters, which can be merged with the base model later for inference.

```python
output_dir = os.path.join("./results", "final_checkpoint")
trainer.model.save_pretrained(output_dir)
```

## Next Steps

You have successfully fine-tuned the Phi-3 Mini model using QLoRA. To use your fine-tuned model:

1.  **Load for Inference:** Use the `PeftModel` class to load your saved adapter and merge it with the base Phi-3 model.
2.  **Push to Hub:** Optionally, push your adapter to the Hugging Face Hub to share or deploy it.
3.  **Experiment Further:** Try different datasets, adjust LoRA rank (`lora_r`), or modify training hyperparameters to optimize performance for your specific use case.

This efficient fine-tuning approach allows you to adapt powerful, compact models like Phi-3 to specialized tasks without requiring extensive computational resources.