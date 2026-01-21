# Fine-Tuning SmolVLM with Direct Preference Optimization (DPO) on a Consumer GPU

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

## Overview

This guide walks you through fine-tuning a **SmolVLM** (a small Vision Language Model) using **Direct Preference Optimization (DPO)** with the **TRL** library. DPO is a powerful technique for aligning model outputs with human preferences, and we'll demonstrate how to apply it efficiently, even on a consumer-grade GPU like an NVIDIA L4.

We'll use the [HuggingFaceH4/rlaif-v_formatted](https://huggingface.co/datasets/HuggingFaceH4/rlaif-v_formatted) dataset, which contains `(prompt + image)` pairs, each with a **chosen** (preferred) and a **rejected** answer. The goal is to teach the model to prefer the chosen responses, thereby reducing hallucinations and improving alignment.

## Prerequisites & Setup

Before starting, ensure you have the necessary libraries installed.

### 1. Install Dependencies

Run the following commands to install the required packages:

```bash
pip install -U -q transformers trl datasets bitsandbytes peft accelerate
pip install -q flash-attn --no-build-isolation
```

> **Note:** This guide was tested with `transformers==4.46.3`, `trl==0.12.2`, `datasets==3.2.0`, `bitsandbytes==0.45.0`, `peft==0.14.0`, and `accelerate==1.2.0`.

### 2. Authenticate with Hugging Face

To push your fine-tuned model to the Hub, authenticate using:

```python
from huggingface_hub import notebook_login

notebook_login()
```

## Step 1: Load and Prepare the Dataset

We'll load the preference dataset and ensure all images are in RGB format.

### Load the Dataset

```python
from datasets import load_dataset

dataset_id = "HuggingFaceH4/rlaif-v_formatted"
train_dataset, test_dataset = load_dataset(dataset_id, split=['train[:6%]', 'test[:1%]'])
```

> **Tip:** In a production setting, use the full dataset for better results. Here we use a small subset for demonstration.

### Convert Images to RGB

```python
from PIL import Image

def ensure_rgb(example):
    image = example['images'][0]
    if isinstance(image, Image.Image) and image.mode != 'RGB':
        image = image.convert('RGB')
        example['images'] = [image]
    return example

# Apply conversion in parallel
train_dataset = train_dataset.map(ensure_rgb, num_proc=32)
test_dataset = test_dataset.map(ensure_rgb, num_proc=32)
```

### Inspect a Sample

Let's examine one entry to understand the dataset structure:

```python
train_dataset[20]
```

The output shows a dictionary with keys:
- `prompt`: The user input (image + text question).
- `chosen`: The preferred assistant response.
- `rejected`: The dispreferred assistant response.
- `images`: A list containing the PIL image.

## Step 2: Configure the Model and QLoRA

We'll load a quantized version of [SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) using bitsandbytes (4-bit quantization) and set up QLoRA for efficient fine-tuning.

### Load the Quantized Model and Processor

```python
import torch
from transformers import Idefics3ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

model_id = "HuggingFaceTB/SmolVLM-Instruct"

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and processor
model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    _attn_implementation="flash_attention_2",  # Use Flash Attention for speed
)
processor = AutoProcessor.from_pretrained(model_id)
```

### Apply QLoRA Configuration

QLoRA (Quantized Low-Rank Adaptation) drastically reduces memory usage by quantizing the adapter weights, making fine-tuning feasible on consumer hardware.

```python
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
    use_dora=True,  # Weight-Decomposed Low-Rank Adaptation
    init_lora_weights="gaussian"
)

# Wrap the model with PEFT
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
```

You should see an output like:
```
trainable params: 11,269,248 || all params: 2,257,542,128 || trainable%: 0.4992
```

Only about 0.5% of parameters are trainable, thanks to QLoRA.

## Step 3: Configure DPO Training

We'll use the `DPOTrainer` from TRL. First, define the training arguments.

### Set Training Arguments

```python
from trl import DPOConfig

training_args = DPOConfig(
    output_dir="smolvlm-instruct-trl-dpo-rlaif-v",
    bf16=True,
    gradient_checkpointing=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,  # Effective batch size = 32
    num_train_epochs=5,
    dataset_num_proc=8,
    dataloader_num_workers=8,
    logging_steps=10,
    report_to="tensorboard",
    push_to_hub=True,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=1,
    eval_steps=10,
    eval_strategy="steps",
)
```

### Initialize the DPOTrainer

The `DPOTrainer` handles tokenization, preference data formatting, and the DPO loss calculation.

```python
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # DPO uses the base model as the reference implicitly
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    processing_class=processor,
)
```

> **Note:** Tokenizing the dataset may take a while and consume significant disk space.

## Step 4: Train the Model

Start the fine-tuning process:

```python
trainer.train()
```

Once training completes, save the model:

```python
trainer.save_model(training_args.output_dir)
```

## Step 5: Evaluate the Fine-Tuned Model

Let's test the model on an unseen example from the test set.

### Clear GPU Memory

First, free up memory to ensure smooth inference.

```python
import gc
import time
import torch

def clear_memory():
    # Delete large objects
    for var in ['inputs', 'model', 'processor', 'trainer', 'peft_model', 'bnb_config']:
        if var in globals():
            del globals()[var]
    time.sleep(2)

    # Garbage collection and CUDA cache clearing
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

clear_memory()
```

### Reload the Base Model and Attach the Adapter

We'll load the base model again and attach the fine-tuned adapter.

```python
# Reload base model
model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_id)

# Load the fine-tuned adapter (replace with your path)
adapter_path = "sergiopaniego/smolvlm-instruct-trl-dpo-rlaif-v"  # Your Hub path or local directory
model.load_adapter(adapter_path)
```

### Create a Helper Function for Inference

This function processes a sample and generates a response.

```python
def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Apply chat template to the prompt
    text_input = processor.apply_chat_template(
        sample['prompt'],
        add_generation_prompt=True
    )

    # Prepare image (ensure RGB)
    image = sample['images'][0]
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_inputs = [[image]]

    # Tokenize inputs
    model_inputs = processor(
        text=text_input,
        images=image_inputs,
        return_tensors="pt",
    ).to(device)

    # Generate
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim input IDs from the output
    trimmed_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode
    output_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text
```

### Run Inference on a Test Sample

Let's examine a test sample and generate a response.

```python
# Inspect the sample
test_sample = test_dataset[20]
print("Prompt:", test_sample['prompt'])
print("\nChosen answer (preferred):", test_sample['chosen'][0]['content'][0]['text'])
print("\nRejected answer:", test_sample['rejected'][0]['content'][0]['text'])
```

Now, generate the model's response:

```python
output = generate_text_from_sample(model, processor, test_sample)
print("Model's generated response:")
print(output)
```

The model should produce a detailed description of the image, ideally closer in quality to the "chosen" answer from the dataset.

## Conclusion

You have successfully fine-tuned a SmolVLM using Direct Preference Optimization with TRL. This process teaches the model to align its outputs with human preferences, leveraging a preference dataset and efficient QLoRA fine-tuning.

**Next Steps:**
- Experiment with the full dataset for better performance.
- Try different hyperparameters (e.g., `per_device_train_batch_size`, `gradient_accumulation_steps`) to balance speed and memory.
- Apply this pipeline to your own custom preference datasets.

For a deeper dive into DPO for vision-language models, refer to the [accompanying blog post](https://huggingface.co/blog/dpo_vlm).