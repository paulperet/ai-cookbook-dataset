# Fine-tuning SmolVLM with TRL on a Consumer GPU

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

## Overview

This guide demonstrates how to fine-tune a small, efficient Vision Language Model (VLM) using the Hugging Face ecosystem and the Transformer Reinforcement Learning (TRL) library. You will customize the **SmolVLM-Instruct** model on the **ChartQA** dataset to enhance its visual question-answering (VQA) capabilities, all on a consumer-grade GPU.

### Prerequisites

This tutorial assumes you have access to a GPU (tested on an L4). Ensure you have sufficient disk space and a Hugging Face account to push your fine-tuned model.

## 1. Setup and Installation

Begin by installing the required libraries.

```bash
pip install -U -q git+https://github.com/huggingface/trl.git bitsandbytes peft qwen-vl-utils trackio
pip install -q flash-attn --no-build-isolation
```

Authenticate with the Hugging Face Hub to save and share your model.

```python
from huggingface_hub import notebook_login

notebook_login()
```

## 2. Load and Prepare the Dataset

You will use the **ChartQA** dataset, which contains chart images paired with questions and answers.

### 2.1 Define the System Prompt

First, create a system message to instruct the model on its role.

```python
system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""
```

### 2.2 Format the Data into a Chat Structure

Define a function to format each dataset sample into the chat structure expected by the model.

```python
def format_data(sample):
    return {
      "images": [sample["image"]],
      "messages": [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample['query'],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample["label"][0]
                }
            ],
        },
        ]
    }
```

### 2.3 Load and Process the Dataset

Load a subset (10%) of the dataset for demonstration. In a production setting, you would use the full dataset.

```python
from datasets import load_dataset

dataset_id = "HuggingFaceM4/ChartQA"
train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=['train[:10%]', 'val[:10%]', 'test[:10%]'])
```

Apply the formatting function to all splits.

```python
train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
test_dataset = [format_data(sample) for sample in test_dataset]
```

Your data is now structured as a list of conversations, each containing system, user (with image), and assistant messages.

## 3. Evaluate the Base Model

Before fine-tuning, let's assess the base model's performance to establish a baseline.

### 3.1 Load the Model and Processor

```python
import torch
from transformers import Idefics3ForConditionalGeneration, AutoProcessor

model_id = "HuggingFaceTB/SmolVLM-Instruct"

model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(model_id)
```

### 3.2 Create an Inference Function

Define a helper function to generate answers from a formatted sample.

```python
def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample['messages'][1:2],  # Use the sample without the system message
        add_generation_prompt=True
    )

    image_inputs = []
    image = sample['images'][0]
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_inputs.append([image])

    # Prepare the inputs for the model
    model_inputs = processor(
        text=text_input,
        images=image_inputs,
        return_tensors="pt",
    ).to(device)

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]
```

### 3.3 Test the Base Model

Run inference on a sample to see the model's initial performance.

```python
output = generate_text_from_sample(model, processor, train_dataset[1])
print(output)
```

The base model may produce incorrect or irrelevant answers, highlighting the need for fine-tuning.

### 3.4 Clear GPU Memory

Before starting training, clean up the GPU memory.

```python
import gc
import time

def clear_memory():
    # Delete variables if they exist in the current global scope
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    if 'bnb_config' in globals(): del globals()['bnb_config']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

clear_memory()
```

## 4. Fine-Tune the Model with TRL and QLoRA

You will now fine-tune the model using Supervised Fine-Tuning (SFT) with QLoRA for memory efficiency.

### 4.1 Load the Quantized Model

Load the model with 4-bit quantization using `bitsandbytes`.

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    _attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_id)
```

### 4.2 Configure QLoRA with PEFT

Set up the LoRA configuration for Parameter-Efficient Fine-Tuning (PEFT).

```python
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
    use_dora=True,
    init_lora_weights="gaussian"
)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
```

### 4.3 Configure the SFT Trainer

Define the training arguments using `SFTConfig` from TRL.

```python
from trl import SFTConfig

training_args = SFTConfig(
    output_dir="smolvlm-instruct-trl-sft-ChartQA",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=25,
    save_total_limit=1,
    optim="adamw_torch_fused",
    bf16=True,
    push_to_hub=True,
    report_to="none",
    max_length=None
)
```

Initialize tracking (optional).

```python
import trackio

trackio.init(
    project="smolvlm-instruct-trl-sft-ChartQA",
    name="smolvlm-instruct-trl-sft-ChartQA",
    config=training_args,
    space_id=training_args.output_dir + "-trackio"
)
```

### 4.4 Initialize the SFTTrainer

Create the trainer object with your model, dataset, and configurations.

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=processor,
)
```

### 4.5 Start Training

Begin the fine-tuning process.

```python
trainer.train()
```

### 4.6 Save the Model

Save the fine-tuned model and adapter weights.

```python
trainer.save_model(training_args.output_dir)
```

## 5. Evaluate the Fine-Tuned Model

Now, test your fine-tuned model on unseen data to evaluate its improvement.

### 5.1 Clear Memory and Reload the Base Model

First, free up GPU resources and reload the original base model.

```python
clear_memory()

model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_id)
```

### 5.2 Load the Fine-Tuned Adapter

Attach the saved adapter (LoRA weights) to the base model.

```python
adapter_path = "smolvlm-instruct-trl-sft-ChartQA"  # Use your saved path or Hugging Face repo
model.load_adapter(adapter_path)
```

### 5.3 Run Inference on a Test Sample

Evaluate the model on a sample from the test set.

```python
# Inspect the test sample
print(test_dataset[20]['messages'][:2])

# Generate the model's answer
output = generate_text_from_sample(model, processor, test_dataset[20])
print("Model Output:", output)
```

The fine-tuned model should now provide accurate, concise answers aligned with the dataset, demonstrating successful adaptation.

## Conclusion

You have successfully fine-tuned the SmolVLM-Instruct model on the ChartQA dataset using TRL and QLoRA. This process enhances the model's VQA capabilities while maintaining efficiency on consumer hardware. You can now push your adapter to the Hugging Face Hub and integrate it into applications.

For further exploration, you can test the model in a live demo via this [Hugging Face Space](https://huggingface.co/spaces/sergiopaniego/SmolVLM-trl-sft-ChartQA) and compare it with the [pre-trained model](https://huggingface.co/spaces/sergiopaniego/SmolVLM-Instruct).