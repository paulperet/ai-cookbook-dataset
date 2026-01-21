# Fine-Tuning a Vision Language Model (Qwen2-VL-7B) with TRL

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

🚨 **Note**: This tutorial is computationally intensive and is designed to run on a GPU with substantial memory, such as an A100.

In this guide, you will learn how to fine-tune a Vision Language Model (VLM) using the Hugging Face ecosystem, specifically leveraging the Transformer Reinforcement Learning library (TRL). We will fine-tune the **Qwen2-VL-7B-Instruct** model on the **ChartQA** dataset to improve its visual question-answering capabilities.

## Prerequisites

Ensure you have the necessary libraries installed and are logged into your Hugging Face account.

### 1. Install Dependencies

```bash
pip install -U -q git+https://github.com/huggingface/trl.git bitsandbytes peft qwen-vl-utils trackio
```

### 2. Hugging Face Login

Authenticate with Hugging Face to push your fine-tuned model to the Hub.

```python
from huggingface_hub import notebook_login

notebook_login()
```

## Step 1: Load and Prepare the Dataset

We will use the **ChartQA** dataset, which contains images of charts paired with questions and answers.

### Define the System Prompt

First, create a system message to instruct the model on its role.

```python
system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""
```

### Format the Dataset

We need to structure the data into a chatbot format compatible with the model.

```python
def format_data(sample):
    return {
        "images": [sample["image"]],
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample['query']},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["label"][0]}],
            },
        ]
    }
```

### Load and Format the Data

Load a subset of the dataset for demonstration and apply the formatting function.

```python
from datasets import load_dataset

dataset_id = "HuggingFaceM4/ChartQA"
train_dataset, eval_dataset, test_dataset = load_dataset(
    dataset_id, split=['train[:10%]', 'val[:10%]', 'test[:10%]']
)

train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
test_dataset = [format_data(sample) for sample in test_dataset]
```

## Step 2: Evaluate the Base Model

Before fine-tuning, let's assess the base model's performance on a sample.

### Load the Model and Processor

```python
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

model_id = "Qwen/Qwen2-VL-7B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
processor = Qwen2VLProcessor.from_pretrained(model_id)
```

### Create an Inference Function

Define a helper function to generate answers from a formatted sample.

```python
from qwen_vl_utils import process_vision_info

def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample['messages'][1:2],  # Use the sample without the system message
        tokenize=False,
        add_generation_prompt=True
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample['messages'])

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
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

### Test the Base Model

Run inference on a sample to see the model's initial performance.

```python
sample_output = generate_text_from_sample(model, processor, train_dataset[0])
print(sample_output)
```

You will likely observe that while the model can process the image, its answer may be inaccurate or verbose, highlighting the need for fine-tuning.

### Clear GPU Memory

Before proceeding to training, clean up the memory.

```python
import gc
import time

def clear_memory():
    # Delete variables if they exist
    if 'model' in globals():
        del model
    if 'processor' in globals():
        del processor

    time.sleep(2)
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

## Step 3: Fine-Tune the Model with TRL

We will now fine-tune the model using QLoRA (Quantized Low-Rank Adaptation) for efficient training.

### 3.1 Load the Quantized Model

Load the model with 4-bit quantization to reduce memory usage.

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
processor = Qwen2VLProcessor.from_pretrained(model_id)
```

### 3.2 Configure QLoRA and Training Arguments

Set up the LoRA configuration and the supervised fine-tuning parameters.

```python
from peft import LoraConfig
from trl import SFTConfig

# LoRA Configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# Training Arguments
training_args = SFTConfig(
    output_dir="qwen2-7b-instruct-trl-sft-ChartQA",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=None,
    optim="adamw_torch_fused",
    learning_rate=2e-4,
    logging_steps=10,
    eval_steps=10,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    push_to_hub=True,
    report_to="trackio",
)
```

### 3.3 Initialize Training Tracking

Set up trackio to log training metrics.

```python
import trackio

trackio.init(
    project="qwen2-7b-instruct-trl-sft-ChartQA",
    name="qwen2-7b-instruct-trl-sft-ChartQA",
    config=training_args,
    space_id=training_args.output_dir + "-trackio"
)
```

### 3.4 Create the SFT Trainer

The `SFTTrainer` handles the fine-tuning process, automatically applying the correct data collator for vision-language models.

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

### 3.5 Start Training

Begin the fine-tuning process.

```python
trainer.train()
```

### 3.6 Save the Model

Once training is complete, save the fine-tuned model.

```python
trainer.save_model(training_args.output_dir)
```

## Step 4: Test the Fine-Tuned Model

Now, let's evaluate the performance of your fine-tuned model.

### 4.1 Clear Memory and Reload the Base Model

First, clean the GPU memory and reload the base model for a fair comparison.

```python
clear_memory()

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
processor = Qwen2VLProcessor.from_pretrained(model_id)
```

### 4.2 Load the Fine-Tuned Adapters

Load the PeftModel with the trained adapters.

```python
from peft import PeftModel

model = PeftModel.from_pretrained(model, training_args.output_dir)
```

### 4.3 Run Inference on Test Samples

Use the same `generate_text_from_sample` function to test the fine-tuned model.

```python
fine_tuned_output = generate_text_from_sample(model, processor, test_dataset[0])
print(fine_tuned_output)
```

Compare the output with the base model's result. You should see more accurate and concise answers, demonstrating the effectiveness of the fine-tuning process.

## Conclusion

You have successfully fine-tuned a Vision Language Model using TRL and QLoRA on the ChartQA dataset. This process enhances the model's ability to answer questions based on visual data from charts. You can now push your model to the Hugging Face Hub and integrate it into your applications.

**Next Steps:**
- Experiment with different hyperparameters (learning rate, batch size, LoRA rank).
- Try fine-tuning on other multimodal datasets.
- Deploy your model using Hugging Face's Inference Endpoints or a similar service.