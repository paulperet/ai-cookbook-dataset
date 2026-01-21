# Fine-tuning Granite Vision 3.1 2B with TRL

_Authored by: [Eli Schwartz](https://huggingface.co/elischwartz)_

_Adapted from [Sergio Paniego](https://github.com/sergiopaniego)'s [Notebook](https://huggingface.co/learn/cookbook/en/fine_tuning_smol_vlm_sft_trl)_

## Overview

This guide will walk you through fine-tuning IBM's **Granite Vision 3.1 2B Model**, a lightweight yet capable Vision Language Model (VLM). We will adapt the model for a specific visual reasoning task using the Hugging Face ecosystem, specifically the **Transformer Reinforcement Learning library (TRL)**. This process is designed to be feasible on consumer-grade GPUs.

### Model & Dataset

*   **Model:** [IBM Granite Vision 3.1 2B Preview](https://huggingface.co/ibm-granite/granite-vision-3.1-2b-preview) - A 2B parameter model fine-tuned for vision and language tasks.
*   **Dataset:** [Geometric Perception](https://huggingface.co/datasets/euclid-multimodal/Geoperception) - Contains images of geometric diagrams from textbooks paired with question-answer pairs. We will focus on the `LineComparison` task.

**Tested Hardware:** A100 GPU.

---

## 1. Setup and Installation

Begin by installing the required libraries.

```bash
pip install -q git+https://github.com/huggingface/transformers.git
pip install -U -q trl datasets bitsandbytes peft accelerate
# Tested with transformers==4.49.0.dev0, trl==0.14.0, datasets==3.2.0, bitsandbytes==0.45.2, peft==0.14.0, accelerate==1.3.0
```

Optionally, install FlashAttention for faster training (requires a compatible GPU and environment).

```bash
pip install -q flash-attn --no-build-isolation
```

---

## 2. Load and Prepare the Dataset

### 2.1 Load the Dataset

We'll load the Geometric Perception dataset and filter it for the `LineComparison` task.

```python
from datasets import load_dataset

system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

dataset_id = "euclid-multimodal/Geoperception"
dataset = load_dataset(dataset_id)

# Filter for the specific task and split the data
dataset_LineComparison = dataset['train'].filter(lambda x: x['predicate'] == 'LineComparison')
train_test = dataset_LineComparison.train_test_split(test_size=0.5, seed=42)

print(train_test)
```

### 2.2 Format Data for Chat

The model expects a specific chat format. We'll structure each sample as a conversation with system, user (containing image and text), and assistant roles.

```python
def format_data(sample):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample['question']},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["answer"]}],
        },
    ]

# Apply formatting
train_dataset = [format_data(x) for x in train_test['train']]
test_dataset = [format_data(x) for x in train_test['test']]

# Inspect a sample
print(train_dataset[200])
```

---

## 3. Load the Model and Evaluate Baseline Performance

Before fine-tuning, let's load the model and assess its initial performance on our task.

### 3.1 Load Model and Processor

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

model_id = "ibm-granite/granite-vision-3.1-2b-preview"

# Check for FlashAttention
try:
    import flash_attn
    USE_FLASH_ATTENTION = True
except ImportError:
    USE_FLASH_ATTENTION = False

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if USE_FLASH_ATTENTION else None,
)

processor = AutoProcessor.from_pretrained(model_id)
```

### 3.2 Create an Inference Function

We'll create a helper function to generate answers from our formatted samples.

```python
def generate_text_from_sample(model, processor, sample, max_new_tokens=100, device="cuda"):
    # Prepare the text input by applying the chat template (exclude assistant response)
    text_input = processor.apply_chat_template(
        sample[:2],
        add_generation_prompt=True
    )

    # Prepare the image
    image_inputs = []
    image = sample[1]['content'][0]['image']
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_inputs.append([image])

    # Prepare model inputs
    model_inputs = processor(
        text=text_input,
        images=image_inputs,
        return_tensors="pt",
    ).to(device)

    # Generate text
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim input IDs from the output
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]
```

### 3.3 Run a Baseline Evaluation

Let's test the model on a sample from our test set.

```python
test_idx = 20
sample = test_dataset[test_idx]
print("Sample:", sample)

output = generate_text_from_sample(model, processor, sample)
print("Model Output:", output)
print("Expected Answer:", sample[2]['content'][0]['text'])
```

**Expected Observation:** The base model will likely fail to correctly compare line lengths from the image, indicating a need for task-specific fine-tuning.

### 3.4 Clear Memory for Training

Before starting the training process, clean up the GPU memory.

```python
import gc
import time

def clear_memory():
    # Delete major variables
    for var in ['model', 'processor']:
        if var in globals():
            del globals()[var]

    time.sleep(2)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

clear_memory()
```

---

## 4. Fine-Tune the Model with TRL

We will use **QLoRA (Quantized Low-Rank Adaptation)** for efficient fine-tuning, which significantly reduces memory usage.

### 4.1 Load the Quantized Model

```python
from transformers import BitsAndBytesConfig

USE_QLORA = True
USE_LORA = True

if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["vision_tower", "lm_head"],  # Skip problematic modules
        llm_int8_enable_fp32_cpu_offload=True
    )
else:
    bnb_config = None

# Reload the model with quantization
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    _attn_implementation="flash_attention_2" if USE_FLASH_ATTENTION else None,
)
processor = AutoProcessor.from_pretrained(model_id)
```

### 4.2 Configure QLoRA Adapters

We'll apply LoRA adapters to specific modules within the model's language component.

```python
if USE_LORA:
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=[name for name, _ in model.named_modules() if 'language_model' in name and '_proj' in name],
        use_dora=True,
        init_lora_weights="gaussian"
    )

    model.add_adapter(peft_config)
    model.enable_adapters()
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    model.print_trainable_parameters()
else:
    peft_config = None
```

### 4.3 Set Up the SFT Trainer

We'll use the `SFTTrainer` from TRL for supervised fine-tuning.

```python
from trl import SFTConfig

training_args = SFTConfig(
    output_dir="./checkpoints/geoperception",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_steps=10,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=1,
    optim="adamw_torch_fused",
    bf16=True,
    push_to_hub=False,
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
)
```

### 4.4 Define a Data Collator

The collator function prepares batches for the trainer, handling images and text tokenization.

```python
def collate_fn(examples):
    # Prepare text inputs using the chat template
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]

    # Prepare image inputs
    image_inputs = []
    for example in examples:
        image = example[1]['content'][0]['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_inputs.append([image])

    # Process the batch
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    # Prepare labels (we will mask the loss on the prompt tokens)
    labels = batch["input_ids"].clone()
    # ... (label masking logic would continue here based on the assistant token)

    return {**batch, "labels": labels}
```

### 4.5 Initialize and Run the Trainer

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

# Start training
trainer.train()
```

### 4.6 Save the Fine-Tuned Model

After training completes, save your adapter weights.

```python
model.save_pretrained("./fine_tuned_granite_vision")
processor.save_pretrained("./fine_tuned_granite_vision")
```

---

## Next Steps

You have successfully fine-tuned the Granite Vision model on a geometric reasoning task. To use your model:

1.  Load the saved model and processor:
    ```python
    from peft import PeftModel
    base_model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, "./fine_tuned_granite_vision")
    processor = AutoProcessor.from_pretrained("./fine_tuned_granite_vision")
    ```
2.  Use the `generate_text_from_sample` function from **Section 3.2** to run inference on new data.
3.  Evaluate the model's performance on the held-out test set to measure improvement.

For more advanced techniques like Reinforcement Learning from Human Feedback (RLHF), refer to the [TRL documentation](https://huggingface.co/docs/trl).