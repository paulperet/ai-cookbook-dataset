# Fine-tuning an LLM to Generate Persian Product Catalogs in JSON Format

_Authored by: [Mohammadreza Esmaeiliyan](https://github.com/MrzEsma)_

This guide demonstrates how to fine-tune a large language model (LLM) to generate structured Persian product catalogs in JSON format. The model is optimized for use on consumer-grade GPUs and is particularly effective for extracting structured information from unstructured product titles and descriptions found on Iranian platforms like Basalam, Divar, and Digikala.

You can find a model fine-tuned using this method on our [Hugging Face account](https://huggingface.co/BaSalam/Llama2-7b-entity-attr-v1). For fast inference, we'll also showcase using the `vllm` engine.

## Prerequisites

First, install the required libraries.

```bash
pip install torch transformers datasets peft trl bitsandbytes accelerate
```

## Step 1: Import Libraries and Set Hyperparameters

Begin by importing the necessary modules and defining your configuration.

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# General parameters
model_name = "NousResearch/Llama-2-7b-chat-hf"
dataset_name = "BaSalam/entity-attribute-dataset-GPT-3.5-generated-v1"
new_model = "llama-persian-catalog-generator"

# LoRA parameters
lora_r = 64
lora_alpha = lora_r * 2
lora_dropout = 0.1
target_modules = ["q_proj", "v_proj", 'k_proj']

# QLoRA parameters
load_in_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
bnb_4bit_use_double_quant = False

# TrainingArguments parameters
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
learning_rate = 0.00015
weight_decay = 0.01
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 25

# SFT parameters
max_seq_length = None
packing = False
device_map = {"": 0}

# Dataset formatting parameters
use_special_template = True
response_template = ' ### Answer:'
instruction_prompt_template = '"### Human:"'
use_llama_like_model = True
```

**Explanation:**
- **LoRA (Low-Rank Adaptation):** This method fine-tunes a model by adding small, trainable matrices to specific layers (here, the query, key, and value projections). It's highly parameter-efficient.
- **QLoRA (Quantized LoRA):** This technique loads the base model in 4-bit precision to drastically reduce memory usage, enabling fine-tuning on smaller GPUs without sacrificing performance.
- **Training Arguments:** These control the training loop, including batch size, learning rate, and scheduler.

## Step 2: Load and Prepare the Dataset

Load your instruction dataset and split it into training and validation sets.

```python
# Load dataset
dataset = load_dataset(dataset_name, split="train")

# Keep only the 'instruction' and 'output' columns
other_columns = [i for i in dataset.column_names if i not in ['instruction', 'output']]
dataset = dataset.remove_columns(other_columns)

# Split the dataset
percent_of_train_dataset = 0.95
split_dataset = dataset.train_test_split(
    train_size=int(dataset.num_rows * percent_of_train_dataset),
    seed=19,
    shuffle=False
)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Size of the train set: {len(train_dataset)}. Size of the validation set: {len(eval_dataset)}")
```

## Step 3: Configure LoRA and QLoRA

Set up the PEFT (Parameter-Efficient Fine-Tuning) configuration.

```python
# Load LoRA configuration
peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules
)

# Load QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=load_in_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
)
```

## Step 4: Load the Base Model and Tokenizer

Load the base LLM with 4-bit quantization and configure the tokenizer.

```python
# Load base model with QLoRA config
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False  # Disable cache for gradient checkpointing

# Load and configure tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Prevents overflow issues with fp16 training

# Set a default chat template if one isn't provided
if not tokenizer.chat_template:
    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
```

## Step 5: Define Prompt Formatting Functions

Create functions to format your instruction data for training. The choice of formatting is crucial for teaching the model the desired input-output structure.

```python
def special_formatting_prompts(example):
    """Formats prompts using a custom template."""
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"{instruction_prompt_template}{example['instruction'][i]}\n{response_template} {example['output'][i]}"
        output_texts.append(text)
    return output_texts

def normal_formatting_prompts(example):
    """Formats prompts using the model's native chat template."""
    output_texts = []
    for i in range(len(example['instruction'])):
        chat_temp = [
            {"role": "system", "content": example['instruction'][i]},
            {"role": "assistant", "content": example['output'][i]}
        ]
        text = tokenizer.apply_chat_template(chat_temp, tokenize=False)
        output_texts.append(text)
    return output_texts

# Choose the formatting function and set up the data collator
if use_special_template:
    formatting_func = special_formatting_prompts
    if use_llama_like_model:
        # Encode the response template for the collator
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template_ids, tokenizer=tokenizer)
    else:
        collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
else:
    formatting_func = normal_formatting_prompts
```

The `DataCollatorForCompletionOnlyLM` ensures the loss is calculated only on the model's response tokens, not the instruction tokens.

## Step 6: Configure Training Arguments and Initialize the Trainer

Set up the training arguments and create the `SFTTrainer`.

```python
# Set training parameters
training_arguments = TrainingArguments(
    output_dir=new_model,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    gradient_checkpointing=gradient_checkpointing,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type
)

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=formatting_func,
    data_collator=collator,
    max_seq_length=max_seq_length,
    processing_class=tokenizer,
    args=training_arguments,
    packing=packing
)
```

## Step 7: Train the Model and Save the Adapter

Start the fine-tuning process and save the resulting LoRA adapter.

```python
# Train model
trainer.train()

# Save the fine-tuned LoRA adapter
trainer.model.save_pretrained(new_model)
```

The adapter weights are saved separately from the base model, making them lightweight and easy to share.

## Step 8: Run Inference with the Fine-Tuned Model

Now, let's test the model on a sample from the validation set.

First, define a helper function to clear GPU memory and a generation function.

```python
import torch
import gc

def clear_hardwares():
    """Clears GPU memory and cache."""
    torch.clear_autocast_cache()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    gc.collect()

clear_hardwares()

def generate(model, prompt: str, kwargs):
    """Generates text from a given prompt."""
    tokenized_prompt = tokenizer(prompt, return_tensors='pt').to(model.device)
    prompt_length = len(tokenized_prompt.get('input_ids')[0])

    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**tokenized_prompt, **kwargs) if kwargs else model.generate(**tokenized_prompt)
        output = tokenizer.decode(output_tokens[0][prompt_length:], skip_special_tokens=True)
    return output
```

Load the fine-tuned adapter and run inference.

```python
# Load the base model and adapter
base_model = AutoModelForCausalLM.from_pretrained(new_model, return_dict=True, device_map='auto', token='')
tokenizer = AutoTokenizer.from_pretrained(new_model, max_length=max_seq_length)
model = PeftModel.from_pretrained(base_model, new_model)
del base_model  # Free up memory

# Prepare a prompt from the evaluation dataset
sample = eval_dataset[0]
if use_special_template:
    prompt = f"{instruction_prompt_template}{sample['instruction']}\n{response_template}"
else:
    chat_temp = [{"role": "system", "content": sample['instruction']}]
    prompt = tokenizer.apply_chat_template(chat_temp, tokenize=False, add_generation_prompt=True)

# Generate a response
gen_kwargs = {"max_new_tokens": 1024}
generated_texts = generate(model=model, prompt=prompt, kwargs=gen_kwargs)
print(generated_texts)
```

## Step 9: Merge the Adapter with the Base Model (Optional)

To create a single, standalone model, you can merge the LoRA adapter with the base model and upload it to the Hugging Face Hub.

```python
clear_hardwares()
merged_model = model.merge_and_unload()
clear_hardwares()
del model

adapter_model_name = 'your_hf_account/your_desired_name'
merged_model.push_to_hub(adapter_model_name)
```

Alternatively, you can push only the lightweight adapter and load it dynamically later:

```python
# Push only the adapter
model.push_to_hub(adapter_model_name)

# Later, load it like this:
from peft import PeftConfig
config = PeftConfig.from_pretrained(adapter_model_name)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    load_in_8bit=True,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, adapter_model_name)
```

## Step 10: Fast Inference with vLLM

For production use, consider using a high-performance inference engine like `vllm`. First, install it:

```bash
pip install vllm
```

Here's how to run inference with a model hosted on Hugging Face.

```python
from vllm import LLM, SamplingParams

# Define your prompt and templates
prompt = """### Question: here is a product title from a Iranian marketplace.
         give me the Product Entity and Attributes of this product in Persian language.
         give the output in this json format: {'attributes': {'attribute_name' : <attribute value>, ...}, 'product_entity': '<product entity>'}.
         Don't make assumptions about what values to plug into json. Just give Json not a single word more.

product title:"""
user_prompt_template = '### Question: '
response_template = ' ### Answer:'

# Initialize the vLLM engine
llm = LLM(model='BaSalam/Llama2-7b-entity-attr-v1', gpu_memory_utilization=0.9, trust_remote_code=True)

# Your product title
product = 'مانتو اسپرت پانیذ قد جلوی کار حدودا 85 سانتی متر قد پشت کار حدودا 88 سانتی متر'

# Create the full prompt
full_prompt = prompt + product + response_template

# Generate the response
sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)
outputs = llm.generate([full_prompt], sampling_params)
generated_text = outputs[0].outputs[0].text
print(generated_text)
```

This will output a structured JSON containing the product entity and its attributes, extracted from the Persian product title.

## Conclusion

You have successfully fine-tuned a Llama 2 model using QLoRA and LoRA to generate structured Persian product catalogs. The process is memory-efficient, suitable for consumer hardware, and produces a model capable of extracting JSON from unstructured text. You can further adapt this pipeline for other languages or structured output tasks.