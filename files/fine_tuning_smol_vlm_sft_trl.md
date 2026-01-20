# Fine-tuning SmolVLM with TRL on a consumer GPU

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

In this recipe, we’ll demonstrate how to fine-tune a smol 🤏 [Vision Language Model (VLM)](https://huggingface.co/blog/vlms) using the Hugging Face ecosystem, leveraging the powerful [Transformer Reinforcement Learning library (TRL)](https://huggingface.co/docs/trl/index). This step-by-step guide will enable you to customize VLMs for your specific tasks, even on consumer GPUs.

### 🌟 Model & Dataset Overview

In this notebook, we will fine-tune the **[SmolVLM](https://huggingface.co/blog/smolvlm)** model using the **[ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)** dataset. SmolVLM is a highly performant and memory-efficient model, making it an ideal choice for this task. The **ChartQA dataset** contains images of various chart types paired with question-answer pairs, offering a valuable resource for enhancing the model's **visual question-answering (VQA)** capabilities. These skills are crucial for a range of practical applications, including data analysis, business intelligence, and educational tools.

💡 _Note:_ The instruct model we are fine-tuning has already been trained on this dataset, so it is familiar with the data. However, this serves as a valuable educational exercise for understanding fine-tuning techniques. For a complete list of datasets used to train this model, check out [this document](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/blob/main/smolvlm-data.pdf).

### 📖 Additional Resources

Expand your knowledge of Vision Language Models and related tools with these resources:

- **[Multimodal Recipes in Cookbook](https://huggingface.co/learn/cookbook/index):** Explore practical recipes for multimodal models, including RAG pipelines and fine-tuning. We already have [a recipe for fine-tuning a VLM with TRL](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl), so refer to it for more details.
- **[TRL Community Tutorials](https://huggingface.co/docs/trl/main/en/community_tutorials):** A treasure trove of tutorials to deepen your understanding of TRL and its applications.

With these resources, you’ll be equipped to dive deeper into the world of VLMs and push the boundaries of what they can achieve!

This notebook is tested using a L4 GPU.

## 1. Install Dependencies

Let’s start by installing the essential libraries we’ll need for fine-tuning! 🚀

```python
!pip install -U -q git+https://github.com/huggingface/trl.git bitsandbytes peft qwen-vl-utils trackio
# Tested with trl==0.22.0.dev0, bitsandbytes==0.47.0, peft==0.17.1, qwen-vl-utils==0.0.11, trackio==0.2.8
```

[First Entry, ..., Last Entry]

```python
!pip install -q flash-attn --no-build-isolation
```

[First Entry, ..., Last Entry]

Authenticate with your Hugging Face account to save and share your model directly from this notebook 🗝️.

```python
from huggingface_hub import notebook_login

notebook_login()
```

## 2. Load Dataset 📁

We’ll load the [HuggingFaceM4/ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) dataset, which provides chart images along with corresponding questions and answers—perfect for fine-tuning visual question-answering models.

We’ll create a system message to make the VLM act as a chart analysis expert, giving concise answers about chart images.

```python
system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""
```

We’ll format the dataset into a chatbot structure, with the system message, image, user query, and answer for each interaction.

💡For more tips on using this model, check out the [Model Card](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct).

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

For educational purposes, we’ll load only 10% of each split in the dataset. In a real-world scenario, you would load the entire dataset.

```python
from datasets import load_dataset

dataset_id = "HuggingFaceM4/ChartQA"
train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=['train[:10%]', 'val[:10%]', 'test[:10%]'])
```

[First Entry, ..., Last Entry]

Let’s take a look at the dataset structure. It includes an image, a query, a label (the answer), and a fourth feature that we’ll be discarding.

```python
train_dataset
```

Now, let’s format the data using the chatbot structure. This will set up the interactions for the model.

```python
train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
test_dataset = [format_data(sample) for sample in test_dataset]
```

```python
train_dataset[200]
```

## 3. Load Model and Check Performance! 🤔

Now that we’ve loaded the dataset, it’s time to load the [HuggingFaceTB/SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct), a 2B parameter Vision Language Model (VLM) that offers state-of-the-art (SOTA) performance while being efficient in terms of memory usage.

```python
import torch
from transformers import Idefics3ForConditionalGeneration, AutoProcessor

model_id = "HuggingFaceTB/SmolVLM-Instruct"
```

Next, we’ll load the model and the tokenizer to prepare for inference.

```python
model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(model_id)
```

[First Entry, ..., Last Entry]

To evaluate the model's performance, we’ll use a sample from the dataset. First, let’s inspect the internal structure of this sample to understand how the data is organized.

```python
train_dataset[1]
```

We’ll use the sample without the system message to assess the VLM's raw understanding. Here’s the input we will use:

```python
train_dataset[0]['messages'][1:2]
```

Now, let’s take a look at the chart corresponding to the sample. Can you answer the query based on the visual information?

```python
train_dataset[0]['images'][0]
```

Let’s create a method that takes the model, processor, and sample as inputs to generate the model's answer. This will allow us to streamline the inference process and easily evaluate the VLM's performance.

```python
def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample['messages'][1:2],  # Use the sample without the system message
        add_generation_prompt=True
    )

    image_inputs = []
    image = sample['images'][0]
    #image = sample[1]['content'][0]['image']
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_inputs.append([image])

    # Prepare the inputs for the model
    model_inputs = processor(
        #text=[text_input],
        text=text_input,
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device

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

    return output_text[0]  # Return the first decoded output text
```

```python
output = generate_text_from_sample(model, processor, train_dataset[1])
output
```

It seems like the model is referencing the wrong line, causing it to fail. To improve its performance, we can fine-tune the model with more relevant data to ensure it better understands the context and provides more accurate responses.

**Remove Model and Clean GPU**

Before we proceed with training the model in the next section, let's clear the current variables and clean the GPU to free up resources.

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

## 4. Fine-Tune the Model using TRL

### 4.1 Load the Quantized Model for Training ⚙️

Next, we’ll load the quantized model using [bitsandbytes](https://huggingface.co/docs/bitsandbytes/main/en/index). If you want to learn more about quantization, check out [this blog post](https://huggingface.co/blog/merve/quantization) or [this one](https://www.maartengrootendorst.com/blog/quantization/).

```python
from transformers import BitsAndBytesConfig

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    _attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(model_id)
```

### 4.2 Set Up QLoRA and SFTConfig 🚀

Next, we’ll configure [QLoRA](https://github.com/artidoro/qlora) for our training setup. QLoRA allows efficient fine-tuning of large models by reducing the memory footprint. Unlike traditional LoRA, which uses low-rank approximation, QLoRA further quantizes the LoRA adapter weights, leading to even lower memory usage and faster training.

To boost efficiency, we can also leverage a **paged optimizer** or **8-bit optimizer** during QLoRA implementation. This approach enhances memory efficiency and speeds up computations, making it ideal for optimizing our model without sacrificing performance.

```python
from peft import LoraConfig, get_peft_model

# Configure LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
    use_dora=True,
    init_lora_weights="gaussian"
)

# Apply PEFT model adaptation
peft_model = get_peft_model(model, peft_config)

# Print trainable parameters
peft_model.print_trainable_parameters()
```

We will use Supervised Fine-Tuning (SFT) to improve our model's performance on the specific task. To achieve this, we'll define the training arguments with the [SFTConfig](https://huggingface.co/docs/trl/sft_trainer) class from the [TRL library](https://huggingface.co/docs/trl/index). SFT leverages labeled data to help the model generate more accurate responses, adapting it to the task. This approach enhances the model's ability to understand and respond to visual queries more effectively.

```python
from trl import SFTConfig

# Configure training arguments using SFTConfig
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

```python
import trackio

trackio.init(
    project="smolvlm-instruct-trl-sft-ChartQA",
    name="smolvlm-instruct-trl-sft-ChartQA",
    config=training_args,
    space_id=training_args.output_dir + "-trackio"
)
```

### 4.3 Training the Model 🏃

We will define the [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer), which is a wrapper around the [transformers.Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) class and inherits its attributes and methods. This class simplifies the fine-tuning process by properly initializing the [PeftModel](https://huggingface.co/docs/peft/v0.6.0/package_reference/peft_model) when a [PeftConfig](https://huggingface.co/docs/peft/v0.6.0/en/package_reference/config#peft.PeftConfig) object is provided. By using `SFTTrainer`, we can efficiently manage the training workflow and ensure a smooth fine-tuning experience for our Vision Language Model.

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

Time to Train the Model! 🎉

```python
trainer.train()
```

[First Entry, ..., Last Entry]

Let's save the results 💾

```python
trainer.save_model(training_args.output_dir)
```

## 5. Testing the Fine-Tuned Model 🔍

Now that our Vision Language Model (VLM) is fine-tuned, it's time to evaluate its performance! In this section, we'll test the model using examples from the ChartQA dataset to assess how accurately it answers questions based on chart images. Let's dive into the results and see how well it performs! 🚀

Let's clean up the GPU memory to ensure optimal performance 🧹

```python
clear_memory()
```

We will reload the base model using the same pipeline as before.

```python
model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(model_id)
```

We will attach the trained adapter to the pretrained model. This adapter contains the fine-tuning adjustments made during training, enabling the base model to leverage the new knowledge while keeping its core parameters intact. By integrating the adapter, we enhance the model's capabilities without altering its original structure.

```python
adapter_path = "sergiopaniego/smolvlm-instruct-trl-sft-ChartQA"
model.load_adapter(adapter_path)
```

Let's evaluate the model on an unseen sample.

```python
test_dataset[20]['messages'][:2]
```

```python
test_dataset[20]['images'][0]
```

```python
output = generate_text_from_sample(model, processor, test_dataset[20])
output
```

The model has successfully learned to respond to the queries as specified in the dataset. We've achieved our goal! 🎉✨

💻 I’ve developed an example application to test the model, which you can find [here](https://huggingface.co/spaces/sergiopaniego/SmolVLM-trl-sft-ChartQA). You can easily compare it with another Space featuring the pre-trained model, available [here](https://hugging