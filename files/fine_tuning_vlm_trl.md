# Fine-Tuning a Vision Language Model (Qwen2-VL-7B) with the Hugging Face Ecosystem (TRL)

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

🚨 **WARNING**: This notebook is resource-intensive and requires substantial computational power. If you’re running this in Colab, it will utilize an A100 GPU.

In this recipe, we’ll demonstrate how to fine-tune a [Vision Language Model (VLM)](https://huggingface.co/blog/vlms) using the Hugging Face ecosystem, specifically with the [Transformer Reinforcement Learning library (TRL)](https://huggingface.co/docs/trl/index).

**🌟 Model & Dataset Overview**

We’ll be fine-tuning the [Qwen2-VL-7B](https://qwenlm.github.io/blog/qwen2-vl/) model on the [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) dataset. This dataset includes images of various chart types paired with question-answer pairs—ideal for enhancing the model's visual question-answering capabilities.

**📖 Additional Resources**

If you’re interested in more VLM applications, check out:
- [Multimodal Retrieval-Augmented Generation (RAG) Recipe](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_vlms): where I guide you through building a RAG system using Document Retrieval (ColPali) and Vision Language Models (VLMs).
- [Phil Schmid's tutorial](https://www.philschmid.de/fine-tune-multimodal-llms-with-trl): an excellent deep dive into fine-tuning multimodal LLMs with TRL.
- [Merve Noyan's **smol-vision** repository](https://github.com/merveenoyan/smol-vision/tree/main): a collection of engaging notebooks on cutting-edge vision and multimodal AI topics.

## 1. Install Dependencies

Let’s start by installing the essential libraries we’ll need for fine-tuning! 🚀

```python
!pip install -U -q git+https://github.com/huggingface/trl.git bitsandbytes peft qwen-vl-utils trackio
# Tested with trl==0.22.0.dev0, bitsandbytes==0.47.0, peft==0.17.1, qwen-vl-utils==0.0.11, trackio==0.2.8
```

Log in to Hugging Face to upload your fine-tuned model! 🗝️

You’ll need to authenticate with your Hugging Face account to save and share your model directly from this notebook.

```python
from huggingface_hub import notebook_login

notebook_login()
```

## 2. Load Dataset 📁

In this section, we’ll load the [HuggingFaceM4/ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) dataset. This dataset contains chart images paired with related questions and answers, making it ideal for training on visual question answering tasks.

Next, we’ll generate a system message for the VLM. In this case, we want to create a system that acts as an expert in analyzing chart images and providing concise answers to questions based on them.

```python
system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""
```

We’ll format the dataset into a chatbot structure for interaction. Each interaction will consist of a system message, followed by the image and the user's query, and finally, the answer to the query.

💡For more usage tips specific to this model, check out the [Model Card](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct#more-usage-tips).

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

For educational purposes, we’ll load only 10% of each split in the dataset. However, in a real-world use case, you would typically load the entire set of samples.

```python
from datasets import load_dataset

dataset_id = "HuggingFaceM4/ChartQA"
train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=['train[:10%]', 'val[:10%]', 'test[:10%]'])
```

Let’s take a look at the structure of the dataset. It includes an image, a query, a label (which is the answer), and a fourth feature that we’ll be discarding.

```python
train_dataset
```

Now, let’s format the data using the chatbot structure. This will allow us to set up the interactions appropriately for our model.

```python
train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
test_dataset = [format_data(sample) for sample in test_dataset]
```

```python
train_dataset[200]
```

## 3. Load Model and Check Performance! 🤔

Now that we’ve loaded the dataset, let’s start by loading the model and evaluating its performance using a sample from the dataset. We’ll be using [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), a Vision Language Model (VLM) capable of understanding both visual data and text.

If you're exploring alternatives, consider these open-source options:
- Meta AI's [Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
- Mistral AI's [Pixtral-12B](https://huggingface.co/mistralai/Pixtral-12B-2409)
- Allen AI's [Molmo-7B-D-0924](https://huggingface.co/allenai/Molmo-7B-D-0924)

Additionally, you can check the Leaderboards, such as the [WildVision Arena](https://huggingface.co/spaces/WildVision/vision-arena) or the [OpenVLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard), to find the best-performing VLMs.

```python
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

model_id = "Qwen/Qwen2-VL-7B-Instruct"
```

Next, we’ll load the model and the tokenizer to prepare for inference.

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

processor = Qwen2VLProcessor.from_pretrained(model_id)
```

To evaluate the model's performance, we’ll use a sample from the dataset. First, let’s take a look at the internal structure of this sample.

```python
train_dataset[0]
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
# Example of how to call the method with sample:
output = generate_text_from_sample(model, processor, train_dataset[0])
output
```

While the model successfully retrieves the correct visual information, it struggles to answer the question accurately. This indicates that fine-tuning might be the key to enhancing its performance. Let’s proceed with the fine-tuning process!

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
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
processor = Qwen2VLProcessor.from_pretrained(model_id)
```

### 4.2 Set Up QLoRA and SFTConfig 🚀

Next, we will configure [QLoRA](https://github.com/artidoro/qlora) for our training setup. QLoRA enables efficient fine-tuning of large language models while significantly reducing the memory footprint compared to traditional methods. Unlike standard LoRA, which reduces memory usage by applying a low-rank approximation, QLoRA takes it a step further by quantizing the weights of the LoRA adapters. This leads to even lower memory requirements and improved training efficiency, making it an excellent choice for optimizing our model's performance without sacrificing quality.

```python
from peft import LoraConfig

# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
```

We will use Supervised Fine-Tuning (SFT) to refine our model’s performance on the task at hand. To do this, we'll define the training arguments using the [SFTConfig](https://huggingface.co/docs/trl/sft_trainer) class from the [TRL library](https://huggingface.co/docs/trl/index). SFT allows us to provide labeled data, helping the model learn to generate more accurate responses based on the input it receives. This approach ensures that the model is tailored to our specific use case, leading to better performance in understanding and responding to visual queries.

```python
from trl import SFTConfig

# Configure training arguments
training_args = SFTConfig(
    output_dir="qwen2-7b-instruct-trl-sft-ChartQA",  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=4,  # Batch size for evaluation
    gradient_accumulation_steps=8,  # Steps to accumulate gradients
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    max_length=None,
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-4,  # Learning rate for training
    # Logging and evaluation
    logging_steps=10,  # Steps interval for logging
    eval_steps=10,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=20,  # Steps interval for saving
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    push_to_hub=True,  # Whether to push model to Hugging Face Hub
    report_to="trackio",  # Reporting tool for tracking metrics
)
```

### 4.3 Training the Model 🏃

We will log our training progress using [trackio](https://huggingface.co/blog/trackio). Let’s connect our notebook to W&B to capture essential information during training.

```python
import trackio

trackio.init(
    project="qwen2-7b-instruct-trl-sft-ChartQA",
    name="qwen2-7b-instruct-trl-sft-ChartQA",
    config=training_args,
    space_id=training_args.output_dir + "-trackio"
)
```

Now, we will define the [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer), which is a wrapper around the [transformers.Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) class and inherits its attributes and methods. This class simplifies the fine-tuning process by properly initializing the [PeftModel](https://huggingface.co/docs/peft/v0.6.0/package_reference/peft_model) when a [PeftConfig](https://huggingface.co/docs/peft/v0.6.0/en/package_reference/config#peft.PeftConfig) object is provided. By using `SFTTrainer`, we can efficiently manage the training workflow and ensure a smooth fine-tuning experience for our Vision Language Model. When doing inference we defined our own `generate_text_from_sample` function which applied the necessary preprocessing before passing the inputs to the model. Here, the SFTTrainer infers automatically that the model is a vision-language model and applies a `DataCollatorForVisionLanguageModeling` which convers the inputs to the appropriate format.

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

Let's save the results 💾

```python
trainer.save_model(training_args.output_dir)
```

## 5. Testing the Fine-Tuned Model 🔍

Now that we've successfully fine-tuned our Vision Language Model (VLM), it's time to evaluate its performance! In this section, we will test the model using examples from the ChartQA dataset to see how well it answers questions based on chart images. Let's dive in and explore the results! 🚀

Let's clean up the GPU memory to ensure optimal performance 🧹

```python
clear_memory()
```

We will reload the base model using the same pipeline as before.

```python
model = Qwen2VLForCondition