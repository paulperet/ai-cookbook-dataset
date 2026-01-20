# Fine-tuning SmolVLM using direct preference optimization (DPO) with TRL on a consumer GPU

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

In this recipe, we’ll guide you through fine-tuning a **smol 🤏 Vision Language Model (VLM)** with **Direct Preference Optimization (DPO)** using the **Transformer Reinforcement Learning (TRL)** library to demonstrate how you can tailor VLMs to suit your specific needs, even when working with consumer-grade GPUs.

We’ll fine-tune [**SmolVLM**](https://huggingface.co/blog/smolvlm) using a **preference dataset** to help the model align with desired outputs. SmolVLM is a highly performant and memory-efficient model, making it an ideal choice for this task.  If you’re new to **Preference Optimization** for language or [vision-language models](https://huggingface.co/blog/vlms), check out [this blog](https://huggingface.co/blog/dpo_vlm) for an in-depth introduction.

The dataset we’ll use is [HuggingFaceH4/rlaif-v_formatted](https://huggingface.co/datasets/HuggingFaceH4/rlaif-v_formatted), which contains pairs of **`prompt + image`** along with a **`chosen`** and **`rejected`** answer for each pair. The goal of this fine-tuning process is to make the model consistently prefer the **chosen answers** from the dataset, reducing hallucinations.

This notebook has been tested using an **NVIDIA L4 GPU**.

## 1. Install Dependencies

Let’s start by installing the essential libraries we’ll need for fine-tuning! 🚀

```python
!pip install  -U -q transformers trl datasets bitsandbytes peft accelerate
# Tested with transformers==4.46.3, trl==0.12.2, datasets==3.2.0, bitsandbytes==0.45.0, peft==0.14.0, accelerate==1.2.0
```

```python
!pip install -q flash-attn --no-build-isolation
```

Authenticate with your Hugging Face account to save and share your model directly from this notebook 🗝️.

```python
from huggingface_hub import notebook_login

notebook_login()
```

## 2. Load Dataset 📁

We’ll work with the [HuggingFaceH4/rlaif-v_formatted](https://huggingface.co/datasets/HuggingFaceH4/rlaif-v_formatted) dataset, which provides pairs of **`prompt + image`** along with a **`chosen`** and **`rejected`** answers for each pair. This structured format is ideal for training models with **Direct Preference Optimization (DPO)**.

The dataset is already preformatted for this task. If you’re working with a custom dataset, you’ll need to preprocess it into the same format.

In this example, we'll use a subset of the dataset to demonstrate the process. However, in a real-world scenario, you should utilize the full dataset for better performance.

```python
from datasets import load_dataset

dataset_id = "HuggingFaceH4/rlaif-v_formatted"
train_dataset, test_dataset = load_dataset(dataset_id, split=['train[:6%]', 'test[:1%]'])
```

We will ensure all the images are RGB formatted:

```python
from PIL import Image

def ensure_rgb(example):
    # Convert the image to RGB if it's not already
    image = example['images'][0]
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        example['images'] = [image]
    return example

# Apply the transformation to the dataset
train_dataset = train_dataset.map(ensure_rgb, num_proc=32)
test_dataset = test_dataset.map(ensure_rgb, num_proc=32)
```

Let’s explore an example from the dataset to better understand its structure and the type of data we’re working with.

```python
train_dataset[20]
```

    {'chosen': [{'content': [{'text': "Yes, the grass in the image appears to be brown. This could indicate that the photo was taken during a dry season or in a region that experiences arid conditions. The brown grass contrasts with the grayish color of the elephant and provides a natural background that highlights the elephant's presence in its environment.",
         'type': 'text'}],
       'role': 'assistant'}],
     'rejected': [{'content': [{'text': 'Yes, the grass in the image appears to be brown. This could be due to a number of reasons such as the season (it might be a dry season), the type of grass, or the specific conditions of the environment where the photo was taken. The brown grass contrasts with the grayish color of the elephant and the white branches of the thorny tree, making it a prominent feature of the landscape.',
         'type': 'text'}],
       'role': 'assistant'}],
     'images': [<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x333>],
     'prompt': [{'content': [{'text': None, 'type': 'image'},
        {'text': 'Does the grass have brown color?', 'type': 'text'}],
       'role': 'user'}]}

```python
train_dataset[20]['images'][0]
```

## 3. Fine-Tune the Model using TRL

### 3.1 Load the Quantized Model for Training ⚙️

Let's first load a quantized version of the SmolVLM-Instruct model using bitsandbytes, and let's also load the processor. We'll use [SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct).

```python
import torch
from transformers import Idefics3ForConditionalGeneration, AutoProcessor

model_id = "HuggingFaceTB/SmolVLM-Instruct"
```

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

### 3.2 Set Up QLoRA and DPOConfig 🚀

In this step, we’ll configure [QLoRA](https://github.com/artidoro/qlora) for our training setup. **QLoRA** is a powerful fine-tuning technique designed to reduce the memory footprint, making it possible to fine-tune large models efficiently, even on limited hardware.

QLoRA builds upon traditional **LoRA** (Low-Rank Adaptation) by introducing quantization for the adapter weights. This enhancement leads to significantly lower memory usage and faster training, making it an ideal choice for resource-constrained environments.

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

    trainable params: 11,269,248 || all params: 2,257,542,128 || trainable%: 0.4992

Next, we will configure the training options using `DPOConfig`.

```python
from trl import DPOConfig

training_args = DPOConfig(
    output_dir="smolvlm-instruct-trl-dpo-rlaif-v",
    bf16=True,
    gradient_checkpointing=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    num_train_epochs=5,
    dataset_num_proc=8,  # tokenization will use 8 processes
    dataloader_num_workers=8,  # data loading will use 8 workers
    logging_steps=10,
    report_to="tensorboard",
    push_to_hub=True,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=1,
    eval_steps=10,  # Steps interval for evaluation
    eval_strategy="steps",
)
```

We will define the training arguments for **Direct Preference Optimization (DPO)** with the [DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer) class from the [TRL library](https://huggingface.co/docs/trl/index).

**DPO** uses labeled preference data to guide the model toward generating responses that align with preferences. TRL's [DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)  will **tokenize the dataset** before training and save it to disk. This process can consume significant disk space, depending on the amount of data used for training. Plan accordingly to avoid running out of storage.

This step may take a while, so feel free to relax and enjoy the process! 😄

```python
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    processing_class=processor,
)
```

Time to train the model! 🎉

```python
trainer.train()
```

[First Entry, ..., Last Entry]

Let's save the results 💾

```python
trainer.save_model(training_args.output_dir)
```

## 4. Testing the Fine-Tuned Model 🔍

With our Vision Language Model (VLM) fine-tuned, it’s time to evaluate its performance! In this section, we’ll test the model using examples from the [HuggingFaceH4/rlaif-v_formatted](https://huggingface.co/datasets/HuggingFaceH4/rlaif-v_formatted) dataset. Let’s dive into the results and assess how well the model aligns with the preferred responses! 🚀

Before we begin, let’s clean up the GPU memory to ensure smooth and optimal performance. 🧹

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

    GPU allocated memory: 1.64 GB
    GPU reserved memory: 2.01 GB

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
adapter_path = "sergiopaniego/smolvlm-instruct-trl-dpo-rlaif-v"
model.load_adapter(adapter_path)
```

Let's evaluate the model on an unseen sample.

```python
test_dataset[20]
```

    {'chosen': [{'content': [{'text': "In the image, there's a dynamic scene at what appears to be a beach or surfing location. The main focus is on a person skillfully riding a wave on a surfboard. This individual is dressed in a yellow shirt and seems to be enjoying the activity. In addition to the surfer, there are other elements in the scene such as waves breaking and creating white foam, indicating the active nature of the water. Nearby, there's another surfboard floating on the surface of the water, suggesting that more people might be participating in surfing or waiting for their turn. The overall atmosphere conveys a sense of fun, adventure, and connection with nature.",
         'type': 'text'}],
       'role': 'assistant'}],
     'rejected': [{'content': [{'text': 'In the image, there is a man enthusiastically surfing on a wave in the ocean. He is wearing a yellow shirt and short pants, adding a vibrant color contrast to the scene. The surfer is skillfully balancing himself on his surfboard as he rides the wave. Additionally, there are other surfers and surfboards scattered throughout the water, indicating that this might be a popular spot for surfing. The overall atmosphere suggests an active and fun beach day with people enjoying the watersport.',
         'type': 'text'}],
       'role': 'assistant'}],
     'images': [<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x428>],
     'prompt': [{'content': [{'text': None, 'type': 'image'},
        {'text': 'Provide an intricate description of every entity in the image.',
         'type': 'text'}],
       'role': 'user'}]}

```python
test_dataset[20]['images'][0]
```

Let’s create a common function that we can call with different samples to streamline the testing process. This function will allow us to evaluate the model’s performance on multiple examples efficiently without needing to rewrite code for each one. By using this reusable function, we can quickly assess how well the model performs across a variety of inputs.

```python
def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample['prompt'],
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

Now, we’re ready to call the function and evaluate the model! 🚀

```python
output = generate_text_from_sample(model, processor, test_dataset[20])
output
```

    " The image depicts a scene of a person surfing in the ocean. The central figure is a man standing on a surfboard, facing the camera with a joyful expression. He is wearing a bright green rash guard and dark-colored board shorts. The surfboard is white and blue, indicating it is a beginner-friendly board suitable for learning to surf.\n\nSurrounding the man, there are several other individuals in the water. Two people are visible in the background, swimming near the shore. The water is moderately choppy, with small waves breaking on the shore. The waves are white and foamy, indicating they are relatively small and gentle.\n\nIn the background, there is a distant coastline with a few buildings and structures visible. The sky is clear, with a few faint clouds visible. The overall setting appears to be a beach or coastal area, with the ocean and waves as the primary focus.\n\nThe image captures a moment of joy and accomplishment, as the surfer is enjoying the experience of surfing. The man's posture and expression suggest he is having a great time, and the overall scene conveys a sense of adventure and excitement.\n\n### Analysis and Description:\n1. **Surfer**: The man is the central figure in the image, standing on a surfboard and facing the camera. He is wearing a rash guard and board shorts, typical attire for surfing.\n2. **Surfboard**: The surfboard is white and blue, indicating it is a beginner-friendly board suitable for learning to surf.\n3. **Waves**: The waves are small and gentle, breaking on the shore.\n4. **Coastline**: The background features a distant coastline with a few buildings and structures visible.\n5. **Sky**: The sky is clear, with a few faint clouds visible.\n\n### Relevant Knowledge:\n- **Surfing**: Surfing is a water sport that involves standing on a surfboard and riding the waves. It is a popular activity in coastal areas, requiring balance, coordination, and a certain level of skill.\n- **Rash guard**: A rash