# Fine-Tuning a Vision Language Model with TRL using MPO

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

In this recipe, we'll demonstrate how to fine-tune a [Vision Language Model (VLM)](https://huggingface.co/blog/vlms-2025) using Mixed Preference Optimization (MPO) with the Transformer Reinforcement Learning (TRL) library.

MPO is a training approach that combines multiple optimization objectives and was introduced in the paper [Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization](https://huggingface.co/papers/2411.10442). It is part of the [Direct Preference Optimization](https://huggingface.co/papers/2305.18290) \([DPO](https://huggingface.co/docs/trl/main/en/dpo_trainer)\) trainer and works by combining multiple loss functions with different weights, enabling more sophisticated optimization strategies.

We'll fine-tune [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), a small VLM with strong performance, using a preference dataset to help the model align with desired outputs. Check out [this blog post](https://huggingface.co/blog/dpo_vlm) to learn more about preference optimization for vision-language models.

The dataset we'll use is [HuggingFaceH4/rlaif-v_formatted](https://huggingface.co/datasets/HuggingFaceH4/rlaif-v_formatted), a specially formatted version of the [RLAIF-V dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset). This dataset contains pairs of `prompt + image`, along with a `chosen` and `rejected` response for each sample. The final goal of the fine-tuning process is to train a model that consistently prefers the `chosen` answers over the `rejected` ones, thereby reducing hallucinations. To achieve this, multiple loss functions will be used in combination.

## 1. Install Dependencies

Let's start by installing the required dependencies.  
We'll install `trl` from source, as the MPO trainer hasn't been included in an official release at the time of writing.

```python
!pip install -U -q git+https://github.com/huggingface/trl.git bitsandbytes qwen-vl-utils==0.0.8
```

We'll authenticate with the Hugging Face Hub using our account to upload and save the fine-tuned model.  
You can generate your access token [here](https://huggingface.co/settings/tokens).

```python
from huggingface_hub import notebook_login

notebook_login()
```

## 2. Load Dataset

For this recipe, we'll use [HuggingFaceH4/rlaif-v_formatted](https://huggingface.co/datasets/HuggingFaceH4/rlaif-v_formatted), a specially formatted version of the [RLAIF-V dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset).

In the [paper](https://internvl.github.io/blog/2024-11-14-InternVL-2.0-MPO/) that introduced MPO, the authors also presented [OpenGVLab/MMPR](https://huggingface.co/datasets/OpenGVLab/MMPR), a large-scale multimodal preference dataset built through an efficient pipeline that combines both samples with and without clear ground truths.

For our educational case, we'll use `HuggingFaceH4/rlaif-v_formatted`. However, for best reproduction of the paper's results, we recommend exploring MMPR.  
We'll work with a subset of the dataset for this example.

```python
from datasets import load_dataset

dataset_id = "HuggingFaceH4/rlaif-v_formatted"
train_dataset, test_dataset = load_dataset(dataset_id, split=["train[:5%]", "test[:1%]"])
```

Let's include a quick check to ensure the images are in RGB format. If not, we'll convert them accordingly.

```python
from PIL import Image

def ensure_rgb(example):
    # Convert the image to RGB if it's not already
    image = example["images"][0]
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        example["images"] = [image]
    return example


# Apply the transformation to the dataset (change num_proc depending on the available compute)
train_dataset = train_dataset.map(ensure_rgb, num_proc=8)
test_dataset = test_dataset.map(ensure_rgb, num_proc=8)
```

Let's inspect a sample to understand its structure.  
As we can see, each sample contains a `chosen`, `rejected`, `image`, and `prompt`.  
Our goal is to fine-tune the model to prefer the `chosen` answers using MPO.

```python
train_dataset[5]
```

Let's check the image for that particular sample:

```python
train_dataset[5]['images'][0]
```

## 3. Fine-Tune the Model with TRL using MPO

As previously described, we'll leverage `trl`, since this library provides everything we need to train using MPO while abstracting away some of the complexity we don't need to handle for this particular case.

The MPO trainer accepts a list of `loss_type`s. A full list of available loss functions is provided in the DPO trainer documentation [here](https://huggingface.co/docs/trl/main/en/dpo_trainer#loss-functions).  
As mentioned earlier, MPO is a particular case of the DPO trainer, so we can use it by specifying a list of loss types and their corresponding weights.

In the image below, you can see the improvements reported in the MPO paper for the InternVL2-8B model using this training strategy.

### 3.1 Load the Quantized Model for Training

Let's load the model. In this example, we'll use [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), a compact Vision Language Model (VLM) with strong performance.

With the original MPO paper, the authors released a [collection](https://huggingface.co/collections/OpenGVLab/internvl25-mpo-6753fed98cd828219b12f849) of checkpoints fine-tuned with this technique for [InternVL2.5](https://huggingface.co/collections/OpenGVLab/internvl25-673e1019b66e2218f68d7c1c), another high-performing VLM.

We chose Qwen2.5-VL-3B-Instruct for its straightforward integration with the `transformers` library, although InternVL2.5 is the original model used in the paper.

```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
```

```python
from transformers import BitsAndBytesConfig

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
```

### 3.2 Set Up QLoRA

Now that we have the model and processor loaded, let's set up QLoRA and the DPOConfig, where we will set up the losses list and its corresponding weights.  
These configurations enable efficient fine-tuning and optimization tailored for our training objectives.

```python
from peft import LoraConfig, get_peft_model

# Configure LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=["down_proj", "o_proj", "k_proj", "q_proj", "gate_proj", "up_proj", "v_proj"],
    use_dora=True,
    init_lora_weights="gaussian",
)

# Apply PEFT model adaptation
peft_model = get_peft_model(model, peft_config)

# Print trainable parameters
peft_model.print_trainable_parameters()
```

### 3.3 MPO via `DPOConfig`

To configure MPO training using the `DPOConfig`, simply provide a list of loss types using the `loss_type` parameter. This can be passed as either a Python list or a comma-separated string. Optionally, you can specify a corresponding list of `loss_weights` to control the relative importance of each loss during optimization. If omitted, all losses default to a weight of `1.0`.

For example, following the setup described in the original MPO paper, you can define:

`loss_type = ["sigmoid", "bco_pair", "sft"]`

`loss_weights = [0.8, 0.2, 1.0]`

This corresponds to:

> *MPO is defined as a combination of the preference loss (L_p), the quality loss (L_q), and the generation loss (L_g).*

The selected `loss_type` are:

- `"sigmoid"`: Sigmoid loss from the original [DPO](https://huggingface.co/papers/2305.18290) paper.
- `"bco_pair"`: Pairwise BCO loss from the [BCO](https://huggingface.co/papers/2404.04656) paper.
- `"sft"`: Negative log-likelihood loss (standard supervised fine-tuning loss).

For more details on each available loss type and how they affect training, refer to the [official documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer#loss-functions).

All other configuration options follow the standard [`DPOConfig`](https://huggingface.co/docs/trl/main/en/dpo_trainer) format and can be adjusted based on your available compute resources.

```python
from trl import DPOConfig

training_args = DPOConfig(
    output_dir="Qwen2.5-VL-3B-Instruct-trl-mpo-rlaif-v",

    loss_type=["sigmoid", "bco_pair", "sft"], # Loss types to combine, as used in the MPO paper
    loss_weights=[0.8, 0.2, 1.0],  # Corresponding weights, as used in the MPO paper

    bf16=False,
    gradient_checkpointing=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    dataset_num_proc=1,  # tokenization will use 1 processes
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

As we can see, setting MPO vs DPO is straightforward and only requires two additional parameters in the `DPOConfig`. Finally, we can initialize the `DPOTrainer` and start training the model.

```python
from trl import DPOTrainer

trainer = DPOTrainer(
    model=peft_model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=processor,
)
```

```python
trainer.train()
```

[Training Loss: 2.622700, ..., Training Loss: 2.568000]

## 4. Testing the Fine-Tuned Model

We have fine-tuned our model using MPO. Now, let's evaluate its performance on a sample to see how it behaves in practice.

```python
trained_model_id = "sergiopaniego/Qwen2.5-VL-3B-Instruct-trl-mpo-rlaif-v"
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
```

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
trained_model = PeftModel.from_pretrained(base_model, trained_model_id).eval()

trained_processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
```

```python
test_dataset[0]
```

```python
test_dataset[0]["images"][0]
```

```python
from qwen_vl_utils import process_vision_info

def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    model.gradient_checkpointing_disable()
    model.config.use_cache = True

    # Prepare the text input by applying the chat template
    sample["prompt"][0]["content"][0]["image"] = sample["images"][0]
    text_input = processor.apply_chat_template(sample["prompt"], add_generation_prompt=True)

    image_inputs, _ = process_vision_info(sample["prompt"])
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]
```

We'll generate outputs from both the pretrained and fine-tuned models to highlight their differences.  
An interesting extension would be to compare the MPO output with the same model fine-tuned using only DPO.  
We'll leave that experiment for you to explore!

```python
pretrained_output = generate_text_from_sample(model, processor, test_dataset[0])
print('\n\n>>> Pretrained model output:\n\n')
print(pretrained_output)
trained_output = generate_text_from_sample(trained_model, trained_processor, test_dataset[0])
print('\n\n>>> Fine tuned model output:\n\n')
print(trained_output)
```

Looking at the outputs, we can already observe clear stylistic differences in the model's responses after training.  
The MPO fine-tuning is now complete!

## 5. Continue Your Learning Journey 🧑‍🎓️

This is not the end of your learning journey! If you enjoyed this content and want to dive deeper into MPO, `trl`, or Vision-Language Models, check out the following resources:

- [Preference Optimization for Vision Language Models with TRL](https://huggingface.co/blog/dpo_vlm)
- [Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization](https://internvl.github.io/blog/2024-11-14-InternVL-2.0-MPO/)
- [MPO in the TRL documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer)
- [Vision Language Models (Better, Faster, Stronger)](https://huggingface.co/blog/vlms-2025)
- [Explore more multimodal recipes in the Hugging Face Open-Source AI Cookbook](https://huggingface.co/learn/cookbook)