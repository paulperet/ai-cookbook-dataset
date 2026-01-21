# Fine-Tuning a VLM for Object Detection Grounding Using TRL

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

> 🚨 **WARNING**: This guide is resource-intensive and requires substantial computational power. If you're running it in Colab, it will utilize an **A100 GPU**.

**🔍 What You'll Learn**

In this tutorial, you will learn how to fine-tune a [Vision-Language Model (VLM)](https://huggingface.co/blog/vlms-2025) for **object detection grounding** using [TRL](https://huggingface.co/docs/trl/en/index).

Traditional object detection identifies a predefined set of classes (e.g., "car", "person") within an image. Models like [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base) introduced **open-ended object detection**, allowing models to detect *any* class described in natural language. Grounding adds contextual understanding, enabling the model to locate objects based on descriptions like "the red car behind the tree."

We will fine-tune [PaliGemma 2](https://huggingface.co/blog/paligemma2), a VLM from Google with built-in object detection capabilities, using the [RefCOCO](https://paperswithcode.com/dataset/refcoco) dataset designed for referring expression comprehension.

This guide builds upon the [VLM Object Understanding Space](https://huggingface.co/spaces/sergiopaniego/vlm_object_understanding), which compares different VLMs on object understanding tasks.

## Prerequisites

Ensure you have the necessary libraries installed.

```bash
pip install -Uq transformers datasets trl supervision albumentations
```

You will also need a Hugging Face account and an access token to access gated models and save checkpoints. Log in using the following cell.

```python
from huggingface_hub import notebook_login

notebook_login()
```

## Step 1: Load and Prepare the Dataset

We'll use a subset of the RefCOCO dataset, which includes grounded object detection annotations.

### 1.1 Load the Dataset

```python
from datasets import load_dataset

refcoco_dataset = load_dataset("jxu124/refcoco", split='train[:5%]')
```

The dataset contains columns like `bbox` (bounding boxes in `xyxy` format) and `captions`. The images are not directly loaded; we'll fetch them from Flickr URLs provided in the `raw_image_info` column.

### 1.2 Download and Add Images

We'll download images from Flickr URLs, handling potential failures.

```python
import json
import requests
from PIL import Image
from io import BytesIO

def add_image(example):
    try:
        raw_info = json.loads(example['raw_image_info'])
        url = raw_info.get('flickr_url', None)
        if url:
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            example['image'] = image
        else:
            example['image'] = None
    except Exception as e:
        print(f"Error loading image: {e}")
        example['image'] = None
    return example

refcoco_dataset_with_images = refcoco_dataset.map(add_image, desc="Adding image from flickr", num_proc=16)
```

### 1.3 Filter and Clean the Dataset

First, remove samples where image download failed.

```python
filtered_dataset = refcoco_dataset_with_images.filter(
    lambda example: example['image'] is not None,
    desc="Removing failed image downloads"
)
```

Next, keep only the essential columns: `'bbox'`, `'captions'`, and `'image'`.

```python
filtered_dataset = filtered_dataset.remove_columns([
    'sent_ids', 'file_name', 'ann_id', 'ref_id', 'image_id', 'split',
    'sentences', 'category_id', 'raw_anns', 'raw_image_info',
    'raw_sentences', 'image_path', 'global_image_id', 'anns_id'
])
```

Finally, split samples with multiple captions into unique entries.

```python
def separate_captions_into_unique_samples(batch):
    new_images = []
    new_bboxes = []
    new_captions = []

    for image, bbox, captions in zip(batch["image"], batch["bbox"], batch["captions"]):
        for caption in captions:
            new_images.append(image)
            new_bboxes.append(bbox)
            new_captions.append(caption)

    return {
        "image": new_images,
        "bbox": new_bboxes,
        "caption": new_captions,
    }

filtered_dataset = filtered_dataset.map(
    separate_captions_into_unique_samples,
    batched=True,
    batch_size=100,
    num_proc=4,
    remove_columns=filtered_dataset.column_names
)
```

### 1.4 Visualize a Sample

Let's create a helper function to visualize bounding boxes using the `supervision` library.

```python
import supervision as sv
import numpy as np

def get_annotated_image(image, parsed_labels):
    if not parsed_labels:
        return image

    xyxys = []
    labels = []

    for label, bbox in parsed_labels:
        xyxys.append(bbox)
        labels.append(label)

    detections = sv.Detections(xyxy=np.array(xyxys))

    bounding_box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels
    )

    return annotated_image
```

Now, inspect a sample from the prepared dataset.

```python
sample_idx = 20
image = filtered_dataset[sample_idx]['image']
caption = filtered_dataset[sample_idx]['caption']
bbox = filtered_dataset[sample_idx]['bbox']

print(f"Caption: {caption}")
print(f"Bounding Box: {bbox}")

# Visualize
labels = [(caption, bbox)]
annotated_image = get_annotated_image(image, labels)
# Display the image (e.g., in a notebook: annotated_image)
```

### 1.5 Split into Training and Validation Sets

```python
split_dataset = filtered_dataset.train_test_split(test_size=0.2, seed=42, shuffle=False)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
```

## Step 2: Evaluate the Pretrained Model

We'll use the pretrained `google/paligemma2-3b-pt-448` model, which supports object detection out of the box.

### 2.1 Load Model and Processor

```python
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
import torch

model_id = "google/paligemma2-3b-pt-448"

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
).eval()
processor = PaliGemmaProcessor.from_pretrained(model_id, use_fast=True)
```

### 2.2 Run Inference on a Sample

PaliGemma 2 is not an instruct model. The input must be formatted as: `<image>detect [CAPTION]`.

```python
image = train_dataset[20]['image']
caption = train_dataset[20]['caption']

prompt = f"<image>detect {caption}"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    output = processor.decode(generation, skip_special_tokens=True)
    print(output)
```

The output will be in PaliGemma's special format, e.g., `<loc0309><loc0240><loc0962><loc0425> middle vase ; ...`.

### 2.3 Parse the Model Output

We need to convert the location tokens (`<locXXXX>`) back to bounding box coordinates. Use the following helper function.

```python
import re

def parse_paligemma_labels(label, width, height):
    predictions = label.strip().split(";")
    results = []

    for pred in predictions:
        pred = pred.strip()
        if not pred:
            continue

        loc_pattern = r"<loc(\d{4})>"
        locations = [int(loc) for loc in re.findall(loc_pattern, pred)]

        if len(locations) != 4:
            continue

        category = pred.split(">")[-1].strip()

        y1_norm, x1_norm, y2_norm, x2_norm = locations
        x1 = (x1_norm / 1024) * width
        y1 = (y1_norm / 1024) * height
        x2 = (x2_norm / 1024) * width
        y2 = (y2_norm / 1024) * height

        results.append((category, [x1, y1, x2, y2]))

    return results
```

Now parse the model's output and visualize the detections.

```python
width, height = image.size
parsed_labels = parse_paligemma_labels(output, width, height)
print(f"Parsed labels: {parsed_labels}")

annotated_image = get_annotated_image(image, parsed_labels)
# Display the annotated image
```

You'll likely observe that the model detects objects well but struggles with grounding—for example, labeling all vases as "middle vase" instead of just the correct one. Our fine-tuning will aim to improve this.

## Step 3: Configure Fine-Tuning with LoRA and TRL

We'll use Parameter-Efficient Fine-Tuning (PEFT) via LoRA and train with TRL's SFTTrainer.

### 3.1 Apply LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

target_modules = [
    "q_proj",
    "v_proj",
    "fc1",
    "fc2",
    "linear",
    "gate_proj",
    "up_proj",
    "down_proj"
]

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=target_modules,
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
```

This will show that only a small percentage of parameters are trainable (e.g., 0.4%), making fine-tuning efficient.

### 3.2 Prepare for Supervised Fine-Tuning (SFT)

The next steps involve setting up the data collator, training arguments, and SFTTrainer. This guide will continue in the next part, where we define the training loop, handle data formatting, and execute the fine-tuning process.

## Next Steps

In the subsequent part of this tutorial, we will:
1. Define a custom data collator to format inputs and labels correctly.
2. Set up training arguments and the SFTTrainer.
3. Run the fine-tuning loop.
4. Evaluate the fine-tuned model on the validation set.
5. Save and push the model to the Hugging Face Hub.

Stay tuned for the continuation, where we complete the fine-tuning pipeline and achieve improved grounding performance.