# Fine tuning a VLM for Object Detection Grounding using TRL

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_


> 🚨 **WARNING**: This notebook is resource-intensive and requires substantial computational power. If you're running it in Colab, it will utilize an **A100 GPU**.

**🔍 What You'll Learn**

In this recipe, we'll demonstrate how to fine-tune a [Vision-Language Model (VLM)](https://huggingface.co/blog/vlms-2025) for **object detection grounding** using [TRL](https://huggingface.co/docs/trl/en/index).

Traditionally, object detection involves identifying a predefined set of classes (e.g., "car", "person", "dog") within an image. However, this paradigm shifted with models like [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base), [GLIP](https://github.com/microsoft/GLIP), or [OWL-ViT](https://arxiv.org/abs/2205.06230), which introduced **open-ended object detection**—enabling models to detect *any* class described in natural language.

Grounding goes a step further by adding contextual understanding. Instead of just detecting a "car", grounded detection can locate the **"car on the left"**, or the **"red car behind the tree"**. This provides a more nuanced and powerful approach to object detection.

In this recipe, we'll walk through how to fine-tune a VLM for this task. Specifically, we'll use [PaliGemma 2](https://huggingface.co/blog/paligemma2), a Vision-Language Model developed by Google that supports object detection out of the box. While not all VLMs offer detection capabilities by default, the concepts and steps in this notebook can be adapted for models without built-in object detection as well.

To train our model, we'll use [RefCOCO](https://paperswithcode.com/dataset/refcoco), an extension of the popular COCO dataset, designed specifically for **referring expression comprehension**—that is, combining object detection with grounding through natural language.

This recipe also builds upon my recent release of [this Space](https://huggingface.co/spaces/sergiopaniego/vlm_object_understanding), which lets you compare different VLMs on object understanding tasks such as object detection, keypoint detection, and more.

📚 **Additional Resources**  
At the end of this notebook, you'll find extra resources if you're interested in exploring the topic further.



## 1. Install dependencies

Let's start by installing the required dependencies:


```python
!pip install -Uq transformers datasets trl supervision albumentations
```

We'll log in to our Hugging Face [account](https://huggingface.co/join) to access gated models and save our trained checkpoints.  
You'll need an access [token](https://huggingface.co/settings/tokens) 🗝️.


```python
from huggingface_hub import notebook_login

notebook_login()
```

## 2. 📁 Load Dataset

For this example, we'll use [RefCOCO](https://paperswithcode.com/dataset/refcoco), a dataset that includes grounded object detection annotations—enabling more robust and context-aware detection.

To keep things simple and efficient, we'll work with a subset of the dataset.




```python
from datasets import load_dataset
refcoco_dataset = load_dataset("jxu124/refcoco",split='train[:5%]')
```

After loading it, let's see what's inside:


```python
refcoco_dataset
```




    Dataset({
        features: ['sent_ids', 'file_name', 'ann_id', 'ref_id', 'image_id', 'split', 'sentences', 'category_id', 'raw_anns', 'raw_image_info', 'raw_sentences', 'image_path', 'bbox', 'captions', 'global_image_id', 'anns_id'],
        num_rows: 2120
    })



We can see that the dataset contains useful information such as the `bbox` and `captions` columns. In this case, bboxes follow a `xyxy` format.

However, the image itself isn't directly accessible from these fields. For more details about the image source, we can inspect the `raw_image_info` column.


```python
refcoco_dataset[13]['raw_image_info']
```




    '{"license": 3, "file_name": "COCO_train2014_000000581719.jpg", "coco_url": "http://mscoco.org/images/581719", "height": 357, "width": 500, "date_captured": "2013-11-17 05:37:38", "flickr_url": "http://farm1.staticflickr.com/73/153665038_3ecb570b2d_z.jpg", "id": 581719}'



### 2.1 🖼️ Add Images to the Dataset

While we could link each example to the corresponding image in the [COCO dataset](https://cocodataset.org/), we'll simplify the process by downloading the images directly from Flickr.

However, this approach may result in some missing images, so we’ll need to handle those cases accordingly.


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

Awesome! Our images are now downloaded and ready to go.


```python
refcoco_dataset_with_images
```




    Dataset({
        features: ['sent_ids', 'file_name', 'ann_id', 'ref_id', 'image_id', 'split', 'sentences', 'category_id', 'raw_anns', 'raw_image_info', 'raw_sentences', 'image_path', 'bbox', 'captions', 'global_image_id', 'anns_id', 'image'],
        num_rows: 2120
    })



Next, let's filter the dataset to include only samples that have an associated image:


```python
filtered_dataset = refcoco_dataset_with_images.filter(
    lambda example: example['image'] is not None,
    desc="Removing failed image downloads"
)
```

### 2.2 Remove Unneeded Columns


```python
filtered_dataset
```




    Dataset({
        features: ['sent_ids', 'file_name', 'ann_id', 'ref_id', 'image_id', 'split', 'sentences', 'category_id', 'raw_anns', 'raw_image_info', 'raw_sentences', 'image_path', 'bbox', 'captions', 'global_image_id', 'anns_id', 'image'],
        num_rows: 1691
    })



The dataset contains many columns that we won't need for this task.  
Let's simplify it by keeping only the `'bbox'`, `'captions'`, and `'image'` columns.


```python
filtered_dataset = filtered_dataset.remove_columns(['sent_ids', 'file_name', 'ann_id', 'ref_id', 'image_id', 'split', 'sentences', 'category_id', 'raw_anns', 'raw_image_info', 'raw_sentences', 'image_path', 'global_image_id', 'anns_id'])
```

It looks much better now!


```python
filtered_dataset
```




    Dataset({
        features: ['bbox', 'captions', 'image'],
        num_rows: 1691
    })



### 2.3 Separate Captions into Unique Samples

One final step: each sample currently has multiple captions. To simplify the dataset, we'll split these so that each caption becomes a unique sample.


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


    Map (num_proc=4):   0%|          | 0/1691 [00:00<?, ? examples/s]


Now that everything is prepared, let's take a look at an example!


```python
filtered_dataset[20]['caption']
```




    'middle vase'




```python
filtered_dataset[20]['bbox']
```




    [265.35, 143.46, 381.84000000000003, 453.03]




```python
filtered_dataset[20]['image']
```

### 2.4 Display a Sample with Bounding Boxes

Our dataset preparation is complete. Now, let's visualize the bounding boxes on an image from a sample.  
To do this, we'll create an auxiliary function that we can reuse throughout the recipe.

We'll use the [supervision](https://supervision.roboflow.com/latest/) library to assist with displaying the bounding boxes.


```python
labels = [(filtered_dataset[20]['caption'], filtered_dataset[20]['bbox'])]
```


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

annotated_image = get_annotated_image(filtered_dataset[20]['image'], labels)
annotated_image
```

Great! We can now see the grounding caption associated with each bounding box.


### 2.5 Divide the Dataset

Our dataset is ready, but before we proceed, let's split it into training and validation sets for proper model evaluation.



```python
split_dataset = filtered_dataset.train_test_split(test_size=0.2, seed=42, shuffle=False)
train_dataset = split_dataset['train']
val_dataset = split_dataset['test']
train_dataset, val_dataset
```




    (Dataset({
         features: ['bbox', 'image', 'caption'],
         num_rows: 3863
     }),
     Dataset({
         features: ['bbox', 'image', 'caption'],
         num_rows: 966
     }))



## 3. Check the Pretrained Model with the Dataset

As mentioned earlier, we'll be using **PaliGemma 2** as our model since it already includes object detection capabilities, which simplifies our workflow.

If we were using a Vision-Language Model (VLM) without built-in object detection capabilities, we would likely need to train it first to acquire them.

For more on this, check out our [project on "Fine-tuning Gemma 3 for Object Detection"](https://github.com/ariG23498/gemma3-object-detection) that covers this training process in detail.

Now, let's load the model and processor. We'll use the pretrained model [google/paligemma2-3b-pt-448](https://huggingface.co/google/paligemma2-3b-pt-448), which is not fine-tuned for conversational tasks.




```python
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
import torch

model_id = "google/paligemma2-3b-pt-448"

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
processor = PaliGemmaProcessor.from_pretrained(model_id, use_fast=True)
```


    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


### 3.1 Inference on One Sample

Let's evaluate the current performance of the model on a single image and caption.


```python
image = train_dataset[20]['image']
caption = train_dataset[20]['caption']
```

Since our model is not an instruct model, the input should be formatted as follows:

```
<image>detect [CAPTION]
```

Here, `<image>` represents the image token, followed by the keyword `detect` to specify the object detection task, and then the caption describing what to detect.

This format will produce a specific output, as we will see next.


```python
prompt = f"<image>detect {caption}"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(model.device)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    output = processor.decode(generation, skip_special_tokens=True)
    print(output)
```

    <loc0309><loc0240><loc0962><loc0425> middle vase ; <loc0309><loc0577><loc0955><loc0774> middle vase ; <loc0303><loc0428><loc0962><loc0593> middle vase


We can see that the model generates location tokens in a special format like `<locXXXX>...`, followed by the detected category. Each detection is separated by a `;`.

These location tokens follow the PaliGemma format, which is specific to the model and relative to the input size—`448x448` in this case, as indicated by the model name.

To display the detections correctly, we need to convert these tokens back to a usable format. Let's create an auxiliary function to handle this conversion:


```python
import re

# https://github.com/ariG23498/gemma3-object-detection/blob/main/utils.py#L17 thanks to Aritra Roy Gosthipaty
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

Now, we can use this function to parse the PaliGemma labels into the common COCO format.


```python
width, height = image.size
parsed_labels = parse_paligemma_labels(output, width, height)
parsed_labels
```




    [('middle vase', [150.0, 144.84375, 265.625, 450.9375]),
     ('middle vase', [360.625, 144.84375, 483.75, 447.65625]),
     ('middle vase', [267.5, 142.03125, 370.625, 450.9375])]



Next, we can use the previous function to retrieve the image.  
Let's display it along with the parsed bounding boxes!



```python
annotated_image = get_annotated_image(image, parsed_labels)
```


```python
annotated_image
```

We can see that the model performs well on object detection, but it struggles a bit with grounding.  
For example, it labels all three vases as the **"middle vase"** instead of just one.  

Let's work on improving that! 🙂

## 4. Fine-Tuning the Model Using the Dataset with LoRA and TRL

To fine-tune the Vision-Language Model (VLM), we will leverage [LoRA](https://huggingface.co/docs/peft/en/package_reference/lora) and [TRL](https://github.com/huggingface/trl).  

Let's start by configuring LoRA:


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

# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=target_modules,
    task_type="CAUSAL_LM",
)

# Apply PEFT model adaptation
peft_model = get_peft_model(model, peft_config)

# Print trainable parameters
peft_model.print_trainable_parameters()
```

    trainable params: 12,165,888 || all params: 3,045,293,040 || trainable%: 0.3995


Next, let's configure the [SFT training](https://huggingface.co