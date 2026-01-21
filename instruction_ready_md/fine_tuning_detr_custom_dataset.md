# Fine-Tuning an Object Detection Model on a Custom Dataset

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

## Overview

In this guide, you will fine-tune a DETR (DEtection TRansformer) object detection model on a custom dataset using the Hugging Face ecosystem. You'll adapt a pretrained model to recognize fashion items from the Fashionpedia dataset, then deploy the fine-tuned model as a Gradio Space on Hugging Face and learn to interact with it via the Gradio API.

## Prerequisites & Setup

Before starting, ensure you have the necessary libraries installed.

```bash
pip install -U -q datasets transformers[torch] timm wandb torchmetrics matplotlib albumentations
```

## Step 1: Load the Dataset

You'll use the Fashionpedia dataset, which contains fashion images annotated with bounding boxes and categories.

```python
from datasets import load_dataset

dataset = load_dataset('detection-datasets/fashionpedia')
```

Let's examine the dataset structure:

```python
print(dataset)
```

The dataset contains two splits: `train` (45,623 examples) and `val` (1,158 examples). Each example includes:
- `image_id`: Unique identifier
- `image`: The PIL image
- `width` & `height`: Image dimensions
- `objects`: Dictionary containing `bbox_id`, `category`, `bbox`, and `area`

## Step 2: Prepare Training and Validation Splits

Separate the dataset into training and validation sets.

```python
train_dataset = dataset['train']
test_dataset = dataset['val']
```

**Optional:** To speed up experimentation, you can sample a small fraction of the data. For production, use the full dataset.

```python
'''
def create_sample(dataset, sample_fraction=0.01, seed=42):
    sample_size = int(sample_fraction * len(dataset))
    sampled_dataset = dataset.shuffle(seed=seed).select(range(sample_size))
    print(f"Original size: {len(dataset)}")
    print(f"Sample size: {len(sampled_dataset)}")
    return sampled_dataset

train_dataset = create_sample(train_dataset)
test_dataset = create_sample(test_dataset)
'''
```

## Step 3: Understand the Label Mapping

Create mappings between category IDs and human-readable labels.

```python
id2label = {
    0: 'shirt, blouse', 1: 'top, t-shirt, sweatshirt', 2: 'sweater', 3: 'cardigan',
    4: 'jacket', 5: 'vest', 6: 'pants', 7: 'shorts', 8: 'skirt', 9: 'coat',
    10: 'dress', 11: 'jumpsuit', 12: 'cape', 13: 'glasses', 14: 'hat',
    15: 'headband, head covering, hair accessory', 16: 'tie', 17: 'glove',
    18: 'watch', 19: 'belt', 20: 'leg warmer', 21: 'tights, stockings',
    22: 'sock', 23: 'shoe', 24: 'bag, wallet', 25: 'scarf', 26: 'umbrella',
    27: 'hood', 28: 'collar', 29: 'lapel', 30: 'epaulette', 31: 'sleeve',
    32: 'pocket', 33: 'neckline', 34: 'buckle', 35: 'zipper', 36: 'applique',
    37: 'bead', 38: 'bow', 39: 'flower', 40: 'fringe', 41: 'ribbon',
    42: 'rivet', 43: 'ruffle', 44: 'sequin', 45: 'tassel'
}

label2id = {v: k for k, v in id2label.items()}
```

## Step 4: Visualize Dataset Examples

Create a helper function to draw bounding boxes and labels on images.

```python
from PIL import Image, ImageDraw

def draw_image_from_idx(dataset, idx):
    sample = dataset[idx]
    image = sample["image"]
    annotations = sample["objects"]
    draw = ImageDraw.Draw(image)
    width, height = sample["width"], sample["height"]

    for i in range(len(annotations["bbox_id"])):
        box = annotations["bbox"][i]
        x1, y1, x2, y2 = tuple(box)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, y1), id2label[annotations["category"][i]], fill="green")

    return image

# Visualize a single example
image_with_boxes = draw_image_from_idx(dataset=train_dataset, idx=10)
```

To visualize multiple examples at once:

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_images(dataset, indices):
    num_cols = 3
    num_rows = int(np.ceil(len(indices) / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

    for i, idx in enumerate(indices):
        row = i // num_cols
        col = i % num_cols
        image = draw_image_from_idx(dataset, idx)
        axes[row, col].imshow(image)
        axes[row, col].axis("off")

    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()

plot_images(train_dataset, range(9))
```

## Step 5: Filter Invalid Bounding Boxes

Some bounding boxes in the dataset have invalid coordinates. Filter these out to ensure clean training data.

```python
def filter_invalid_bboxes(example):
    valid_bboxes = []
    valid_bbox_ids = []
    valid_categories = []
    valid_areas = []

    for i, bbox in enumerate(example['objects']['bbox']):
        x_min, y_min, x_max, y_max = bbox[:4]
        if x_min < x_max and y_min < y_max:
            valid_bboxes.append(bbox)
            valid_bbox_ids.append(example['objects']['bbox_id'][i])
            valid_categories.append(example['objects']['category'][i])
            valid_areas.append(example['objects']['area'][i])
        else:
            print(f"Image with invalid bbox: {example['image_id']}")

    example['objects']['bbox'] = valid_bboxes
    example['objects']['bbox_id'] = valid_bbox_ids
    example['objects']['category'] = valid_categories
    example['objects']['area'] = valid_areas

    return example

train_dataset = train_dataset.map(filter_invalid_bboxes)
test_dataset = test_dataset.map(filter_invalid_bboxes)
```

## Step 6: Analyze Class Distribution

Understanding class distribution helps identify potential imbalances that could affect model performance.

```python
category_examples = {}
for example in train_dataset:
    for category in example['objects']['category']:
        label = id2label[category]
        category_examples[label] = category_examples.get(label, 0) + 1

# Visualize the distribution
categories = list(category_examples.keys())
values = list(category_examples.values())

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(categories, values, color='skyblue')
ax.set_xlabel('Categories', fontsize=14)
ax.set_ylabel('Number of Occurrences', fontsize=14)
ax.set_title('Number of Occurrences by Category', fontsize=16)
ax.set_xticklabels(categories, rotation=90, ha='right')
ax.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', 
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
```

You'll notice classes like "shoe" and "sleeve" are overrepresented, indicating class imbalance that may need addressing during training.

## Step 7: Apply Data Augmentation

Data augmentation improves model robustness by creating variations of training images. Use Albumentations for transformations that properly adjust bounding boxes.

```python
import albumentations as A

train_transform = A.Compose(
    [
        A.LongestMaxSize(500),
        A.PadIfNeeded(500, 500, border_mode=0, value=(0, 0, 0)),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.GaussianBlur(p=0.5),
        A.GaussNoise(p=0.5),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["category"]
    ),
)

val_transform = A.Compose(
    [
        A.LongestMaxSize(500),
        A.PadIfNeeded(500, 500, border_mode=0, value=(0, 0, 0)),
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["category"]
    ),
)
```

## Step 8: Initialize the Image Processor

The image processor prepares images for the DETR model by applying the appropriate resizing, normalization, and formatting.

```python
from transformers import AutoImageProcessor

checkpoint = "facebook/detr-resnet-50-dc5"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
```

## Next Steps

You've now prepared your dataset for fine-tuning. In the next part of this tutorial, you'll:
1. Apply the augmentations to your dataset
2. Configure the DETR model for fine-tuning
3. Set up the training loop
4. Evaluate the model's performance
5. Deploy the model to Hugging Face Spaces

The dataset is now cleaned, visualized, and augmented, ready for the model training phase.