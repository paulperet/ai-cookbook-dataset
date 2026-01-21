# Fine-Tuning a Semantic Segmentation Model on a Custom Dataset

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

In this guide, you will learn how to fine-tune a semantic segmentation model on a custom dataset and deploy it via an inference API. We'll use the pretrained [Segformer](https://huggingface.co/docs/transformers/model_doc/segformer) model and the [segments/sidewalk-semantic](https://huggingface.co/datasets/segments/sidewalk-semantic) dataset, which contains labeled images of sidewalks—ideal for applications like autonomous delivery robots.

## Prerequisites

Before starting, ensure you have the necessary libraries installed.

```bash
pip install -q datasets transformers evaluate wandb
```

## Step 1: Load and Authenticate with the Dataset

The dataset is gated, so you need to log in to the Hugging Face Hub and accept its license.

```python
from huggingface_hub import notebook_login

notebook_login()
```

Define the dataset identifier and load it.

```python
sidewalk_dataset_identifier = "segments/sidewalk-semantic"

from datasets import load_dataset

dataset = load_dataset(sidewalk_dataset_identifier)
```

Inspect the dataset structure.

```python
dataset
```

Output:
```
DatasetDict({
    train: Dataset({
        features: ['pixel_values', 'label'],
        num_rows: 1000
    })
})
```

## Step 2: Split the Dataset

Since the dataset only has a training split, we'll manually split it into training and test sets (80/20 split).

```python
dataset = dataset.shuffle(seed=42)
dataset = dataset["train"].train_test_split(test_size=0.2)
train_ds = dataset["train"]
test_ds = dataset["test"]
```

## Step 3: Explore the Dataset

Let's examine a single example to understand the data format.

```python
image = train_ds[0]
image
```

Output:
```
{'pixel_values': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1920x1080>,
 'label': <PIL.PngImagePlugin.PngImageFile image mode=L size=1920x1080>}
```

The `pixel_values` key contains the RGB image, and `label` contains the corresponding segmentation mask.

## Step 4: Load Label Mappings

The dataset includes an `id2label.json` file that maps category IDs to human-readable names. We'll load it to understand the 34 distinct categories.

```python
import json
from huggingface_hub import hf_hub_download

filename = "id2label.json"
id2label = json.load(open(hf_hub_download(repo_id=sidewalk_dataset_identifier, filename=filename, repo_type="dataset"), "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)
print("Id2label:", id2label)
```

Output:
```
Id2label: {0: 'unlabeled', 1: 'flat-road', 2: 'flat-sidewalk', ... , 34: 'void-unclear'}
```

## Step 5: Define a Color Palette for Visualization

Assign a unique color to each category to help visualize the segmentation masks.

```python
sidewalk_palette = [
  [0, 0, 0], # unlabeled
  [216, 82, 24], # flat-road
  [255, 255, 0], # flat-sidewalk
  # ... (full palette for 34 categories)
  [255, 170, 127], # void-unclear
]
```

## Step 6: Visualize Dataset Samples

Let's visualize a few examples from the training set to see the original images, their segmentation masks, and an overlay.

First, create a legend for the categories.

```python
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches

# Create and show the legend separately
fig, ax = plt.subplots(figsize=(18, 2))

legend_patches = [patches.Patch(color=np.array(color)/255, label=label) for label, color in zip(id2label.values(), sidewalk_palette)]

ax.legend(handles=legend_patches, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=5, fontsize=8)
ax.axis('off')

plt.show()
```

Now, visualize five samples.

```python
for i in range(5):
    image = train_ds[i]

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Show the original image
    ax[0].imshow(image['pixel_values'])
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    mask_np = np.array(image['label'])

    # Create a new empty RGB image
    colored_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)

    # Assign colors to each value in the mask
    for label_id, color in enumerate(sidewalk_palette):
        colored_mask[mask_np == label_id] = color

    colored_mask_img = Image.fromarray(colored_mask, 'RGB')

    # Show the segmentation mask
    ax[1].imshow(colored_mask_img)
    ax[1].set_title('Segmentation Mask')
    ax[1].axis('off')

    # Convert the original image to RGBA to support transparency
    image_rgba = image['pixel_values'].convert("RGBA")
    colored_mask_rgba = colored_mask_img.convert("RGBA")

    # Adjust transparency of the mask
    alpha = 128  # Transparency level (0 fully transparent, 255 fully opaque)
    image_2_with_alpha = Image.new("RGBA", colored_mask_rgba.size)
    for x in range(colored_mask_rgba.width):
        for y in range(colored_mask_rgba.height):
            r, g, b, a = colored_mask_rgba.getpixel((x, y))
            image_2_with_alpha.putpixel((x, y), (r, g, b, alpha))

    superposed = Image.alpha_composite(image_rgba, image_2_with_alpha)

    # Show the mask overlay
    ax[2].imshow(superposed)
    ax[2].set_title('Mask Overlay')
    ax[2].axis('off')

    plt.show()
```

## Step 7: Analyze Class Distribution

Understanding the frequency of each class helps identify potential dataset imbalances.

```python
import matplotlib.pyplot as plt
import numpy as np

class_counts = np.zeros(len(id2label))

for example in train_ds:
    mask_np = np.array(example['label'])
    unique, counts = np.unique(mask_np, return_counts=True)
    for u, c in zip(unique, counts):
        class_counts[u] += c
```

Visualize the class distribution.

```python
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import patches

labels = list(id2label.values())

# Normalize colors to be in the range [0, 1]
normalized_palette = [tuple(c / 255 for c in color) for color in sidewalk_palette]

# Visualization
fig, ax = plt.subplots(figsize=(12, 8))

bars = ax.bar(range(len(labels)), class_counts, color=[normalized_palette[i] for i in range(len(labels))])

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90, ha="right")

ax.set_xlabel("Categories", fontsize=14)
ax.set_ylabel("Number of Occurrences", fontsize=14)
ax.set_title("Number of Occurrences by Category", fontsize=16)

ax.grid(axis="y", linestyle="--", alpha=0.7)

# Adjust the y-axis limit
y_max = max(class_counts)
ax.set_ylim(0, y_max * 1.25)

for bar in bars:
    height = int(bar.get_height())
    offset = 10  # Adjust the text location
    ax.text(bar.get_x() + bar.get_width() / 2.0, height + offset, f"{height}",
            ha="center", va="bottom", rotation=90, fontsize=10, color='black')

fig.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=8)

plt.tight_layout()
plt.show()
```

## Step 8: Initialize Image Processor and Apply Data Augmentation

We'll use the `SegformerImageProcessor` and apply data augmentation with Albumentations to improve model robustness.

```python
import albumentations as A
from transformers import SegformerImageProcessor

image_processor = SegformerImageProcessor()

albumentations_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.7),
    A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=20, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.4),
])

def train_transforms(example_batch):
    augmented = [
        albumentations_transform(image=np.array(image), mask=np.array(label))
        for image, label in zip(example_batch['pixel_values'], example_batch['label'])
    ]
    augmented_images = [item['image'] for item in augmented]
    augmented_labels = [item['mask'] for item in augmented]
    inputs = image_processor(augmented_images, augmented_labels)
    return inputs

def val_transforms(example_batch):
    images = [x for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = image_processor(images, labels)
    return inputs

# Apply transforms
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)
```

## Step 9: Load the Pretrained Model

Initialize a Segformer model from the `nvidia/mit-b0` checkpoint, providing the label mappings.

```python
from transformers import SegformerForSemanticSegmentation

pretrained_model_name = "nvidia/mit-b0"
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    id2label=id2label,
    label2id=label2id
)
```

## Step 10: Configure Training Arguments and Weights & Biases

Set up training parameters and connect to Weights & Biases for experiment tracking.

```python
from transformers import TrainingArguments

output_dir = "test-segformer-b0-segments-sidewalk-finetuned"

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=6e-5,
    num_train_epochs=20,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_total_limit=2,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_steps=1,
    eval_accumulation_steps=5,
    load_best_model_at_end=True,
    push_to_hub=True,
    report_to="wandb"
)
```

Initialize the W&B run.

```python
import wandb

wandb.init(
    project="test-segformer-b0-segments-sidewalk-finetuned",
    name="test-segformer-b0-segments-sidewalk-finetuned",
    config=training_args,
)
```

## Step 11: Define Evaluation Metrics

We'll use mean Intersection over Union (IoU) as the primary evaluation metric. This step also suppresses warnings that may appear when a category is not present in an image.

```python
import evaluate
import numpy as np

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=False,
        )
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()
        return metrics
```

## Next Steps

You have now prepared the dataset, configured the model, and set up training and evaluation. The next part of this tutorial will cover:
* Setting up the Trainer and starting the fine-tuning process.
* Evaluating the model on the test set.
* Uploading the fine-tuned model to the Hugging Face Hub.
* Deploying the model using the Serverless Inference API.

Proceed to the next section to continue with model training and deployment.