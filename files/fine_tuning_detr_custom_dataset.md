# Fine-Tuning Object Detection Model on a Custom Dataset 🖼, Deployment in Spaces, and Gradio API Integration

_Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_

In this notebook, we will fine-tune an [object detection](https://huggingface.co/docs/transformers/tasks/object_detection) model—specifically, [DETR](https://huggingface.co/docs/transformers/model_doc/detr)—using a custom dataset. We will leverage the [Hugging Face ecosystem](https://huggingface.co/docs) to accomplish this task.

Our approach involves starting with a pretrained DETR model and fine-tuning it on a custom dataset of annotated fashion images, namely [Fashionpedia](https://huggingface.co/datasets/detection-datasets/fashionpedia). By doing so, we'll adapt the model to better recognize and detect objects within the fashion domain.

After successfully fine-tuning the model, we will deploy it as a Gradio Space on Hugging Face. Additionally, we’ll explore how to interact with the deployed model using the Gradio API, enabling seamless communication with the hosted Space and unlocking new possibilities for real-world applications.

## 1. Install Dependencies

Let's start by installing the necessary libraries for fine-tuning our object detection model.

```python
!pip install -U -q datasets transformers[torch] timm wandb torchmetrics matplotlib albumentations
# Tested with datasets==2.21.0, transformers==4.44.2 timm==1.0.9, wandb==0.17.9 torchmetrics==1.4.1
```

## 2. Load Dataset 📁

📁 The dataset we will use is [Fashionpedia](https://huggingface.co/datasets/detection-datasets/fashionpedia), which comes from the paper [Fashionpedia: Ontology, Segmentation, and an Attribute Localization Dataset](https://arxiv.org/abs/2004.12276). The authors describe it as follows:

````
Fashionpedia is a dataset which consists of two parts: (1) an ontology built by fashion experts containing 27 main apparel categories, 19 apparel parts, 294 fine-grained attributes and their relationships; (2) a dataset with 48k everyday and celebrity event fashion images annotated with segmentation masks and their associated per-mask fine-grained attributes, built upon the Fashionpedia ontology.
````

The dataset includes:

* **46,781 images** 🖼
* **342,182 bounding boxes** 📦

It is available on Hugging Face: [Fashionpedia Dataset](https://huggingface.co/datasets/detection-datasets/fashionpedia)

```python
from datasets import load_dataset

dataset = load_dataset('detection-datasets/fashionpedia')
```

```python
dataset
```

    DatasetDict({
        train: Dataset({
            features: ['image_id', 'image', 'width', 'height', 'objects'],
            num_rows: 45623
        })
        val: Dataset({
            features: ['image_id', 'image', 'width', 'height', 'objects'],
            num_rows: 1158
        })
    })

Review the internal structure of one of the examples

```python
dataset["train"][0]
```

    {'image_id': 23,
     'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=682x1024>,
     'width': 682,
     'height': 1024,
     'objects': {'bbox_id': [150311, 150312, 150313, 150314],
      'category': [23, 23, 33, 10],
      'bbox': [[445.0, 910.0, 505.0, 983.0],
       [239.0, 940.0, 284.0, 994.0],
       [298.0, 282.0, 386.0, 352.0],
       [210.0, 282.0, 448.0, 665.0]],
      'area': [1422, 843, 373, 56375]}}

## 3. Get Splits of the Dataset for Training and Testing ➗

The dataset comes with two splits: **train** and **test**. We will use the training split to fine-tune the model and the test split for validation.

```python
train_dataset = dataset['train']
test_dataset = dataset['val']
```

**Optional**

In the next commented cell, we randomly sample 1% of the original dataset for both the training and test splits. This approach is used to speed up the training process, as the dataset contains a large number of examples.

For the best results, we recommend skipping these two cells and using the full dataset. However, you can uncomment them if needed.

```python
'''
def create_sample(dataset, sample_fraction=0.01, seed=42):
    sample_size = int(sample_fraction * len(dataset))
    sampled_dataset = dataset.shuffle(seed=seed).select(range(sample_size))
    print(f"Original size: {len(dataset)}")
    print(f"Sample size: {len(sampled_dataset)}")
    return sampled_dataset

# Apply function to both splits
train_dataset = create_sample(train_dataset)
test_dataset = create_sample(test_dataset)
'''
```

## 4. Visualize One Example from the Dataset with Its Objects 👀

Now that we've loaded the dataset, let's visualize an example along with its annotated objects.

### Generate `id2label` and `label2id`

These variables contain the mappings between object IDs and their corresponding labels. `id2label` maps from IDs to labels, while `label2id` maps from labels to IDs.

```python
import numpy as np
from PIL import Image, ImageDraw


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

### Let's Draw One Image! 🎨

Now, let's visualize one image from the dataset to better understand what it looks like.

```python
def draw_image_from_idx(dataset, idx):
    sample = dataset[idx]
    image = sample["image"]
    annotations = sample["objects"]
    draw = ImageDraw.Draw(image)
    width, height = sample["width"], sample["height"]

    print(annotations)

    for i in range(len(annotations["bbox_id"])):
        box = annotations["bbox"][i]
        x1, y1, x2, y2 = tuple(box)
        draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
        draw.text((x1, y1), id2label[annotations["category"][i]], fill="green")

    return image

draw_image_from_idx(dataset=train_dataset, idx=10) # You can test changing this id
```

    {'bbox_id': [158977, 158978, 158979, 158980, 158981, 158982, 158983], 'category': [1, 23, 23, 6, 31, 31, 33], 'bbox': [[210.0, 225.0, 536.0, 784.0], [290.0, 897.0, 350.0, 1015.0], [464.0, 950.0, 534.0, 1021.0], [313.0, 407.0, 524.0, 954.0], [268.0, 229.0, 333.0, 563.0], [489.0, 247.0, 528.0, 591.0], [387.0, 225.0, 450.0, 253.0]], 'area': [69960, 2449, 1788, 75418, 15149, 5998, 479]}

### Let's Visualize Some More Images 📸

Now, let's take a look at a few more images from the dataset to get a broader view of the data.

```python
import matplotlib.pyplot as plt

def plot_images(dataset, indices):
    """
    Plot images and their annotations.
    """
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

    [{'bbox_id': [150311, 150312, 150313, 150314], 'category': [23, 23, 33, 10], 'bbox': [[445.0, 910.0, 505.0, 983.0], [239.0, 940.0, 284.0, 994.0], [298.0, 282.0, 386.0, 352.0], [210.0, 282.0, 448.0, 665.0]], 'area': [1422, 843, 373, 56375]}, ..., {'bbox_id': [158972, 158973, 158974, 158975, 158976], 'category': [23, 23, 28, 10, 5], 'bbox': [[412.0, 588.0, 451.0, 631.0], [333.0, 585.0, 357.0, 627.0], [361.0, 243.0, 396.0, 257.0], [303.0, 243.0, 447.0, 517.0], [330.0, 259.0, 425.0, 324.0]], 'area': [949, 737, 133, 17839, 2916]}]

## 5. Filter Invalid Bboxes ❌

As the first step in preprocessing the dataset, we will filter out some invalid bounding boxes. After reviewing the dataset, we found that some bounding boxes did not have a valid structure. Therefore, we will discard these invalid entries.

```python
from datasets import Dataset

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
            print(f"Image with invalid bbox: {example['image_id']} Invalid bbox detected and discarded: {bbox} - bbox_id: {example['objects']['bbox_id'][i]} - category: {example['objects']['category'][i]}")

    example['objects']['bbox'] = valid_bboxes
    example['objects']['bbox_id'] = valid_bbox_ids
    example['objects']['category'] = valid_categories
    example['objects']['area'] = valid_areas

    return example

train_dataset = train_dataset.map(filter_invalid_bboxes)
test_dataset = test_dataset.map(filter_invalid_bboxes)
```

    Map:   0%|          | 0/45623 [00:00<?, ? examples/s]

    [Image with invalid bbox: 8396 Invalid bbox detected and discarded: [0.0, 0.0, 0.0, 0.0] - bbox_id: 139952 - category: 42, ..., Image with invalid bbox: 34253 Invalid bbox detected and discarded: [0.0, 0.0, 0.0, 0.0] - bbox_id: 315750 - category: 19]

    Map:   0%|          | 0/1158 [00:00<?, ? examples/s]

```python
print(train_dataset)
print(test_dataset)
```

    Dataset({
        features: ['image_id', 'image', 'width', 'height', 'objects'],
        num_rows: 45623
    })
    Dataset({
        features: ['image_id', 'image', 'width', 'height', 'objects'],
        num_rows: 1158
    })

## 6. Visualize Class Occurrences 👀

Let's explore the dataset further by plotting the occurrences of each class. This will help us understand the distribution of classes and identify any potential biases.

```python
id_list = []
category_examples = {}
for example in train_dataset:
  id_list += example['objects']['bbox_id']
  for category in example['objects']['category']:
    if id2label[category] not in category_examples:
      category_examples[id2label[category]] = 1
    else:
      category_examples[id2label[category]] += 1

id_list.sort()
```

```python
import matplotlib.pyplot as plt

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
    ax.text(
        bar.get_x() + bar.get_width() / 2.0, height,
        f'{height}', ha='center', va='bottom', fontsize=10
    )

plt.tight_layout()
plt.show()
```

    <ipython-input-66-aa111d1e000d>:14: UserWarning: FixedFormatter should only be used together with FixedLocator
      ax.set_xticklabels(categories, rotation=90, ha='right')

We can observe that some classes, such as "shoe" or "sleeve," are overrepresented in the dataset. This indicates that the dataset may have an imbalance, with certain classes appearing more frequently than others. Identifying these imbalances is crucial for addressing potential biases in model training.

## 7. Add Data Augmentation to the Dataset

Data augmentation 🪄 is crucial for enhancing performance in object detection tasks. In this section, we will leverage the capabilities of [Albumentations](https://albumentations.ai/) to augment our dataset effectively.

Albumentations provides a range of powerful augmentation techniques tailored for object detection. It allows for various transformations, all while ensuring that bounding boxes are accurately adjusted. These capabilities help in generating a more diverse dataset, improving the model’s robustness and generalization.

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

## 8. Initialize Image Processor from Model Checkpoint 🎆

We will instantiate the image processor using a pretrained model checkpoint. In this case, we are using the [facebook/detr-resnet-50-dc5](https://huggingface.co/facebook/detr-resnet-50-dc5) model.

```python
from transformers import AutoImageProcessor

checkpoint = "facebook/detr-resnet-50-dc5"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
```

    preprocessor_config.json:   0