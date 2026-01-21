# Fine-tuning a Vision Transformer on a Biomedical Dataset

_Authored by: [Emre Albayrak](https://github.com/emre570)_

This guide walks you through fine-tuning a Vision Transformer (ViT) model on a custom biomedical dataset for ultrasound image classification. You will learn how to prepare the data, configure the model, run the training loop, and evaluate the results.

## Prerequisites

Ensure you have the following libraries installed. Run this command in your environment:

```bash
pip install datasets transformers accelerate torch torchvision scikit-learn matplotlib wandb
```

**Optional:** If you plan to upload your fine-tuned model to the Hugging Face Hub, you will need to log in. Uncomment and run the following cell in a notebook environment:

```python
# from huggingface_hub import notebook_login
# notebook_login()
```

## Step 1: Load and Explore the Dataset

We'll use the `datasets` library to load a custom breast cancer ultrasound image dataset.

```python
from datasets import load_dataset

dataset = load_dataset("emre570/breastcancer-ultrasound-images")
dataset
```

The dataset is provided as a `DatasetDict` with `train` and `test` splits. Let's inspect its structure:

```python
DatasetDict({
    train: Dataset({
        features: ['image', 'label'],
        num_rows: 624
    })
    test: Dataset({
        features: ['image', 'label'],
        num_rows: 156
    })
})
```

### 1.1 Create a Validation Split

The dataset lacks a validation set. We'll create one by splitting the training data. The validation size will be proportional to the test set size.

```python
# Calculate the size of the validation set
test_num = len(dataset["test"])
train_num = len(dataset["train"])
val_size = test_num / train_num

# Split the training set
train_val_split = dataset["train"].train_test_split(test_size=val_size)
train_val_split
```

Now, let's merge all splits into a single `DatasetDict` with clear `train`, `validation`, and `test` keys.

```python
from datasets import DatasetDict

dataset = DatasetDict({
    "train": train_val_split["train"],
    "validation": train_val_split["test"],
    "test": dataset["test"]
})
dataset
```

Assign each split to a variable for easy reference:

```python
train_ds = dataset['train']
val_ds = dataset['validation']
test_ds = dataset['test']
```

### 1.2 Inspect the Data

Each sample is a dictionary containing a PIL image and an integer label.

```python
train_ds[0]
```

```python
{'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=460x391>,
 'label': 0}
```

Check the feature schema to understand the label mapping:

```python
train_ds.features
```

```python
{'image': Image(mode=None, decode=True, id=None),
 'label': ClassLabel(names=['benign', 'malignant', 'normal'], id=None)}
```

The labels correspond to three classes: `benign`, `malignant`, and `normal`.

## Step 2: Prepare the Data for Training

We need to process the images into a format the ViT model expects. This involves:
1.  Creating label mappings.
2.  Defining image transformations (different for training and evaluation).
3.  Applying those transformations.
4.  Setting up data loaders.

### 2.1 Create Label Mappings

Create dictionaries to convert between label IDs and names.

```python
id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label:id for id,label in id2label.items()}
id2label
```

```python
{0: 'benign', 1: 'malignant', 2: 'normal'}
```

### 2.2 Initialize the Image Processor

We'll use the `ViTImageProcessor` corresponding to our base model, `google/vit-large-patch16-224`. This processor knows the required image size and normalization statistics.

```python
from transformers import ViTImageProcessor

model_name = "google/vit-large-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
```

### 2.3 Define Image Transformations

We apply different transformations for training (data augmentation) and evaluation (simple resizing and cropping).

```python
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor, Resize

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)

# Training transforms include augmentation
train_transforms = Compose([        
    RandomResizedCrop(size),
    RandomHorizontalFlip(),
    ToTensor(),
    normalize,
])

# Validation and test transforms are deterministic
val_transforms = Compose([
    Resize(size),
    CenterCrop(size),
    ToTensor(),
    normalize,
])

test_transforms = Compose([
    Resize(size),
    CenterCrop(size),
    ToTensor(),
    normalize,
])
```

### 2.4 Apply Transforms to the Datasets

We define functions to apply the transforms and then set them as the dataset's transform method.

```python
def apply_train_transforms(examples):
    examples['pixel_values'] = [train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def apply_val_transforms(examples):
    examples['pixel_values'] = [val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def apply_test_transforms(examples):
    examples['pixel_values'] = [test_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# Apply the transforms
train_ds.set_transform(apply_train_transforms)
val_ds.set_transform(apply_val_transforms)
test_ds.set_transform(apply_test_transforms)
```

Now a sample from the training set includes the `pixel_values` tensor.

```python
train_ds[0]['pixel_values'].shape
```

```python
torch.Size([3, 224, 224])
```

### 2.5 Create Data Loaders

We need a custom collate function to stack the processed images and labels into batches.

```python
import torch
from torch.utils.data import DataLoader

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

train_dl = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)
```

Let's verify a batch has the correct shape.

```python
batch = next(iter(train_dl))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k, v.shape)
```

```python
pixel_values torch.Size([4, 3, 224, 224])
labels torch.Size([4])
```

Perfect! The data is ready: batches of 4 images, each with 3 color channels and a 224x224 resolution.

## Step 3: Initialize and Configure the Model

Load the pre-trained Vision Transformer model for image classification. A key parameter is `ignore_mismatched_sizes=True`. This is necessary because the original model was trained on 1000 ImageNet classes, but our dataset has only 3 classes. This setting allows the model's final classification layer to be resized automatically.

```python
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    model_name,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
```

You will see a warning about newly initialized weights in the classifier layer—this is expected and correct.

## Step 4: Set Up Training Arguments

We use the `TrainingArguments` class to define all hyperparameters and logistics for training.

**Note:** The following configuration logs metrics to **Weights & Biases (W&B)**. If you don't have a W&B account or prefer not to use it, remove the `report_to="wandb"` line.

```python
from transformers import TrainingArguments

train_args = TrainingArguments(
    output_dir = "output-models",          # Directory for saving model checkpoints
    save_total_limit=2,                    # Keep only the last 2 checkpoints
    report_to="wandb",                     # Log metrics to Weights & Biases
    save_strategy="epoch",                 # Save a checkpoint at the end of each epoch
    evaluation_strategy="epoch",           # Evaluate on the validation set each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=40,
    weight_decay=0.01,
    load_best_model_at_end=True,           # Load the best model at the end of training
    logging_dir='logs',
    remove_unused_columns=False,           # Required for our custom dataset format
)
```

## Step 5: Train the Model

Instantiate the `Trainer` object with the model, arguments, datasets, collator, and processor (which acts as the tokenizer for image data).

```python
from transformers import Trainer

trainer = Trainer(
    model,
    train_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    tokenizer=processor,
)

# Start the training process
trainer.train()
```

Training will run for 40 epochs. The trainer will print logs for each epoch, showing training and validation loss.

## Step 6: Evaluate the Model

After training, evaluate the model's performance on the held-out test set.

```python
outputs = trainer.predict(test_ds)
print(outputs.metrics)
```

```python
{'test_loss': 0.3219967782497406,
 'test_accuracy': 0.9102564102564102,
 'test_runtime': 4.0543,
 'test_samples_per_second': 38.478,
 'test_steps_per_second': 9.619}
```

Our fine-tuned model achieves **91% accuracy** on the test set.

### 6.1 (Optional) Push the Model to the Hub

To share your model, you can push it to your Hugging Face Hub repository.

```python
model.push_to_hub("your-username/your-model-name")
```

## Step 7: Analyze Results

Let's gain deeper insights into the model's performance using a confusion matrix and per-class recall scores.

### 7.1 Generate the Confusion Matrix

A confusion matrix visualizes how the model's predictions compare to the true labels.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Get true labels and model predictions
y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

# Create and plot the confusion matrix
labels = train_ds.features['label'].names
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)
plt.show()
```

### 7.2 Calculate Recall Scores

Recall measures the model's ability to correctly identify all samples of a given class.

```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred, average=None)

for label, score in zip(labels, recall):
    print(f'Recall for {label}: {score:.2f}')
```

```python
Recall for benign: 0.90
Recall for malignant: 0.93
Recall for normal: 0.89
```

## Conclusion

You have successfully fine-tuned a Vision Transformer model on a custom biomedical dataset. The process involved:
1.  Loading and partitioning the data.
2.  Applying appropriate image transformations and augmentations.
3.  Configuring the model to handle a different number of classes.
4.  Training the model with the Hugging Face `Trainer` API.
5.  Evaluating the model, achieving 91% test accuracy, and analyzing per-class performance.

You can now use this model for inference on new ultrasound images or further iterate on the training process to improve performance.