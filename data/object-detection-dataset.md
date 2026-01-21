# Object Detection Dataset Tutorial

This guide walks you through working with a custom object detection dataset. We'll use a small, synthetic dataset of banana images to demonstrate the core concepts of loading and preparing data for object detection models.

## Prerequisites

First, ensure you have the necessary libraries installed. This tutorial provides code for both PyTorch and MXNet frameworks.

### For MXNet Users
```bash
pip install mxnet pandas matplotlib
```

### For PyTorch Users
```bash
pip install torch torchvision pandas matplotlib
```

## 1. Downloading the Dataset

We'll use a custom banana detection dataset that contains 1000 artificially generated images with labeled bounding boxes. The dataset is available for download through the D2L data hub.

```python
# Common setup for both frameworks
from d2l import torch as d2l  # or 'from d2l import mxnet as d2l'
import pandas as pd
import os

# Register the dataset with D2L's data hub
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## 2. Reading the Dataset

Next, we'll create a function to read the dataset files. The dataset includes CSV files with bounding box coordinates and class labels.

### MXNet Implementation
```python
from mxnet import gluon, image, np, npx
npx.set_np()

def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train 
                             else 'bananas_val', 'label.csv')
    
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else 
                         'bananas_val', 'images', f'{img_name}')))
        # Target contains: (class, upper-left x, upper-left y, 
        # lower-right x, lower-right y)
        targets.append(list(target))
    
    return images, np.expand_dims(np.array(targets), 1) / 256
```

### PyTorch Implementation
```python
import torch
import torchvision

def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train 
                             else 'bananas_val', 'label.csv')
    
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else 
                         'bananas_val', 'images', f'{img_name}')))
        # Target contains: (class, upper-left x, upper-left y, 
        # lower-right x, lower-right y)
        targets.append(list(target))
    
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

## 3. Creating a Custom Dataset Class

Now we'll wrap our data loading function into a proper Dataset class that can be used with framework-specific data loaders.

### MXNet Dataset Class
```python
class BananasDataset(gluon.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print(f'read {len(self.features)} ' + 
              ('training examples' if is_train else 'validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

### PyTorch Dataset Class
```python
class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print(f'read {len(self.features)} ' + 
              ('training examples' if is_train else 'validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

## 4. Creating Data Loaders

Let's create a function that returns data loader instances for both training and validation sets.

### MXNet Data Loader Function
```python
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

### PyTorch Data Loader Function
```python
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

## 5. Testing the Data Pipeline

Let's test our data loading pipeline by reading a minibatch and examining the shapes of the data.

```python
# Set batch size and image size
batch_size, edge_size = 32, 256

# Load the data
train_iter, _ = load_data_bananas(batch_size)

# Get one batch
batch = next(iter(train_iter))

# Check shapes
print(f"Image batch shape: {batch[0].shape}")
print(f"Label batch shape: {batch[1].shape}")
```

You should see output similar to:
```
read 1000 training examples
read 100 validation examples
Image batch shape: (32, 3, 256, 256)
Label batch shape: (32, 1, 5)
```

**Understanding the output:**
- The image batch has shape `(batch_size, channels, height, width)`
- The label batch has shape `(batch_size, m, 5)` where `m` is the maximum number of bounding boxes per image
- For our banana dataset, `m=1` since each image contains exactly one banana
- Each bounding box label is an array of length 5: `[class, x1, y1, x2, y2]` where coordinates are normalized between 0 and 1

## 6. Visualizing the Data

Let's visualize some images with their ground-truth bounding boxes to understand what our dataset looks like.

### MXNet Visualization
```python
import matplotlib.pyplot as plt

imgs = (batch[0][:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)

for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

### PyTorch Visualization
```python
import matplotlib.pyplot as plt

imgs = (batch[0][:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)

for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

This will display 10 images with white bounding boxes drawn around the bananas. You'll notice the bananas vary in rotation, size, and position within the images.

## Key Takeaways

1. **Dataset Structure**: Object detection datasets require both images and bounding box annotations, unlike image classification datasets which only need class labels.

2. **Data Loading**: The process involves:
   - Downloading and extracting the dataset
   - Reading image files and corresponding label files
   - Creating a Dataset class for framework compatibility
   - Setting up DataLoader instances for batching

3. **Label Format**: Bounding boxes are typically represented as `[class_id, x1, y1, x2, y2]` where coordinates are normalized between 0 and 1.

4. **Padding**: Since images can have varying numbers of objects, datasets often pad with "illegal" bounding boxes (class = -1) to ensure consistent batch sizes.

## Exercises

1. **Explore the Dataset**: Try visualizing different batches of images. How do the bounding boxes and object positions vary across the dataset?

2. **Consider Data Augmentation**: Think about how you might apply data augmentation techniques like random cropping to object detection. What challenges arise when a crop contains only part of an object?

## Next Steps

Now that you have a working data pipeline for object detection, you can:
- Use this dataset to train simple object detection models
- Extend the pipeline to handle more complex, real-world datasets
- Implement data augmentation techniques specific to object detection
- Experiment with different batch sizes and image resolutions

Remember that real-world object detection datasets are typically much larger and more complex than this synthetic banana dataset, but the fundamental data loading principles remain the same.