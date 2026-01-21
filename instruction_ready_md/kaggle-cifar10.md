# Image Classification on CIFAR-10: A Kaggle Competition Guide

This guide walks you through building an image classifier for the CIFAR-10 dataset and preparing a submission for the Kaggle competition. We'll start from raw image files, organize the dataset, apply data augmentation, define a ResNet-18 model, train it, and generate predictions.

## Prerequisites

First, ensure you have the necessary libraries installed. This guide provides code for both PyTorch and MXNet frameworks.

```bash
# Install required packages (choose your framework)
# For PyTorch:
pip install torch torchvision pandas

# For MXNet:
pip install mxnet pandas
```

## Step 1: Setup and Imports

Import the required libraries and set up your environment.

```python
import collections
import math
import os
import pandas as pd
import shutil

# Framework-specific imports
# For PyTorch:
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# For MXNet:
# from mxnet import gluon, init, npx
# from mxnet.gluon import nn
# npx.set_np()
# from d2l import mxnet as d2l
```

## Step 2: Obtain and Organize the Dataset

The competition dataset consists of 50,000 training and 300,000 test images. For initial experimentation, we'll use a small sample.

### 2.1 Download the Sample Dataset

We provide a tiny dataset for quick prototyping. Set `demo = False` to use the full Kaggle dataset.

```python
# Download the small sample dataset
d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

demo = True  # Set to False for the full competition dataset

if demo:
    data_dir = d2l.download_extract('cifar10_tiny')
else:
    data_dir = '../data/cifar-10/'
```

### 2.2 Read Dataset Labels

Create a mapping from image filenames to their labels by reading the CSV file.

```python
def read_csv_labels(fname):
    """Read `fname` to return a filename to label dictionary."""
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
print(f'# training examples: {len(labels)}')
print(f'# classes: {len(set(labels.values()))}')
```

### 2.3 Split Training and Validation Sets

Organize the dataset by splitting a validation set from the training data. This function ensures each class is proportionally represented in the validation set.

```python
def copyfile(filename, target_dir):
    """Copy a file into a target directory."""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def reorg_train_valid(data_dir, labels, valid_ratio):
    """Split the validation set out of the original training set."""
    # Find the class with the fewest examples
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # Calculate validation examples per class
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        # Copy to train_valid directory
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        # Assign to validation set if quota not met
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label
```

### 2.4 Organize the Test Set

Prepare the test set for prediction, placing all test images under an 'unknown' class folder.

```python
def reorg_test(data_dir):
    """Organize the testing set for data loading during prediction."""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))
```

### 2.5 Execute Dataset Organization

Run the complete organization pipeline with your chosen validation ratio.

```python
def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)

# Set parameters
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)
```

## Step 3: Apply Image Augmentation

Data augmentation helps prevent overfitting. We'll define separate transformations for training and testing.

### 3.1 Training Transformations

For training, apply random cropping, flipping, and normalization.

```python
# PyTorch version
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                             ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# MXNet version
# transform_train = gluon.data.vision.transforms.Compose([
#     gluon.data.vision.transforms.Resize(40),
#     gluon.data.vision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
#                                                    ratio=(1.0, 1.0)),
#     gluon.data.vision.transforms.RandomFlipLeftRight(),
#     gluon.data.vision.transforms.ToTensor(),
#     gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
#                                            [0.2023, 0.1994, 0.2010])])
```

### 3.2 Testing Transformations

For testing, only apply normalization to ensure consistent evaluation.

```python
# PyTorch version
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# MXNet version
# transform_test = gluon.data.vision.transforms.Compose([
#     gluon.data.vision.transforms.ToTensor(),
#     gluon.data.vision.transforms.Normalize([0.4914, 0.4822, 0.4465],
#                                            [0.2023, 0.1994, 0.2010])])
```

## Step 4: Create Data Loaders

Load the organized dataset using framework-specific data loaders.

### 4.1 PyTorch Data Loaders

```python
# Load datasets
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]

# Create data loaders
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)
```

### 4.2 MXNet Data Loaders

```python
# Load datasets
# train_ds, valid_ds, train_valid_ds, test_ds = [
#     gluon.data.vision.ImageFolderDataset(
#         os.path.join(data_dir, 'train_valid_test', folder))
#     for folder in ['train', 'valid', 'train_valid', 'test']]

# Create data loaders with transformations
# train_iter, train_valid_iter = [gluon.data.DataLoader(
#     dataset.transform_first(transform_train), batch_size, shuffle=True,
#     last_batch='discard') for dataset in (train_ds, train_valid_ds)]

# valid_iter = gluon.data.DataLoader(
#     valid_ds.transform_first(transform_test), batch_size, shuffle=False,
#     last_batch='discard')

# test_iter = gluon.data.DataLoader(
#     test_ds.transform_first(transform_test), batch_size, shuffle=False,
#     last_batch='keep')
```

## Step 5: Define the Model

We'll use a ResNet-18 architecture for this classification task.

### 5.1 PyTorch Model Definition

```python
def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)  # Using D2L's ResNet-18 implementation
    return net

loss = nn.CrossEntropyLoss(reduction="none")
```

### 5.2 MXNet Model Definition

```python
# Define Residual block
# class Residual(nn.HybridBlock):
#     def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
#         super(Residual, self).__init__(**kwargs)
#         self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
#                                strides=strides)
#         self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
#         if use_1x1conv:
#             self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
#                                    strides=strides)
#         else:
#             self.conv3 = None
#         self.bn1 = nn.BatchNorm()
#         self.bn2 = nn.BatchNorm()
# 
#     def hybrid_forward(self, F, X):
#         Y = F.npx.relu(self.bn1(self.conv1(X)))
#         Y = self.bn2(self.conv2(Y))
#         if self.conv3:
#             X = self.conv3(X)
#         return F.npx.relu(Y + X)

# Define ResNet-18
# def resnet18(num_classes):
#     net = nn.HybridSequential()
#     net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
#             nn.BatchNorm(), nn.Activation('relu'))
# 
#     def resnet_block(num_channels, num_residuals, first_block=False):
#         blk = nn.HybridSequential()
#         for i in range(num_residuals):
#             if i == 0 and not first_block:
#                 blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
#             else:
#                 blk.add(Residual(num_channels))
#         return blk
# 
#     net.add(resnet_block(64, 2, first_block=True),
#             resnet_block(128, 2),
#             resnet_block(256, 2),
#             resnet_block(512, 2))
#     net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
#     return net

# def get_net(devices):
#     num_classes = 10
#     net = resnet18(num_classes)
#     net.initialize(ctx=devices, init=init.Xavier())
#     return net
# 
# loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Step 6: Define the Training Function

Create a training function that handles both training and validation.

```python
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # PyTorch optimizer and scheduler
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    
    # MXNet alternative:
    # trainer = gluon.Trainer(net.collect_params(), 'sgd',
    #                         {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    
    # PyTorch multi-GPU setup
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3)
        
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            # PyTorch training step
            l, acc = d2l.train_batch_ch13(net, features, labels,
                                          loss, trainer, devices)
            
            # MXNet alternative:
            # l, acc = d2l.train_batch_ch13(
            #     net, features, labels.astype('float32'), loss, trainer,
            #     devices, d2l.split_batch)
            
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            
            # Update visualization
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[2],
                              None))
        
        # Validation
        if valid_iter is not None:
            # PyTorch validation
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            
            # MXNet alternative:
            # valid_acc = d2l.evaluate_accuracy_gpus(net, valid_iter,
            #                                        d2l.split_batch)
            
            animator.add(epoch + 1, (None, None, valid_acc))
        
        # PyTorch learning rate scheduling
        scheduler.step()
        
        # MXNet learning rate adjustment:
        # if epoch > 0 and epoch % lr_period == 0:
        #     trainer.set_learning_rate(trainer.learning_rate * lr_decay)
    
    # Print final metrics
    measures = (f'train loss {metric[0] / metric[2]:.3f}, '
                f'train acc {metric[1] / metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## Step 7: Train and Validate the Model

Now train the model with the chosen hyperparameters. We'll use 20 epochs for demonstration.

```python
# Set hyperparameters
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-4
lr_period, lr_decay = 4, 0.9

# Get model
net = get_net()

# For PyTorch, do an initial forward pass
net(next(iter(train_iter))[0])

# For MXNet, hybridize the model
# net.hybridize()

# Train the model
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## Step 8: Generate Predictions for Submission

After hyperparameter tuning, retrain on the combined training and validation set, then predict on the test set.

```python
# Retrain on full training data (including validation)
net, preds = get_net(), []

# For PyTorch, do an initial forward pass
net(next(iter(train_valid_iter))[0])

# For MXNet, hybridize
# net.hybridize()

train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

# Generate predictions
for X, _ in test_iter:
    # PyTorch prediction
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
    
    # MXNet prediction:
    # y_hat = net(X.as_in_ctx(devices[0]))
    # preds.extend(y_hat.argmax(axis=1).astype(int).asnumpy())

# Prepare submission file
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))

# Map predictions to class names
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])  #