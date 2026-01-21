# Dog Breed Identification: A Kaggle Competition Guide

This guide walks you through building a model to identify 120 different dog breeds using a subset of the ImageNet dataset. You'll learn how to fine-tune a pre-trained ResNet model for this multi-class classification task.

## Prerequisites

First, ensure you have the necessary libraries installed. This guide provides code for both PyTorch and MXNet.

```bash
# Install required packages (choose your framework)
# For PyTorch:
pip install torch torchvision

# For MXNet:
pip install mxnet gluoncv
```

## Step 1: Setup and Imports

Import the required libraries and set up your environment.

```python
# PyTorch Version
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import os

# MXNet Version
# from d2l import mxnet as d2l
# from mxnet import autograd, gluon, init, npx
# from mxnet.gluon import nn
# import os
# npx.set_np()
```

## Step 2: Obtain and Organize the Dataset

The competition dataset contains 10,222 training images and 10,357 test images across 120 dog breeds. For this tutorial, we'll use a smaller sample dataset to speed up experimentation.

### Download the Sample Dataset

```python
# Download a small sample dataset for quick experimentation
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

demo = True  # Set to False to use the full Kaggle dataset
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')
```

### Organize the Dataset Structure

We need to split the training data into training and validation sets, and organize images into label-based subdirectories.

```python
def reorg_dog_data(data_dir, valid_ratio):
    """Reorganize the dog breed dataset."""
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)

# Set hyperparameters
batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)
```

## Step 3: Prepare Image Augmentation

Since ImageNet images are larger than typical datasets like CIFAR-10, we use specific augmentations suitable for higher-resolution images.

### Training Augmentations

```python
# PyTorch Version
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

# MXNet Version
# transform_train = gluon.data.vision.transforms.Compose([
#     gluon.data.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
#                                                    ratio=(3.0/4.0, 4.0/3.0)),
#     gluon.data.vision.transforms.RandomFlipLeftRight(),
#     gluon.data.vision.transforms.RandomColorJitter(brightness=0.4,
#                                                    contrast=0.4,
#                                                    saturation=0.4),
#     gluon.data.vision.transforms.RandomLighting(0.1),
#     gluon.data.vision.transforms.ToTensor(),
#     gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
#                                            [0.229, 0.224, 0.225])])
```

### Validation/Test Transformations

For validation and testing, we use deterministic transformations without randomness.

```python
# PyTorch Version
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

# MXNet Version
# transform_test = gluon.data.vision.transforms.Compose([
#     gluon.data.vision.transforms.Resize(256),
#     gluon.data.vision.transforms.CenterCrop(224),
#     gluon.data.vision.transforms.ToTensor(),
#     gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
#                                            [0.229, 0.224, 0.225])])
```

## Step 4: Create Data Loaders

Now we'll create data loaders for training, validation, and testing.

```python
# PyTorch Version
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]

train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)

# MXNet Version
# train_ds, valid_ds, train_valid_ds, test_ds = [
#     gluon.data.vision.ImageFolderDataset(
#         os.path.join(data_dir, 'train_valid_test', folder))
#     for folder in ('train', 'valid', 'train_valid', 'test')]
# 
# train_iter, train_valid_iter = [gluon.data.DataLoader(
#     dataset.transform_first(transform_train), batch_size, shuffle=True,
#     last_batch='discard') for dataset in (train_ds, train_valid_ds)]
# 
# valid_iter = gluon.data.DataLoader(
#     valid_ds.transform_first(transform_test), batch_size, shuffle=False,
#     last_batch='discard')
# 
# test_iter = gluon.data.DataLoader(
#     test_ds.transform_first(transform_test), batch_size, shuffle=False,
#     last_batch='keep')
```

## Step 5: Build the Model with Transfer Learning

We'll use a pre-trained ResNet-34 model and replace its final layer with a custom classifier for our 120 dog breeds.

### Define the Model Architecture

```python
# PyTorch Version
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # Define a new output network for 120 dog breeds
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # Move to device
    finetune_net = finetune_net.to(devices[0])
    # Freeze feature extraction layers
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

# MXNet Version
# def get_net(devices):
#     finetune_net = gluon.model_zoo.vision.resnet34_v2(pretrained=True)
#     # Define a new output network
#     finetune_net.output_new = nn.HybridSequential(prefix='')
#     finetune_net.output_new.add(nn.Dense(256, activation='relu'))
#     finetune_net.output_new.add(nn.Dense(120))  # 120 breeds
#     # Initialize and distribute parameters
#     finetune_net.output_new.initialize(init.Xavier(), ctx=devices)
#     finetune_net.collect_params().reset_ctx(devices)
#     return finetune_net
```

### Define Loss Function and Evaluation

```python
# PyTorch Version
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return l_sum / n

# MXNet Version
# loss = gluon.loss.SoftmaxCrossEntropyLoss()
# 
# def evaluate_loss(data_iter, net, devices):
#     l_sum, n = 0.0, 0
#     for features, labels in data_iter:
#         X_shards, y_shards = d2l.split_batch(features, labels, devices)
#         output_features = [net.features(X_shard) for X_shard in X_shards]
#         outputs = [net.output_new(feature) for feature in output_features]
#         ls = [loss(output, y_shard).sum() for output, y_shard
#               in zip(outputs, y_shards)]
#         l_sum += sum([float(l.sum()) for l in ls])
#         n += labels.size
#     return l_sum / n
```

## Step 6: Train the Model

We'll only train the custom output network while keeping the pre-trained feature extractor frozen.

### Define the Training Function

```python
# PyTorch Version
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().cpu()))
        scheduler.step()
    
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')

# MXNet Version (see original code for complete implementation)
```

### Train and Validate

Now let's train our model with the chosen hyperparameters.

```python
# Set hyperparameters
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay = 2, 0.9

# Get and train the model
net = get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## Step 7: Generate Predictions for Kaggle Submission

After training, we'll use the entire training data (including validation) to train a final model and generate predictions for the test set.

```python
# Train on combined training and validation data
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

# Generate predictions
preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)
    preds.extend(output.cpu().detach().numpy())

# Create submission file
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))

with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

The code above generates a `submission.csv` file that you can upload to the Kaggle competition.

## Summary

In this tutorial, you learned how to:

1. Organize and preprocess a dog breed identification dataset
2. Apply appropriate image augmentations for higher-resolution images
3. Implement transfer learning using a pre-trained ResNet-34 model
4. Fine-tune only the final classification layers for efficiency
5. Generate predictions for Kaggle submission

## Next Steps and Exercises

1. **Experiment with hyperparameters**: Try increasing `batch_size` and `num_epochs` while adjusting learning rate schedules.
2. **Use deeper models**: Experiment with ResNet-50 or ResNet-101 architectures. Remember to adjust the input size of your custom classifier accordingly.
3. **Data augmentation**: Try different augmentation strategies or intensities to improve generalization.
4. **Full dataset**: Set `demo = False` to train on the complete Kaggle dataset and compare results.

Remember that the key advantage of this approach is computational efficiency—by freezing the pre-trained feature extractor, we significantly reduce training time and memory requirements while still achieving good performance on the dog breed classification task.