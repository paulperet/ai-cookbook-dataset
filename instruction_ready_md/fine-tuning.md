# Fine-Tuning a Pre-trained Model for Hot Dog Recognition

This guide walks you through the process of fine-tuning a pre-trained ResNet model to recognize hot dogs in images. Fine-tuning is a powerful transfer learning technique that allows you to leverage knowledge from a large dataset (like ImageNet) and adapt it to a smaller, specific task.

## Prerequisites

First, ensure you have the necessary libraries installed. This tutorial provides code for both PyTorch and MXNet frameworks.

### PyTorch Setup
```bash
pip install torch torchvision matplotlib
```

### MXNet Setup
```bash
pip install mxnet gluoncv matplotlib
```

## Step 1: Understanding the Problem and Dataset

We'll use a custom hot dog dataset containing:
- 1400 positive images (with hot dogs)
- 1400 negative images (other foods)

The dataset is split into training (1000 images per class) and testing (400 images per class) sets.

Let's download and explore the dataset structure:

```python
# Download and extract the dataset
import d2l
import os

# Define dataset URL and download
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')
```

## Step 2: Load and Visualize the Dataset

Now, let's load the dataset and examine some sample images:

```python
# PyTorch version
import torchvision

train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# Display sample images
import matplotlib.pyplot as plt

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]

fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for i, img in enumerate(hotdogs + not_hotdogs):
    ax = axes[i // 8, i % 8]
    ax.imshow(img)
    ax.axis('off')
plt.show()
```

```python
# MXNet version
from mxnet import gluon

train_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'train'))
test_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'test'))

# Display sample images (similar visualization logic applies)
```

You'll notice the images vary in size and aspect ratio, which we'll need to standardize during preprocessing.

## Step 3: Define Data Augmentation and Preprocessing

To ensure consistent input to our model and improve generalization, we'll apply transformations:

```python
# PyTorch transformations
import torchvision.transforms as transforms

# Normalization values for ImageNet pretrained models
normalize = transforms.Normalize(
    [0.485, 0.456, 0.406],  # Mean for RGB channels
    [0.229, 0.224, 0.225]   # Std for RGB channels
)

# Training augmentations
train_augs = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

# Testing transformations
test_augs = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
```

```python
# MXNet transformations
from mxnet.gluon.data.vision import transforms

normalize = transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.ToTensor(),
    normalize
])

test_augs = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
```

## Step 4: Load the Pre-trained Model

We'll use ResNet-18, which was pre-trained on ImageNet, as our base model:

```python
# PyTorch
import torch
import torchvision.models as models

pretrained_net = models.resnet18(pretrained=True)
print(pretrained_net.fc)  # View the final classification layer
```

```python
# MXNet
from mxnet.gluon.model_zoo import vision as models

pretrained_net = models.resnet18_v2(pretrained=True)
print(pretrained_net.output)  # View the final classification layer
```

The pre-trained model has 1000 output classes (for ImageNet). We need to modify this for our binary classification task.

## Step 5: Create the Fine-tuning Model

We'll create a new model that copies all layers except the final classification layer:

```python
# PyTorch
import torch.nn as nn

finetune_net = models.resnet18(pretrained=True)
# Replace the final fully connected layer for binary classification
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
# Initialize the new layer's weights
nn.init.xavier_uniform_(finetune_net.fc.weight)

# Set different learning rates for different parts
# The pre-trained layers will use a smaller learning rate
# The new classification layer will use a larger learning rate
```

```python
# MXNet
from mxnet import init

finetune_net = models.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
# Set learning rate multiplier for output layer
finetune_net.output.collect_params().setattr('lr_mult', 10)
```

## Step 6: Define the Training Function

Create a training function that handles the fine-tuning process:

```python
# PyTorch training function
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    # Create data loaders
    train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train'), 
            transform=train_augs
        ),
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'test'), 
            transform=test_augs
        ),
        batch_size=batch_size
    )
    
    # Set up loss function and optimizer
    loss = nn.CrossEntropyLoss(reduction="none")
    devices = d2l.try_all_gpus()
    
    if param_group:
        # Separate parameters: pre-trained vs new
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([
            {'params': params_1x},  # Pre-trained layers
            {'params': net.fc.parameters(), 'lr': learning_rate * 10}  # New layer
        ], lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(
            net.parameters(), 
            lr=learning_rate, 
            weight_decay=0.001
        )
    
    # Train the model
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, 
                   num_epochs, devices)
```

```python
# MXNet training function
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gluon.data.DataLoader(
        train_imgs.transform_first(train_augs), 
        batch_size, 
        shuffle=True
    )
    test_iter = gluon.data.DataLoader(
        test_imgs.transform_first(test_augs), 
        batch_size
    )
    
    devices = d2l.try_all_gpus()
    net.collect_params().reset_ctx(devices)
    net.hybridize()
    
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 
        'wd': 0.001
    })
    
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, 
                   num_epochs, devices)
```

## Step 7: Fine-tune the Model

Now let's train our fine-tuned model with a small learning rate for the pre-trained layers:

```python
# PyTorch
train_fine_tuning(finetune_net, 5e-5)
```

```python
# MXNet
train_fine_tuning(finetune_net, 0.01)
```

## Step 8: Compare with Training from Scratch

To demonstrate the value of fine-tuning, let's compare with a model trained from scratch:

```python
# PyTorch comparison
scratch_net = models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
```

```python
# MXNet comparison
scratch_net = models.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train_fine_tuning(scratch_net, 0.1)
```

You should observe that the fine-tuned model achieves better performance with fewer epochs, thanks to the pre-trained weights.

## Key Takeaways

1. **Transfer Learning**: Fine-tuning leverages knowledge from a large source dataset (ImageNet) to improve performance on a smaller target dataset.

2. **Layer-specific Learning Rates**: Pre-trained layers typically use smaller learning rates, while newly added layers use larger learning rates.

3. **Data Augmentation**: Crucial for preventing overfitting when working with small datasets.

4. **Performance Benefit**: Fine-tuned models generally converge faster and achieve better accuracy than models trained from scratch on small datasets.

## Exercises for Further Exploration

1. Experiment with different learning rates for the fine-tuned model. How does accuracy change with larger learning rates?

2. Try different hyperparameter combinations for both the fine-tuned and scratch models. Does the performance gap persist?

3. Freeze the pre-trained layers and only train the new classification layer. How does this affect model accuracy?

4. Investigate if there's a "hotdog" class in the original ImageNet dataset and explore how you might leverage its weights.

## Conclusion

Fine-tuning is an essential technique in modern deep learning, allowing you to build effective models even with limited data. By starting with pre-trained weights and carefully adjusting learning rates, you can create specialized models that perform well on specific tasks while saving significant training time and computational resources.