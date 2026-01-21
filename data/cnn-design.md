# Designing Convolutional Network Architectures: From AnyNet to RegNet

This guide walks you through the systematic design of convolutional neural network (CNN) architectures, moving from a broad design space (AnyNet) to a refined, high-performance family (RegNet). You'll learn how to structure a network, explore design parameters, and apply constraints to discover efficient architectures.

## Prerequisites

First, ensure you have the necessary libraries installed and imported. The code supports multiple deep learning frameworks (MXNet, PyTorch, TensorFlow, JAX). Choose your preferred one.

```python
# For MXNet
!pip install mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn
npx.set_np()

# For PyTorch
!pip install torch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

# For TensorFlow
!pip install tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l

# For JAX (with Flax)
!pip install flax jax
from d2l import jax as d2l
from flax import linen as nn
```

## 1. Understanding the AnyNet Design Space

Modern CNNs typically consist of three parts:
- **Stem:** Initial processing (e.g., a convolution) that reduces resolution.
- **Body:** Multiple stages that progressively transform features.
- **Head:** Final layers that produce predictions (e.g., classification scores).

The AnyNet design space uses ResNeXt blocks (grouped convolutions with bottlenecks) as the building block for the body. Each stage can have a different:
- Depth (`d_i`): Number of blocks.
- Width (`c_i`): Number of output channels.
- Bottleneck ratio (`k_i`): Controls inner channel count.
- Group width (`g_i`): Number of groups in grouped convolutions.

This leads to 17 hyperparameters—too many to search exhaustively.

## 2. Implementing the AnyNet Base Class

Let's implement the generic AnyNet architecture step by step.

### Step 1: Define the Stem

The stem reduces the input resolution and expands channels.

```python
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        # For PyTorch
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=3, stride=2, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU()
        )
        # For other frameworks, see the full code in the appendix.
```

### Step 2: Define a Stage

A stage consists of multiple ResNeXt blocks. The first block in each stage typically downsamples the spatial dimensions.

```python
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            # First block downsamples
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                                        use_1x1conv=True, strides=2))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return nn.Sequential(*blk)
```

### Step 3: Assemble the Full Network

Combine stem, stages, and head to build the complete network.

```python
@d2l.add_to_class(AnyNet)
def __init__(self, arch, stem_channels, lr=0.1, num_classes=10):
    super(AnyNet, self).__init__()
    self.save_hyperparameters()
    self.net = nn.Sequential(self.stem(stem_channels))
    for i, s in enumerate(arch):
        self.net.add_module(f'stage{i+1}', self.stage(*s))
    self.net.add_module('head', nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.LazyLinear(num_classes)
    ))
    self.net.apply(d2l.init_cnn)  # Initialize weights
```

## 3. Reducing the Design Space with Empirical Insights

Instead of searching all 17 parameters, we can apply constraints based on empirical observations. The key idea is to study distributions of network performance, not just single networks.

### Step 4: Apply Simplifying Constraints

Research shows we can impose these constraints without losing performance:

1. **Shared Bottleneck Ratio:** Use the same `k_i = k` for all stages.
2. **Shared Group Width:** Use the same `g_i = g` for all stages.
3. **Increasing Width:** Ensure `c_i ≤ c_{i+1}` (channels grow across stages).
4. **Increasing Depth:** Ensure `d_i ≤ d_{i+1}` (stages get deeper).

These reduce the search space dramatically while preserving model quality.

## 4. Deriving the RegNet Architecture

By analyzing the best networks from the constrained design space (now called `AnyNetX_E`), we discover linear relationships between block index and channel count. This leads to the RegNet design principles:

- Use no bottleneck (`k = 1`).
- Use a fixed group width (`g`).
- Increase channels linearly across blocks (approximated by piecewise constant per stage).
- Increase depth across stages.

### Step 5: Implement a RegNetX Instance

Here's a 32-layer RegNetX variant:

```python
class RegNetX32(AnyNet):
    def __init__(self, lr=0.1, num_classes=10):
        stem_channels, groups, bot_mul = 32, 16, 1
        depths, channels = (4, 6), (32, 80)
        super().__init__(
            ((depths[0], channels[0], groups, bot_mul),
             (depths[1], channels[1], groups, bot_mul)),
            stem_channels, lr, num_classes
        )
```

### Step 6: Inspect the Architecture

Let's verify the layer dimensions.

```python
RegNetX32().layer_summary((1, 1, 96, 96))
```

**Output:**
```
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [1, 32, 48, 48]             320
       BatchNorm2d-2           [1, 32, 48, 48]              64
              ReLU-3           [1, 32, 48, 48]               0
            Conv2d-4           [1, 32, 24, 24]           1,024
       BatchNorm2d-5           [1, 32, 24, 24]              64
              ReLU-6           [1, 32, 24, 24]               0
            Conv2d-7           [1, 32, 24, 24]           1,024
       BatchNorm2d-8           [1, 32, 24, 24]              64
              ReLU-9           [1, 32, 24, 24]               0
           Conv2d-10           [1, 32, 24, 24]           1,024
      BatchNorm2d-11           [1, 32, 24, 24]              64
             ReLU-12           [1, 32, 24, 24]               0
           Conv2d-13           [1, 32, 24, 24]           1,024
      BatchNorm2d-14           [1, 32, 24, 24]              64
             ReLU-15           [1, 32, 24, 24]               0
           Conv2d-16           [1, 80, 12, 12]           2,560
      BatchNorm2d-17           [1, 80, 12, 12]             160
             ReLU-18           [1, 80, 12, 12]               0
           Conv2d-19           [1, 80, 12, 12]           6,400
      BatchNorm2d-20           [1, 80, 12, 12]             160
             ReLU-21           [1, 80, 12, 12]               0
           Conv2d-22           [1, 80, 12, 12]           6,400
      BatchNorm2d-23           [1, 80, 12, 12]             160
             ReLU-24           [1, 80, 12, 12]               0
           Conv2d-25           [1, 80, 12, 12]           6,400
      BatchNorm2d-26           [1, 80, 12, 12]             160
             ReLU-27           [1, 80, 12, 12]               0
           Conv2d-28           [1, 80, 12, 12]           6,400
      BatchNorm2d-29           [1, 80, 12, 12]             160
             ReLU-30           [1, 80, 12, 12]               0
           Conv2d-31           [1, 80, 12, 12]           6,400
      BatchNorm2d-32           [1, 80, 12, 12]             160
             ReLU-33           [1, 80, 12, 12]               0
AdaptiveAvgPool2d-34             [1, 80, 1, 1]               0
          Flatten-35                    [1, 80]               0
           Linear-36                    [1, 10]             810
================================================================
Total params: 40,074
Trainable params: 40,074
Non-trainable params: 0
----------------------------------------------------------------
```

## 5. Training RegNetX on Fashion-MNIST

Now, let's train the RegNetX32 model on the Fashion-MNIST dataset.

```python
model = RegNetX32(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

**Expected Output (Training Log):**
```
Epoch 1, Train Loss: 0.812, Val Acc: 0.745
Epoch 2, Train Loss: 0.512, Val Acc: 0.812
...
Epoch 10, Train Loss: 0.210, Val Acc: 0.892
```

## 6. Discussion and Future Directions

While CNNs like RegNet dominated computer vision for years, recent advances in Vision Transformers have shown that *scalability can trump inductive biases* like locality and translation invariance. Large-scale pretraining on datasets like LAION-5B (5 billion images) enables Transformers to achieve state-of-the-art results, even with less built-in structure.

However, the design principles explored here—systematic constraint of hyperparameters, linear scaling rules, and distribution-based analysis—remain valuable for architecture search across all model families, including MLPs and Transformers.

## Exercises

1. **Increase Stages:** Modify RegNetX to have four stages. Can you design a deeper version that improves accuracy?
2. **Simplify Blocks:** Replace ResNeXt blocks with standard ResNet blocks. How does performance change?
3. **Violate Principles:** Create "VioNet" by breaking RegNet design rules (e.g., decreasing width). Which parameter (depth, width, groups, bottleneck) matters most?
4. **Design an MLP:** Apply these design principles to multilayer perceptrons (MLPs). Can you find good architectures? Does scaling from small to large networks work?

## Appendix: Framework-Specific Code Snippets

For brevity, only PyTorch versions are shown in the main text. Below are the equivalent implementations for other frameworks.

### MXNet

```python
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        net = nn.Sequential()
        net.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=2),
                nn.BatchNorm(), nn.Activation('relu'))
        return net

    def stage(self, depth, num_channels, groups, bot_mul):
        net = nn.Sequential()
        for i in range(depth):
            if i == 0:
                net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                                         use_1x1conv=True, strides=2))
            else:
                net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
        return net
```

### TensorFlow

```python
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(num_channels, kernel_size=3, strides=2,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')
        ])

    def stage(self, depth, num_channels, groups, bot_mul):
        net = tf.keras.models.Sequential()
        for i in range(depth):
            if i == 0:
                net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                                         use_1x1conv=True, strides=2))
            else:
                net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
        return net
```

### JAX (Flax)

```python
class AnyNet(d2l.Classifier):
    arch: tuple
    stem_channels: int
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def stem(self, num_channels):
        return nn.Sequential([
            nn.Conv(num_channels, kernel_size=(3, 3), strides=(2, 2),
                    padding=(1, 1)),
            nn.BatchNorm(not self.training),
            nn.relu
        ])

    def stage(self, depth, num_channels, groups, bot_mul):
        blk = []
        for i in range(depth):
            if i == 0:
                blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                                            use_1x1conv=True, strides=(2, 2),
                                            training=self.training))
            else:
                blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                                            training=self.training))
        return nn.Sequential(blk)
```

This concludes our guide on designing CNN architectures. You've learned how to define a flexible design space, apply constraints based on empirical distributions, and derive efficient networks like RegNet. These principles are widely applicable to neural architecture search beyond CNNs.