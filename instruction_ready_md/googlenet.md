# Implementing GoogLeNet: A Multi-Branch Network Architecture

## Introduction

GoogLeNet, the 2014 ImageNet Challenge winner, introduced a groundbreaking architecture that combined multiple convolutional kernel sizes within a single block. This tutorial will guide you through implementing GoogLeNet's key innovation - the Inception block - and building the complete network architecture. You'll learn how to create a network that efficiently extracts features at multiple scales while managing computational complexity.

## Prerequisites

First, let's set up our environment by importing the necessary libraries. The code supports multiple deep learning frameworks - choose the one you're most comfortable with.

```python
# Framework selection - choose one
import os
framework = "pytorch"  # Change to "mxnet", "tensorflow", or "jax" as needed

if framework == "mxnet":
    from mxnet import np, npx, init
    from mxnet.gluon import nn
    npx.set_np()
elif framework == "pytorch":
    import torch
    from torch import nn
    from torch.nn import functional as F
elif framework == "tensorflow":
    import tensorflow as tf
elif framework == "jax":
    from flax import linen as nn
    from jax import numpy as jnp
    import jax

# Import D2L utilities for all frameworks
from d2l import torch as d2l  # Adjust import based on your framework choice
```

## Step 1: Understanding the Inception Block

The core innovation of GoogLeNet is the Inception block, which processes input through four parallel branches with different convolutional kernel sizes. This allows the network to capture features at multiple scales simultaneously.

### Inception Block Architecture

The Inception block consists of four parallel branches:
1. **Branch 1**: 1×1 convolution
2. **Branch 2**: 1×1 convolution followed by 3×3 convolution
3. **Branch 3**: 1×1 convolution followed by 5×5 convolution
4. **Branch 4**: 3×3 max-pooling followed by 1×1 convolution

All branches use appropriate padding to maintain spatial dimensions, and their outputs are concatenated along the channel dimension.

## Step 2: Implementing the Inception Block

Let's implement the Inception block for your chosen framework. The hyperparameters `c1`, `c2`, `c3`, and `c4` control the number of output channels for each branch.

```python
if framework == "mxnet":
    class Inception(nn.Block):
        def __init__(self, c1, c2, c3, c4, **kwargs):
            super(Inception, self).__init__(**kwargs)
            # Branch 1: 1x1 convolution
            self.b1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
            # Branch 2: 1x1 followed by 3x3 convolution
            self.b2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
            self.b2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1, activation='relu')
            # Branch 3: 1x1 followed by 5x5 convolution
            self.b3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
            self.b3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2, activation='relu')
            # Branch 4: Max pooling followed by 1x1 convolution
            self.b4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
            self.b4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

        def forward(self, x):
            b1 = self.b1_1(x)
            b2 = self.b2_2(self.b2_1(x))
            b3 = self.b3_2(self.b3_1(x))
            b4 = self.b4_2(self.b4_1(x))
            return np.concatenate((b1, b2, b3, b4), axis=1)

elif framework == "pytorch":
    class Inception(nn.Module):
        def __init__(self, c1, c2, c3, c4, **kwargs):
            super(Inception, self).__init__(**kwargs)
            # Branch 1: 1x1 convolution
            self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
            # Branch 2: 1x1 followed by 3x3 convolution
            self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
            self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
            # Branch 3: 1x1 followed by 5x5 convolution
            self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
            self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
            # Branch 4: Max pooling followed by 1x1 convolution
            self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

        def forward(self, x):
            b1 = F.relu(self.b1_1(x))
            b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
            b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
            b4 = F.relu(self.b4_2(self.b4_1(x)))
            return torch.cat((b1, b2, b3, b4), dim=1)

elif framework == "tensorflow":
    class Inception(tf.keras.Model):
        def __init__(self, c1, c2, c3, c4):
            super().__init__()
            self.b1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
            self.b2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
            self.b2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same', activation='relu')
            self.b3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
            self.b3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same', activation='relu')
            self.b4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
            self.b4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')

        def call(self, x):
            b1 = self.b1_1(x)
            b2 = self.b2_2(self.b2_1(x))
            b3 = self.b3_2(self.b3_1(x))
            b4 = self.b4_2(self.b4_1(x))
            return tf.keras.layers.Concatenate()([b1, b2, b3, b4])

elif framework == "jax":
    class Inception(nn.Module):
        c1: int
        c2: tuple
        c3: tuple
        c4: int

        def setup(self):
            self.b1_1 = nn.Conv(self.c1, kernel_size=(1, 1))
            self.b2_1 = nn.Conv(self.c2[0], kernel_size=(1, 1))
            self.b2_2 = nn.Conv(self.c2[1], kernel_size=(3, 3), padding='same')
            self.b3_1 = nn.Conv(self.c3[0], kernel_size=(1, 1))
            self.b3_2 = nn.Conv(self.c3[1], kernel_size=(5, 5), padding='same')
            self.b4_1 = lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(1, 1), padding='same')
            self.b4_2 = nn.Conv(self.c4, kernel_size=(1, 1))

        def __call__(self, x):
            b1 = nn.relu(self.b1_1(x))
            b2 = nn.relu(self.b2_2(nn.relu(self.b2_1(x))))
            b3 = nn.relu(self.b3_2(nn.relu(self.b3_1(x))))
            b4 = nn.relu(self.b4_2(self.b4_1(x)))
            return jnp.concatenate((b1, b2, b3, b4), axis=-1)
```

## Step 3: Building the GoogLeNet Architecture

Now let's construct the complete GoogLeNet model. The architecture consists of five main blocks (b1 through b5), each with specific purposes.

### Block 1: Initial Feature Extraction

The first block serves as the network stem, performing initial feature extraction with a 7×7 convolutional layer followed by max-pooling.

```python
class GoogleNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        
        if framework == "mxnet":
            self.net = nn.Sequential()
            self.net.add(self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
                         nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        elif framework == "pytorch":
            self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), self.b4(),
                                     self.b5(), nn.LazyLinear(num_classes))
        elif framework == "tensorflow":
            self.net = tf.keras.Sequential([
                self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
                tf.keras.layers.Dense(num_classes)
            ])
        elif framework == "jax":
            self.net = nn.Sequential([
                self.b1(), self.b2(), self.b3(), self.b4(), self.b5(),
                nn.Dense(num_classes)
            ])

    def b1(self):
        """First block: 7x7 convolution and max pooling"""
        if framework == "mxnet":
            net = nn.Sequential()
            net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
                    nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            return net
        elif framework == "pytorch":
            return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        elif framework == "tensorflow":
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
        elif framework == "jax":
            return nn.Sequential([
                nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
                nn.relu,
                lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='same')
            ])
```

### Block 2: Channel Expansion

The second block expands the channel dimension using 1×1 and 3×3 convolutions.

```python
@d2l.add_to_class(GoogleNet)
def b2(self):
    """Second block: Channel expansion"""
    if framework == "mxnet":
        net = nn.Sequential()
        net.add(nn.Conv2D(64, kernel_size=1, activation='relu'),
               nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    elif framework == "pytorch":
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    elif framework == "tensorflow":
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 1, activation='relu'),
            tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
    elif framework == "jax":
        return nn.Sequential([
            nn.Conv(64, kernel_size=(1, 1)), nn.relu,
            nn.Conv(192, kernel_size=(3, 3), padding='same'), nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='same')
        ])
```

### Block 3: First Inception Blocks

The third block introduces the first Inception blocks, carefully balancing channel capacities across branches.

```python
@d2l.add_to_class(GoogleNet)
def b3(self):
    """Third block: First Inception blocks"""
    if framework == "mxnet":
        net = nn.Sequential()
        net.add(Inception(64, (96, 128), (16, 32), 32),
               Inception(128, (128, 192), (32, 96), 64),
               nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    elif framework == "pytorch":
        return nn.Sequential(
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    elif framework == "tensorflow":
        return tf.keras.models.Sequential([
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
    elif framework == "jax":
        return nn.Sequential([
            Inception(64, (96, 128), (16, 32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='same')
        ])
```

### Block 4: Deep Inception Stack

The fourth block contains five Inception blocks in series, creating a deep feature extraction pipeline.

```python
@d2l.add_to_class(GoogleNet)
def b4(self):
    """Fourth block: Deep Inception stack"""
    if framework == "mxnet":
        net = nn.Sequential()
        net.add(Inception(192, (96, 208), (16, 48), 64),
                Inception(160, (112, 224), (24, 64), 64),
                Inception(128, (128, 256), (24, 64), 64),
                Inception(112, (144, 288), (32, 64), 64),
                Inception(256, (160, 320), (32, 128), 128),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        return net
    elif framework == "pytorch":
        return nn.Sequential(
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    elif framework == "tensorflow":
        return tf.keras.Sequential([
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
    elif framework == "jax":
        return nn.Sequential([
            Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24, 64), 64),
            Inception(128, (128, 256), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='same')
        ])
```

### Block 5: Final Inception Blocks and Classification Head

The fifth block contains the final Inception blocks followed by global average pooling and