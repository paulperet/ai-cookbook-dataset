# Implementing Residual Networks (ResNet) and ResNeXt

## Overview

In this tutorial, we'll explore Residual Networks (ResNet) and ResNeXt, two groundbreaking architectures that revolutionized deep learning by enabling the training of very deep neural networks. You'll learn the mathematical motivation behind residual connections, implement residual blocks, build complete ResNet models, and extend this knowledge to ResNeXt architectures.

## Prerequisites

First, let's set up our environment with the necessary imports. The code supports multiple deep learning frameworks - choose the one you're most comfortable with.

```python
# For MXNet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn
npx.set_np()

# For PyTorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F

# For TensorFlow
import tensorflow as tf
from d2l import tensorflow as d2l

# For JAX
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
```

## Understanding the Motivation: Function Classes

Before diving into implementation, let's understand why residual connections are so important. Consider $\mathcal{F}$, the class of functions that a specific network architecture can reach through training. We want to find the best function $f^*_\mathcal{F}$ within this class.

The key insight is that if we design a more powerful architecture $\mathcal{F}'$, we expect $f^*_{\mathcal{F}'}$ to be better than $f^*_{\mathcal{F}}$. However, this is only guaranteed if $\mathcal{F} \subseteq \mathcal{F}'$ - that is, if the larger function class contains the smaller one.

Residual networks address this by ensuring that every additional layer can easily learn the identity function $f(\mathbf{x}) = \mathbf{x}$. If the identity mapping is optimal, the residual mapping becomes $g(\mathbf{x}) = 0$, which is easier to learn than an arbitrary transformation.

## Step 1: Implementing the Residual Block

The core building block of ResNet is the residual block. Instead of learning the desired mapping $f(\mathbf{x})$ directly, we learn the residual $g(\mathbf{x}) = f(\mathbf{x}) - \mathbf{x}$.

### 1.1 Basic Residual Block Implementation

Let's implement the residual block for each framework:

```python
# MXNet implementation
class Residual(nn.Block):
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,
                               strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return npx.relu(Y + X)

# PyTorch implementation
class Residual(nn.Module):
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# TensorFlow implementation
class Residual(tf.keras.Model):
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(num_channels, padding='same',
                                            kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                            padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                                                strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)

# JAX implementation
class Residual(nn.Module):
    """The Residual block of ResNet models."""
    num_channels: int
    use_1x1conv: bool = False
    strides: tuple = (1, 1)
    training: bool = True

    def setup(self):
        self.conv1 = nn.Conv(self.num_channels, kernel_size=(3, 3),
                             padding='same', strides=self.strides)
        self.conv2 = nn.Conv(self.num_channels, kernel_size=(3, 3),
                             padding='same')
        if self.use_1x1conv:
            self.conv3 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                                 strides=self.strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm(not self.training)
        self.bn2 = nn.BatchNorm(not self.training)

    def __call__(self, X):
        Y = nn.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return nn.relu(Y)
```

### 1.2 Testing the Residual Block

Now let's test our residual block implementation. First, we'll create a block where input and output have the same shape:

```python
# Create a residual block with 3 output channels
blk = Residual(3)

# Test with random input
X = d2l.randn(4, 3, 6, 6)  # Batch of 4, 3 channels, 6x6 images
output_shape = blk(X).shape
print(f"Output shape: {output_shape}")
```

Next, let's test a block that halves the spatial dimensions while increasing channels:

```python
# Create a residual block with 1x1 convolution and stride 2
blk = Residual(6, use_1x1conv=True, strides=2)
output_shape = blk(X).shape
print(f"Output shape with downsampling: {output_shape}")
```

## Step 2: Building the Complete ResNet Model

Now that we have our residual blocks, let's build a complete ResNet model. We'll implement ResNet-18, which has 18 layers total.

### 2.1 Defining the Initial Block

The first block of ResNet is similar to other CNN architectures but includes batch normalization:

```python
class ResNet(d2l.Classifier):
    def b1(self):
        # MXNet implementation
        if framework == 'mxnet':
            net = nn.Sequential()
            net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
                    nn.BatchNorm(), nn.Activation('relu'),
                    nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            return net
        
        # PyTorch implementation
        elif framework == 'pytorch':
            return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                nn.LazyBatchNorm2d(), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        # TensorFlow implementation
        elif framework == 'tensorflow':
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, kernel_size=7, strides=2,
                                       padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2,
                                          padding='same')])
        
        # JAX implementation
        elif framework == 'jax':
            return nn.Sequential([
                nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
                nn.BatchNorm(not self.training), nn.relu,
                lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2),
                                      padding='same')])
```

### 2.2 Creating Residual Block Groups

ResNet organizes residual blocks into groups. Each group has the same number of output channels, and the first block in each group (except the first) uses stride 2 to downsample:

```python
@d2l.add_to_class(ResNet)
def block(self, num_residuals, num_channels, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # First block in group (except first group) downsamples
            blk.append(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels))
    
    # Convert to sequential container based on framework
    if framework == 'pytorch':
        return nn.Sequential(*blk)
    elif framework == 'jax':
        return nn.Sequential(blk)
    else:
        # MXNet and TensorFlow handle this differently
        return container_type(blk)
```

### 2.3 Assembling the Complete ResNet

Now let's assemble the complete ResNet model:

```python
@d2l.add_to_class(ResNet)
def __init__(self, arch, lr=0.1, num_classes=10):
    super(ResNet, self).__init__()
    self.save_hyperparameters()
    
    # Build the network
    self.net = self.b1()
    
    # Add residual block groups
    for i, (num_residuals, num_channels) in enumerate(arch):
        self.net.add(self.block(num_residuals, num_channels, 
                               first_block=(i==0)))
    
    # Add final layers
    self.net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
```

### 2.4 Creating ResNet-18

Let's create a specific ResNet-18 configuration:

```python
class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=10):
        # Architecture: (num_residuals, num_channels) for each group
        arch = ((2, 64), (2, 128), (2, 256), (2, 512))
        super().__init__(arch, lr, num_classes)
```

### 2.5 Visualizing the Architecture

Let's examine how the input shape changes through the network:

```python
model = ResNet18()
# For PyTorch/MXNet
model.layer_summary((1, 1, 96, 96))
# For TensorFlow/JAX
model.layer_summary((1, 96, 96, 1))
```

## Step 3: Training ResNet on Fashion-MNIST

Now let's train our ResNet-18 model on the Fashion-MNIST dataset:

```python
# Prepare the data
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))

# Create and train the model
model = ResNet18(lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)

# For PyTorch, we need to initialize the weights
if framework == 'pytorch':
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], 
                     d2l.init_cnn)

# Train the model
trainer.fit(model, data)
```

## Step 4: Implementing ResNeXt

ResNeXt extends ResNet by using grouped convolutions, which are more computationally efficient while maintaining or improving accuracy.

### 4.1 Understanding Grouped Convolutions

A grouped convolution splits the input channels into $g$ groups and performs separate convolutions on each group. This reduces computational cost from $\mathcal{O}(c_\textrm{i} \cdot c_\textrm{o})$ to $\mathcal{O}(c_\textrm{i} \cdot c_\textrm{o} / g)$.

### 4.2 Implementing the ResNeXt Block

```python
# MXNet implementation
class ResNeXtBlock(nn.Block):
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul,
                 use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.Conv2D(bot_channels, kernel_size=1, padding=0,
                               strides=1)
        self.conv2 = nn.Conv2D(bot_channels, kernel_size=3, padding=1, 
                               strides=strides, groups=bot_channels//groups)
        self.conv3 = nn.Conv2D(num_channels, kernel_size=1, padding=0,
                               strides=1)
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
        self.bn3 = nn.BatchNorm()
        if use_1x1conv:
            self.conv4 = nn.Conv2D(num_channels, kernel_size=1,
                                   strides=strides)
            self.bn4 = nn.BatchNorm()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = npx.relu(self.bn1(self.conv1(X)))
        Y = npx.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return npx.relu(Y + X)

# PyTorch implementation
class ResNeXtBlock(nn.Module):
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=bot_channels//groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1, 
                                       stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)

# TensorFlow implementation
class ResNeXtBlock(tf.keras.Model):
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = tf.keras.layers.Conv2D(bot_channels, 1, strides=1)
        self.conv2 = tf.keras.layers.Conv2D(bot_channels, 3, strides=strides,
                                            padding="same",
                                            groups=bot_channels//groups)
        self.conv3 = tf.keras.layers.Conv2D(num_channels, 1, strides=1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        if use_1x1conv:
            self.conv4 = tf.keras.layers.Conv2D(num_channels, 1,
                                                strides=strides)
            self.bn4 = tf.keras.layers.BatchNormalization()
        else:
            self.conv4 = None

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = tf.keras.activations.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4