# Network in Network (NiN) Tutorial

This guide walks you through implementing the Network in Network (NiN) architecture. NiN addresses key limitations of earlier CNNs like AlexNet and VGG by replacing fully connected layers with global average pooling and using 1×1 convolutions to add nonlinearity.

## Prerequisites

First, install the required libraries and import the necessary modules. The code supports multiple frameworks (MXNet, PyTorch, TensorFlow, JAX).

```bash
# Install d2l library if needed
# pip install d2l
```

Choose your framework and import accordingly:

```python
# For PyTorch
from d2l import torch as d2l
import torch
from torch import nn

# For TensorFlow
import tensorflow as tf
from d2l import tensorflow as d2l

# For JAX
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp

# For MXNet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn
npx.set_np()
```

## 1. Understanding the NiN Block

The core innovation of NiN is the *NiN block*. Instead of stacking multiple large convolutional layers, a NiN block uses:
1. A standard convolution (e.g., 11×11, 5×5, or 3×3)
2. Two consecutive 1×1 convolutions that act as per-pixel fully connected layers, adding channel-wise nonlinearity

This design reduces parameters and increases nonlinearity without destroying spatial structure.

### Implementing the NiN Block

Define the `nin_block` function for your chosen framework:

```python
# PyTorch version
def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())

# TensorFlow version  
def nin_block(out_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(out_channels, kernel_size, strides=strides,
                               padding=padding),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(out_channels, 1),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(out_channels, 1),
        tf.keras.layers.Activation('relu')])

# JAX version
def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential([
        nn.Conv(out_channels, kernel_size, strides, padding),
        nn.relu,
        nn.Conv(out_channels, kernel_size=(1, 1)), nn.relu,
        nn.Conv(out_channels, kernel_size=(1, 1)), nn.relu])

# MXNet version
def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding,
                      activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk
```

## 2. Building the Complete NiN Model

Now, construct the full NiN architecture. The model follows this pattern:
- Three NiN blocks with decreasing kernel sizes (11×11, 5×5, 3×3)
- Max pooling after each block
- A final NiN block with output channels equal to the number of classes
- Global average pooling instead of fully connected layers

```python
# PyTorch, MXNet, TensorFlow version
class NiN(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        
        # PyTorch
        self.net = nn.Sequential(
            nin_block(96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            nin_block(num_classes, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())
        
        # Initialize weights (PyTorch specific)
        self.net.apply(d2l.init_cnn)

# JAX version
class NiN(d2l.Classifier):
    lr: float = 0.1
    num_classes = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nin_block(96, kernel_size=(11, 11), strides=(4, 4), padding=(0, 0)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nin_block(256, kernel_size=(5, 5), strides=(1, 1), padding=(2, 2)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nin_block(384, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nn.Dropout(0.5, deterministic=not self.training),
            nin_block(self.num_classes, kernel_size=(3, 3), strides=1, padding=(1, 1)),
            lambda x: nn.avg_pool(x, (5, 5)),  # global average pooling
            lambda x: x.reshape((x.shape[0], -1))  # flatten
        ])
```

## 3. Inspecting the Model Architecture

Let's verify the output shape at each layer to ensure our architecture is correct:

```python
# For PyTorch/MXNet (NCHW format)
NiN().layer_summary((1, 1, 224, 224))

# For TensorFlow/JAX (NHWC format)  
NiN().layer_summary((1, 224, 224, 1))
```

You should see the tensor dimensions progressively reduce in spatial size while increasing in channel depth, ending with a 10-element vector (for 10 classes).

## 4. Training the NiN Model

Now train the model on the Fashion-MNIST dataset. We'll use the same training setup as with AlexNet and VGG for comparison.

```python
# PyTorch/MXNet/JAX version
model = NiN(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))

# For PyTorch, initialize weights
if using_pytorch:
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)

trainer.fit(model, data)

# TensorFlow version
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = NiN(lr=0.05)
    trainer.fit(model, data)
```

## Key Insights

1. **Parameter Efficiency**: NiN eliminates giant fully connected layers, dramatically reducing parameters compared to AlexNet/VGG.
2. **Global Average Pooling**: This simple operation aggregates spatial information without learnable parameters, maintaining accuracy while reducing computation.
3. **1×1 Convolutions**: These add nonlinearity across channels at each spatial location, functioning like mini fully connected layers per pixel.

## Exercises for Further Exploration

1. Experiment with the number of 1×1 convolutions in each NiN block. Try one or three instead of two.
2. Replace 1×1 convolutions with 3×3 convolutions and observe the impact on performance and parameters.
3. Compare global average pooling with a fully connected layer in terms of speed, accuracy, and parameter count.
4. Calculate NiN's resource usage: number of parameters, FLOPs, and memory requirements during training and inference.
5. Consider the implications of reducing 384×5×5 features directly to 10×5×5 features in the final block.
6. Design a family of NiN networks inspired by VGG's depth variations (like VGG-11, VGG-16, VGG-19).

## Summary

NiN introduced two influential concepts to CNN design: 1×1 convolutions for cross-channel nonlinearity and global average pooling for parameter-efficient classification. These innovations address the parameter explosion problem in traditional CNNs while maintaining competitive accuracy, making NiN particularly suitable for resource-constrained environments.