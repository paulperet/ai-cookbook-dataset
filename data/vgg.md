# Networks Using Blocks (VGG): A Practical Guide
:label:`sec_vgg`

This guide introduces the VGG network architecture, a pivotal model that popularized the use of repeated convolutional blocks to build deep neural networks. You will learn the core concepts, implement the VGG block, construct a full VGG network, and train it on a dataset.

## Prerequisites

Ensure you have the necessary deep learning framework installed. The code supports MXNet, PyTorch, TensorFlow, and JAX. Import the required libraries.

```python
# For MXNet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn
npx.set_np()
```

```python
# For PyTorch
from d2l import torch as d2l
import torch
from torch import nn
```

```python
# For TensorFlow
import tensorflow as tf
from d2l import tensorflow as d2l
```

```python
# For JAX
from d2l import jax as d2l
from flax import linen as nn
import jax
```

## 1. Understanding VGG Blocks

The key innovation of VGG is the **VGG block**, a repeating pattern of layers that forms the building block of the network. A standard VGG block consists of:
1.  A sequence of convolutional layers with 3x3 kernels and padding of 1 (preserving spatial dimensions).
2.  A ReLU activation function after each convolution.
3.  A single 2x2 max-pooling layer with stride 2 (halving the spatial dimensions).

This design allows the network to be deep while controlling resolution reduction. Using multiple small (3x3) convolutions in sequence is computationally similar to a larger receptive field but with more non-linearity and fewer parameters.

## 2. Implementing a VGG Block

Let's implement a function `vgg_block` that creates one of these blocks. It takes two arguments: `num_convs` (number of convolutional layers in the block) and `num_channels` (number of output channels for those convolutions).

```python
# For MXNet
def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```python
# For PyTorch
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```python
# For TensorFlow
def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(
            tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                   padding='same', activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```python
# For JAX
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv(out_channels, kernel_size=(3, 3), padding=(1, 1)))
        layers.append(nn.relu)
    layers.append(lambda x: nn.max_pool(x, window_shape=(2, 2), strides=(2, 2)))
    return nn.Sequential(layers)
```

## 3. Constructing the VGG Network

The full VGG network connects several VGG blocks in sequence, followed by a classifier head of fully connected layers. The architecture is defined by a list of tuples specifying the configuration for each block.

We'll define a `VGG` class that inherits from a base `Classifier`. The `arch` parameter is a list like `[(num_convs_block1, channels_block1), (num_convs_block2, channels_block2), ...]`.

```python
# For PyTorch, MXNet, TensorFlow
class VGG(d2l.Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            conv_blks = []
            for (num_convs, out_channels) in arch:
                conv_blks.append(vgg_block(num_convs, out_channels))
            self.net = nn.Sequential(
                *conv_blks, nn.Flatten(),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(num_classes))
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(
                tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes)]))
```

```python
# For JAX
class VGG(d2l.Classifier):
    arch: list
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        conv_blks = []
        for (num_convs, out_channels) in self.arch:
            conv_blks.append(vgg_block(num_convs, out_channels))

        self.net = nn.Sequential([
            *conv_blks,
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(4096), nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(4096), nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(self.num_classes)])
```

## 4. Exploring VGG-11

The original VGG-11 network uses the following architecture: five blocks with configurations `((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))`. This results in 11 weight layers (8 convolutional + 3 fully connected). Let's instantiate it and inspect the layer dimensions.

```python
# For PyTorch/MXNet
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary((1, 1, 224, 224))
```

```python
# For TensorFlow
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary((1, 224, 224, 1))
```

```python
# For JAX
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)), training=False).layer_summary((1, 224, 224, 1))
```

The output shows how each block reduces the spatial dimensions (height and width) by half, eventually reaching 7x7 before the flattening operation.

## 5. Training a Compact VGG Model

VGG-11 is computationally intensive. For a practical demonstration on the Fashion-MNIST dataset, we'll use a smaller variant with fewer channels. We'll resize the images to 224x224 to match the network's expected input size.

```python
# For MXNet, PyTorch, JAX
model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```python
# For TensorFlow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
    trainer.fit(model, data)
```

Training for 10 epochs should show a close match between training and validation loss, indicating minimal overfitting with this configuration.

## Summary

VGG established the template of using repeated convolutional blocks to build deep networks. Its preference for deep, narrow architectures with small 3x3 convolutions became a standard design pattern. By implementing VGG, you've seen how to construct complex networks from reusable blocks—a practice that modern deep learning frameworks make straightforward through modular code.

## Exercises

1.  **Computational Analysis**: Compare AlexNet and VGG in terms of the number of parameters and floating-point operations. How could you reduce the computational cost of the fully connected layers in VGG?
2.  **Layer Counting**: The VGG-11 network summary shows information for eight blocks, but the network is described as having 11 layers. Account for the missing layers.
3.  **Architecture Variants**: Using Table 1 from the VGG paper, construct the VGG-16 or VGG-19 architectures.
4.  **Input Resolution**: The 224x224 input for Fashion-MNIST is computationally expensive. Experiment with modifying the network architecture and input resolution (e.g., 56x56 or 84x84). Can you maintain accuracy? Refer to the VGG paper for ideas on adding nonlinearities before downsampling.

---
*For discussion on this topic, please visit the [D2L.ai forum](https://discuss.d2l.ai).*