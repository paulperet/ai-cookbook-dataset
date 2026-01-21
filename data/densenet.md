# Densely Connected Networks (DenseNet)

This guide walks you through implementing DenseNet, a powerful convolutional neural network architecture that builds upon the concepts introduced by ResNet. DenseNet enhances feature reuse by connecting all layers directly with each other, leading to more efficient and accurate models.

## Prerequisites

Ensure you have the necessary libraries installed. This tutorial supports multiple deep learning frameworks.

```bash
# Install the d2l library which provides common functions and datasets
pip install d2l
```

Depending on your chosen framework, you may also need:

```bash
# For MXNet
pip install mxnet

# For PyTorch
pip install torch torchvision

# For TensorFlow
pip install tensorflow

# For JAX (with Flax)
pip install flax jax jaxlib
```

Now, import the required modules for your framework.

```python
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```python
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```python
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```python
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
```

## Understanding DenseNet: From ResNet to Dense Connections

ResNet introduced the idea of adding an input to the output of a block via skip connections: `f(x) = x + g(x)`. DenseNet extends this by concatenating, rather than adding, the input with the output of each layer. This creates a dense connectivity pattern where each layer receives feature maps from all preceding layers.

Mathematically, if ResNet represents a first-order expansion, DenseNet captures higher-order terms by repeatedly concatenating transformed features:

```
x → [x, f₁(x), f₂([x, f₁(x)]), f₃([x, f₁(x), f₂([x, f₁(x)])]), ...]
```

This dense reuse of features improves gradient flow and parameter efficiency.

## Step 1: Building the Convolutional Block

The fundamental building block in DenseNet is a convolutional block that performs batch normalization, applies a ReLU activation, and then a 3×3 convolution. This structure is similar to the pre-activation ResNet block.

Define the convolutional block for your framework:

```python
%%tab mxnet
def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

```python
%%tab pytorch
def conv_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1))
```

```python
%%tab tensorflow
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3), padding='same')
        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x, y], axis=-1)
        return y
```

```python
%%tab jax
class ConvBlock(nn.Module):
    num_channels: int
    training: bool = True

    @nn.compact
    def __call__(self, X):
        Y = nn.relu(nn.BatchNorm(not self.training)(X))
        Y = nn.Conv(self.num_channels, kernel_size=(3, 3), padding=(1, 1))(Y)
        Y = jnp.concatenate((X, Y), axis=-1)
        return Y
```

## Step 2: Creating a Dense Block

A dense block stacks multiple convolutional blocks. The key operation is that each block's output is concatenated with the current feature map along the channel dimension. This causes the number of channels to grow with each block.

Implement the dense block:

```python
%%tab mxnet
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels):
        super().__init__()
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = np.concatenate((X, Y), axis=1)
        return X
```

```python
%%tab pytorch
class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X
```

```python
%%tab tensorflow
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers:
            x = layer(x)
        return x
```

```python
%%tab jax
class DenseBlock(nn.Module):
    num_convs: int
    num_channels: int
    training: bool = True

    def setup(self):
        layer = []
        for i in range(self.num_convs):
            layer.append(ConvBlock(self.num_channels, self.training))
        self.net = nn.Sequential(layer)

    def __call__(self, X):
        return self.net(X)
```

### Testing the Dense Block

Let's verify the dense block works correctly. We'll create a block with 2 convolutional layers, each producing 10 output channels. Given an input with 3 channels, the output should have `3 + 10 + 10 = 23` channels.

```python
%%tab pytorch, mxnet, tensorflow
blk = DenseBlock(2, 10)
if tab.selected('mxnet'):
    X = np.random.uniform(size=(4, 3, 8, 8))
    blk.initialize()
if tab.selected('pytorch'):
    X = torch.randn(4, 3, 8, 8)
if tab.selected('tensorflow'):
    X = tf.random.uniform((4, 8, 8, 3))
Y = blk(X)
print(f'Output shape: {Y.shape}')
```

```python
%%tab jax
blk = DenseBlock(2, 10)
X = jnp.zeros((4, 8, 8, 3))
Y = blk.init_with_output(d2l.get_key(), X)[0]
print(f'Output shape: {Y.shape}')
```

## Step 3: Adding Transition Layers

To control model complexity and reduce spatial dimensions, we insert transition layers between dense blocks. A transition layer consists of a 1×1 convolution (to reduce channels) followed by average pooling (to halve height and width).

Define the transition layer:

```python
%%tab mxnet
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

```python
%%tab pytorch
def transition_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

```python
%%tab tensorflow
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)
```

```python
%%tab jax
class TransitionBlock(nn.Module):
    num_channels: int
    training: bool = True

    @nn.compact
    def __call__(self, X):
        X = nn.BatchNorm(not self.training)(X)
        X = nn.relu(X)
        X = nn.Conv(self.num_channels, kernel_size=(1, 1))(X)
        X = nn.avg_pool(X, window_shape=(2, 2), strides=(2, 2))
        return X
```

Apply the transition layer to the output from our dense block test. This should reduce the channels to 10 and halve the spatial dimensions.

```python
%%tab mxnet
blk = transition_block(10)
blk.initialize()
print(f'Transition output shape: {blk(Y).shape}')
```

```python
%%tab pytorch
blk = transition_block(10)
print(f'Transition output shape: {blk(Y).shape}')
```

```python
%%tab tensorflow
blk = TransitionBlock(10)
print(f'Transition output shape: {blk(Y).shape}')
```

```python
%%tab jax
blk = TransitionBlock(10)
print(f'Transition output shape: {blk.init_with_output(d2l.get_key(), Y)[0].shape}')
```

## Step 4: Assembling the Full DenseNet Model

Now, we'll construct the complete DenseNet. The model begins with a single convolutional layer and max-pooling (similar to ResNet), followed by a series of dense blocks and transition layers, and ends with global average pooling and a fully connected layer.

Define the initial block:

```python
%%tab pytorch, mxnet, tensorflow
class DenseNet(d2l.Classifier):
    def b1(self):
        if tab.selected('mxnet'):
            net = nn.Sequential()
            net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
                nn.BatchNorm(), nn.Activation('relu'),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            return net
        if tab.selected('pytorch'):
            return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                nn.LazyBatchNorm2d(), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if tab.selected('tensorflow'):
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(
                    64, kernel_size=7, strides=2, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(
                    pool_size=3, strides=2, padding='same')])
```

```python
%%tab jax
class DenseNet(d2l.Classifier):
    num_channels: int = 64
    growth_rate: int = 32
    arch: tuple = (4, 4, 4, 4)
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def b1(self):
        return nn.Sequential([
            nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
            nn.BatchNorm(not self.training),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3),
                                  strides=(2, 2), padding='same')
        ])
```

Now, define the full network architecture. We'll use four dense blocks with 4 convolutional layers each and a growth rate of 32 (meaning each convolution adds 32 channels).

```python
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(DenseNet)
def __init__(self, num_channels=64, growth_rate=32, arch=(4, 4, 4, 4),
             lr=0.1, num_classes=10):
    super(DenseNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add(DenseBlock(num_convs, growth_rate))
            num_channels += num_convs * growth_rate
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add(transition_block(num_channels))
        self.net.add(nn.BatchNorm(), nn.Activation('relu'),
                     nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add_module(f'dense_blk{i+1}', DenseBlock(num_convs,
                                                              growth_rate))
            num_channels += num_convs * growth_rate
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add_module(f'tran_blk{i+1}', transition_block(
                    num_channels))
        self.net.add_module('last', nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add(DenseBlock(num_convs, growth_rate))
            num_channels += num_convs * growth_rate
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add(TransitionBlock(num_channels))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes)]))
```

```python
%%tab jax
@d2l.add_to_class(DenseNet)
def create_net(self):
    net = self.b1()
    for i, num_convs in enumerate(self.arch):
        net.layers.extend([DenseBlock(num_convs, self.growth_rate,
                                      training=self.training)])
        num_channels = self.num_channels + (num_convs * self.growth_rate)
        if i != len(self.arch) - 1:
            num_channels //= 2
            net.layers.extend([TransitionBlock(num_channels,
                                               training=self.training)])
    net.layers.extend([
        nn.BatchNorm(not self.training),
        nn.relu,
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                              strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)
    ])
    return net
```

## Step 5: Training the Model

Let's train our DenseNet on the Fashion-MNIST dataset. To keep training time manageable, we'll resize the images to 96×96 pixels.

```python
%%tab mxnet, pytorch, jax
model = DenseNet(lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```python
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = DenseNet(lr=0.01)
    trainer.fit(model, data)
```

## Summary

You've successfully implemented and trained a DenseNet model. Key takeaways:

- **Dense Blocks** concatenate feature maps from all preceding layers, promoting feature reuse.
- **Transition Layers** control model complexity by reducing channel counts and spatial dimensions.
- Compared to ResNet, DenseNet typically requires fewer parameters but can be more memory-intensive due to concatenation operations.

## Exercises

1. Why does DenseNet use average pooling instead of max-pooling in transition layers?
2. How does DenseNet achieve a smaller parameter count than ResNet?
3. DenseNet is known for high memory consumption. Propose a method to mitigate this.
4. Implement the DenseNet variants (DenseNet-121, DenseNet-169, etc.) from the original paper.
5. Design an MLP version of DenseNet and apply it to a regression task like house price prediction.

For further discussion, visit the [D2L.ai forum](https://discuss.d2l.ai).