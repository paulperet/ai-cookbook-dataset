# Convolutional Layers for Images
:label:`sec_conv_layer`

Now that we understand the theory behind convolutional layers, let's see how they work in practice. We'll focus on images as our primary example, building on the motivation that convolutional neural networks are efficient architectures for exploring spatial structure in image data.

## Prerequisites

First, let's import the necessary libraries. We'll provide implementations for multiple frameworks - choose the one that matches your setup.

```python
# For MXNet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()

# For PyTorch
from d2l import torch as d2l
import torch
from torch import nn

# For TensorFlow
from d2l import tensorflow as d2l
import tensorflow as tf

# For JAX
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## The Cross-Correlation Operation

Strictly speaking, what we call "convolutional layers" actually perform cross-correlation operations. In this operation, an input tensor and a kernel tensor are combined to produce an output tensor.

Let's start with two-dimensional data (ignoring channels for now). Consider an input tensor with height 3 and width 3, and a kernel with height 2 and width 2. The cross-correlation operation works by sliding the kernel across the input tensor, multiplying elementwise at each position, and summing the results.

The output size is slightly smaller than the input because the kernel needs to fit entirely within the image. For an input of size $n_h \times n_w$ and kernel of size $k_h \times k_w$, the output size is:

$$(n_h - k_h + 1) \times (n_w - k_w + 1)$$

Let's implement this operation:

```python
def corr2d(X, K):
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i:i + h, j:j + w] * K))
    return Y
```

Now let's test our implementation with a simple example:

```python
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

The output should be:
```
[[19., 25.],
 [37., 43.]]
```

## Implementing a Convolutional Layer

A convolutional layer performs cross-correlation between the input and kernel, then adds a scalar bias. The layer has two parameters: the kernel weights and the bias. During training, we typically initialize these randomly.

Let's implement a two-dimensional convolutional layer:

```python
# For MXNet
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()

# For PyTorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# For TensorFlow
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1,),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias

# For JAX
class Conv2D(nn.Module):
    kernel_size: int

    def setup(self):
        self.weight = nn.param('w', nn.initializers.uniform, self.kernel_size)
        self.bias = nn.param('b', nn.initializers.zeros, 1)

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

## Practical Application: Edge Detection

Let's apply convolutional layers to a practical problem: detecting edges in images. We'll create a simple image with a vertical edge and use a kernel to detect it.

First, construct a 6×8 pixel image where the middle four columns are black (0) and the rest are white (1):

```python
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

Now create a kernel that detects vertical edges:

```python
K = d2l.tensor([[1.0, -1.0]])
```

This kernel computes the difference between horizontally adjacent pixels. When applied to our image, it will detect transitions from white to black (positive values) and from black to white (negative values).

Let's apply the cross-correlation:

```python
Y = corr2d(X, K)
Y
```

The output shows 1s where we have white-to-black transitions and -1s where we have black-to-white transitions. All other positions are 0.

Notice that this kernel only detects vertical edges. If we transpose the image, the kernel won't detect any edges:

```python
corr2d(d2l.transpose(X), K)
```

## Learning Kernels from Data

Instead of manually designing kernels, we can learn them from data. Let's see if we can learn the edge detection kernel we just used.

We'll use a built-in convolutional layer and train it to reproduce the output `Y` from input `X`:

```python
# For MXNet
conv2d = nn.Conv2D(1, kernel_size=(1, 2), use_bias=False)
conv2d.initialize()

X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)
lr = 3e-2

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    conv2d.weight.data()[:] -= lr * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {float(l.sum()):.3f}')

# For PyTorch
conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')

# For TensorFlow
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))
lr = 3e-2

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        update = tf.multiply(lr, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'epoch {i + 1}, loss {tf.reduce_sum(l):.3f}')

# For JAX
conv2d = nn.Conv(1, kernel_size=(1, 2), use_bias=False, padding='VALID')
X = X.reshape((1, 6, 8, 1))
Y = Y.reshape((1, 6, 7, 1))
lr = 3e-2

params = conv2d.init(jax.random.PRNGKey(d2l.get_seed()), X)

def loss(params, X, Y):
    Y_hat = conv2d.apply(params, X)
    return ((Y_hat - Y) ** 2).sum()

for i in range(10):
    l, grads = jax.value_and_grad(loss)(params, X, Y)
    params = jax.tree_map(lambda p, g: p - lr * g, params, grads)
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l:.3f}')
```

After training, let's examine the learned kernel:

```python
# For MXNet
d2l.reshape(conv2d.weight.data(), (1, 2))

# For PyTorch
d2l.reshape(conv2d.weight.data, (1, 2))

# For TensorFlow
d2l.reshape(conv2d.get_weights()[0], (1, 2))

# For JAX
params['params']['kernel'].reshape((1, 2))
```

The learned kernel should be close to `[1, -1]`, which is exactly the edge detection kernel we designed manually!

## Key Concepts

### Cross-Correlation vs. Convolution
In deep learning, we use the term "convolution" even though we're technically performing cross-correlation. True convolution would require flipping the kernel both horizontally and vertically before the cross-correlation operation. However, since kernels are learned from data, the distinction doesn't matter in practice - the network will learn the appropriate weights regardless.

### Feature Maps and Receptive Fields
- **Feature Map**: The output of a convolutional layer, representing learned features in spatial dimensions.
- **Receptive Field**: All elements from previous layers that affect the calculation of a given element during forward propagation.

The receptive field grows as we stack more convolutional layers, allowing deeper networks to capture broader contextual information from the input.

## Summary

In this tutorial, we've learned:
1. How to implement the cross-correlation operation from scratch
2. How to build convolutional layers that can learn from data
3. How convolutional layers can detect edges and other patterns in images
4. That we can learn effective kernels automatically rather than designing them manually

The local nature of convolutional operations makes them highly efficient for hardware implementation, which has been crucial for the success of deep learning in computer vision. Moreover, these operations correspond well to biological visual processing, giving us confidence in their effectiveness.

## Exercises

1. Experiment with different image patterns:
   - Create an image with diagonal edges and apply the kernel `K`
   - Try transposing the image or the kernel
   
2. Design kernels manually for different purposes:
   - Create an edge detector for a specific direction
   - Design a kernel for detecting second derivatives
   - Create a blur kernel
   
3. Explore gradient computation for custom convolutional layers

4. Consider how to represent cross-correlation as matrix multiplication

These exercises will help deepen your understanding of how convolutional layers work and how they can be applied to different problems.