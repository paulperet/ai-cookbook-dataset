# Padding and Stride in Convolutional Neural Networks

## Introduction

In convolutional neural networks (CNNs), applying convolution layers often reduces the spatial dimensions of the input. For example, a 240×240 pixel image processed through ten 5×5 convolution layers would shrink to 200×200 pixels, losing 30% of the original image area and potentially discarding important boundary information.

This tutorial explores two essential techniques for controlling output size:
- **Padding**: Adding extra pixels around the input to preserve spatial dimensions
- **Stride**: Controlling how much the convolution window moves, enabling downsampling

## Prerequisites

First, let's import the necessary libraries. Choose your preferred deep learning framework:

```python
# MXNet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```python
# PyTorch
import torch
from torch import nn
```

```python
# TensorFlow
import tensorflow as tf
```

```python
# JAX
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Understanding the Problem

Recall that for an input of size $n_h × n_w$ and a kernel of size $k_h × k_w$, the output size without padding is:

$$(n_h - k_h + 1) × (n_w - k_w + 1)$$

This reduction occurs because the kernel can only shift until it runs out of input pixels. The corners of the input are particularly underutilized, as shown in research on pixel utilization patterns.

## 1. Implementing Padding

Padding adds extra pixels (typically zeros) around the input boundary to control output size. With $p_h$ rows and $p_w$ columns of padding, the output becomes:

$$(n_h - k_h + p_h + 1) × (n_w - k_w + p_w + 1)$$

### 1.1 Helper Function

We'll create a helper function to simplify convolution operations across different frameworks:

```python
def comp_conv2d(conv2d, X):
    """Compute 2D convolution and return output shape."""
    # Framework-specific implementations
    pass
```

### 1.2 Symmetric Padding Example

Let's create a convolution layer with a 3×3 kernel and 1 pixel of padding on all sides. Given an 8×8 input, the output should remain 8×8:

```python
# MXNet
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = np.random.uniform(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)  # Output: (8, 8)
```

```python
# PyTorch
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)  # Output: torch.Size([8, 8])
```

```python
# TensorFlow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
print(comp_conv2d(conv2d, X).shape)  # Output: (8, 8)
```

```python
# JAX
conv2d = nn.Conv(1, kernel_size=(3, 3), padding='SAME')
X = jax.random.uniform(jax.random.PRNGKey(d2l.get_seed()), shape=(8, 8))
print(comp_conv2d(conv2d, X).shape)  # Output: (8, 8)
```

### 1.3 Asymmetric Padding

When kernel dimensions differ, we can use different padding values for height and width:

```python
# Example with 5×3 kernel, padding 2 on height, 1 on width
# All frameworks produce (8, 8) output from 8×8 input
```

## 2. Implementing Stride

Stride controls how many pixels the convolution window moves between operations. With strides $s_h$ and $s_w$, the output size becomes:

$$\lfloor(n_h - k_h + p_h + s_h)/s_h\rfloor × \lfloor(n_w - k_w + p_w + s_w)/s_w\rfloor$$

### 2.1 Stride of 2 Example

Using stride 2 effectively halves the spatial dimensions:

```python
# All frameworks: 3×3 kernel, padding 1, stride 2
# Input: 8×8, Output: 4×4
```

### 2.2 Complex Stride Example

Let's examine a more complex case with different strides and padding:

```python
# Kernel: 3×5, Padding: (0, 1), Stride: (3, 4)
# Input: 8×8
# Output calculation: floor((8-3+0+3)/3) × floor((8-5+1+4)/4) = 2×2
```

## 3. Practical Considerations

### 3.1 Why Odd Kernel Sizes?

CNNs commonly use odd-sized kernels (1, 3, 5, 7) because:
- They allow symmetric padding (same number of pixels on all sides)
- They preserve input dimensions when using appropriate padding
- The output pixel at position (i, j) corresponds exactly to the input window centered at (i, j)

### 3.2 Benefits of Stride > 1

1. **Computational Efficiency**: Fewer output positions to compute
2. **Downsampling**: Reduces spatial dimensions without pooling layers
3. **Receptive Field**: Each output pixel covers a larger input area

### 3.3 Zero Padding vs. Alternatives

While zero-padding is computationally efficient and easy to implement, other padding strategies exist:
- Mirror padding (reflecting boundary values)
- Replication padding (repeating edge values)
- Learnable padding

Zero-padding has the advantage of allowing CNNs to learn positional information by identifying "whitespace" boundaries.

## 4. Exercises

Test your understanding with these exercises:

1. Verify the output shape for the complex example: kernel (3,5), padding (0,1), stride (3,4) on an 8×8 input
2. For audio signals, what does a stride of 2 represent?
3. Implement mirror padding in your framework of choice
4. Analyze the computational benefits of stride > 1
5. Explore statistical advantages of larger strides
6. How would you implement a stride of ½? What applications might this have?

## Summary

In this tutorial, you've learned:

- **Padding** preserves spatial dimensions by adding pixels around input boundaries
- **Stride** controls downsampling rate by adjusting window movement
- The relationship between input size, kernel size, padding, stride, and output size
- Practical considerations for choosing padding and stride values

These techniques are fundamental for designing CNN architectures that maintain appropriate spatial dimensions throughout the network while controlling computational complexity.

## Further Reading

For more advanced padding techniques and their applications, see the research on non-zero padding strategies and their effects on CNN performance. The default choices of zero-padding and unit stride work well for many applications, but understanding these parameters gives you finer control over your network architecture.