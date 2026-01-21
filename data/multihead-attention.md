# Implementing Multi-Head Attention: A Step-by-Step Guide

## Introduction

Multi-head attention is a fundamental building block in modern neural network architectures, particularly Transformers. It allows a model to jointly attend to information from different representation subspaces at different positions. This tutorial will guide you through the mathematical formulation and practical implementation of multi-head attention across four major deep learning frameworks: PyTorch, TensorFlow, JAX, and MXNet.

## Prerequisites

Before starting, ensure you have the necessary libraries installed. The implementation uses framework-specific imports:

```python
# For PyTorch
import torch
from torch import nn

# For TensorFlow
import tensorflow as tf

# For JAX
from flax import linen as nn
import jax
import jax.numpy as jnp

# For MXNet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

## Understanding Multi-Head Attention

### Mathematical Formulation

Given a query $\mathbf{q} \in \mathbb{R}^{d_q}$, key $\mathbf{k} \in \mathbb{R}^{d_k}$, and value $\mathbf{v} \in \mathbb{R}^{d_v}$, each attention head $\mathbf{h}_i$ (where $i = 1, \ldots, h$) is computed as:

$$\mathbf{h}_i = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v}$$

Where:
- $\mathbf W_i^{(q)}\in\mathbb R^{p_q\times d_q}$, $\mathbf W_i^{(k)}\in\mathbb R^{p_k\times d_k}$, and $\mathbf W_i^{(v)}\in\mathbb R^{p_v\times d_v}$ are learnable parameters
- $f$ is an attention pooling function (like scaled dot product attention)

The multi-head attention output is produced by concatenating all heads and applying a final linear transformation:

$$\mathbf W_o \begin{bmatrix}\mathbf h_1\\\vdots\\\mathbf h_h\end{bmatrix} \in \mathbb{R}^{p_o}$$

### Design Choices

For computational efficiency, we set $p_q = p_k = p_v = p_o / h$. This allows parallel computation of all heads by setting the output dimensions of the linear transformations to $p_q h = p_k h = p_v h = p_o$.

## Step 1: Implementing the MultiHeadAttention Class

We'll implement multi-head attention using scaled dot product attention for each head. The implementation varies slightly by framework, but follows the same core logic.

### PyTorch Implementation

```python
class MultiHeadAttention(d2l.Module):
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Transform queries, keys, and values
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        # Handle valid lengths for masking
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Apply attention
        output = self.attention(queries, keys, values, valid_lens)
        
        # Concatenate and transform output
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
```

### TensorFlow Implementation

```python
class MultiHeadAttention(d2l.Module):
    """Multi-head attention."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
    
    def call(self, queries, keys, values, valid_lens, **kwargs):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        
        if valid_lens is not None:
            valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)
            
        output = self.attention(queries, keys, values, valid_lens, **kwargs)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
```

### JAX Implementation

```python
class MultiHeadAttention(nn.Module):
    num_hiddens: int
    num_heads: int
    dropout: float
    bias: bool = False

    def setup(self):
        self.attention = d2l.DotProductAttention(self.dropout)
        self.W_q = nn.Dense(self.num_hiddens, use_bias=self.bias)
        self.W_k = nn.Dense(self.num_hiddens, use_bias=self.bias)
        self.W_v = nn.Dense(self.num_hiddens, use_bias=self.bias)
        self.W_o = nn.Dense(self.num_hiddens, use_bias=self.bias)

    @nn.compact
    def __call__(self, queries, keys, values, valid_lens, training=False):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            valid_lens = jnp.repeat(valid_lens, self.num_heads, axis=0)

        output, attention_weights = self.attention(
            queries, keys, values, valid_lens, training=training)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat), attention_weights
```

### MXNet Implementation

```python
class MultiHeadAttention(d2l.Module):
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, use_bias=False,
                 **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_k = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_v = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_o = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            valid_lens = valid_lens.repeat(self.num_heads, axis=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
```

## Step 2: Implementing Tensor Transposition Methods

To enable parallel computation of multiple attention heads, we need methods to reshape tensors appropriately. These methods handle the transformation between batch-first and head-first representations.

### PyTorch Transposition Methods

```python
@d2l.add_to_class(MultiHeadAttention)
def transpose_qkv(self, X):
    """Transposition for parallel computation of multiple attention heads."""
    # Reshape to separate heads
    X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
    # Permute dimensions to put heads first
    X = X.permute(0, 2, 1, 3)
    # Flatten batch and heads dimensions
    return X.reshape(-1, X.shape[2], X.shape[3])

@d2l.add_to_class(MultiHeadAttention)
def transpose_output(self, X):
    """Reverse the operation of transpose_qkv."""
    # Reshape back to separate batch and heads
    X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
    # Permute dimensions back to original order
    X = X.permute(0, 2, 1, 3)
    # Combine heads dimension with hidden dimension
    return X.reshape(X.shape[0], X.shape[1], -1)
```

### TensorFlow Transposition Methods

```python
@d2l.add_to_class(MultiHeadAttention)
def transpose_qkv(self, X):
    X = tf.reshape(X, shape=(X.shape[0], X.shape[1], self.num_heads, -1))
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    return tf.reshape(X, shape=(-1, X.shape[2], X.shape[3]))

@d2l.add_to_class(MultiHeadAttention)
def transpose_output(self, X):
    X = tf.reshape(X, shape=(-1, self.num_heads, X.shape[1], X.shape[2]))
    X = tf.transpose(X, perm=(0, 2, 1, 3))
    return tf.reshape(X, shape=(X.shape[0], X.shape[1], -1))
```

### JAX Transposition Methods

```python
@d2l.add_to_class(MultiHeadAttention)
def transpose_qkv(self, X):
    X = X.reshape((X.shape[0], X.shape[1], self.num_heads, -1))
    X = jnp.transpose(X, (0, 2, 1, 3))
    return X.reshape((-1, X.shape[2], X.shape[3]))

@d2l.add_to_class(MultiHeadAttention)
def transpose_output(self, X):
    X = X.reshape((-1, self.num_heads, X.shape[1], X.shape[2]))
    X = jnp.transpose(X, (0, 2, 1, 3))
    return X.reshape((X.shape[0], X.shape[1], -1))
```

### MXNet Transposition Methods

```python
@d2l.add_to_class(MultiHeadAttention)
def transpose_qkv(self, X):
    X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

@d2l.add_to_class(MultiHeadAttention)
def transpose_output(self, X):
    X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

## Step 3: Testing the Implementation

Let's verify our implementation works correctly with a simple test case where keys and values are identical.

### PyTorch Test

```python
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
Y = d2l.ones((batch_size, num_kvpairs, num_hiddens))

# Check output shape
d2l.check_shape(attention(X, Y, Y, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

### TensorFlow Test

```python
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = d2l.tensor([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
Y = tf.ones((batch_size, num_kvpairs, num_hiddens))

d2l.check_shape(attention(X, Y, Y, valid_lens, training=False),
                (batch_size, num_queries, num_hiddens))
```

### JAX Test

```python
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
Y = d2l.ones((batch_size, num_kvpairs, num_hiddens))

output_shape = attention.init_with_output(d2l.get_key(), X, Y, Y, valid_lens,
                                         training=False)[0][0].shape
print(f"Output shape: {output_shape}")
```

### MXNet Test

```python
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()

batch_size, num_queries, num_kvpairs = 2, 4, 6
valid_lens = d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
Y = d2l.ones((batch_size, num_kvpairs, num_hiddens))

d2l.check_shape(attention(X, Y, Y, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

## Summary

In this tutorial, you've learned how to implement multi-head attention from scratch. Key takeaways include:

1. **Mathematical Foundation**: Multi-head attention computes multiple attention heads in parallel, each with its own learned linear projections for queries, keys, and values.

2. **Parallel Computation**: By properly reshaping tensors, we can compute all attention heads simultaneously, improving computational efficiency.

3. **Framework Adaptability**: The core logic remains consistent across different deep learning frameworks, with only minor syntax differences for tensor operations.

4. **Practical Implementation**: The implementation includes proper handling of attention masks (via `valid_lens`) and dropout for regularization.

## Exercises

1. **Visualize Attention Weights**: Extend the implementation to return and visualize attention weights from multiple heads. This can help understand what each head is focusing on.

2. **Head Importance Analysis**: Design experiments to measure the importance of individual attention heads for model pruning. Consider approaches like:
   - Measuring the variance of attention weights per head
   - Ablation studies (removing heads and measuring performance drop)
   - Gradient-based importance metrics

3. **Optimization Opportunities**: Explore how different tensor reshaping strategies affect computational performance on your hardware.

## Further Reading

For more detailed discussions and community insights, refer to the framework-specific discussion forums:
- [PyTorch Discussions](https://discuss.d2l.ai/t/1635)
- [TensorFlow Discussions](https://discuss.d2l.ai/t/3869)
- [JAX Discussions](https://discuss.d2l.ai/t/18029)
- [MXNet Discussions](https://discuss.d2l.ai/t/1634)

This implementation provides a solid foundation for building more complex attention-based architectures like Transformers, which have revolutionized natural language processing and other sequence modeling tasks.