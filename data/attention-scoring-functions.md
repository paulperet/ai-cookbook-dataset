# Attention Scoring Functions: A Practical Guide

## Introduction

In attention mechanisms, we need a way to measure compatibility between queries and keys. While we previously used distance-based kernels like the Gaussian kernel, these can be computationally expensive. This guide explores more efficient attention scoring functions, focusing on dot product attention and additive attention, which form the foundation of modern Transformer architectures.

## Prerequisites

First, let's set up our environment with the necessary imports. The code supports multiple deep learning frameworks - choose the one you're working with.

```python
# For MXNet
import math
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```python
# For PyTorch
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```python
# For TensorFlow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```python
# For JAX
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
import math
```

## Understanding Dot Product Attention

### From Gaussian Kernel to Dot Product

Recall the Gaussian kernel attention function (without exponentiation):

$$
a(\mathbf{q}, \mathbf{k}_i) = -\frac{1}{2} \|\mathbf{q} - \mathbf{k}_i\|^2 = \mathbf{q}^\top \mathbf{k}_i -\frac{1}{2} \|\mathbf{k}_i\|^2 -\frac{1}{2} \|\mathbf{q}\|^2
$$

Let's analyze this step by step:

1. **Last term elimination**: The final term $-\frac{1}{2} \|\mathbf{q}\|^2$ depends only on $\mathbf{q}$ and is identical for all $(\mathbf{q}, \mathbf{k}_i)$ pairs. When we normalize attention weights using softmax, this term disappears entirely.

2. **Key norm simplification**: With batch and layer normalization, the norms $\|\mathbf{k}_i\|$ become well-bounded and often constant. This allows us to drop the middle term without significantly affecting results.

3. **Variance control**: When query $\mathbf{q} \in \mathbb{R}^d$ and key $\mathbf{k}_i \in \mathbb{R}^d$ have elements drawn from distributions with zero mean and unit variance, their dot product has variance $d$. To maintain unit variance regardless of vector length, we scale by $1/\sqrt{d}$.

This leads us to the **scaled dot product attention** function:

$$ a(\mathbf{q}, \mathbf{k}_i) = \mathbf{q}^\top \mathbf{k}_i / \sqrt{d} $$

Applying softmax gives us the final attention weights:

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(a(\mathbf{q}, \mathbf{k}_i)) = \frac{\exp(\mathbf{q}^\top \mathbf{k}_i / \sqrt{d})}{\sum_{j=1} \exp(\mathbf{q}^\top \mathbf{k}_j / \sqrt{d})}$$

## Essential Utility Functions

Before implementing attention mechanisms, we need two crucial utility functions.

### 1. Masked Softmax Operation

When processing sequences of variable lengths in minibatches, we often pad shorter sequences with dummy tokens. We need to ensure these padding tokens don't contribute to attention calculations.

Here's how to implement a masked softmax that ignores padding:

```python
# MXNet implementation
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    if valid_lens is None:
        return npx.softmax(X)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = valid_lens.repeat(shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = npx.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, True,
                              value=-1e6, axis=1)
        return npx.softmax(X).reshape(shape)
```

```python
# PyTorch implementation
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X
    
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
```

```python
# TensorFlow implementation
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[
            None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)

        if len(X.shape) == 3:
            return tf.where(tf.expand_dims(mask, axis=-1), X, value)
        else:
            return tf.where(mask, X, value)
    
    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])
        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)
        X = _sequence_mask(tf.reshape(X, shape=(-1, shape[-1])), valid_lens,
                           value=-1e6)    
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)
```

```python
# JAX implementation
def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = jnp.arange((maxlen),
                          dtype=jnp.float32)[None, :] < valid_len[:, None]
        return jnp.where(mask, X, value)

    if valid_lens is None:
        return nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = jnp.repeat(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.softmax(X.reshape(shape), axis=-1)
```

**How it works**: The function replaces elements beyond the valid length with a very large negative value (-1e6), causing their exponentials to approach zero in the softmax operation.

Let's test it with a simple example:

```python
# Create a 2x2x4 tensor with valid lengths [2, 3]
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
```

For more fine-grained control, you can specify valid lengths for each vector:

```python
# Specify different valid lengths for each vector
masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]]))
```

### 2. Batch Matrix Multiplication

When working with minibatches of queries, keys, and values, we need efficient batch matrix multiplication. Given batches of matrices:

$$\mathbf{Q} = [\mathbf{Q}_1, \mathbf{Q}_2, \ldots, \mathbf{Q}_n] \in \mathbb{R}^{n \times a \times b}$$
$$\mathbf{K} = [\mathbf{K}_1, \mathbf{K}_2, \ldots, \mathbf{K}_n] \in \mathbb{R}^{n \times b \times c}$$

Batch matrix multiplication computes:

$$\textrm{BMM}(\mathbf{Q}, \mathbf{K}) = [\mathbf{Q}_1 \mathbf{K}_1, \mathbf{Q}_2 \mathbf{K}_2, \ldots, \mathbf{Q}_n \mathbf{K}_n] \in \mathbb{R}^{n \times a \times c}$$

Here's how to use it in different frameworks:

```python
# Create sample tensors
Q = d2l.ones((2, 3, 4))  # Batch of 2, each 3x4
K = d2l.ones((2, 4, 6))  # Batch of 2, each 4x6

# MXNet
npx.batch_dot(Q, K)  # Result: (2, 3, 6)

# PyTorch
torch.bmm(Q, K)  # Result: (2, 3, 6)

# TensorFlow
tf.matmul(Q, K)  # Result: (2, 3, 6)

# JAX
jax.lax.batch_matmul(Q, K)  # Result: (2, 3, 6)
```

## Implementing Scaled Dot Product Attention

Now let's implement the complete scaled dot product attention mechanism. Given queries $\mathbf{Q}\in\mathbb{R}^{n\times d}$, keys $\mathbf{K}\in\mathbb{R}^{m\times d}$, and values $\mathbf{V}\in\mathbb{R}^{m\times v}$, the operation is:

$$ \mathrm{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top }{\sqrt{d}}\right) \mathbf{V} \in \mathbb{R}^{n\times v}$$

Here's the implementation with dropout for regularization:

```python
# MXNet implementation
class DotProductAttention(nn.Block):
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = npx.batch_dot(queries, keys, transpose_b=True) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```python
# PyTorch implementation
class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```python
# TensorFlow implementation
class DotProductAttention(tf.keras.layers.Layer):
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, queries, keys, values, valid_lens=None, **kwargs):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
            tf.cast(d, dtype=tf.float32))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)
```

```python
# JAX implementation
class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    dropout: float

    @nn.compact
    def __call__(self, queries, keys, values, valid_lens=None, training=False):
        d = queries.shape[-1]
        scores = queries@(keys.swapaxes(1, 2)) / math.sqrt(d)
        attention_weights = masked_softmax(scores, valid_lens)
        dropout_layer = nn.Dropout(self.dropout, deterministic=not training)
        return dropout_layer(attention_weights)@values, attention_weights
```

### Testing Dot Product Attention

Let's test our implementation with a toy example:

```python
# Create sample data
queries = d2l.normal(0, 1, (2, 1, 2))      # 2 batches, 1 query each, dimension 2
keys = d2l.normal(0, 1, (2, 10, 2))        # 2 batches, 10 keys each, dimension 2
values = d2l.normal(0, 1, (2, 10, 4))      # 2 batches, 10 values each, dimension 4
valid_lens = d2l.tensor([2, 6])            # Valid lengths: 2 for first batch, 6 for second

# Initialize attention mechanism
attention = DotProductAttention(dropout=0.5)

# For PyTorch, set to evaluation mode
attention.eval()

# Apply attention
output = attention(queries, keys, values, valid_lens)
print(f"Output shape: {output.shape}")  # Should be (2, 1, 4)
```

The attention weights should be zero beyond the valid lengths (2 for the first sequence, 6 for the second).

## Additive Attention for Different Dimensionalities

When queries and keys have different dimensions, we can't use dot product directly. Additive attention solves this by using a small neural network to compute compatibility scores.

The additive attention scoring function is:

$$a(\mathbf q, \mathbf k) = \mathbf w_v^\top \textrm{tanh}(\mathbf W_q\mathbf q + \mathbf W_k \mathbf k) \in \mathbb{R}$$

where $\mathbf W_q\in\mathbb R^{h\times q}$, $\mathbf W_k\in\mathbb R^{h\times k}$, and $\mathbf w_v\in\mathbb R^{h}$ are learnable parameters.

Here's the implementation:

```python
# MXNet implementation
class AdditiveAttention(nn.Block):
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.W_q = nn.Dense(num_hiddens, use_bias=False, flatten=False)
        self.w_v = nn.Dense(1, use_bias=False, flatten=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = np.expand_dims(queries, axis=2) + np.expand_dims(keys, axis=1)
        features = np.tanh(features)
        scores = np.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return npx.batch_dot(self.dropout(self.attention_weights), values)
```

```python
# PyTorch implementation
class AdditiveAttention(nn.Module):
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

```python
# TensorFlow implementation
class AdditiveAttention(tf.keras.layers.Layer):
    """Additive attention."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=False)
        self.w_v = tf.keras.layers.Dense(1, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, queries, keys, values, valid_lens, **kwargs):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = tf.expand_dims(queries, axis=2) + tf.expand_dims(keys, axis=1)
        features = tf.nn.tanh(features)
        scores = tf.squeeze(self.w_v(features), axis=-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)
```

```python
# JAX implementation
class AdditiveAttention(nn.Module):
    num_hiddens: int
    dropout: float

    def setup(self):
        self.W_k = nn.Dense(self.num_hiddens, use_bias=False)
        self.W_q = nn.Dense(self.num_hiddens, use_bias=False)
        self.w_v = nn.Dense(1, use_bias=False)

    @nn