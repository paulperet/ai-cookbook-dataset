# Self-Attention and Positional Encoding

In this guide, we will explore the concepts of self-attention and positional encoding, which are fundamental building blocks of modern transformer architectures. Unlike recurrent neural networks (RNNs) that process sequences sequentially, self-attention allows for parallel computation by enabling each token in a sequence to attend to all other tokens simultaneously. However, this parallel processing loses the inherent order of the sequence. To address this, we introduce positional encoding, which injects information about the position of each token.

## Prerequisites

Before we begin, ensure you have the necessary libraries installed. This tutorial is designed to work with multiple deep learning frameworks. Choose your preferred framework and import the required modules.

```python
# For PyTorch
import torch
from torch import nn
import math
from d2l import torch as d2l

# For TensorFlow
import tensorflow as tf
import numpy as np
from d2l import tensorflow as d2l

# For JAX
import jax
import jax.numpy as jnp
from flax import linen as nn
from d2l import jax as d2l

# For MXNet
import mxnet as mx
from mxnet import np, npx, autograd, gluon
from mxnet.gluon import nn
import math
from d2l import mxnet as d2l
npx.set_np()
```

## Step 1: Understanding Self-Attention

Self-attention is a mechanism where each token in a sequence generates its own query, key, and value vectors. The output for each token is a weighted sum of the values of all tokens, with weights determined by the compatibility between the token's query and the keys of all tokens.

Given a sequence of input tokens $\mathbf{x}_1, \ldots, \mathbf{x}_n$ where each $\mathbf{x}_i \in \mathbb{R}^d$, the self-attention output $\mathbf{y}_i$ for token $i$ is computed as:

$$
\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d
$$

This allows every token to directly interact with every other token in the sequence.

### Implementing Self-Attention

We'll use a multi-head attention layer to compute self-attention. The following code initializes the attention mechanism and applies it to a sample input tensor.

```python
# Configuration
num_hiddens, num_heads = 100, 5
dropout = 0.5
batch_size, num_queries = 2, 4
valid_lens = d2l.tensor([3, 2])  # Valid sequence lengths for each batch

# For PyTorch
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
X = d2l.ones((batch_size, num_queries, num_hiddens))
output = attention(X, X, X, valid_lens)
print(f"Output shape: {output.shape}")

# For TensorFlow
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, dropout)
X = tf.ones((batch_size, num_queries, num_hiddens))
output = attention(X, X, X, valid_lens, training=False)
print(f"Output shape: {output.shape}")

# For JAX
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
X = d2l.ones((batch_size, num_queries, num_hiddens))
params = attention.init(d2l.get_key(), X, X, X, valid_lens, training=False)
output, _ = attention.apply(params, X, X, X, valid_lens, training=False)
print(f"Output shape: {output.shape}")

# For MXNet
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
attention.initialize()
X = d2l.ones((batch_size, num_queries, num_hiddens))
output = attention(X, X, X, valid_lens)
print(f"Output shape: {output.shape}")
```

The output tensor has the same shape as the input `(batch_size, num_queries, num_hiddens)`, confirming that self-attention preserves the sequence length and embedding dimension.

## Step 2: Comparing CNNs, RNNs, and Self-Attention

To understand the advantages of self-attention, let's compare it with convolutional neural networks (CNNs) and RNNs in terms of computational complexity, sequential operations, and maximum path length.

- **CNNs**: Process local features (like n-grams) using convolutional kernels. Computational complexity is $\mathcal{O}(knd^2)$ for kernel size $k$, sequence length $n$, and embedding dimension $d$. They have $\mathcal{O}(1)$ sequential operations and a maximum path length of $\mathcal{O}(n/k)$.
- **RNNs**: Process sequences step-by-step. Computational complexity is $\mathcal{O}(nd^2)$. They require $\mathcal{O}(n)$ sequential operations (not parallelizable) and have a maximum path length of $\mathcal{O}(n)$.
- **Self-Attention**: Computes interactions between all token pairs. Computational complexity is $\mathcal{O}(n^2d)$. It allows $\mathcal{O}(1)$ sequential operations (fully parallel) and has a maximum path length of $\mathcal{O}(1)$.

Self-attention excels in parallel computation and capturing long-range dependencies but becomes slow for very long sequences due to its quadratic complexity.

## Step 3: Introducing Positional Encoding

Since self-attention does not inherently preserve token order, we need to explicitly provide positional information. Positional encoding adds a fixed embedding to each token's representation based on its position in the sequence.

The positional encoding matrix $\mathbf{P} \in \mathbb{R}^{n \times d}$ uses sine and cosine functions of different frequencies:

$$
\begin{aligned}
p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\
p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).
\end{aligned}
$$

Here, $i$ is the position and $j$ is the dimension index.

### Implementing Positional Encoding

Let's implement the `PositionalEncoding` class for each framework.

```python
# For PyTorch
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

# For TensorFlow
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)

# For JAX
class PositionalEncoding(nn.Module):
    num_hiddens: int
    dropout: float
    max_len: int = 1000

    def setup(self):
        self.P = d2l.zeros((1, self.max_len, self.num_hiddens))
        X = d2l.arange(self.max_len, dtype=jnp.float32).reshape(-1, 1) / jnp.power(
            10000, jnp.arange(0, self.num_hiddens, 2, dtype=jnp.float32) / self.num_hiddens)
        self.P = self.P.at[:, :, 0::2].set(jnp.sin(X))
        self.P = self.P.at[:, :, 1::2].set(jnp.cos(X))

    @nn.compact
    def __call__(self, X, training=False):
        self.sow('intermediates', 'P', self.P)
        X = X + self.P[:, :X.shape[1], :]
        return nn.Dropout(self.dropout)(X, deterministic=not training)

# For MXNet
class PositionalEncoding(nn.Block):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)
```

### Visualizing Positional Encodings

Let's create a positional encoding matrix and visualize its structure to understand how frequencies vary across dimensions.

```python
encoding_dim, num_steps = 32, 60

# For PyTorch
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(d2l.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=[f"Col {d}" for d in d2l.arange(6, 10)])

# For TensorFlow
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(tf.zeros((1, num_steps, encoding_dim)), training=False)
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(np.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=[f"Col {d}" for d in np.arange(6, 10)])

# For JAX
pos_encoding = PositionalEncoding(encoding_dim, 0)
params = pos_encoding.init(d2l.get_key(), d2l.zeros((1, num_steps, encoding_dim)))
X, inter_vars = pos_encoding.apply(params, d2l.zeros((1, num_steps, encoding_dim)),
                                   mutable='intermediates')
P = inter_vars['intermediates']['P'][0]
P = P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=[f"Col {d}" for d in d2l.arange(6, 10)])

# For MXNet
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.initialize()
X = pos_encoding(np.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=[f"Col {d}" for d in d2l.arange(6, 10)])
```

The plot shows that columns 6 and 7 have higher frequencies than columns 8 and 9, demonstrating the alternating sine and cosine pattern.

## Step 4: Absolute vs. Relative Positional Information

### Absolute Positional Information

The sinusoidal design captures absolute positional information by assigning unique patterns to each position. Lower dimensions (columns) change more rapidly, similar to lower bits in a binary representation.

```python
for i in range(8):
    print(f'{i} in binary is {i:>03b}')
```

### Relative Positional Information

A key advantage of sinusoidal positional encodings is that they enable the model to easily learn relative positions. For any fixed offset $\delta$, the encoding at position $i + \delta$ can be expressed as a linear projection of the encoding at position $i$. This property arises from trigonometric identities and allows the model to generalize to unseen sequence lengths.

Mathematically, for frequency $\omega_j = 1/10000^{2j/d}$:

$$
\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\ -\sin(\delta \omega_j) & \cos(\delta \omega_j) \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\ p_{i, 2j+1} \end{bmatrix} =
\begin{bmatrix} p_{i+\delta, 2j} \\ p_{i+\delta, 2j+1} \end{bmatrix}
$$

The projection matrix depends only on the offset $\delta$, not on the absolute position $i$.

## Summary

- **Self-attention** allows each token to attend to all other tokens in parallel, enabling efficient capture of long-range dependencies but with quadratic complexity in sequence length.
- **Positional encoding** injects sequence order information into the model. Sinusoidal encodings provide both absolute and relative positional information through a fixed, deterministic pattern.
- Compared to CNNs and RNNs, self-attention offers superior parallelizability and shorter maximum path lengths, making it ideal for many sequence modeling tasks despite its computational cost for long sequences.

## Exercises

1. Consider stacking multiple self-attention layers with positional encoding. What potential issues might arise, such as vanishing gradients or overfitting to positional patterns?
2. Design a learnable positional encoding method. How would you initialize and train these embeddings?
3. Explore relative position embeddings, where the attention mechanism considers the offset between query and key positions. How might this improve model performance on tasks like machine translation or music generation?

For further discussion, refer to the D2L.ai forums for your specific framework:
- [MXNet Discussions](https://discuss.d2l.ai/t/1651)
- [PyTorch Discussions](https://discuss.d2l.ai/t/1652)
- [TensorFlow Discussions](https://discuss.d2l.ai/t/3870)
- [JAX Discussions](https://discuss.d2l.ai/t/18030)