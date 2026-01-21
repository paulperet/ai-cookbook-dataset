# The Transformer Architecture: A Step-by-Step Implementation Guide

## Introduction

The Transformer architecture, introduced in the seminal paper "Attention Is All You Need" (Vaswani et al., 2017), revolutionized sequence modeling by relying entirely on attention mechanisms without convolutional or recurrent layers. This architecture achieves both parallel computation and the shortest maximum path length, making it highly effective for various tasks including machine translation, text generation, and beyond.

In this tutorial, we'll implement the complete Transformer model from scratch, breaking down each component with clear explanations and code examples.

## Prerequisites

First, let's import the necessary libraries. The implementation supports multiple deep learning frameworks (MXNet, PyTorch, TensorFlow, JAX) - choose the one you're most comfortable with.

```python
# Framework-specific imports
# For MXNet
from d2l import mxnet as d2l
import math
from mxnet import autograd, init, np, npx
from mxnet.gluon import nn
npx.set_np()

# For PyTorch
from d2l import torch as d2l
import math
import torch
from torch import nn

# For TensorFlow
from d2l import tensorflow as d2l
import tensorflow as tf

# For JAX
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
import math
```

## 1. Positionwise Feed-Forward Networks

The positionwise feed-forward network applies the same multilayer perceptron (MLP) to each position in the sequence independently. This allows the model to transform representations at all positions using shared parameters.

```python
# MXNet implementation
class PositionWiseFFN(nn.Block):
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Dense(ffn_num_hiddens, flatten=False, activation='relu')
        self.dense2 = nn.Dense(ffn_num_outputs, flatten=False)

    def forward(self, X):
        return self.dense2(self.dense1(X))

# PyTorch implementation  
class PositionWiseFFN(nn.Module):
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

# TensorFlow implementation
class PositionWiseFFN(tf.keras.layers.Layer):
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, X):
        return self.dense2(self.relu(self.dense1(X)))

# JAX implementation
class PositionWiseFFN(nn.Module):
    """The positionwise feed-forward network."""
    ffn_num_hiddens: int
    ffn_num_outputs: int

    def setup(self):
        self.dense1 = nn.Dense(self.ffn_num_hiddens)
        self.dense2 = nn.Dense(self.ffn_num_outputs)

    def __call__(self, X):
        return self.dense2(nn.relu(self.dense1(X)))
```

Let's test the positionwise feed-forward network to see how it transforms the input dimensions:

```python
# Test the FFN
ffn = PositionWiseFFN(4, 8)
if framework == 'pytorch':
    ffn.eval()
    output = ffn(d2l.ones((2, 3, 4)))[0]
elif framework == 'tensorflow':
    output = ffn(tf.ones((2, 3, 4)))[0]
# ... similar for other frameworks
print(f"Output shape: {output.shape}")
```

The output shows that the innermost dimension changes from 4 to 8, while the batch size and sequence length remain unchanged.

## 2. Residual Connection and Layer Normalization

The "add & norm" component in the Transformer consists of a residual connection followed by layer normalization. This combination helps with gradient flow and training stability.

```python
# MXNet implementation
class AddNorm(nn.Block):
    """The residual connection followed by layer normalization."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm()

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

# PyTorch implementation
class AddNorm(nn.Module):
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

# TensorFlow implementation
class AddNorm(tf.keras.layers.Layer):
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(norm_shape)

    def call(self, X, Y, **kwargs):
        return self.ln(self.dropout(Y, **kwargs) + X)

# JAX implementation
class AddNorm(nn.Module):
    """The residual connection followed by layer normalization."""
    dropout: int

    @nn.compact
    def __call__(self, X, Y, training=False):
        return nn.LayerNorm()(
            nn.Dropout(self.dropout)(Y, deterministic=not training) + X)
```

The residual connection requires that both inputs have the same shape, ensuring the output maintains the same dimensions.

## 3. Building the Transformer Encoder

### 3.1 Encoder Block

Each encoder block contains two sublayers: multi-head self-attention and positionwise feed-forward network, with residual connections and layer normalization around each.

```python
# MXNet implementation
class TransformerEncoderBlock(nn.Block):
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

# PyTorch implementation
class TransformerEncoderBlock(nn.Module):
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

# TensorFlow implementation
class TransformerEncoderBlock(tf.keras.layers.Layer):
    """The Transformer encoder block."""
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, 
                 ffn_num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(key_size, query_size, value_size, 
                                                num_hiddens, num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def call(self, X, valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs), **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)

# JAX implementation
class TransformerEncoderBlock(nn.Module):
    """The Transformer encoder block."""
    num_hiddens: int
    ffn_num_hiddens: int
    num_heads: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.attention = d2l.MultiHeadAttention(self.num_hiddens, self.num_heads,
                                                self.dropout, self.use_bias)
        self.addnorm1 = AddNorm(self.dropout)
        self.ffn = PositionWiseFFN(self.ffn_num_hiddens, self.num_hiddens)
        self.addnorm2 = AddNorm(self.dropout)

    def __call__(self, X, valid_lens, training=False):
        output, attention_weights = self.attention(X, X, X, valid_lens, training=training)
        Y = self.addnorm1(X, output, training=training)
        return self.addnorm2(Y, self.ffn(Y), training=training), attention_weights
```

### 3.2 Complete Encoder

The Transformer encoder stacks multiple identical encoder blocks and includes embedding layers with positional encoding.

```python
# MXNet implementation
class TransformerEncoder(d2l.Encoder):
    """The Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads, 
                 num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for _ in range(num_blks):
            self.blks.add(TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))
        self.initialize()

    def forward(self, X, valid_lens):
        # Scale embeddings before adding positional encoding
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

# PyTorch implementation
class TransformerEncoder(d2l.Encoder):
    """The Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads, 
                 num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

# TensorFlow implementation
class TransformerEncoder(d2l.Encoder):
    """The Transformer encoder."""
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, 
                 norm_shape, ffn_num_hiddens, num_heads, num_blks, dropout, bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = [TransformerEncoderBlock(key_size, query_size, value_size, 
                     num_hiddens, norm_shape, ffn_num_hiddens, num_heads, dropout, bias) 
                     for _ in range(num_blks)]

    def call(self, X, valid_lens, **kwargs):
        X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(
            tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens, **kwargs)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

# JAX implementation
class TransformerEncoder(d2l.Encoder):
    """The Transformer encoder."""
    vocab_size: int
    num_hiddens:int
    ffn_num_hiddens: int
    num_heads: int
    num_blks: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(self.num_hiddens, self.dropout)
        self.blks = [TransformerEncoderBlock(self.num_hiddens, self.ffn_num_hiddens,
                                             self.num_heads, self.dropout, self.use_bias)
                     for _ in range(self.num_blks)]

    def __call__(self, X, valid_lens, training=False):
        X = self.embedding(X) * math.sqrt(self.num_hiddens)
        X = self.pos_encoding(X, training=training)
        attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X, attention_w = blk(X, valid_lens, training=training)
            attention_weights[i] = attention_w
        return X
```

## 4. Building the Transformer Decoder

### 4.1 Decoder Block

The decoder block contains three sublayers: masked multi-head self-attention, encoder-decoder attention, and positionwise feed-forward network.

```python
# MXNet implementation
class TransformerDecoderBlock(nn.Block):
    """The i-th block in the Transformer decoder."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # Handle key values for autoregressive generation
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = np.concatenate((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        # Create decoder valid lengths for masking
        if autograd.is_training():
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = np.tile(np.arange(1, num_steps + 1, ctx=X.ctx),
                                     (batch_size, 1))
        else:
            dec_valid_lens = None
        
        # Self-attention with masking
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        
        # Encoder-decoder attention
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        
        return self.addnorm3(Z, self.ffn(Z)), state

# PyTorch implementation
class TransformerDecoderBlock(nn.Module):
    """The i-th block in the Transformer decoder."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def