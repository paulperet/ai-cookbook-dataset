# The Encoder-Decoder Architecture

## Introduction

Sequence-to-sequence problems, such as machine translation, involve inputs and outputs of varying, unaligned lengths. The standard solution is the **encoder-decoder architecture**. This design consists of two core components:
1.  An **Encoder** that processes a variable-length input sequence.
2.  A **Decoder** that acts as a conditional language model, generating the output sequence token-by-token based on the encoded input and previously generated tokens.

For example, to translate "They are watching ." from English to French ("Ils regardent ."), the architecture first encodes the English sequence into a state, then decodes that state to produce the French tokens sequentially.

This guide will define the core interfaces for the Encoder and Decoder, which form the foundation for specific sequence-to-sequence models built in subsequent tutorials.

## Prerequisites

First, import the necessary libraries. The code is framework-agnostic; select your preferred deep learning framework.

```python
# Install d2l if needed: !pip install d2l
# Framework Selection (choose one):
# For MXNet
from d2l import mxnet as d2l
from mxnet.gluon import nn

# For PyTorch
from d2l import torch as d2l
from torch import nn

# For TensorFlow
from d2l import tensorflow as d2l
import tensorflow as tf

# For JAX
from d2l import jax as d2l
from flax import linen as nn
```

## Step 1: Define the Encoder Interface

The encoder's role is to transform a variable-length input sequence into a fixed representation. We define a base `Encoder` class that specifies this interface. Concrete encoder models (like RNNs or Transformers) will inherit from this class and implement the forward pass.

```python
# For MXNet
class Encoder(nn.Block):  #@save
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError

# For PyTorch
class Encoder(nn.Module):  #@save
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError

# For TensorFlow
class Encoder(tf.keras.layers.Layer):  #@save
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def call(self, X, *args):
        raise NotImplementedError

# For JAX
class Encoder(nn.Module):  #@save
    """The base encoder interface for the encoder-decoder architecture."""
    def setup(self):
        raise NotImplementedError

    # Later there can be additional arguments (e.g., length excluding padding)
    def __call__(self, X, *args):
        raise NotImplementedError
```

## Step 2: Define the Decoder Interface

The decoder generates the output sequence. Its interface includes an `init_state` method to convert the encoder's output into an initial state for decoding, and a forward method to produce the next token given an input and the current state.

```python
# For MXNet
class Decoder(nn.Block):  #@save
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

# For PyTorch
class Decoder(nn.Module):  #@save
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

# For TensorFlow
class Decoder(tf.keras.layers.Layer):  #@save
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def call(self, X, state):
        raise NotImplementedError

# For JAX
class Decoder(nn.Module):  #@save
    """The base decoder interface for the encoder-decoder architecture."""
    def setup(self):
        raise NotImplementedError

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def __call__(self, X, state):
        raise NotImplementedError
```

## Step 3: Assemble the Full Architecture

The `EncoderDecoder` class connects the encoder and decoder. In its forward pass, it:
1.  Encodes the input sequence.
2.  Initializes the decoder's state using the encoder's output.
3.  Passes the decoder's input (e.g., the target sequence shifted right) and state to the decoder, returning the output.

```python
# For MXNet and PyTorch
class EncoderDecoder(d2l.Classifier):  #@save
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]

# For TensorFlow
class EncoderDecoder(d2l.Classifier):  #@save
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args, training=True)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state, training=True)[0]

# For JAX
class EncoderDecoder(d2l.Classifier):  #@save
    """The base class for the encoder-decoder architecture."""
    encoder: nn.Module
    decoder: nn.Module
    training: bool

    def __call__(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args, training=self.training)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state, training=self.training)[0]
```

## Summary

You have now implemented the foundational encoder-decoder architecture. This design is powerful for sequence-to-sequence tasks because:
*   The **encoder** converts a variable-length input into a fixed-shape state.
*   The **decoder** maps this state back to a variable-length output sequence.

In the next tutorial, you will see how to implement this architecture using Recurrent Neural Networks (RNNs) to build a practical machine translation model.

## Exercises

1.  Do the encoder and decoder have to be the same type of neural network (e.g., both RNNs)? Why or why not?
2.  Besides machine translation, can you think of another application suitable for an encoder-decoder architecture? (e.g., text summarization, speech recognition).

---
*For further discussion, visit the [D2L.ai forum](https://discuss.d2l.ai).*