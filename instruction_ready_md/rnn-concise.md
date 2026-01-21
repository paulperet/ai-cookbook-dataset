# Implementing Recurrent Neural Networks with High-Level APIs

This guide demonstrates how to implement a Recurrent Neural Network (RNN) for language modeling using the high-level APIs provided by popular deep learning frameworks. While building from scratch offers valuable insights, production code benefits from optimized library implementations that save both development and computation time.

## Prerequisites

First, ensure you have the necessary libraries installed. This tutorial supports multiple frameworks. Choose the one you are using.

```bash
# Install the d2l library which provides common utilities
pip install d2l
```

Now, import the required modules for your chosen framework.

```python
# For PyTorch
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# For TensorFlow
import tensorflow as tf
from d2l import tensorflow as d2l

# For MXNet
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()
from d2l import mxnet as d2l

# For JAX
from jax import numpy as jnp
from flax import linen as nn
from d2l import jax as d2l
```

## Step 1: Define the RNN Model

We will create an RNN model class using the framework's built-in RNN layer. This abstracts away the manual implementation of the recurrent loop.

### PyTorch Implementation

```python
class RNN(d2l.Module):
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)
        
    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

### TensorFlow Implementation

```python
class RNN(d2l.Module):
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()            
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, return_sequences=True, return_state=True,
            time_major=True)
        
    def forward(self, inputs, H=None):
        outputs, H = self.rnn(inputs, H)
        return outputs, H
```

### MXNet Implementation

```python
class RNN(d2l.Module):
    """The RNN model implemented with high-level APIs."""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()        
        self.rnn = rnn.RNN(num_hiddens)
        
    def forward(self, inputs, H=None):
        if H is None:
            H, = self.rnn.begin_state(inputs.shape[1], ctx=inputs.ctx)
        outputs, (H, ) = self.rnn(inputs, (H, ))
        return outputs, H
```

**Note for JAX/Flax Users:** Flax's `linen` API does not currently provide a built-in RNN cell for vanilla RNNs. It offers more advanced variants like LSTMs and GRUs. For this tutorial, we will focus on the other frameworks.

## Step 2: Build the Complete Language Model

We now define the full language model by inheriting from a base class (`RNNLMScratch`). This model adds a linear output layer to predict the next token from the RNN's hidden states.

### PyTorch Language Model

```python
class RNNLM(d2l.RNNLMScratch):
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)
        
    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```

### TensorFlow & MXNet Language Model

```python
class RNNLM(d2l.RNNLMScratch):
    """The RNN-based language model implemented with high-level APIs."""
    def init_params(self):
        if framework == 'mxnet':
            self.linear = nn.Dense(self.vocab_size, flatten=False)
            self.initialize()
        if framework == 'tensorflow':
            self.linear = tf.keras.layers.Dense(self.vocab_size)
        
    def output_layer(self, hiddens):
        if framework == 'mxnet':
            return d2l.swapaxes(self.linear(hiddens), 0, 1)        
        if framework == 'tensorflow':
            return d2l.transpose(self.linear(hiddens), (1, 0, 2))
```

## Step 3: Load the Dataset

We'll use *The Time Machine* dataset, prepared into minibatches for training.

```python
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
```

## Step 4: Instantiate the Model and Make a Prediction

Before training, let's verify the model structure by making a prediction with randomly initialized weights. The output will be nonsensical.

```python
# Instantiate the RNN and the language model
if framework in ['mxnet', 'tensorflow']:
    rnn_layer = RNN(num_hiddens=32)
if framework == 'pytorch':
    rnn_layer = RNN(num_inputs=len(data.vocab), num_hiddens=32)

model = RNNLM(rnn_layer, vocab_size=len(data.vocab), lr=1)

# Generate a sequence
prediction = model.predict('it has', 20, data.vocab)
print(prediction)
```

## Step 5: Train the Model

Now, we train the model using a high-level trainer. The training loop, gradient clipping, and device management are handled for you.

```python
# Configure and run the trainer
if framework in ['mxnet', 'pytorch']:
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if framework == 'tensorflow':
    with d2l.try_gpu():
        trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)

trainer.fit(model, data)
```

The model should converge to a perplexity comparable to a from-scratch implementation, but much faster due to the optimized backend.

## Step 6: Generate Text with the Trained Model

Finally, let's use the trained model to generate text following a given prefix.

```python
# Generate a sequence after training
prediction = model.predict('it has', 20, data.vocab, d2l.try_gpu())
print(prediction)
```

## Summary

In this tutorial, you implemented an RNN-based language model using high-level framework APIs. This approach offers two key advantages:

1.  **Development Speed:** You avoid reimplementing standard, complex components.
2.  **Computational Efficiency:** Framework implementations are heavily optimized, leading to faster training and inference.

## Exercises

1.  **Overfitting Experiment:** Can you adjust the model or training parameters (e.g., reduce dataset size, increase model capacity) to make the RNN overfit using these high-level APIs?
2.  **Autoregressive Model:** Try implementing the simple autoregressive model from the sequence chapter using the built-in RNN layer as a building block.

For further discussion, refer to the framework-specific forums linked in the original material.