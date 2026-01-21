# Implementing a Recurrent Neural Network (RNN) from Scratch

In this guide, you will implement a Recurrent Neural Network (RNN) from scratch. You will train this RNN to function as a character-level language model using the text from H. G. Wells' *The Time Machine*. This tutorial follows a step-by-step approach, covering model definition, training, and text generation.

## Prerequisites

Before starting, ensure you have the necessary libraries installed. The code supports multiple deep learning frameworks. Choose the one you are using and install the required packages.

```bash
# Example installation for PyTorch
# pip install torch matplotlib
```

Import the required modules for your chosen framework.

```python
# For PyTorch
import math
import torch
from torch import nn
from torch.nn import functional as F

# For MXNet
# import math
# from mxnet import autograd, gluon, np, npx
# npx.set_np()

# For TensorFlow
# import math
# import tensorflow as tf

# For JAX
# from flax import linen as nn
# import jax
# from jax import numpy as jnp
# import math

# Common imports from d2l
from d2l import torch as d2l  # Change to mxnet, tensorflow, or jax as needed
```

## Step 1: Define the RNN Model

You will start by implementing the RNN model. The number of hidden units (`num_hiddens`) is a tunable hyperparameter.

```python
class RNNScratch(d2l.Module):
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W_xh = nn.Parameter(d2l.randn(num_inputs, num_hiddens) * sigma)
        self.W_hh = nn.Parameter(d2l.randn(num_hiddens, num_hiddens) * sigma)
        self.b_h = nn.Parameter(d2l.zeros(num_hiddens))
```

The `forward` method defines how to compute the output and hidden state at each time step. It processes the input sequence step-by-step, updating the hidden state using a tanh activation function.

```python
@d2l.add_to_class(RNNScratch)
def forward(self, inputs, state=None):
    if state is None:
        # Initial state with shape: (batch_size, num_hiddens)
        state = d2l.zeros((inputs.shape[1], self.num_hiddens), device=inputs.device)
    else:
        state, = state
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
        state = d2l.tanh(d2l.matmul(X, self.W_xh) + d2l.matmul(state, self.W_hh) + self.b_h)
        outputs.append(state)
    return outputs, state
```

Now, test the RNN with a dummy input to ensure it produces outputs of the correct shape.

```python
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)

# Check shapes
def check_len(a, n):
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'

def check_shape(a, shape):
    assert a.shape == shape, f'tensor\'s shape {a.shape} != expected shape {shape}'

check_len(outputs, num_steps)
check_shape(outputs[0], (batch_size, num_hiddens))
check_shape(state, (batch_size, num_hiddens))
print("All shape checks passed!")
```

## Step 2: Build an RNN-Based Language Model

Next, you will create a language model that uses the RNN. The `RNNLMScratch` class integrates the RNN and adds an output layer for vocabulary-sized predictions.

```python
class RNNLMScratch(d2l.Classifier):
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()

    def init_params(self):
        self.W_hq = nn.Parameter(d2l.randn(self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.vocab_size))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

### One-Hot Encoding

Since the model deals with categorical data (character indices), you will represent each token using one-hot encoding. This transforms each index into a vector where all elements are zero except the one corresponding to the token.

```python
@d2l.add_to_class(RNNLMScratch)
def one_hot(self, X):
    # Output shape: (num_steps, batch_size, vocab_size)
    return F.one_hot(X.T, self.vocab_size).type(torch.float32)
```

### Transforming RNN Outputs

Add an output layer to convert the RNN's hidden states into predictions for each token in the vocabulary.

```python
@d2l.add_to_class(RNNLMScratch)
def output_layer(self, rnn_outputs):
    outputs = [d2l.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
    return d2l.stack(outputs, 1)

@d2l.add_to_class(RNNLMScratch)
def forward(self, X, state=None):
    embs = self.one_hot(X)
    rnn_outputs, _ = self.rnn(embs, state)
    return self.output_layer(rnn_outputs)
```

Verify that the forward pass produces outputs with the correct shape.

```python
model = RNNLMScratch(rnn, num_inputs)
outputs = model(d2l.ones((batch_size, num_steps), dtype=torch.int64))
check_shape(outputs, (batch_size, num_steps, num_inputs))
print("Language model output shape is correct.")
```

## Step 3: Implement Gradient Clipping

Training RNNs can lead to exploding gradients. Gradient clipping is a common technique to limit the magnitude of gradients, ensuring stable training.

The following method clips gradients if their norm exceeds a specified threshold.

```python
@d2l.add_to_class(d2l.Trainer)
def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

## Step 4: Train the Model

Now, you will train the language model on *The Time Machine* dataset. The training process uses gradient clipping to prevent exploding gradients.

```python
# Load the dataset
data = d2l.TimeMachine(batch_size=1024, num_steps=32)

# Initialize the model and trainer
rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)

# Train the model
trainer.fit(model, data)
```

## Step 5: Generate Text with the Trained Model

Once trained, the model can generate text continuations given a prefix. The `predict` method first warms up the model with the prefix, then generates new characters one at a time.

```python
@d2l.add_to_class(RNNLMScratch)
def predict(self, prefix, num_preds, vocab, device=None):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        X = d2l.tensor([[outputs[-1]]], device=device)
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn(embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.output_layer(rnn_outputs)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

Test the text generation by providing a prefix and asking for 20 additional characters.

```python
generated = model.predict('it has', 20, data.vocab, d2l.try_gpu())
print(f'Generated text: {generated}')
```

## Summary

In this tutorial, you implemented an RNN from scratch and trained it as a character-level language model. You learned how to:

1. Define an RNN model with customizable hidden units.
2. Build a language model using one-hot encoding and an output layer.
3. Apply gradient clipping to stabilize training.
4. Train the model on a text dataset.
5. Generate new text continuations based on a given prefix.

This foundational RNN, while educational, has limitations like vanishing gradients. In practice, you would use more advanced architectures (e.g., LSTMs or GRUs) and leverage optimized deep learning frameworks for better performance.

## Exercises

1. Does the model use the entire history from the first token for predictions?
2. Which hyperparameter controls the length of the history considered?
3. Show that one-hot encoding is equivalent to using a different embedding for each token.
4. Experiment with hyperparameters (epochs, hidden units, time steps, learning rate) to lower perplexity.
5. Replace one-hot encoding with learnable embeddings. Does performance improve?
6. Test the model on other books by H. G. Wells, like *The War of the Worlds*.
7. Evaluate the model on books by different authors.
8. Modify the prediction to use sampling instead of argmax. What happens? Try biasing the sampling with a temperature parameter.
9. Run training without gradient clipping. What occurs?
10. Replace tanh with ReLU activation. Is gradient clipping still necessary? Why?