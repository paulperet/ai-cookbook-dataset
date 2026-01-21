# Implementing Gated Recurrent Units (GRU) from Scratch

## Overview

Gated Recurrent Units (GRUs) offer a streamlined alternative to LSTMs, maintaining the ability to capture both short-term and long-term dependencies in sequential data while being computationally faster. In this tutorial, you will implement a GRU from scratch and compare it with a high-level API implementation.

## Prerequisites

First, ensure you have the necessary libraries installed. This tutorial supports multiple deep learning frameworks.

```bash
# Installation commands would go here for each framework
# For example: pip install torch tensorflow jax flax
```

Import the required modules. The code below handles imports for different frameworks.

```python
# Framework selection and imports
import sys

# Simulating framework selection - choose one
framework = 'pytorch'  # Change to 'mxnet', 'tensorflow', or 'jax'

if framework == 'pytorch':
    import torch
    from torch import nn
    import d2l.torch as d2l
elif framework == 'mxnet':
    from mxnet import np, npx
    from mxnet.gluon import rnn
    npx.set_np()
    import d2l.mxnet as d2l
elif framework == 'tensorflow':
    import tensorflow as tf
    import d2l.tensorflow as d2l
elif framework == 'jax':
    import jax
    from jax import numpy as jnp
    from flax import linen as nn
    import d2l.jax as d2l
```

## Understanding GRU Architecture

GRUs simplify the LSTM architecture by using only two gates instead of three:

1. **Reset Gate**: Controls how much of the previous hidden state to remember
2. **Update Gate**: Controls how much of the new candidate state to incorporate

The mathematical formulation for a minibatch input $\mathbf{X}_t$ and previous hidden state $\mathbf{H}_{t-1}$ is:

- Reset gate: $\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xr}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hr}} + \mathbf{b}_\textrm{r})$
- Update gate: $\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{\textrm{xz}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hz}} + \mathbf{b}_\textrm{z})$
- Candidate hidden state: $\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + (\mathbf{R}_t \odot \mathbf{H}_{t-1}) \mathbf{W}_{\textrm{hh}} + \mathbf{b}_\textrm{h})$
- Final hidden state: $\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1} + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t$

Where $\odot$ denotes elementwise multiplication and $\sigma$ is the sigmoid function.

## Step 1: Implementing GRU from Scratch

### 1.1 Initialize Model Parameters

Create a class that initializes all necessary parameters for the GRU gates.

```python
class GRUScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize parameters based on framework
        if framework == 'mxnet':
            init_weight = lambda *shape: d2l.randn(*shape) * sigma
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              d2l.zeros(num_hiddens))
        elif framework == 'pytorch':
            init_weight = lambda *shape: nn.Parameter(d2l.randn(*shape) * sigma)
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              nn.Parameter(d2l.zeros(num_hiddens)))
        elif framework == 'tensorflow':
            init_weight = lambda *shape: tf.Variable(d2l.normal(shape) * sigma)
            triple = lambda: (init_weight(num_inputs, num_hiddens),
                              init_weight(num_hiddens, num_hiddens),
                              tf.Variable(d2l.zeros(num_hiddens)))
        elif framework == 'jax':
            # JAX uses a different parameter initialization pattern
            pass  # Handled differently in JAX
        
        # Initialize gate parameters
        self.W_xz, self.W_hz, self.b_z = triple()  # Update gate
        self.W_xr, self.W_hr, self.b_r = triple()  # Reset gate
        self.W_xh, self.W_hh, self.b_h = triple()  # Candidate hidden state
```

### 1.2 Define the Forward Pass

Implement the forward computation that processes input sequences through the GRU cell.

```python
def forward(self, inputs, H=None):
    if H is None:
        # Initialize hidden state
        batch_size = inputs.shape[1]
        if framework in ['mxnet', 'pytorch']:
            H = d2l.zeros((batch_size, self.num_hiddens), 
                          ctx=inputs.ctx if framework == 'mxnet' else inputs.device)
        elif framework == 'tensorflow':
            H = d2l.zeros((batch_size, self.num_hiddens))
        elif framework == 'jax':
            H = jnp.zeros((batch_size, self.num_hiddens))
    
    outputs = []
    # Process each time step
    for X in inputs:
        # Update gate
        Z = d2l.sigmoid(d2l.matmul(X, self.W_xz) + 
                        d2l.matmul(H, self.W_hz) + self.b_z)
        
        # Reset gate
        R = d2l.sigmoid(d2l.matmul(X, self.W_xr) + 
                        d2l.matmul(H, self.W_hr) + self.b_r)
        
        # Candidate hidden state
        H_tilde = d2l.tanh(d2l.matmul(X, self.W_xh) + 
                           d2l.matmul(R * H, self.W_hh) + self.b_h)
        
        # Final hidden state
        H = Z * H + (1 - Z) * H_tilde
        outputs.append(H)
    
    return outputs, H

# Add forward method to class
GRUScratch.forward = forward
```

## Step 2: Train the GRU Model

Now let's train our GRU implementation on a language modeling task using "The Time Machine" dataset.

```python
# Load and prepare data
data = d2l.TimeMachine(batch_size=1024, num_steps=32)

# Initialize model
gru_scratch = GRUScratch(num_inputs=len(data.vocab), num_hiddens=32)
model = d2l.RNNLMScratch(gru_scratch, vocab_size=len(data.vocab), lr=4)

# Train the model
trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1)
trainer.fit(model, data)
```

## Step 3: Concise Implementation Using High-Level APIs

Most deep learning frameworks provide built-in GRU implementations that are optimized and easier to use.

### 3.1 Define the Concise GRU Class

```python
class GRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        
        if framework == 'mxnet':
            self.rnn = rnn.GRU(num_hiddens)
        elif framework == 'pytorch':
            self.rnn = nn.GRU(num_inputs, num_hiddens)
        elif framework == 'tensorflow':
            self.rnn = tf.keras.layers.GRU(num_hiddens, 
                                           return_sequences=True, 
                                           return_state=True)
        elif framework == 'jax':
            self.num_hiddens = num_hiddens
```

### 3.2 Train the Concise Model

```python
# Initialize the concise model
if framework == 'jax':
    gru_concisse = GRU(num_hiddens=32)
else:
    gru_concisse = GRU(num_inputs=len(data.vocab), num_hiddens=32)

# Create and train the language model
model_concisse = d2l.RNNLM(gru_concisse, vocab_size=len(data.vocab), lr=4)
trainer.fit(model_concisse, data)
```

## Step 4: Generate Text with the Trained Model

Let's test our trained model by generating text starting with a given prefix.

```python
# Generate text prediction
if framework == 'jax':
    prediction = model_concisse.predict('it has', 20, data.vocab, trainer.state.params)
else:
    prediction = model_concisse.predict('it has', 20, data.vocab)
    
print(f"Generated text: {prediction}")
```

## Key Takeaways

1. **GRUs vs LSTMs**: GRUs provide similar performance to LSTMs but with fewer parameters and faster computation due to having only two gates instead of three.

2. **Gate Functions**:
   - The **reset gate** controls short-term dependencies by determining how much of the previous state to remember
   - The **update gate** controls long-term dependencies by balancing between old and new information

3. **Implementation Choices**:
   - From-scratch implementation helps understand the underlying mechanics
   - High-level API implementations are more efficient and production-ready

4. **Training Observations**: The concise implementation typically trains faster due to optimized compiled operations.

## Exercises for Further Exploration

1. **Gate Analysis**: What would be the optimal reset and update gate values if you only want to use input from time step $t'$ to predict output at time step $t > t'$?

2. **Hyperparameter Tuning**: Experiment with different values for `num_hiddens`, learning rate, and batch size. Observe how they affect:
   - Training time
   - Model perplexity
   - Quality of generated text

3. **Architecture Comparison**: Compare the performance of:
   - Basic RNN vs GRU implementations
   - From-scratch vs high-level API implementations
   Measure differences in runtime, perplexity, and output quality

4. **Ablation Studies**: What happens if you modify the GRU architecture by:
   - Using only the reset gate (always set update gate to 1)
   - Using only the update gate (always set reset gate to 1)
   How does this affect the model's ability to capture dependencies?

## Conclusion

GRUs offer an efficient alternative to LSTMs for sequence modeling tasks. By implementing them from scratch, you gain valuable insight into how gating mechanisms help RNNs capture both short-term and long-term dependencies. The high-level API implementations provide optimized versions suitable for production use while maintaining the same core functionality.