# Building Deep Recurrent Neural Networks

## Introduction

In previous sections, we've worked with RNNs that have a single hidden layer between the input and output. While these networks can capture temporal dependencies over many time steps, we often need more expressive power to model complex relationships between inputs and outputs at the same time step. This is where deep RNNs come in—they add depth not just through time but also through multiple hidden layers.

In this tutorial, you'll learn how to build and train deep RNNs from scratch and using high-level frameworks. We'll implement stacked RNN layers and apply them to text generation using *The Time Machine* dataset.

## Prerequisites

First, let's set up our environment by importing the necessary libraries. The code below handles framework-specific imports:

```python
# Framework selection and imports
from d2l import torch as d2l  # or mxnet, tensorflow, jax
import torch
from torch import nn
```

## Understanding Deep RNN Architecture

A deep RNN stacks multiple RNN layers on top of each other. Each layer processes the sequence and passes its output to the next layer. The hidden state at each time step depends on both the previous time step (temporal dependency) and the previous layer's output at the current time step (depth dependency).

Mathematically, for layer *l* at time step *t*, the hidden state is calculated as:

```
H_t^(l) = φ_l(H_t^(l-1) W_xh^(l) + H_(t-1)^(l) W_hh^(l) + b_h^(l))
```

Where:
- `H_t^(l-1)` is the previous layer's output
- `H_(t-1)^(l)` is the same layer's previous hidden state
- `W_xh^(l)`, `W_hh^(l)` are weight matrices
- `b_h^(l)` is the bias term
- `φ_l` is the activation function

The final output layer uses only the last hidden layer's state.

## Step 1: Implementing a Deep RNN from Scratch

Let's start by building a multi-layer RNN from the ground up. We'll create a `StackedRNNScratch` class that stacks individual RNN layers.

### 1.1 Define the Stacked RNN Class

```python
class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        # Create a list of RNN layers
        self.rnns = nn.Sequential(*[d2l.RNNScratch(
            num_inputs if i==0 else num_hiddens,  # First layer gets input size
            num_hiddens, sigma)
            for i in range(num_layers)])
```

The constructor creates `num_layers` RNN instances. The first layer receives the original input size, while subsequent layers receive the hidden size as input.

### 1.2 Implement the Forward Pass

```python
@d2l.add_to_class(StackedRNNScratch)
def forward(self, inputs, Hs=None):
    outputs = inputs
    if Hs is None: 
        Hs = [None] * self.num_layers
    
    # Process through each layer
    for i in range(self.num_layers):
        outputs, Hs[i] = self.rnns[i](outputs, Hs[i])
        outputs = d2l.stack(outputs, 0)
    
    return outputs, Hs
```

The forward method sequentially processes the input through each RNN layer. Each layer receives the previous layer's output and returns both its output and hidden state.

## Step 2: Training the Deep RNN

Now let's train our deep RNN on *The Time Machine* dataset. We'll use a 2-layer architecture with 32 hidden units per layer.

### 2.1 Prepare the Data

```python
# Load and preprocess the dataset
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
```

### 2.2 Initialize the Model

```python
# Create the stacked RNN
rnn_block = StackedRNNScratch(
    num_inputs=len(data.vocab),  # Vocabulary size
    num_hiddens=32,              # Hidden units per layer
    num_layers=2                 # Number of RNN layers
)

# Wrap in language model with cross-entropy loss
model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=2)
```

### 2.3 Train the Model

```python
# Set up trainer with gradient clipping
trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)

# Start training
trainer.fit(model, data)
```

The training process will optimize the model to predict the next character in the sequence. Gradient clipping prevents exploding gradients, which is common in deep RNNs.

## Step 3: Concise Implementation with High-Level APIs

While building from scratch is educational, in practice we use framework-provided RNN implementations for better performance and convenience. Let's create a concise GRU-based deep RNN.

### 3.1 Define the Multi-Layer GRU Class

```python
class GRU(d2l.RNN):
    """The multilayer GRU model."""
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.save_hyperparameters()
        # Use PyTorch's built-in GRU
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers, dropout=dropout)
```

This class leverages PyTorch's optimized `nn.GRU` module, which handles the multi-layer architecture internally.

### 3.2 Train the Concise Model

```python
# Initialize the GRU model
gru = GRU(
    num_inputs=len(data.vocab),  # Input size matches vocabulary
    num_hiddens=32,              # Hidden units
    num_layers=2                 # Number of layers
)

# Create the language model
model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)

# Train the model
trainer.fit(model, data)
```

### 3.3 Generate Text with the Trained Model

```python
# Generate text starting with "it has"
generated_text = model.predict('it has', 20, data.vocab, d2l.try_gpu())
print(generated_text)
```

The model will generate 20 additional characters based on the initial prompt "it has".

## Key Concepts and Best Practices

### Hyperparameter Selection
- **Hidden units**: Typically between 64 and 2056
- **Number of layers**: Usually between 1 and 8
- **Learning rate**: Requires careful tuning for convergence
- **Gradient clipping**: Essential for training stability

### Framework-Specific Notes
- **PyTorch/MXNet/TensorFlow**: Provide built-in multi-layer RNN implementations
- **JAX/Flax**: Requires more manual setup but offers flexibility
- **Dropout**: Can be added between layers to prevent overfitting

### Training Considerations
1. Deep RNNs require careful initialization
2. Learning rate scheduling often helps convergence
3. Gradient clipping is crucial for deep architectures
4. More layers don't always mean better performance—start simple

## Exercises

1. **Experiment with LSTM**: Replace the GRU with an LSTM and compare training speed and accuracy. Which converges faster? Which gives better perplexity?

2. **Scale the Data**: Train on multiple books instead of just *The Time Machine*. How low can you drive the perplexity with more data?

3. **Multi-Author Training**: Consider combining texts from different authors. What are the benefits? What potential issues might arise from style mixing?

## Summary

In this tutorial, you've learned how to:
- Build deep RNNs from scratch by stacking layers
- Implement forward propagation through multiple RNN layers
- Train deep RNNs on sequence data with gradient clipping
- Use high-level API implementations for efficiency
- Generate text with trained multi-layer RNNs

Deep RNNs extend the power of recurrent networks by adding depth in the input-output direction while maintaining temporal dependencies. They're particularly useful for complex sequence modeling tasks where relationships between inputs and outputs at the same time step are as important as long-term dependencies.

Remember that while deep RNNs are powerful, they require careful tuning of hyperparameters and regularization techniques to train effectively. Start with shallow architectures and gradually add depth as needed for your specific task.