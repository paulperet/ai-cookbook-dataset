# Implementing Bidirectional Recurrent Neural Networks
:label:`sec_bi_rnn`

## Introduction

In previous tutorials on sequence learning, we focused on tasks like language modeling, where predicting the next token depends only on the preceding context. This unidirectional approach is natural for such tasks. However, many sequence learning problems benefit from conditioning predictions on both past *and* future contexts. For example:
*   **Part-of-speech tagging:** The grammatical role of a word often depends on surrounding words in both directions.
*   **Masked token prediction:** A common pretraining task where a model must predict a missing word. The correct prediction depends heavily on the words that come after the blank (e.g., "I am `___`" vs. "I am `___` hungry").

Bidirectional Recurrent Neural Networks (BiRNNs) :cite:`Schuster.Paliwal.1997` elegantly solve this by processing sequences in both forward and reverse directions simultaneously.

## Core Concept

A BiRNN consists of two separate unidirectional RNN layers:
1.  A **forward RNN** processes the input sequence from the first element to the last.
2.  A **backward RNN** processes the input sequence from the last element to the first.

At each time step `t`, the hidden states from both directions are concatenated to form the final hidden state, which incorporates context from the entire sequence.

Formally, for a minibatch input `X_t` at time step `t`:
*   The forward hidden state is: `H_t_forward = φ(X_t * W_xh_f + H_{t-1}_forward * W_hh_f + b_h_f)`
*   The backward hidden state is: `H_t_backward = φ(X_t * W_xh_b + H_{t+1}_backward * W_hh_b + b_h_b)`

The combined hidden state `H_t` is the concatenation `[H_t_forward, H_t_backward]`. This state is then passed to the output layer.

## Prerequisites

Before we begin, ensure you have the necessary libraries installed. This guide provides implementations for multiple frameworks.

```bash
# Install the d2l library which provides common utilities
# The framework-specific imports are handled in the code blocks below.
```

## Implementation from Scratch

Let's first build a bidirectional RNN from fundamental components to understand the mechanics.

### Step 1: Define the Bidirectional RNN Class

We'll create a `BiRNNScratch` class that contains two independent `RNNScratch` instances—one for the forward pass and one for the backward pass.

```python
class BiRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        # Instantiate two separate RNNs
        self.f_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        # The output dimension doubles due to concatenation
        self.num_hiddens *= 2
```

### Step 2: Implement the Forward Pass

The forward method must:
1.  Run the forward RNN on the input sequence.
2.  Run the backward RNN on the *reversed* input sequence.
3.  Concatenate the outputs from both directions at each time step.

```python
@d2l.add_to_class(BiRNNScratch)
def forward(self, inputs, Hs=None):
    # Unpack initial hidden states if provided
    f_H, b_H = Hs if Hs is not None else (None, None)
    
    # Forward RNN pass
    f_outputs, f_H = self.f_rnn(inputs, f_H)
    # Backward RNN pass (note the reversed input)
    b_outputs, b_H = self.b_rnn(reversed(inputs), b_H)
    
    # Concatenate outputs. Reverse the backward outputs to align time steps.
    outputs = [d2l.concat((f, b), -1) for f, b in zip(
        f_outputs, reversed(b_outputs))]
    
    return outputs, (f_H, b_H)
```

**Key Insight:** We reverse the `b_outputs` list before concatenation because the backward RNN processes the sequence in reverse order. This step aligns the hidden states from both directions for the same original time step.

## Concise Implementation Using High-Level APIs

Manually managing two RNNs is instructive but verbose. Most deep learning frameworks provide built-in support for bidirectional RNNs. Here's how to implement a Bidirectional GRU concisely.

### For PyTorch and MXNet

These frameworks offer a `bidirectional=True` argument in their RNN layer constructors.

```python
class BiGRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        if framework == 'mxnet':
            from mxnet.gluon import rnn
            self.rnn = rnn.GRU(num_hiddens, bidirectional=True)
        if framework == 'pytorch':
            from torch import nn
            self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional=True)
        # Hidden dimension is doubled
        self.num_hiddens *= 2
```

### Note on JAX/Flax

As of this writing, Flax's `nn` module does not include a built-in RNN layer with a `bidirectional` flag. To create a bidirectional RNN in JAX, you would manually implement the logic shown in the "Implementation from Scratch" section, using Flax's `nn.Module` setup.

## Summary

*   **Bidirectional RNNs** leverage both past and future context by combining two unidirectional RNNs that process the sequence in opposite directions.
*   They are exceptionally powerful for **sequence encoding** tasks (e.g., part-of-speech tagging, named entity recognition) where full-sentence context is crucial.
*   The primary trade-off is **computational cost and training complexity** due to the longer gradient chains that result from processing the sequence in both directions.

## Exercises

Test your understanding by tackling these challenges:

1.  **Shape Analysis:** If the forward RNN uses `h_f` hidden units and the backward RNN uses `h_b` hidden units, what will the dimension of the concatenated hidden state `H_t` be?
2.  **Deep BiRNNs:** Extend the scratch implementation to design a bidirectional RNN with multiple hidden layers. Consider how the hidden states should be passed between layers.
3.  **Handling Polysemy:** Words like "bank" have different meanings based on context. How could you design a neural model that, given a context sequence and a target word, produces a context-aware vector representation for that word? What architectural features (like attention or bidirectional layers) would be particularly useful?

---
*Discussions: [MXNet](https://discuss.d2l.ai/t/339) | [PyTorch](https://discuss.d2l.ai/t/1059) | [JAX](https://discuss.d2l.ai/t/18019)*