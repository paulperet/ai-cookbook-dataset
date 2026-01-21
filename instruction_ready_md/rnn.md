# Recurrent Neural Networks (RNNs): A Practical Guide

## Introduction

In traditional language models like Markov models and n-grams, the prediction of the next token depends only on a fixed number of previous tokens. This approach has limitations: to capture longer-term dependencies, we need to increase `n`, which causes model parameters to grow exponentially.

Recurrent Neural Networks (RNNs) solve this problem by introducing a *hidden state* that maintains information across time steps. This hidden state acts as a memory, allowing the model to incorporate information from all previous tokens when making predictions.

## Prerequisites

Before we begin, ensure you have the necessary imports. The code below supports multiple deep learning frameworks—choose the one you're working with.

```python
# Select your framework: 'mxnet', 'pytorch', 'tensorflow', or 'jax'
framework = 'pytorch'  # Change this to your preferred framework

if framework == 'mxnet':
    from d2l import mxnet as d2l
    from mxnet import np, npx
    npx.set_np()
elif framework == 'pytorch':
    from d2l import torch as d2l
    import torch
elif framework == 'tensorflow':
    from d2l import tensorflow as d2l
    import tensorflow as tf
elif framework == 'jax':
    from d2l import jax as d2l
    import jax
    from jax import numpy as jnp
```

## Step 1: Understanding Neural Networks Without Hidden States

Let's start by reviewing a simple Multilayer Perceptron (MLP) with one hidden layer. This will help us contrast it with RNNs later.

Given a minibatch of examples `X` with shape `(n, d)` (batch size `n`, input dimension `d`), the hidden layer output `H` is computed as:

```python
H = ϕ(X @ W_xh + b_h)
```

Where:
- `W_xh` is the weight matrix of shape `(d, h)`
- `b_h` is the bias vector of shape `(1, h)`
- `ϕ` is the activation function
- `h` is the number of hidden units

The output layer then produces:
```python
O = H @ W_hq + b_q
```

For classification, we apply `softmax(O)` to get probability distributions. This architecture has **no memory**—each input is processed independently.

## Step 2: Introducing Hidden States

RNNs differ by maintaining a hidden state that evolves over time. At each time step `t`, the hidden state `H_t` depends on both the current input `X_t` and the previous hidden state `H_{t-1}`:

```python
H_t = ϕ(X_t @ W_xh + H_{t-1} @ W_hh + b_h)
```

Key components:
- `W_xh`: Weights for current input (shape `(d, h)`)
- `W_hh`: Weights for previous hidden state (shape `(h, h)`)
- `b_h`: Bias term (shape `(1, h)`)

The output at time step `t` is:
```python
O_t = H_t @ W_hq + b_q
```

Crucially, **the same parameters** (`W_xh`, `W_hh`, `b_h`, `W_hq`, `b_q`) are reused across all time steps. This parameter sharing makes RNNs efficient for sequential data.

## Step 3: Visualizing the RNN Computation

The following diagram illustrates how information flows through an RNN across three time steps:

```
Time t-1: [X_{t-1}] → [H_{t-1}] → [O_{t-1}]
                    ↓
Time t:   [X_t] → [H_t] → [O_t]
                    ↓
Time t+1: [X_{t+1}] → [H_{t+1}] → [O_{t+1}]
```

At each step:
1. Concatenate (conceptually) the current input `X_t` and previous hidden state `H_{t-1}`
2. Pass through a fully connected layer with activation `ϕ` to get `H_t`
3. Use `H_t` to compute the output `O_t`
4. Pass `H_t` to the next time step

## Step 4: Implementation Insight - Concatenation Equivalence

The computation `X_t @ W_xh + H_{t-1} @ W_hh` is mathematically equivalent to concatenating the matrices and performing a single matrix multiplication. Let's verify this with code:

```python
# Initialize matrices
if framework in ['mxnet', 'pytorch']:
    X, W_xh = d2l.randn(3, 1), d2l.randn(1, 4)
    H, W_hh = d2l.randn(3, 4), d2l.randn(4, 4)
elif framework == 'tensorflow':
    X, W_xh = d2l.normal((3, 1)), d2l.normal((1, 4))
    H, W_hh = d2l.normal((3, 4)), d2l.normal((4, 4))
elif framework == 'jax':
    X, W_xh = jax.random.normal(d2l.get_key(), (3, 1)), jax.random.normal(
        d2l.get_key(), (1, 4))
    H, W_hh = jax.random.normal(d2l.get_key(), (3, 4)), jax.random.normal(
        d2l.get_key(), (4, 4))

# Method 1: Separate multiplications
result1 = d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
print("Separate multiplication result shape:", result1.shape)

# Method 2: Concatenation approach
X_H_concat = d2l.concat((X, H), 1)  # Shape: (3, 5)
W_concat = d2l.concat((W_xh, W_hh), 0)  # Shape: (5, 4)
result2 = d2l.matmul(X_H_concat, W_concat)
print("Concatenation result shape:", result2.shape)

# Verify they're equivalent (within numerical precision)
print("Results are close:", d2l.norm(result1 - result2).item() < 1e-6)
```

Both approaches yield the same result, demonstrating the mathematical equivalence. In practice, the concatenation approach can be more efficient for implementation.

## Step 5: Building a Character-Level Language Model

Let's apply RNNs to a practical task: character-level language modeling. Given a sequence of characters, we want to predict the next character at each time step.

Consider the word "machine". Our training setup would be:
- Input sequence: "m", "a", "c", "h", "i", "n"
- Target sequence: "a", "c", "h", "i", "n", "e"

At time step 3:
- Input: "c" (along with hidden state from processing "m" and "a")
- Target: "h"
- The RNN uses the history ("m", "a", "c") to predict the probability distribution for the next character

During training:
1. We compute outputs `O_t` at each time step
2. Apply softmax to get probability distributions
3. Use cross-entropy loss between predictions and targets
4. Backpropagate through time to update parameters

## Step 6: Key Advantages of RNNs

1. **Parameter Efficiency**: The same parameters are reused across all time steps, so model size doesn't grow with sequence length.
2. **Variable-Length Inputs**: RNNs can process sequences of any length.
3. **Temporal Dynamics**: The hidden state captures dependencies across time steps.

## Common Challenges and Solutions

While RNNs are powerful, they face several challenges:

1. **Vanishing/Exploding Gradients**: During backpropagation through many time steps, gradients can become extremely small or large. Solutions include:
   - Gradient clipping
   - Using gated architectures (LSTM, GRU)
   - Proper weight initialization

2. **Computational Complexity**: Processing long sequences can be slow. Techniques to address this:
   - Truncated backpropagation through time
   - Parallelization where possible

3. **Limited Context**: Basic RNNs struggle with very long-term dependencies. Advanced architectures like LSTMs and Transformers often perform better for these cases.

## Exercises for Practice

1. **Output Dimension**: If we use an RNN to predict the next character in a text sequence with a vocabulary size of `V`, what should be the output dimension?
   
2. **Conditional Probability**: Explain why RNNs can express the conditional probability of a token based on all previous tokens, unlike n-gram models.

3. **Gradient Analysis**: What happens to gradients when backpropagating through a very long sequence? How does this affect training?

4. **Model Limitations**: Identify potential problems with the character-level language model described here and suggest improvements.

## Summary

Recurrent Neural Networks provide a powerful framework for modeling sequential data by maintaining a hidden state that serves as memory. Through recurrent computation and parameter sharing, RNNs can capture temporal dependencies efficiently. While they have limitations (particularly with very long sequences), they form the foundation for more advanced sequence models and remain essential tools in natural language processing, time series analysis, and other sequential data applications.

In the next guide, we'll implement a complete RNN-based language model and explore practical training techniques.