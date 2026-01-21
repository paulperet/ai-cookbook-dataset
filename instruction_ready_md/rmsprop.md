# RMSProp Optimization Algorithm: A Practical Guide

## Overview

RMSProp (Root Mean Square Propagation) addresses a key limitation of Adagrad by decoupling learning rate scheduling from coordinate-adaptive scaling. While Adagrad's learning rate decreases too aggressively for non-convex problems, RMSProp maintains per-parameter adaptability through exponential moving averages of squared gradients.

## Core Algorithm

The RMSProp update equations are:

1. **State update**: $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2$
2. **Parameter update**: $\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t$

Where:
- $\gamma$: Decay rate (typically 0.9)
- $\eta$: Learning rate
- $\epsilon$: Small constant (1e-6) for numerical stability
- $\mathbf{g}_t$: Gradient at time step $t$

## Prerequisites

First, install the required libraries and import necessary modules:

```bash
pip install d2l torch matplotlib numpy
```

```python
import math
import matplotlib.pyplot as plt
import torch
from d2l import torch as d2l
```

## Visualizing RMSProp's Weight Decay

Let's examine how different decay rates ($\gamma$) affect the weighting of past gradients:

```python
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]

for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')

plt.xlabel('Time steps')
plt.ylabel('Weight')
plt.legend()
plt.show()
```

This visualization shows how $\gamma$ controls the "memory" of the algorithm. Higher $\gamma$ values give more weight to historical gradients.

## Implementing RMSProp from Scratch

### 1. 2D Test Function

We'll test RMSProp on the quadratic function $f(\mathbf{x})=0.1x_1^2+2x_2^2$:

```python
def rmsprop_2d(x1, x2, s1, s2):
    """RMSProp update for 2D parameters."""
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    """Test function to optimize."""
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

# Set hyperparameters
eta, gamma = 0.4, 0.9

# Visualize the optimization trajectory
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

Notice how RMSProp maintains reasonable progress throughout optimization, unlike Adagrad which slows down excessively.

### 2. General Implementation for Neural Networks

Now let's implement a general RMSProp optimizer:

```python
def init_rmsprop_states(feature_dim):
    """Initialize state variables for RMSProp."""
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def rmsprop(params, states, hyperparams):
    """RMSProp update rule."""
    gamma, eps = hyperparams['gamma'], 1e-6
    
    for p, s in zip(params, states):
        with torch.no_grad():
            # Update state with exponential moving average of squared gradients
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            # Update parameters with adaptive learning rate
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

### 3. Training Example

Let's test our implementation on a simple dataset:

```python
# Prepare data
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)

# Train with RMSProp
d2l.train_ch11(
    rmsprop,
    init_rmsprop_states(feature_dim),
    {'lr': 0.01, 'gamma': 0.9},
    data_iter,
    feature_dim
)
```

With $\gamma=0.9$, the state $\mathbf{s}$ aggregates approximately 10 past observations ($1/(1-0.9)$).

## Using Built-in RMSProp

Most deep learning frameworks provide RMSProp implementations. Here's how to use them:

### PyTorch
```python
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9}, data_iter)
```

### TensorFlow/Keras
```python
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9}, data_iter)
```

Note: Parameter names differ between frameworks (`alpha` in PyTorch, `rho` in TensorFlow).

## Key Insights

1. **Adaptive Learning Rates**: RMSProp maintains per-parameter learning rates based on recent gradient magnitudes
2. **Decoupled Scheduling**: Unlike Adagrad, learning rate scheduling is separate from gradient scaling
3. **Exponential Decay**: Recent gradients have more influence than distant ones, controlled by $\gamma$
4. **Numerical Stability**: The $\epsilon$ term prevents division by extremely small values

## Practical Considerations

- **Typical Values**: $\gamma=0.9$, $\eta=0.001$, $\epsilon=10^{-6}$
- **Learning Rate**: May still require scheduling for complex problems
- **Memory**: Only stores one additional state variable per parameter
- **Compatibility**: Works well with momentum and other optimization enhancements

## Exercises

1. **Extreme Decay**: Set $\gamma = 1$ and observe the behavior. What happens and why?
2. **Rotated Problem**: Minimize $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. How does convergence change?
3. **Real Dataset**: Test RMSProp on Fashion-MNIST with different learning rates.
4. **Adaptive $\gamma$**: Would adjusting $\gamma$ during training be beneficial? How sensitive is RMSProp to $\gamma$ changes?

## Further Reading

- Original RMSProp paper: Tieleman & Hinton (2012)
- Comparison with other adaptive methods: Adam, Adagrad, Adadelta
- Practical tuning guides for deep learning optimization

RMSProp strikes a balance between Adagrad's per-parameter adaptability and the need for reasonable learning rates throughout training, making it a popular choice for non-convex optimization in deep learning.