# Implementing the Adadelta Optimizer: A Step-by-Step Guide

## Overview

Adadelta is an adaptive learning rate optimization algorithm that builds upon AdaGrad and RMSProp. Unlike other optimizers, Adadelta doesn't require a manually set learning rate parameter—it automatically adapts the learning rate based on the historical gradient information and parameter changes. This guide walks you through implementing Adadelta from scratch and using high-level API implementations.

## Prerequisites

First, ensure you have the necessary libraries installed. This tutorial provides implementations for MXNet, PyTorch, and TensorFlow.

```bash
# Install the deep learning framework of your choice
# For MXNet:
pip install mxnet

# For PyTorch:
pip install torch

# For TensorFlow:
pip install tensorflow
```

## Understanding the Adadelta Algorithm

Adadelta maintains two state variables for each parameter:
1. **sₜ**: A leaky average of the squared gradients (similar to RMSProp)
2. **Δxₜ**: A leaky average of the squared parameter updates

The update equations are:

1. Update the squared gradient average:
   ```
   sₜ = ρ·sₜ₋₁ + (1-ρ)·gₜ²
   ```

2. Compute the rescaled gradient:
   ```
   gₜ' = (√(Δxₜ₋₁ + ε) / √(sₜ + ε)) ⊙ gₜ
   ```

3. Update the parameters:
   ```
   xₜ = xₜ₋₁ - gₜ'
   ```

4. Update the squared update average:
   ```
   Δxₜ = ρ·Δxₜ₋₁ + (1-ρ)·(gₜ')²
   ```

Where ρ is the decay rate (typically 0.9) and ε is a small constant (1e-5) for numerical stability.

## Step 1: Implementing Adadelta from Scratch

### MXNet Implementation

```python
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    """Initialize state variables for Adadelta."""
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    """Update parameters using the Adadelta algorithm."""
    rho, eps = hyperparams['rho'], 1e-5
    
    for p, (s, delta) in zip(params, states):
        # Update squared gradient average
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        
        # Compute rescaled gradient
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        
        # Update parameters
        p[:] -= g
        
        # Update squared update average
        delta[:] = rho * delta + (1 - rho) * g * g
```

### PyTorch Implementation

```python
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    """Initialize state variables for Adadelta."""
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    """Update parameters using the Adadelta algorithm."""
    rho, eps = hyperparams['rho'], 1e-5
    
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # Update squared gradient average
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            
            # Compute rescaled gradient
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            
            # Update parameters
            p[:] -= g
            
            # Update squared update average
            delta[:] = rho * delta + (1 - rho) * g * g
        
        # Clear gradients for next iteration
        p.grad.data.zero_()
```

### TensorFlow Implementation

```python
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    """Initialize state variables for Adadelta."""
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    """Update parameters using the Adadelta algorithm."""
    rho, eps = hyperparams['rho'], 1e-5
    
    for p, (s, delta), grad in zip(params, states, grads):
        # Update squared gradient average
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        
        # Compute rescaled gradient
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        
        # Update parameters
        p[:].assign(p - g)
        
        # Update squared update average
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

## Step 2: Testing the Implementation

Now let's test our Adadelta implementation on a sample dataset. We'll use ρ = 0.9, which corresponds to a half-life of 10 updates for the moving averages.

```python
# Load sample data
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)

# Train using our custom Adadelta implementation
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim)
```

When you run this code, you'll see the training progress showing decreasing loss over epochs, demonstrating that Adadelta is successfully optimizing the model parameters.

## Step 3: Using High-Level API Implementations

For practical applications, you can use the built-in Adadelta implementations in each framework.

### MXNet

```python
# Using MXNet's built-in Adadelta optimizer
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

### PyTorch

```python
# Using PyTorch's built-in Adadelta optimizer
import torch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

### TensorFlow

```python
# Using TensorFlow's built-in Adadelta optimizer
import tensorflow as tf

# Note: Adadelta may require a different learning rate in TensorFlow
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate': 5.0, 'rho': 0.9}, data_iter)
```

**Note for TensorFlow users**: The default learning rate might not converge well for some problems. You may need to adjust it, as shown above with `learning_rate=5.0`.

## Key Takeaways

1. **Learning Rate Free**: Adadelta doesn't require a manually set learning rate parameter. It automatically adapts based on historical gradient and parameter change information.

2. **Dual State Variables**: Adadelta maintains two state variables per parameter—one for squared gradients and one for squared parameter updates.

3. **Leaky Averages**: The algorithm uses exponential moving averages (leaky averages) to maintain running estimates, with ρ controlling the decay rate.

4. **Numerical Stability**: The small ε constant (1e-5) prevents division by zero in the gradient rescaling calculation.

## Exercises for Further Exploration

1. **Experiment with ρ**: Try different values of ρ (e.g., 0.5, 0.95). Observe how it affects convergence speed and stability.

2. **Alternative Implementation**: Implement Adadelta without explicitly computing gₜ'. Consider how this might improve numerical stability.

3. **Limitations Analysis**: Identify optimization problems where Adadelta might struggle. Is it truly "learning rate free" in all scenarios?

4. **Comparative Study**: Compare Adadelta's convergence behavior with AdaGrad and RMSProp on different types of optimization landscapes.

## Next Steps

Now that you understand how Adadelta works, you can:
- Apply it to your own deep learning models
- Experiment with different ρ values for your specific problems
- Combine Adadelta with other optimization techniques like momentum
- Analyze its performance compared to other adaptive optimizers like Adam

Remember that while Adadelta eliminates the need to manually set a learning rate, you still need to choose an appropriate ρ value for your problem. The typical default of ρ = 0.9 works well for many applications, but don't hesitate to experiment with different values based on your specific use case.