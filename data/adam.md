# Adam Optimizer: A Comprehensive Tutorial

## Introduction

Adam (Adaptive Moment Estimation) is one of the most popular optimization algorithms in deep learning, combining the strengths of several previous optimization techniques. In this tutorial, you'll learn how Adam works, implement it from scratch, and explore its improved variant called Yogi.

## Prerequisites

Before we begin, ensure you have the necessary libraries installed:

```bash
pip install d2l torch torchvision  # For PyTorch
# or
pip install d2l tensorflow         # For TensorFlow
# or
pip install d2l mxnet              # For MXNet
```

## 1. Understanding the Adam Algorithm

Adam combines several optimization techniques we've seen before:
- Momentum for accelerating convergence
- RMSProp for adaptive learning rates per parameter
- Bias correction for better initialization

### 1.1 The Core Equations

Adam maintains two state variables for each parameter:

1. **First moment (momentum) estimate**:
   ```
   v_t = β₁ * v_{t-1} + (1 - β₁) * g_t
   ```

2. **Second moment (uncentered variance) estimate**:
   ```
   s_t = β₂ * s_{t-1} + (1 - β₂) * g_t²
   ```

Where:
- `g_t` is the gradient at time step `t`
- `β₁` and `β₂` are decay rates (typically 0.9 and 0.999)
- `v_t` and `s_t` are initialized to 0

### 1.2 Bias Correction

Since `v_t` and `s_t` are initialized to 0, they're biased toward 0 during early training. Adam corrects this with:

```python
v̂_t = v_t / (1 - β₁^t)
ŝ_t = s_t / (1 - β₂^t)
```

### 1.3 Parameter Update

The final parameter update is:

```python
g_t' = η * v̂_t / (√(ŝ_t) + ε)
x_t = x_{t-1} - g_t'
```

Where:
- `η` is the learning rate
- `ε` is a small constant (typically 10⁻⁶) for numerical stability

## 2. Implementing Adam from Scratch

Let's implement Adam step by step. We'll focus on the PyTorch version, but the concepts apply to all frameworks.

### 2.1 Initialize State Variables

First, we need to initialize the momentum and second moment variables:

```python
def init_adam_states(feature_dim):
    """Initialize state variables for Adam."""
    # For weights
    v_w = torch.zeros((feature_dim, 1))
    s_w = torch.zeros((feature_dim, 1))
    
    # For biases
    v_b = torch.zeros(1)
    s_b = torch.zeros(1)
    
    return ((v_w, s_w), (v_b, s_b))
```

### 2.2 Implement the Adam Update

Now, let's implement the core Adam update logic:

```python
def adam(params, states, hyperparams):
    """Adam optimizer update step."""
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    
    for param, (v, s) in zip(params, states):
        with torch.no_grad():
            # Update biased first moment estimate
            v[:] = beta1 * v + (1 - beta1) * param.grad
            
            # Update biased second moment estimate
            s[:] = beta2 * s + (1 - beta2) * torch.square(param.grad)
            
            # Compute bias-corrected estimates
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            
            # Update parameters
            param[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)
        
        # Zero out gradients for next iteration
        param.grad.data.zero_()
    
    # Increment time step
    hyperparams['t'] += 1
```

### 2.3 Training with Our Adam Implementation

Let's test our implementation on a simple problem:

```python
# Load data
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)

# Train the model
d2l.train_ch11(
    adam, 
    init_adam_states(feature_dim),
    {'lr': 0.01, 't': 1}, 
    data_iter, 
    feature_dim
)
```

## 3. Using Built-in Adam Optimizers

While implementing from scratch is educational, in practice you'll use built-in optimizers. Here's how to use them in different frameworks:

### 3.1 PyTorch

```python
import torch

# Create optimizer
trainer = torch.optim.Adam

# Train the model
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

### 3.2 TensorFlow

```python
import tensorflow as tf

# Create optimizer
trainer = tf.keras.optimizers.Adam

# Train the model
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

### 3.3 MXNet/Gluon

```python
# Using Gluon's trainer
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```

## 4. Understanding and Implementing Yogi

Yogi is an improved variant of Adam that addresses convergence issues in certain scenarios. The key difference is in how it updates the second moment estimate.

### 4.1 The Problem with Adam

In Adam, the second moment update is:
```
s_t = s_{t-1} + (1 - β₂) * (g_t² - s_{t-1})
```

When gradients have high variance or are sparse, `s_t` can forget past values too quickly, potentially causing divergence.

### 4.2 Yogi's Solution

Yogi modifies the update to:
```
s_t = s_{t-1} + (1 - β₂) * g_t² ⊙ sign(g_t² - s_{t-1})
```

This ensures the update magnitude doesn't depend on the deviation amount, improving stability.

### 4.3 Implementing Yogi

```python
def yogi(params, states, hyperparams):
    """Yogi optimizer update step."""
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    
    for param, (v, s) in zip(params, states):
        with torch.no_grad():
            # Update momentum (same as Adam)
            v[:] = beta1 * v + (1 - beta1) * param.grad
            
            # Yogi's modified second moment update
            grad_sq = torch.square(param.grad)
            s[:] = s + (1 - beta2) * torch.sign(grad_sq - s) * grad_sq
            
            # Bias correction
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            
            # Parameter update
            param[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr) + eps)
        
        param.grad.data.zero_()
    
    hyperparams['t'] += 1
```

### 4.4 Testing Yogi

```python
# Train with Yogi
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(
    yogi,
    init_adam_states(feature_dim),
    {'lr': 0.01, 't': 1},
    data_iter,
    feature_dim
)
```

## 5. Practical Considerations

### 5.1 Learning Rate Selection
- Start with the default learning rate of 0.001
- For Adam, typical learning rates range from 0.0001 to 0.1
- Always use learning rate scheduling or decay as training progresses

### 5.2 When to Use Adam vs Yogi
- **Use Adam** for most standard deep learning tasks
- **Consider Yogi** when:
  - You have very sparse gradients
  - You're experiencing convergence issues with Adam
  - Working with non-stationary objectives

### 5.3 Common Hyperparameters
```python
# Recommended defaults
hyperparams = {
    'lr': 0.001,      # Learning rate
    'beta1': 0.9,     # Momentum decay
    'beta2': 0.999,   # Second moment decay
    'eps': 1e-8       # Numerical stability
}
```

## 6. Exercises for Practice

1. **Learning Rate Experiment**: Try training with different learning rates (0.1, 0.01, 0.001, 0.0001) and observe how it affects convergence speed and final performance.

2. **Bias Correction Analysis**: Modify the Adam implementation to remove bias correction. Compare the training curves with and without bias correction, especially during the first few epochs.

3. **Learning Rate Decay**: Implement learning rate decay (e.g., reduce by 10% every 10 epochs) and observe how it affects convergence in the later stages of training.

4. **Divergence Case**: Try to create a scenario where Adam diverges but Yogi converges. Hint: Use extremely sparse gradients or a pathological objective function.

## Summary

In this tutorial, you've learned:

1. **How Adam works**: It combines momentum and RMSProp with bias correction
2. **Implementation details**: How to implement Adam from scratch in PyTorch
3. **Built-in usage**: How to use Adam in popular deep learning frameworks
4. **Yogi optimizer**: An improved variant that addresses Adam's convergence issues
5. **Practical considerations**: Tips for using Adam effectively in real projects

Adam remains one of the most robust and widely-used optimizers in deep learning. While Yogi offers improvements in certain edge cases, standard Adam with proper hyperparameter tuning works well for most applications.

## Further Reading

- Original Adam paper: [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- Yogi paper: [Adaptive Methods for Nonconvex Optimization](https://papers.nips.cc/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html)
- D2L Book: [Deep Learning with Adam](https://d2l.ai/chapter_optimization/adam.html)

Remember that while Adam is a powerful optimizer, it's not a silver bullet. Always monitor your training, experiment with different optimizers, and choose what works best for your specific problem.