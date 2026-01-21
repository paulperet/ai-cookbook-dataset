# Stochastic Gradient Descent (SGD) Cookbook

## Overview
This guide explains Stochastic Gradient Descent (SGD), a fundamental optimization algorithm for training machine learning models. You'll learn how SGD reduces computational costs compared to standard gradient descent, implement it with different learning rate schedules, and understand its convergence properties.

## Prerequisites
Ensure you have the required libraries installed. This guide provides code for three frameworks: MXNet, PyTorch, and TensorFlow.

```bash
# Install the d2l library which contains utility functions
pip install d2l
```

Choose your framework and import the necessary modules.

```python
# For MXNet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()
```

```python
# For PyTorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```python
# For TensorFlow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

## 1. Understanding the Stochastic Gradient Update

In deep learning, the objective function is typically the average loss over all training examples. For `n` examples with loss functions `f_i(x)` and parameter vector `x`, the objective is:

```
f(x) = (1/n) * Σ f_i(x)
```

The gradient is:
```
∇f(x) = (1/n) * Σ ∇f_i(x)
```

Standard gradient descent computes this full gradient at each iteration, costing `O(n)`. **Stochastic Gradient Descent (SGD)** reduces this cost by randomly sampling a single example `i` at each iteration and updating parameters using only `∇f_i(x)`:

```
x ← x - η * ∇f_i(x)
```

Where `η` is the learning rate. This reduces per-iteration cost to `O(1)`. The stochastic gradient is an unbiased estimate of the true gradient since `E[∇f_i(x)] = ∇f(x)`.

## 2. Implementing Basic SGD with Simulated Noise

Let's implement SGD on a simple convex function and compare it to gradient descent by adding simulated noise to the gradient.

First, define the objective function and its gradient:

```python
def f(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Gradient of the objective function
    return 2 * x1, 4 * x2
```

Now implement the SGD update function with added Gaussian noise (mean=0, variance=1) to simulate stochasticity:

```python
# MXNet version
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```python
# PyTorch version  
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```python
# TensorFlow version
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

Define a constant learning rate function and run SGD:

```python
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

You'll observe that the optimization path is much noisier than with standard gradient descent. Even after 50 steps, we haven't converged to the optimum at (0,0). The constant learning rate is insufficient—we need dynamic learning rate scheduling.

## 3. Implementing Dynamic Learning Rate Schedules

Fixed learning rates struggle with SGD's noise. Let's implement three common decay strategies:

### 3.1 Exponential Decay
The learning rate decreases exponentially: `η(t) = η₀ * exp(-λt)`

```python
def exponential_lr():
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

Exponential decay reduces parameter variance but may decay too quickly, preventing convergence to the optimum.

### 3.2 Polynomial Decay
A more gradual decay: `η(t) = η₀ * (βt + 1)^{-α}`. With α=0.5, this gives inverse square root decay.

```python
def polynomial_lr():
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Polynomial decay provides better convergence, getting closer to the optimum in just 50 steps.

## 4. Convergence Analysis for Convex Objectives (Optional)

For convex functions, we can analyze SGD's convergence. Assuming the stochastic gradient has bounded norm `∥∇f_i(x)∥ ≤ L` and initial distance `∥x₁ - x*∥² ≤ r²`, with learning rate `η = r/(L√T)`, SGD converges at rate `O(1/√T)`.

The key inequality bounding the expected suboptimality after T steps is:

```
E[R(x̄)] - R* ≤ (r² + L² Σ η_t²) / (2 Σ η_t)
```

Where `x̄` is a weighted average of the optimization path. This shows convergence depends on gradient bound `L`, initial distance `r`, and learning rate schedule.

## 5. Practical Considerations: Sampling Methods

In practice, we typically sample training examples **without replacement** for each epoch. Compared to sampling with replacement:
- **Without replacement**: Each example seen exactly once per epoch, lower variance
- **With replacement**: Examples may be repeated or missed, higher variance

The probability of missing a specific example when sampling with replacement is approximately 37%, making it less data-efficient.

## 6. Key Takeaways

1. **Computational Efficiency**: SGD reduces per-iteration cost from O(n) to O(1), crucial for large datasets.
2. **Learning Rate Scheduling**: Dynamic learning rates (polynomial decay) work better than constant rates for SGD.
3. **Convergence Guarantees**: For convex problems, SGD converges with rate O(1/√T) given proper learning rates.
4. **Non-Convex Challenges**: For deep learning (non-convex objectives), theoretical guarantees are limited, but SGD remains effective in practice.
5. **Sampling Strategy**: Sample without replacement for better data efficiency and lower variance.

## 7. Exercises for Practice

1. Experiment with different learning rate schedules (constant, exponential, polynomial) and iteration counts. Plot the distance from optimum (0,0) versus iterations.
2. Prove that adding Gaussian noise to the gradient of `f(x₁,x₂)=x₁²+2x₂²` is equivalent to minimizing a loss with normally distributed inputs.
3. Compare convergence when sampling with versus without replacement.
4. How would you modify SGD if certain gradient coordinates are consistently larger?
5. Analyze the function `f(x)=x²(1+sin x)`. How many local minima exist? Can you modify it to require evaluating all local minima for optimization?

## Next Steps
SGD forms the foundation for more advanced optimizers like Momentum, RMSProp, and Adam. Understanding SGD's properties will help you choose and tune these more complex algorithms effectively.