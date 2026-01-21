# Momentum: A Practical Guide to Accelerating Gradient Descent

## Introduction

In standard stochastic gradient descent (SGD), noisy gradients can cause optimization challenges. If we decrease the learning rate too quickly, convergence stalls. If we're too lenient, noise prevents us from reaching a good solution. Momentum addresses this by introducing a "velocity" term that averages past gradients, providing more stable and accelerated convergence.

In this guide, you'll implement momentum from scratch, understand its mathematical properties, and apply it to practical optimization problems.

## Prerequisites

First, ensure you have the necessary libraries installed. This guide supports multiple frameworks.

```bash
# Install d2l library for educational utilities
pip install d2l
```

```python
# Framework-specific imports
# For MXNet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

# For PyTorch
from d2l import torch as d2l
import torch

# For TensorFlow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Understanding the Problem: Ill-Conditioned Optimization

To understand why momentum helps, let's examine an ill-conditioned optimization problem. Consider the function:

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$$

This function is very flat in the $x_1$ direction but steep in the $x_2$ direction, creating a narrow canyon-like optimization landscape.

### Step 1: Visualizing Standard Gradient Descent

Let's see how standard gradient descent performs on this function with a learning rate of 0.4.

```python
eta = 0.4

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

# Visualize the optimization path
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

The gradient in the $x_2$ direction is much larger and changes more rapidly than in the $x_1$ direction. With a small learning rate, we avoid divergence in $x_2$ but converge slowly in $x_1$. With a larger learning rate, we progress faster in $x_1$ but diverge in $x_2$.

### Step 2: Increasing the Learning Rate

Let's see what happens when we increase the learning rate to 0.6:

```python
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

Convergence in the $x_1$ direction improves, but the solution quality worsens overall due to oscillations in the $x_2$ direction.

## Implementing Momentum

The momentum method solves this problem by maintaining a velocity vector that accumulates past gradients. The update equations are:

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1} \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t
\end{aligned}
$$

Here, $\beta$ is the momentum parameter (typically between 0 and 1), and $\mathbf{g}_{t, t-1}$ is the gradient at time $t$.

### Step 3: Implementing Momentum for 2D Optimization

Let's implement momentum for our 2D problem:

```python
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

# Test with beta = 0.5
eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Even with the same learning rate that caused divergence before, momentum converges well. The velocity term averages gradients, reducing oscillations in the $x_2$ direction while maintaining progress in the $x_1$ direction.

### Step 4: Reducing the Momentum Parameter

What happens if we reduce the momentum parameter?

```python
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

With $\beta = 0.25$, convergence is slower but still better than without momentum. This demonstrates that momentum provides robustness even with suboptimal hyperparameters.

## Understanding Effective Sample Weight

The velocity term can be expanded as:

$$\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$$

In the limit, the sum of weights is $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$. This means momentum effectively increases the step size to $\frac{\eta}{1-\beta}$ while providing a better-behaved descent direction.

### Step 5: Visualizing Weight Decay

Let's visualize how different $\beta$ values weight past gradients:

```python
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend()
```

Larger $\beta$ values give more weight to past gradients, creating a longer memory. Smaller $\beta$ values focus more on recent gradients.

## Practical Implementation

Now let's implement momentum for a real optimization problem using minibatch stochastic gradient descent.

### Step 6: Initializing Momentum States

First, we need to initialize velocity variables with the same shape as our parameters:

```python
# For MXNet and PyTorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)

# For TensorFlow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

### Step 7: Implementing the Momentum Update

Now let's implement the momentum update rule:

```python
# For MXNet
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v

# For PyTorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()

# For TensorFlow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
        v[:].assign(hyperparams['momentum'] * v + g)
        p[:].assign(p - hyperparams['lr'] * v)
```

### Step 8: Training with Momentum

Let's train a model using momentum:

```python
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

# Get data
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)

# Train with momentum = 0.5
train_momentum(0.02, 0.5)
```

### Step 9: Increasing Momentum

Let's increase the momentum to 0.9 (effective sample size of 10) and adjust the learning rate:

```python
train_momentum(0.01, 0.9)
```

### Step 10: Fine-tuning Learning Rate

For even smoother convergence, we can reduce the learning rate further:

```python
train_momentum(0.005, 0.9)
```

## Concise Implementation

Most deep learning frameworks have built-in momentum support. Here's how to use it:

```python
# For MXNet
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)

# For PyTorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)

# For TensorFlow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## Theoretical Insights

### Quadratic Convex Functions

For a quadratic convex function $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$, momentum decomposes the optimization into coordinate-wise updates along the eigenvectors of $\mathbf{Q}$. This explains why momentum is particularly effective for ill-conditioned problems where eigenvalues vary significantly.

### Convergence Analysis

For the scalar function $f(x) = \frac{\lambda}{2} x^2$, gradient descent converges when $|1 - \eta \lambda| < 1$. With momentum, convergence occurs for $0 < \eta \lambda < 2 + 2\beta$, which is a larger feasible region than the $0 < \eta \lambda < 2$ for standard gradient descent.

Let's visualize convergence rates for different $\lambda$ values:

```python
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend()
```

## Summary

In this guide, you've learned:

1. **Momentum replaces gradients** with a leaky average over past gradients, accelerating convergence significantly.
2. **It benefits both noise-free gradient descent** and stochastic gradient descent by preventing stalling.
3. **The effective number of gradients** considered is $\frac{1}{1-\beta}$ due to exponential downweighting of past data.
4. **Implementation requires storing** an additional state vector (velocity $\mathbf{v}$) but is straightforward.
5. **Momentum is particularly effective** for ill-conditioned problems with varying curvature directions.

## Exercises

1. Experiment with different combinations of momentum hyperparameters and learning rates. Observe how they affect convergence.
2. Apply gradient descent and momentum to a quadratic problem with multiple eigenvalues: $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$ where $\lambda_i = 2^{-i}$. Plot how values decrease from initialization $x_i = 1$.
3. Derive the minimum value and minimizer for $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$.
4. Investigate how stochastic gradient descent with momentum differs from minibatch SGD with momentum. Experiment with different parameter settings.

## Further Reading

For a deeper understanding of momentum, consult:
- The [Distill article](https://distill.pub/2017/momentum/) by Goh (2017) for interactive visualizations
- Polyak (1964) for the original proposal
- Nesterov (2018) for theoretical analysis in convex optimization
- Sutskever et al. (2013) for applications in deep learning