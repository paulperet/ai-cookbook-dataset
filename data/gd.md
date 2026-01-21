# Gradient Descent: A Practical Guide

## Introduction

Gradient descent is a fundamental optimization algorithm used to minimize functions. While rarely used directly in modern deep learning, understanding gradient descent is crucial for grasping more advanced algorithms like stochastic gradient descent. In this guide, we'll explore gradient descent from basic principles to multivariate implementations, examining key concepts like learning rates, local minima, and adaptive methods.

## Prerequisites

We'll use a standard deep learning library interface. The code is framework-agnostic and will work with MXNet, PyTorch, or TensorFlow.

```python
%matplotlib inline
from d2l import torch as d2l  # Can be mxnet or tensorflow instead
import numpy as np
import torch  # Or the appropriate framework
```

## 1. One-Dimensional Gradient Descent

Let's start with the simplest case: optimizing a function of one variable. The core idea is to iteratively move in the direction opposite to the gradient (derivative) of the function.

### 1.1 Mathematical Foundation

For a continuously differentiable function $f: \mathbb{R} \rightarrow \mathbb{R}$, we can use a Taylor expansion:

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2)$$

If we choose $\epsilon = -\eta f'(x)$ where $\eta > 0$ is the learning rate, we get:

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x))$$

When $f'(x) \neq 0$, we have $\eta f'^2(x) > 0$, so for sufficiently small $\eta$, the function value decreases:

$$f(x - \eta f'(x)) \lessapprox f(x)$$

This gives us the gradient descent update rule:

$$x \leftarrow x - \eta f'(x)$$

### 1.2 Implementing Basic Gradient Descent

Let's implement gradient descent for a simple quadratic function $f(x) = x^2$, where we know the minimum is at $x=0$.

```python
def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x

def gd(eta, f_grad):
    """Gradient descent algorithm."""
    x = 10.0  # Initial value
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)  # Update x
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

# Run gradient descent with learning rate 0.2
results = gd(0.2, f_grad)
```

### 1.3 Visualizing the Optimization Path

Let's create a helper function to visualize how gradient descent progresses:

```python
def show_trace(results, f):
    """Plot the optimization trajectory."""
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], 
             [[f(x) for x in f_line], [f(x) for x in results]], 
             'x', 'f(x)', fmts=['-', '-o'])
    
show_trace(results, f)
```

## 2. The Critical Role of Learning Rate

The learning rate $\eta$ significantly impacts gradient descent performance. Let's examine what happens with different learning rates.

### 2.1 Learning Rate Too Small

With $\eta = 0.05$, progress is very slow:

```python
show_trace(gd(0.05, f_grad), f)
```

Even after 10 steps, we're still far from the optimal solution at $x=0$.

### 2.2 Learning Rate Too Large

With $\eta = 1.1$, the algorithm overshoots and diverges:

```python
show_trace(gd(1.1, f_grad), f)
```

The update steps are too large, causing $x$ to oscillate and move away from the minimum.

### 2.3 Dealing with Nonconvex Functions

For nonconvex functions, gradient descent can get stuck in local minima. Consider $f(x) = x \cdot \cos(cx)$:

```python
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

# With a high learning rate, we get stuck in a poor local minimum
show_trace(gd(2, f_grad), f)
```

## 3. Multivariate Gradient Descent

Now let's extend gradient descent to functions of multiple variables. For $f: \mathbb{R}^d \to \mathbb{R}$, the gradient is a vector of partial derivatives:

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top$$

The update rule becomes:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x})$$

### 3.1 Implementing 2D Gradient Descent

Let's optimize $f(\mathbf{x}) = x_1^2 + 2x_2^2$ with gradient $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$:

```python
def f_2d(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # Gradient of the objective function
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    """2D gradient descent update."""
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

def train_2d(trainer, steps=20, f_grad=None):
    """Optimize a 2D objective function."""
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results):
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')

# Run 2D gradient descent with η = 0.1
eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

The algorithm converges to the minimum at $[0, 0]$, though progress is somewhat slow due to the fixed learning rate.

## 4. Adaptive Methods

Choosing the right learning rate is challenging. Second-order methods that use curvature information can help, though they're often too expensive for deep learning.

### 4.1 Newton's Method

Newton's method uses second-order Taylor expansion:

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3)$$

Where $\mathbf{H} = \nabla^2 f(\mathbf{x})$ is the Hessian matrix. The Newton update is:

$$\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x})$$

Let's implement Newton's method:

```python
c = d2l.tensor(0.5)

def f(x):  # Objective function
    return d2l.cosh(c * x)

def f_grad(x):  # Gradient
    return c * d2l.sinh(c * x)

def f_hess(x):  # Hessian
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    """Newton's method with optional learning rate."""
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)  # Newton update
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

### 4.2 Newton's Method on Nonconvex Functions

For nonconvex functions, Newton's method can fail spectacularly:

```python
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # Hessian
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)  # Diverges!
```

Adding a learning rate can help stabilize the method:

```python
show_trace(newton(0.5), f)  # Converges with smaller step
```

### 4.3 Preconditioning

Since computing the full Hessian is expensive, preconditioning uses only diagonal entries:

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \textrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x})$$

This effectively gives each variable its own learning rate, which is helpful when variables have different scales.

## 5. Gradient Descent with Line Search

A hybrid approach combines gradient descent with line search to find the optimal step size:

1. Use $\nabla f(\mathbf{x})$ as the search direction
2. Perform binary search to find $\eta$ that minimizes $f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$

While this converges rapidly, it's too expensive for deep learning as it requires evaluating the objective function multiple times per update.

## Summary

- **Learning rates are crucial**: Too large causes divergence, too small leads to slow progress
- **Local minima**: Gradient descent can get stuck, especially in nonconvex functions
- **High dimensions**: Adjusting learning rates becomes increasingly complex
- **Preconditioning**: Helps handle variables with different scales
- **Newton's method**: Much faster for convex problems but can fail on nonconvex ones
- **Practical considerations**: Full second-order methods are often too expensive for deep learning

## Exercises

1. Experiment with different learning rates and objective functions for gradient descent
2. Implement line search for convex minimization:
   - Determine if derivatives are needed for the binary search decision
   - Analyze the convergence rate
   - Apply to minimizing $\log(\exp(x) + \exp(-2x - 3))$
3. Design a 2D objective function where gradient descent is very slow (hint: use different scales for each coordinate)
4. Implement lightweight Newton's method with diagonal preconditioning
5. Test the algorithm on various functions and observe what happens when coordinates are rotated by 45 degrees

## Next Steps

This foundation in gradient descent prepares you for understanding more advanced optimization algorithms used in deep learning, particularly stochastic gradient descent and its variants. The key insights about learning rates, convergence, and preconditioning will reappear in these more complex algorithms.