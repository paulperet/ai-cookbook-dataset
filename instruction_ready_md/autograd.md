# Automatic Differentiation: A Practical Guide

## Introduction

Automatic differentiation (autograd) is a foundational technique in modern deep learning that automates the calculation of derivatives. This guide walks you through the core concepts and practical implementation of autograd across four major frameworks: MXNet, PyTorch, TensorFlow, and JAX.

## Prerequisites

First, let's import the necessary libraries for each framework:

```python
# MXNet
from mxnet import autograd, np, npx
npx.set_np()

# PyTorch
import torch

# TensorFlow
import tensorflow as tf

# JAX
from jax import numpy as jnp
from jax import grad, random
import jax
```

## Step 1: Basic Gradient Calculation

Let's start with a simple function: $y = 2\mathbf{x}^{\top}\mathbf{x}$, where $\mathbf{x}$ is a column vector. We'll compute its gradient with respect to $\mathbf{x}$.

### 1.1 Initialize the Input Vector

First, create the input vector $\mathbf{x}$:

```python
# MXNet
x = np.arange(4.0)

# PyTorch
x = torch.arange(4.0)

# TensorFlow
x = tf.range(4, dtype=tf.float32)

# JAX
x = jnp.arange(4.0)
```

### 1.2 Prepare for Gradient Storage

Before computing gradients, we need to allocate storage. The gradient of a scalar-valued function with respect to a vector has the same shape as the vector:

```python
# MXNet: Attach gradient storage
x.attach_grad()

# PyTorch: Enable gradient tracking
x.requires_grad_(True)

# TensorFlow: Convert to Variable
x = tf.Variable(x)

# JAX: No explicit storage needed - gradients are computed on demand
```

### 1.3 Compute the Function

Now compute $y = 2\mathbf{x}^{\top}\mathbf{x}$:

```python
# MXNet: Record computation in autograd scope
with autograd.record():
    y = 2 * np.dot(x, x)

# PyTorch: Direct computation
y = 2 * torch.dot(x, x)

# TensorFlow: Record computation on tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)

# JAX: Define as function
y_func = lambda x: 2 * jnp.dot(x, x)
y = y_func(x)
```

### 1.4 Compute the Gradient

Now compute the gradient $\frac{dy}{d\mathbf{x}}$:

```python
# MXNet: Backward pass
y.backward()
gradient = x.grad

# PyTorch: Backward pass
y.backward()
gradient = x.grad

# TensorFlow: Get gradient from tape
gradient = t.gradient(y, x)

# JAX: Use grad transform
gradient = grad(y_func)(x)
```

### 1.5 Verify the Result

The theoretical gradient is $4\mathbf{x}$. Let's verify:

```python
# All frameworks
expected = 4 * x
print(gradient == expected)  # Should return True for all elements
```

## Step 2: Gradient Accumulation and Reset

Different frameworks handle gradient accumulation differently. Let's compute a new function and observe the behavior.

### 2.1 Compute Another Function

Compute $y = \sum_i x_i$:

```python
# MXNet: New computation overwrites previous gradient
with autograd.record():
    y = x.sum()
y.backward()
print(x.grad)  # Shows gradient of sum(x)

# PyTorch: Gradients accumulate by default
x.grad.zero_()  # Explicitly reset gradients
y = x.sum()
y.backward()
print(x.grad)  # Shows [1, 1, 1, 1]

# TensorFlow: New tape, new gradient
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
print(t.gradient(y, x))  # Shows [1, 1, 1, 1]

# JAX: Compute gradient of sum
sum_func = lambda x: x.sum()
print(grad(sum_func)(x))  # Shows [1, 1, 1, 1]
```

**Key Difference**: PyTorch accumulates gradients by default and requires manual resetting with `x.grad.zero_()`, while MXNet and TensorFlow start fresh with each new computation scope.

## Step 3: Gradients for Non-Scalar Outputs

When dealing with vector-valued functions, frameworks handle gradients differently. Typically, they sum the gradients of each output component.

### 3.1 Vector-Valued Function

Consider $y = \mathbf{x} \odot \mathbf{x}$ (element-wise square):

```python
# MXNet: Automatically sums before computing gradient
with autograd.record():
    y = x * x
y.backward()
print(x.grad)  # Shows 2*x

# PyTorch: Need to specify reduction
x.grad.zero_()
y = x * x
# Either sum then backward, or provide gradient vector
y.sum().backward()  # Preferred method
# Or: y.backward(gradient=torch.ones(len(y)))
print(x.grad)  # Shows 2*x

# TensorFlow: Automatically sums
with tf.GradientTape() as t:
    y = x * x
print(t.gradient(y, x))  # Shows 2*x

# JAX: grad() only works with scalar outputs
square_func = lambda x: (x * x).sum()
print(grad(square_func)(x))  # Shows 2*x
```

**Important**: PyTorch requires explicit handling of non-scalar outputs, while other frameworks automatically sum the outputs before differentiation.

## Step 4: Detaching Computations

Sometimes you need to break the computational graph to prevent gradient flow through certain operations.

### 4.1 The Detach Operation

Consider $z = x \cdot y$ where $y = x^2$, but we want to treat $y$ as a constant:

```python
# MXNet: Use detach()
with autograd.record():
    y = x * x
    u = y.detach()  # Break connection to x
    z = u * x
z.backward()
print(x.grad == u)  # Should be True

# PyTorch: Use detach()
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)  # Should be True

# TensorFlow: Use stop_gradient()
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x
print(t.gradient(z, x) == u)  # Should be True

# JAX: Use stop_gradient
y_func = lambda x: x * x
u = jax.lax.stop_gradient(y_func(x))
z_func = lambda x: u * x
print(grad(lambda x: z_func(x).sum())(x) == y_func(x))  # Should be True
```

**Note**: The original computation graph for $y$ still exists and can be differentiated separately.

## Step 5: Gradients with Python Control Flow

Automatic differentiation works even with complex control flow (conditionals, loops, etc.).

### 5.1 Function with Control Flow

Define a function with conditional logic:

```python
# MXNet
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

# PyTorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

# TensorFlow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c

# JAX
def f(a):
    b = a * 2
    while jnp.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

### 5.2 Compute Gradient with Dynamic Control Flow

```python
# MXNet
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
print(a.grad == d / a)  # Should be True

# PyTorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d / a)  # Should be True

# TensorFlow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
print(d_grad == d / a)  # Should be True

# JAX
key = random.PRNGKey(1)
a = random.normal(key, ())
d = f(a)
d_grad = grad(f)(a)
print(d_grad == d / a)  # Should be True
```

**Remark**: Despite the complex control flow, the gradient computation works correctly because autograd dynamically traces the execution path.

## Key Takeaways

1. **Framework Differences**: 
   - MXNet and TensorFlow create new gradient contexts for each computation
   - PyTorch accumulates gradients by default (requires manual reset)
   - JAX uses functional transformations and immutable arrays

2. **Non-Scalar Outputs**: Most frameworks automatically sum vector outputs before differentiation, but PyTorch requires explicit handling.

3. **Graph Manipulation**: Use `detach()` (PyTorch/MXNet) or `stop_gradient()` (TensorFlow/JAX) to break connections in the computational graph.

4. **Control Flow**: Automatic differentiation handles dynamic control flow by tracing the actual execution path.

## Best Practices

1. Always check whether your framework accumulates or resets gradients between operations
2. For vector-valued functions, ensure you understand how gradients are aggregated
3. Use detachment operations judiciously to control gradient flow
4. Remember that autograd works with dynamic control structures, enabling flexible model architectures

Automatic differentiation has revolutionized deep learning by freeing practitioners from manual gradient calculations. While each framework has its nuances, the core concepts remain consistent across implementations.