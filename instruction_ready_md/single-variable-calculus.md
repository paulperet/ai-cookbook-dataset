# Single Variable Calculus: A Practical Guide

This guide provides a practical, code-first introduction to single variable calculus, focusing on concepts essential for machine learning. We'll explore derivatives, their computational rules, and applications through hands-on examples.

## Prerequisites

First, let's import the necessary libraries. We'll use a plotting utility from the D2L library for visualization.

```python
import numpy as np
import matplotlib.pyplot as plt
```

## 1. Understanding Derivatives Through Finite Differences

Differential calculus studies how functions change under small input variations. In machine learning, this helps us understand how to adjust model parameters to minimize loss.

### 1.1 Visualizing Local Linearity

Consider the function `f(x) = sin(x^x)`. When we zoom in sufficiently, any smooth function appears linear locally.

Let's examine this behavior at different scales:

```python
# Define our function
def f(x):
    return np.sin(x**x)

# Plot at different scales
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Large scale
x_big = np.arange(0.01, 3.01, 0.01)
axes[0].plot(x_big, f(x_big))
axes[0].set_title('Large scale [0, 3]')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')

# Medium scale
x_med = np.arange(1.75, 2.25, 0.001)
axes[1].plot(x_med, f(x_med))
axes[1].set_title('Medium scale [1.75, 2.25]')
axes[1].set_xlabel('x')

# Small scale
x_small = np.arange(2.0, 2.01, 0.0001)
axes[2].plot(x_small, f(x_small))
axes[2].set_title('Small scale [2.0, 2.01]')
axes[2].set_xlabel('x')

plt.tight_layout()
plt.show()
```

Notice how the function appears increasingly linear as we zoom in. This observation is fundamental: in a small enough interval, we can approximate any smooth function with a straight line.

### 1.2 Computing Derivatives via Finite Differences

The derivative measures how much the output changes relative to a small input change. We can approximate it using the difference quotient:

```python
def L(x):
    return x**2 + 1701*(x-4)**3

# Approximate derivative at x=4 using different epsilon values
x = 4
for epsilon in [0.1, 0.001, 0.0001, 0.00001]:
    derivative_approx = (L(x + epsilon) - L(x)) / epsilon
    print(f'epsilon = {epsilon:.5f} -> derivative ≈ {derivative_approx:.5f}')
```

As ε approaches 0, the approximation converges to the true derivative value of 8. Mathematically, we write:

$$
\lim_{\epsilon \rightarrow 0}\frac{L(4+\epsilon) - L(4)}{\epsilon} = 8
$$

## 2. Derivative Rules and Their Applications

While finite differences work, they're computationally expensive. Fortunately, we have rules that let us compute derivatives analytically.

### 2.1 Common Derivatives

Here are essential derivatives to remember:

- Constants: `d/dx c = 0`
- Linear functions: `d/dx (ax) = a`
- Power rule: `d/dx x^n = n*x^(n-1)`
- Exponential: `d/dx e^x = e^x`
- Logarithm: `d/dx log(x) = 1/x`

### 2.2 Combining Functions: Sum, Product, and Chain Rules

When functions combine, we use these rules:

1. **Sum Rule**: `d/dx [g(x) + h(x)] = g'(x) + h'(x)`
2. **Product Rule**: `d/dx [g(x)·h(x)] = g(x)h'(x) + g'(x)h(x)`
3. **Chain Rule**: `d/dx g(h(x)) = g'(h(x))·h'(x)`

Let's apply these rules to compute a complex derivative:

```python
# Example: Compute derivative of log(1 + (x-1)^10)
# Using chain rule and other derivative rules
x = 2.5  # Example point
derivative = 10 * (x-1)**9 / (1 + (x-1)**10)
print(f"Derivative of log(1+(x-1)^10) at x={x}: {derivative:.4f}")
```

### 2.3 Linear Approximation

The derivative provides the best linear approximation to a function near a point:

```python
def linear_approximation(f, df, x0, xs):
    """Return linear approximation f(x0) + f'(x0)*(x-x0)"""
    return f(x0) + df(x0) * (xs - x0)

# Example with sin(x)
xs = np.arange(-np.pi, np.pi, 0.01)
f_vals = np.sin(xs)
df_vals = np.cos(xs)  # Derivative of sin is cos

# Plot sin(x) and its linear approximations at various points
plt.figure(figsize=(10, 6))
plt.plot(xs, f_vals, label='sin(x)', linewidth=2)

for x0 in [-1.5, 0, 2]:
    approx = linear_approximation(np.sin, np.cos, x0, xs)
    plt.plot(xs, approx, '--', label=f'Approximation at x={x0}')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.ylim([-1.5, 1.5])
plt.legend()
plt.title('Linear Approximations of sin(x)')
plt.show()
```

## 3. Higher-Order Derivatives and Taylor Series

### 3.1 Second Derivatives and Curvature

The second derivative tells us about a function's curvature:

```python
def analyze_curvature():
    """Demonstrate how second derivative relates to curvature"""
    # Quadratic functions with different curvatures
    x = np.linspace(-2, 2, 100)
    
    # Positive curvature (concave up)
    f1 = x**2
    # Negative curvature (concave down)
    f2 = -x**2
    # Zero curvature (straight line)
    f3 = 2*x + 1
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(x, f1)
    plt.title('Positive curvature\nf\'\'(x) > 0')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(x, f2)
    plt.title('Negative curvature\nf\'\'(x) < 0')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(x, f3)
    plt.title('Zero curvature\nf\'\'(x) = 0')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

analyze_curvature()
```

### 3.2 Quadratic Approximations

We can get better approximations using second derivatives:

```python
def quadratic_approximation(f, df, d2f, x0, xs):
    """Quadratic Taylor approximation: f(x0) + f'(x0)(x-x0) + f''(x0)(x-x0)^2/2"""
    return f(x0) + df(x0)*(xs - x0) + 0.5*d2f(x0)*(xs - x0)**2

# Compare linear and quadratic approximations for sin(x)
xs = np.arange(-np.pi, np.pi, 0.01)
x0 = 1.0

linear_approx = linear_approximation(np.sin, np.cos, x0, xs)
quadratic_approx = quadratic_approximation(np.sin, np.cos, lambda x: -np.sin(x), x0, xs)

plt.figure(figsize=(10, 6))
plt.plot(xs, np.sin(xs), label='sin(x)', linewidth=2)
plt.plot(xs, linear_approx, '--', label='Linear approximation')
plt.plot(xs, quadratic_approx, '-.', label='Quadratic approximation')
plt.axvline(x=x0, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title(f'Approximations of sin(x) near x={x0}')
plt.show()
```

### 3.3 Taylor Series for Exponential Function

Taylor series provide polynomial approximations using derivatives of all orders:

```python
def taylor_approximation(degree, x0, xs):
    """Taylor approximation of exp(x) around x0"""
    approximation = np.zeros_like(xs)
    for n in range(degree + 1):
        # nth derivative of exp(x) is exp(x)
        term = np.exp(x0) * (xs - x0)**n / np.math.factorial(n)
        approximation += term
    return approximation

# Compare Taylor approximations of exp(x)
xs = np.arange(0, 3, 0.01)
true_values = np.exp(xs)

plt.figure(figsize=(10, 6))
plt.plot(xs, true_values, 'k-', label='exp(x)', linewidth=2)

degrees = [1, 2, 5]
for degree in degrees:
    approx = taylor_approximation(degree, 0, xs)
    plt.plot(xs, approx, '--', label=f'Degree {degree} Taylor')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Taylor Series Approximations of exp(x)')
plt.grid(True, alpha=0.3)
plt.show()
```

## 4. Practical Applications in Machine Learning

### 4.1 Gradient-Based Optimization

Derivatives enable gradient descent, the workhorse of neural network training:

```python
def gradient_descent_step(f, df, x_current, learning_rate=0.1):
    """Perform one step of gradient descent"""
    gradient = df(x_current)
    x_next = x_current - learning_rate * gradient
    return x_next

# Example: Minimize f(x) = x^2
def f_example(x):
    return x**2

def df_example(x):
    return 2*x

# Starting point
x = 3.0
print(f"Initial: x = {x}, f(x) = {f_example(x):.4f}")

# Perform gradient descent steps
for i in range(10):
    x = gradient_descent_step(f_example, df_example, x, learning_rate=0.1)
    print(f"Step {i+1}: x = {x:.4f}, f(x) = {f_example(x):.4f}")
```

### 4.2 Analyzing Loss Functions

Derivatives help us understand loss landscape geometry:

```python
def analyze_loss_landscape():
    """Analyze a simple loss function and its derivatives"""
    x = np.linspace(-2, 2, 100)
    loss = x**4 - 2*x**2 + 0.5*x  # Example loss function
    gradient = 4*x**3 - 4*x + 0.5  # First derivative
    hessian = 12*x**2 - 4  # Second derivative
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(x, loss)
    axes[0].set_title('Loss Function')
    axes[0].set_xlabel('Parameter x')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    axes[1].plot(x, gradient)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_title('Gradient (First Derivative)')
    axes[1].set_xlabel('Parameter x')
    axes[1].set_ylabel('Gradient')
    axes[1].grid(True)
    
    axes[2].plot(x, hessian)
    axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[2].set_title('Hessian (Second Derivative)')
    axes[2].set_xlabel('Parameter x')
    axes[2].set_ylabel('Hessian')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

analyze_loss_landscape()
```

## 5. Exercises

Test your understanding with these problems:

1. Compute the derivative of `f(x) = x^3 - 4x + 1`
2. Find the derivative of `log(1/x)`
3. If `f'(x) = 0`, does `f` necessarily have a maximum or minimum at `x`?
4. Find the minimum of `f(x) = x*log(x)` for `x ≥ 0`

### Solutions Template

```python
# Exercise 1: Derivative of x^3 - 4x + 1
def derivative_ex1(x):
    # Your code here
    pass

# Exercise 2: Derivative of log(1/x)
def derivative_ex2(x):
    # Your code here
    pass

# Exercise 4: Find minimum of x*log(x)
def find_minimum():
    # Your code here
    pass
```

## Summary

In this guide, we've covered:

1. **Derivatives as local linear approximations**: Functions appear linear when zoomed in sufficiently
2. **Computational rules**: Sum, product, and chain rules for computing derivatives analytically
3. **Higher-order derivatives**: Second derivatives describe curvature, enabling better approximations
4. **Taylor series**: Polynomial approximations using multiple derivatives
5. **Machine learning applications**: Gradient-based optimization and loss landscape analysis

These concepts form the mathematical foundation for understanding and implementing optimization algorithms in machine learning, particularly for training neural networks via backpropagation.

Remember: While we used finite differences for conceptual understanding, in practice we use automatic differentiation (as implemented in frameworks like PyTorch and TensorFlow) for efficient gradient computation.