# Multivariable Calculus: Gradients, Backpropagation, and Matrix Derivatives

## Introduction
This guide explores the fundamentals of multivariable calculus as they apply to machine learning. You'll learn how to compute gradients for functions with billions of parameters, understand the geometry behind gradient descent, and master the chain rule for complex neural network computations.

## Prerequisites
Before starting, ensure you have the necessary libraries installed:

```bash
pip install numpy matplotlib torch tensorflow
```

## 1. Higher-Dimensional Differentiation

### Understanding Partial Derivatives
When working with loss functions that depend on billions of weights, we need to understand how changing individual parameters affects the output. For a function $L(w_1, w_2, \ldots, w_N)$, the partial derivative with respect to $w_1$ is:

$$L(w_1+\epsilon_1, w_2, \ldots, w_N) \approx L(w_1, w_2, \ldots, w_N) + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N).$$

### The Gradient Vector
By extending this reasoning to all parameters, we get the gradient approximation:

$$L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}),$$

where $\nabla_{\mathbf{w}} L = \left[\frac{\partial L}{\partial w_1}, \ldots, \frac{\partial L}{\partial w_N}\right]^\top$ is the gradient vector.

### Practical Example: Testing the Gradient Approximation
Let's verify this approximation with a concrete function:

```python
import numpy as np

def f(x, y):
    return np.log(np.exp(x) + np.exp(y))

def grad_f(x, y):
    return np.array([np.exp(x) / (np.exp(x) + np.exp(y)),
                     np.exp(y) / (np.exp(x) + np.exp(y))])

# Test point
epsilon = np.array([0.01, -0.03])
grad_approx = f(0, np.log(2)) + epsilon.dot(grad_f(0, np.log(2)))
true_value = f(0 + epsilon[0], np.log(2) + epsilon[1])

print(f'Approximation: {grad_approx}, True Value: {true_value}')
```

The output shows the gradient approximation closely matches the true function value, confirming our mathematical derivation.

## 2. Geometry of Gradients and Gradient Descent

### The Direction of Steepest Descent
From our approximation $L(\mathbf{w} + \mathbf{v}) \approx L(\mathbf{w}) + \mathbf{v}\cdot \nabla_{\mathbf{w}} L(\mathbf{w})$, we can see that to decrease $L$ most rapidly, we should move in the direction opposite to the gradient. This gives us the gradient descent algorithm:

1. Start with random initial parameters $\mathbf{w}$
2. Compute $\nabla_{\mathbf{w}} L(\mathbf{w})$
3. Update: $\mathbf{w} \leftarrow \mathbf{w} - \epsilon\nabla_{\mathbf{w}} L(\mathbf{w})$
4. Repeat until convergence

### Critical Points and Optimization
At a minimum, the gradient must be zero: $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$. Points satisfying this are called critical points. However, not all critical points are global minima—some could be local minima, maxima, or saddle points.

Let's examine a function to find its minima:

```python
import matplotlib.pyplot as plt

def f(x):
    return 3*x**4 - 4*x**3 - 12*x**2

def df(x):
    return 12*x**3 - 12*x**2 - 24*x

# Find critical points
x = np.linspace(-2, 3, 500)
critical_points = [-1, 0, 2]
values = [f(pt) for pt in critical_points]

print("Critical points and values:")
for pt, val in zip(critical_points, values):
    print(f"x={pt}: f(x)={val}")

# Plot
plt.figure(figsize=(8, 4))
plt.plot(x, f(x), label='f(x)')
plt.scatter(critical_points, values, color='red', zorder=5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()
```

The global minimum occurs at $x=2$ with $f(2)=-32$.

## 3. Multivariate Chain Rule

### Understanding Function Composition
When functions are composed, like in neural networks, we need the chain rule to compute derivatives efficiently. Consider:

$$\begin{aligned}
f(u, v) & = (u+v)^{2} \\
u(a, b) & = (a+b)^{2}, \quad v(a, b) = (a-b)^{2} \\
a(w, x, y, z) & = (w+x+y+z)^{2}, \quad b(w, x, y, z) = (w+x-y-z)^2
\end{aligned}$$

### Applying the Chain Rule
To compute $\frac{\partial f}{\partial w}$, we apply:

$$\frac{\partial f}{\partial w} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial w} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial w},$$

with further expansions for $\frac{\partial u}{\partial w}$ and $\frac{\partial v}{\partial w}$.

## 4. The Backpropagation Algorithm

### Manual Computation Example
Let's compute derivatives manually to understand the process:

```python
# Define inputs
w, x, y, z = -1, 0, -2, 1

# Forward pass
a = (w + x + y + z)**2
b = (w + x - y - z)**2
u = (a + b)**2
v = (a - b)**2
f = (u + v)**2

print(f'f at {w}, {x}, {y}, {z} is {f}')

# Compute partial derivatives
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db = 2*(a + b), 2*(a + b)
dv_da, dv_db = 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)

# Backward pass
du_dw = du_da*da_dw + du_db*db_dw
dv_dw = dv_da*da_dw + dv_db*db_dw
df_dw = df_du*du_dw + df_dv*dv_dw

print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
```

### Automated Backpropagation
Modern frameworks automate this process. Here's how to compute all gradients automatically:

**Using PyTorch:**
```python
import torch

w = torch.tensor([-1.], requires_grad=True)
x = torch.tensor([0.], requires_grad=True)
y = torch.tensor([-2.], requires_grad=True)
z = torch.tensor([1.], requires_grad=True)

# Forward pass
a = (w + x + y + z)**2
b = (w + x - y - z)**2
u = (a + b)**2
v = (a - b)**2
f = (u + v)**2

# Backward pass
f.backward()

print(f'df/dw: {w.grad.item()}')
print(f'df/dx: {x.grad.item()}')
print(f'df/dy: {y.grad.item()}')
print(f'df/dz: {z.grad.item()}')
```

**Using TensorFlow:**
```python
import tensorflow as tf

w = tf.Variable(-1.)
x = tf.Variable(0.)
y = tf.Variable(-2.)
z = tf.Variable(1.)

with tf.GradientTape(persistent=True) as tape:
    a = (w + x + y + z)**2
    b = (w + x - y - z)**2
    u = (a + b)**2
    v = (a - b)**2
    f = (u + v)**2

print(f'df/dw: {tape.gradient(f, w).numpy()}')
print(f'df/dx: {tape.gradient(f, x).numpy()}')
print(f'df/dy: {tape.gradient(f, y).numpy()}')
print(f'df/dz: {tape.gradient(f, z).numpy()}')
```

## 5. Hessians and Second-Order Approximations

### The Hessian Matrix
For a function $f(x_1, \ldots, x_n)$, the Hessian matrix contains all second partial derivatives:

$$\mathbf{H}_f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}.$$

### Quadratic Approximation
The Hessian enables better quadratic approximations:

$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0) \cdot (\mathbf{x}-\mathbf{x}_0) + \frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H} f(\mathbf{x}_0) (\mathbf{x}-\mathbf{x}_0).$$

## 6. Matrix Calculus

### Basic Matrix Derivatives
When differentiating matrix expressions, we often get clean results that resemble single-variable calculus:

1. For $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$: $\frac{d}{d\mathbf{x}} f = \boldsymbol{\beta}$
2. For $f(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x}$: $\frac{d}{d\mathbf{x}} f = (A + A^\top)\mathbf{x}$

### Practical Matrix Derivative Example
Consider matrix factorization where we want to minimize $\|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^2$:

```python
import numpy as np

def matrix_derivative_example():
    # Example dimensions
    n, m, r = 3, 4, 2
    
    # Random matrices
    X = np.random.randn(n, m)
    U = np.random.randn(n, r)
    V = np.random.randn(r, m)
    
    # Compute derivative manually
    manual_deriv = -2 * U.T @ (X - U @ V)
    
    # Verify using finite differences
    eps = 1e-6
    numerical_deriv = np.zeros_like(V)
    
    for i in range(r):
        for j in range(m):
            V_plus = V.copy()
            V_plus[i, j] += eps
            f_plus = np.sum((X - U @ V_plus)**2)
            
            V_minus = V.copy()
            V_minus[i, j] -= eps
            f_minus = np.sum((X - U @ V_minus)**2)
            
            numerical_deriv[i, j] = (f_plus - f_minus) / (2*eps)
    
    print(f"Manual derivative shape: {manual_deriv.shape}")
    print(f"Numerical derivative shape: {numerical_deriv.shape}")
    print(f"Maximum difference: {np.max(np.abs(manual_deriv - numerical_deriv))}")
    
    return manual_deriv, numerical_deriv

manual, numerical = matrix_derivative_example()
```

The derivative is $\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2} = -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V})$.

## Summary

In this guide, you've learned:

1. **Partial derivatives and gradients** extend single-variable calculus to functions with many parameters
2. **Gradient descent** uses the direction of steepest descent to minimize functions
3. **The chain rule** enables efficient computation of derivatives in composed functions
4. **Backpropagation** organizes chain rule computations for neural networks
5. **Hessian matrices** provide second-order approximations for better optimization
6. **Matrix calculus** gives concise rules for differentiating matrix expressions

These concepts form the mathematical foundation for training deep neural networks and understanding how they learn from data.

## Exercises

1. Compute derivatives of $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$ and $g(\mathbf{x}) = \mathbf{x}^\top\boldsymbol{\beta}$. Why are they the same?
2. Find $\frac{\partial}{\partial\mathbf{v}}\|\mathbf{v}\|_2$ for an $n$-dimensional vector $\mathbf{v}$
3. For $L(x, y) = \log(e^x + e^y)$, compute the gradient and sum its components
4. Analyze $f(x, y) = x^2y + xy^2$ to determine if $(0,0)$ is a maximum, minimum, or neither
5. Interpret $\nabla f = 0$ geometrically when minimizing $f(\mathbf{x}) = g(\mathbf{x}) + h(\mathbf{x})$