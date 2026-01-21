# A Practical Guide to Integral Calculus

This guide introduces the core concepts of integral calculus, focusing on its geometric interpretation, the fundamental theorem, and practical computation techniques. We'll explore how integration connects to differentiation and how to compute areas under curves, including in multiple dimensions.

## Prerequisites

Before we begin, ensure you have the necessary libraries installed. We'll use `matplotlib` for visualization and a deep learning framework of your choice (MXNet, PyTorch, or TensorFlow) for numerical operations.

```bash
pip install matplotlib
# Install one of: mxnet, torch, tensorflow
```

## 1. Geometric Interpretation of Integration

Integration answers the question: "What is the area under a curve?" For a non-negative function \( f(x) \), the area between the curve and the x-axis over an interval \([a, b]\) is denoted by:

\[
\text{Area}(\mathcal{A}) = \int_a^b f(x) \;dx.
\]

The variable inside the integral is a dummy variable—it can be replaced with any other symbol without changing the integral's value.

### 1.1 Visualizing the Area Under a Curve

Let's visualize the area under the curve \( f(x) = e^{-x^2} \). First, we'll set up our environment and plot the function.

```python
import matplotlib.pyplot as plt
import numpy as np

# Define the function and interval
x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

# Plot the function and shade the area
plt.figure(figsize=(8, 4))
plt.plot(x, f, color='black')
plt.fill_between(x, f, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Area under f(x) = e^{-x^2}')
plt.grid(True)
plt.show()
```

The shaded region represents the total area under the curve from \( x = -2 \) to \( x = 2 \). However, this area might be infinite or undefined for some functions. Typically, we compute the area between two finite endpoints.

### 1.2 Approximating Areas with Rectangles

One way to approximate an integral is to chop the area into thin vertical slices, approximate each slice as a rectangle, and sum their areas. This is known as the Riemann sum approach.

Let's approximate \( \int_0^2 \frac{x}{1+x^2} dx \) using this method.

```python
epsilon = 0.05  # Width of each rectangle
a = 0
b = 2

x = np.arange(a, b, epsilon)
f = x / (1 + x**2)

# Approximate the integral by summing areas of rectangles
approx = np.sum(epsilon * f)
true_value = np.log(2) / 2  # Known exact value

# Visualize the approximation
plt.figure(figsize=(8, 4))
plt.bar(x, f, width=epsilon, align='edge', alpha=0.5, edgecolor='black')
plt.plot(x, f, color='black', linewidth=2)
plt.ylim([0, 1])
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'Approximation: {approx:.4f}, True Value: {true_value:.4f}')
plt.grid(True)
plt.show()
```

This approximation improves as \( \epsilon \to 0 \), but for complex functions, this numerical approach can be inefficient. We need a more analytical method.

## 2. The Fundamental Theorem of Calculus

The fundamental theorem of calculus connects integration with differentiation. Define:

\[
F(x) = \int_0^x f(y) dy.
\]

This function measures the area from 0 to \( x \). The key insight is:

\[
\int_a^b f(x) \;dx = F(b) - F(a).
\]

### 2.1 The Derivative of the Area Function

Consider a small change \( \epsilon \) in \( x \). The change in area is:

\[
F(x+\epsilon) - F(x) = \int_x^{x+\epsilon} f(y) \; dy.
\]

For small \( \epsilon \), this area is approximately a rectangle with height \( f(x) \) and width \( \epsilon \):

\[
F(x+\epsilon) - F(x) \approx \epsilon f(x).
\]

This is exactly the definition of the derivative:

\[
\frac{dF}{dx}(x) = f(x).
\]

Thus, the fundamental theorem states:

\[
\frac{d}{dx}\int_0^x f(y) \; dy = f(x).
\]

This means integration is essentially the reverse of differentiation. To find an integral, we look for a function whose derivative is the integrand.

### 2.2 Applying the Fundamental Theorem

Using known derivatives, we can compute integrals directly. For example:

- Since \( \frac{d}{dx} x^n = n x^{n-1} \), we have:
  \[
  \int_0^x n y^{n-1} \; dy = x^n.
  \]

- Since \( \frac{d}{dx} e^x = e^x \), we have:
  \[
  \int_0^x e^y \; dy = e^x - 1.
  \]

This approach transforms integration from a geometric slicing problem into an algebraic one.

## 3. Change of Variables

The change of variables formula (also called substitution) is a powerful technique for simplifying integrals. It's the integral version of the chain rule.

### 3.1 The Formula

Suppose we have an integral in terms of \( x \), but we substitute \( x = u(t) \). Then:

\[
\int_{u(a)}^{u(b)} f(x) \; dx = \int_a^b f(u(t)) \cdot u'(t) \; dt.
\]

### 3.2 Example Application

Compute \( \int_0^1 y e^{-y^2} dy \).

Let \( u = -y^2 \), so \( du = -2y dy \). Then:

\[
\int_0^1 y e^{-y^2} dy = -\frac{1}{2} \int_0^{-1} e^u du = -\frac{1}{2} (e^{-1} - e^0) = \frac{1 - e^{-1}}{2}.
\]

We can verify this numerically:

```python
from scipy.integrate import quad

def integrand(y):
    return y * np.exp(-y**2)

result, error = quad(integrand, 0, 1)
print(f"Numerical result: {result:.6f}")
print(f"Analytical result: {(1 - np.exp(-1)) / 2:.6f}")
```

## 4. Signed Areas and Integration Direction

Areas computed by integration can be negative in two cases:

1. **Negative Functions**: If \( f(x) < 0 \) over an interval, the integral is negative.
   \[
   \int_0^1 (-1) dx = -1.
   \]

2. **Reversed Limits**: Integrating from right to left flips the sign.
   \[
   \int_0^{-1} 1 dx = -1.
   \]

The sign represents orientation, similar to the determinant in linear algebra. Two sign flips cancel:

\[
\int_0^{-1} (-1) dx = 1.
\]

## 5. Multiple Integrals

For functions of two variables \( f(x, y) \), the double integral:

\[
\int_{[a,b]\times[c,d]} f(x, y) \; dx dy
\]

represents the volume under the surface \( z = f(x, y) \) over the rectangle.

### 5.1 Visualizing a Multivariable Function

Let's plot \( f(x, y) = e^{-x^2 - y^2} \):

```python
from mpl_toolkits.mplot3d import Axes3D

# Create grid
x = np.linspace(-2, 2, 101)
y = np.linspace(-2, 2, 101)
X, Y = np.meshgrid(x, y)
Z = np.exp(-X**2 - Y**2)

# 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('f(x, y) = e^{-x^2 - y^2}')
plt.show()
```

### 5.2 Iterated Integration

Fubini's Theorem allows us to compute double integrals as iterated single integrals:

\[
\int_{[a,b]\times[c,d]} f(x, y) \; dx dy = \int_c^d \left( \int_a^b f(x, y) \; dx \right) dy.
\]

The order can be swapped:

\[
= \int_a^b \left( \int_c^d f(x, y) \; dy \right) dx.
\]

## 6. Change of Variables in Multiple Integrals

For multivariable functions, the change of variables formula involves the Jacobian determinant.

### 6.1 The Jacobian

Given a transformation \( \phi: \mathbb{R}^n \to \mathbb{R}^n \), the Jacobian matrix is:

\[
D\phi = \begin{bmatrix}
\frac{\partial \phi_1}{\partial x_1} & \cdots & \frac{\partial \phi_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial \phi_n}{\partial x_1} & \cdots & \frac{\partial \phi_n}{\partial x_n}
\end{bmatrix}.
\]

The change of variables formula is:

\[
\int_{\phi(U)} f(\mathbf{x}) d\mathbf{x} = \int_U f(\phi(\mathbf{x})) |\det(D\phi(\mathbf{x}))| d\mathbf{x}.
\]

### 6.2 Polar Coordinates Example

Compute \( \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} e^{-x^2 - y^2} dx dy \).

Use polar coordinates: \( x = r\cos\theta, y = r\sin\theta \). The Jacobian determinant is \( r \), so:

\[
\int_0^\infty \int_0^{2\pi} e^{-r^2} r d\theta dr = 2\pi \int_0^\infty r e^{-r^2} dr = \pi.
\]

This famous result will reappear when studying probability distributions.

## 7. Exercises

Test your understanding with these problems:

1. Compute \( \int_1^2 \frac{1}{x} dx \).
2. Use substitution to find \( \int_0^{\sqrt{\pi}} x\sin(x^2) dx \).
3. Evaluate \( \int_{[0,1]^2} xy dx dy \).
4. For \( f(x, y) = \frac{xy(x^2-y^2)}{(x^2+y^2)^3} \), compute both \( \int_0^2\int_0^1 f(x,y) dy dx \) and \( \int_0^1\int_0^2 f(x,y) dx dy \). Notice they differ—this function violates Fubini's Theorem conditions.

## Summary

- Integration computes areas under curves and volumes under surfaces.
- The fundamental theorem of calculus links integration to differentiation: \( \frac{d}{dx} \int_a^x f(t) dt = f(x) \).
- Change of variables simplifies integrals by substitution.
- Multiple integrals compute higher-dimensional volumes via iterated integration.
- The Jacobian determinant accounts for distortion during multivariable change of coordinates.

With these tools, you can approach a wide range of integration problems encountered in machine learning and data science.