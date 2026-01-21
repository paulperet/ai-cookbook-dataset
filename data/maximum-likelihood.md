# Maximum Likelihood Estimation: A Practical Guide

## Introduction

Maximum likelihood estimation (MLE) is a fundamental concept in machine learning and statistics. It provides a principled approach to estimating model parameters by finding the values that make the observed data most probable. This guide will walk you through the core concepts with practical examples.

## Prerequisites

This tutorial uses Python with common numerical libraries. We'll provide examples in three frameworks for comparison:

```python
# For MXNet users
# !pip install mxnet
from mxnet import autograd, np, npx
npx.set_np()

# For PyTorch users  
# !pip install torch
import torch

# For TensorFlow users
# !pip install tensorflow
import tensorflow as tf
```

## The Maximum Likelihood Principle

The core idea is simple: given a probabilistic model with unknown parameters θ and observed data X, the best parameters are those that maximize the probability of observing the data:

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta})
$$

This follows from Bayesian reasoning with an uninformative prior. The quantity $P(X \mid \boldsymbol{\theta})$ is called the **likelihood**.

## Step 1: A Concrete Coin Flipping Example

Let's start with a simple example to build intuition. Suppose we have a coin with probability θ of landing heads. After flipping 13 times, we observe the sequence "HHHTHTTHHHHHT" (9 heads, 4 tails).

The likelihood function is:

$$
P(X \mid \theta) = \theta^9(1-\theta)^4
$$

### Step 1.1: Visualizing the Likelihood

Let's plot this function to see where it reaches its maximum:

```python
# MXNet version
theta = np.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4
# Plotting code would go here

# PyTorch version  
theta = torch.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4

# TensorFlow version
theta = tf.range(0, 1, 0.001)
p = theta**9 * (1 - theta)**4
```

The plot shows the maximum occurs near 9/13 ≈ 0.692, which matches our intuition.

### Step 1.2: Analytical Solution

We can find the exact maximum by taking the derivative and setting it to zero:

```python
# Derivative calculation
# d/dθ [θ^9(1-θ)^4] = 9θ^8(1-θ)^4 - 4θ^9(1-θ)^3
#                   = θ^8(1-θ)^3(9-13θ)
```

Setting this equal to zero gives solutions at θ = 0, 1, and 9/13. The first two give zero probability to our data, so the maximum likelihood estimate is:

$$
\hat{\theta} = \frac{9}{13}
$$

## Step 2: Numerical Optimization with Negative Log-Likelihood

For real-world problems with billions of parameters, we need numerical methods. Directly working with likelihood causes numerical underflow. The solution is to use the **negative log-likelihood**.

### Step 2.1: Why Use Log-Likelihood?

1. **Numerical stability**: Products of probabilities become sums of log-probabilities
2. **Computational efficiency**: Derivatives become simpler to compute
3. **Theoretical connections**: Relates to information theory and entropy

For our coin example, the negative log-likelihood is:

$$
-\log(P(X \mid \theta)) = -(n_H\log(\theta) + n_T\log(1-\theta))
$$

### Step 2.2: Gradient Descent Implementation

Let's implement gradient descent to find the MLE, even for massive datasets:

```python
# MXNet implementation
n_H = 8675309  # Number of heads
n_T = 256245   # Number of tails

theta = np.array(0.5)
theta.attach_grad()

lr = 1e-9
for iter in range(100):
    with autograd.record():
        loss = -(n_H * np.log(theta) + n_T * np.log(1 - theta))
    loss.backward()
    theta -= lr * theta.grad

print(f"Estimated θ: {theta}")
print(f"True ratio: {n_H / (n_H + n_T)}")
```

```python
# PyTorch implementation
n_H = 8675309
n_T = 256245

theta = torch.tensor(0.5, requires_grad=True)
lr = 1e-9

for iter in range(100):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()

print(f"Estimated θ: {theta}")
print(f"True ratio: {n_H / (n_H + n_T)}")
```

```python
# TensorFlow implementation
n_H = 8675309
n_T = 256245

theta = tf.Variable(tf.constant(0.5))
lr = 1e-9

for iter in range(100):
    with tf.GradientTape() as t:
        loss = -(n_H * tf.math.log(theta) + n_T * tf.math.log(1 - theta))
    theta.assign_sub(lr * t.gradient(loss, theta))

print(f"Estimated θ: {theta}")
print(f"True ratio: {n_H / (n_H + n_T)}")
```

All implementations converge to the correct value, demonstrating the robustness of this approach.

## Step 3: Mathematical Advantages of Log-Likelihood

### Step 3.1: Computational Efficiency

For independent data points, the likelihood is a product:

$$
P(X\mid\boldsymbol{\theta}) = \prod_{i=1}^n p(x_i\mid\boldsymbol{\theta})
$$

The derivative requires O(n²) operations due to the product rule. The log-likelihood transforms this to a sum:

$$
\log(P(X\mid\boldsymbol{\theta})) = \sum_{i=1}^n \log(p(x_i\mid\boldsymbol{\theta}))
$$

Whose derivative requires only O(n) operations.

### Step 3.2: Connection to Information Theory

The average negative log-likelihood relates to cross-entropy and information theory:

$$
H(p) = -\sum_{i} p_i \log_2(p_i)
$$

This provides a theoretical foundation for using negative log-likelihood as a loss function.

## Step 4: Continuous Variables

MLE extends naturally to continuous variables by replacing probabilities with probability densities. For a dataset ${x_i}_{i=1}^N$, we maximize:

$$
-\sum_{i=1}^N \log(p(x_i\mid\boldsymbol{\theta}))
$$

Where $p(x_i\mid\boldsymbol{\theta})$ is the probability density function.

### Step 4.1: Why This Works

Even though the probability of any exact value is zero, we consider ε-intervals around each observation. The negative log-probability becomes:

$$
-N\log(\epsilon) - \sum_{i=1}^N \log(p(x_i\mid\boldsymbol{\theta}))
$$

The first term doesn't depend on θ, so maximizing the likelihood is equivalent to minimizing the negative log-density sum.

## Summary

You've learned how to:
1. **Understand the maximum likelihood principle** for parameter estimation
2. **Apply MLE to discrete problems** like coin flipping with analytical solutions
3. **Implement numerical optimization** using negative log-likelihood for scalability
4. **Appreciate the mathematical advantages** of log-likelihood for computation and theory
5. **Extend MLE to continuous variables** using probability densities

## Exercises

1. **Exponential distribution**: Given a random variable with density $\alpha e^{-\alpha x}$ for $\alpha>0$, and a single observation $x=3$, find the MLE for $\alpha$.

2. **Gaussian mean estimation**: Given samples ${x_i}_{i=1}^N$ from a Gaussian with unknown mean μ and variance 1, find the MLE for μ.

## Next Steps

Maximum likelihood estimation forms the foundation for many machine learning algorithms. In practice, you'll encounter it in:
- Linear and logistic regression
- Neural network training (cross-entropy loss)
- Gaussian mixture models
- And many other probabilistic models

The key insight is that by maximizing the (log-)likelihood, we're finding the parameters that make our observed data most probable under our model.