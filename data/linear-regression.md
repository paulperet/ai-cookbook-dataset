# Linear Regression: A Foundational Guide
:label:`sec_linear_regression`

Regression is a fundamental problem in machine learning where we aim to predict a continuous numerical value. Common applications include forecasting house prices, predicting patient hospital stays, and estimating product demand. This guide introduces **linear regression**, the simplest and most widely-used tool for regression tasks, dating back to the early 19th century.

We'll use a practical example: predicting house prices based on their area (square feet) and age (years). To build a model, we need a **training dataset**—a collection of examples, each containing features (area, age) and a **label** (the actual sale price).

## Prerequisites

First, ensure you have the necessary libraries installed. This guide supports multiple frameworks. Choose your preferred one by setting the `tab` environment.

```bash
# Installation commands would typically go here.
# Since this is a conceptual guide, we assume the environment is set up.
```

Import the required modules for your chosen framework.

```python
# Framework-agnostic placeholder for imports
# In practice, you would import d2l, math, numpy, and your deep learning framework (MXNet, PyTorch, TensorFlow, or JAX)
import math
import time
```

## 1. Understanding the Linear Model

The core assumption of linear regression is that the relationship between features \(\mathbf{x}\) and target \(y\) is approximately linear. This means the expected value of the target can be expressed as a weighted sum of the features, plus a constant bias.

For our house price example with features area and age, the model is:

\[
\textrm{price} = w_{\textrm{area}} \cdot \textrm{area} + w_{\textrm{age}} \cdot \textrm{age} + b.
\]
:label:`eq_price-area`

Here, \(w_{\textrm{area}}\) and \(w_{\textrm{age}}\) are **weights**, and \(b\) is the **bias**. The weights determine each feature's influence, while the bias represents the predicted price when all features are zero.

### 1.1 Vectorized Form

For datasets with many features, linear algebra notation is more efficient. With \(d\) features, we represent them as a vector \(\mathbf{x} \in \mathbb{R}^d\) and weights as \(\mathbf{w} \in \mathbb{R}^d\). The prediction \(\hat{y}\) becomes:

\[
\hat{y} = \mathbf{w}^\top \mathbf{x} + b.
\]
:label:`eq_linreg-y`

For a dataset of \(n\) examples stored in a **design matrix** \(\mathbf{X} \in \mathbb{R}^{n \times d}\), predictions are computed as:

\[
{\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b,
\]
:label:`eq_linreg-y-vec`

where broadcasting handles the bias addition.

## 2. Defining a Loss Function

To train our model, we need a measure of how well its predictions match the true labels. For regression, the **squared error loss** is standard. For a single example \(i\) with prediction \(\hat{y}^{(i)}\) and true label \(y^{(i)}\), the loss is:

\[
l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.
\]
:label:`eq_mse`

The constant \(\frac{1}{2}\) simplifies derivatives. The average loss over the entire training set is:

\[
L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.
\]

Our goal is to find parameters \(\mathbf{w}^*, b^*\) that minimize this total loss.

## 3. Solving Linear Regression

### 3.1 Analytic Solution

Linear regression has a closed-form solution. By appending a column of ones to \(\mathbf{X}\) to absorb the bias \(b\), the optimal weights are given by:

\[
\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y},
\]

provided \(\mathbf X^\top \mathbf X\) is invertible (i.e., features are linearly independent). While useful for analysis, most real-world problems lack such neat solutions, necessitating iterative optimization.

### 3.2 Minibatch Stochastic Gradient Descent (SGD)

In practice, we often use **minibatch SGD** to optimize models. This iterative algorithm updates parameters using small random subsets (minibatches) of the data, balancing efficiency and stability.

For each iteration \(t\):
1. Sample a minibatch \(\mathcal{B}_t\) of size \(|\mathcal{B}|\).
2. Compute the gradient of the average loss on the minibatch.
3. Update parameters by moving opposite the gradient, scaled by a **learning rate** \(\eta\):

\[
(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).
\]

For our squared loss and linear model, the updates expand to:

\[
\begin{aligned}
\mathbf{w} & \leftarrow \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)\\
b &\leftarrow b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
\]
:label:`eq_linreg_batch_update`

Parameters like minibatch size and learning rate are **hyperparameters**, tuned separately (e.g., via validation).

## 4. Implementing Efficient Computation

Training models efficiently requires vectorized operations, leveraging fast linear algebra libraries instead of slow Python loops.

### 4.1 Demonstrating Vectorization Speed

Let's compare adding two vectors element-by-element versus using a vectorized operation.

```python
# Initialize two large vectors
n = 10000
a = d2l.ones(n)  # Vector of 1s
b = d2l.ones(n)  # Vector of 1s
```

**Method 1: Slow Python for-loop**

```python
c = d2l.zeros(n)
t = time.time()
for i in range(n):
    c[i] = a[i] + b[i]  # Framework-specific assignment
print(f'Loop time: {time.time() - t:.5f} sec')
```

**Method 2: Fast vectorized addition**

```python
t = time.time()
d = a + b  # Vectorized operation
print(f'Vectorized time: {time.time() - t:.5f} sec')
```

The vectorized approach is orders of magnitude faster, highlighting the importance of using library-optimized operations.

## 5. Probabilistic Interpretation

Linear regression with squared loss has a deep connection to probability. Assume observations are generated by:

\[
y = \mathbf{w}^\top \mathbf{x} + b + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2),
\]

where noise \(\epsilon\) follows a Gaussian (normal) distribution. The normal distribution with mean \(\mu\) and variance \(\sigma^2\) is:

\[
p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).
\]

Let's visualize it for different parameters.

```python
def normal(x, mu, sigma):
    """Compute the normal distribution N(mu, sigma^2)."""
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 * (x - mu)**2 / sigma**2)

# Visualization
x = np.arange(-7, 7, 0.01)
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params],
         xlabel='x', ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

Under this noise model, **maximizing the likelihood** of the data is equivalent to **minimizing the squared error loss**. This provides a statistical justification for our chosen loss function.

## 6. Linear Regression as a Neural Network

Linear regression can be viewed as a **single-layer neural network** with \(d\) input neurons (one per feature) connected directly to a single output neuron. This perspective bridges classic regression to modern deep learning.

While biological neurons inspired early models, today's deep learning draws from diverse fields like mathematics, statistics, and computer science.

## 7. Summary

In this guide, we covered:
- The **linear model** for regression.
- **Squared error loss** to measure prediction quality.
- **Analytic and iterative optimization** (minibatch SGD) for training.
- The importance of **vectorization** for performance.
- A **probabilistic interpretation** linking squared loss to Gaussian noise.
- Representing linear regression as a **simple neural network**.

Linear regression introduces key components used throughout deep learning: parametric models, differentiable objectives, gradient-based optimization, and evaluation on unseen data.

## Exercises

1. **Optimal Constant Prediction**
    - Find \(b\) minimizing \(\sum_i (x_i - b)^2\).
    - Relate this to the normal distribution.
    - Solve for the optimal \(b\) if using absolute loss \(\sum_i |x_i-b|\).

2. **Affine Function Equivalence**
    - Prove that \(\mathbf{x}^\top \mathbf{w} + b\) is linear in \((\mathbf{x}, 1)\).

3. **Quadratic Features in a Network**
    - How would you model \(f(\mathbf{x}) = b + \sum_i w_i x_i + \sum_{j \leq i} w_{ij} x_{i} x_{j}\) in a neural network?

4. **Rank-Deficient Design Matrix**
    - What if \(\mathbf{X}^\top \mathbf{X}\) is not full rank?
    - How does adding Gaussian noise to \(\mathbf{X}\) affect this?
    - What happens during SGD in this case?

5. **Exponential Noise Model**
    - Assume noise \(\epsilon\) follows \(p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)\).
    - Write the negative log-likelihood.
    - Propose a minibatch SGD algorithm and discuss potential issues.

6. **Composing Linear Layers**
    - Why doesn't stacking two linear layers (without nonlinearity) create a more powerful model?

7. **Regression for Prices**
    - Why is Gaussian noise inappropriate for price data?
    - Why is predicting the logarithm of price better?
    - Discuss challenges with penny stocks.

8. **Count Data and Poisson Distribution**
    - Why is Gaussian noise unsuitable for count data (e.g., apples sold)?
    - The Poisson distribution \(p(k \mid \lambda) = \lambda^k e^{-\lambda}/k!\) models counts. Show \(\lambda\) is the expected count.
    - Design a loss function for Poisson-distributed data and for estimating \(\log \lambda\).

---
*For further discussion, refer to the [D2L.ai forum](https://discuss.d2l.ai).*