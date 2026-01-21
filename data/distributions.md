# A Practical Guide to Common Probability Distributions

## Introduction

Probability distributions form the foundation of statistical modeling and machine learning. In this guide, you'll explore several fundamental distributions, learn their properties, and implement them in code. We'll cover both discrete and continuous distributions, showing how to work with them programmatically.

## Prerequisites

First, let's set up our environment by importing the necessary libraries. We'll use a multi-framework approach to demonstrate implementation across different deep learning platforms.

```python
# For MXNet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from math import erf, factorial
import numpy as np

# For PyTorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from math import erf, factorial
import torch
torch.pi = torch.acos(torch.zeros(1)) * 2  # Define pi in torch

# For TensorFlow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from math import erf, factorial
import tensorflow as tf
import tensorflow_probability as tfp
tf.pi = tf.acos(tf.zeros(1)) * 2  # Define pi in TensorFlow
```

## 1. Bernoulli Distribution

The Bernoulli distribution models a binary outcome—like a coin flip—where success (1) occurs with probability `p` and failure (0) occurs with probability `1-p`.

### Mathematical Definition
If $X$ follows a Bernoulli distribution, we write:
$$X \sim \textrm{Bernoulli}(p)$$

The cumulative distribution function (CDF) is:
$$F(x) = \begin{cases} 0 & x < 0, \\ 1-p & 0 \le x < 1, \\ 1 & x \ge 1 \end{cases}$$

### Implementation

Let's visualize the probability mass function (PMF):

```python
p = 0.3
d2l.set_figsize()
d2l.plt.stem([0, 1], [1 - p, p], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Now, let's plot the cumulative distribution function:

```python
# For MXNet
x = np.arange(-1, 2, 0.01)
def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p
d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')

# For PyTorch
x = torch.arange(-1, 2, 0.01)
def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p
d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')

# For TensorFlow
x = tf.range(-1, 2, 0.01)
def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p
d2l.plot(x, tf.constant([F(y) for y in x]), 'x', 'c.d.f.')
```

### Key Properties
- Mean: $\mu_X = p$
- Variance: $\sigma_X^2 = p(1-p)$

### Sampling
You can sample from a Bernoulli distribution as follows:

```python
# MXNet
1*(np.random.rand(10, 10) < p)

# PyTorch
1*(torch.rand(10, 10) < p)

# TensorFlow
tf.cast(tf.random.uniform((10, 10)) < p, dtype=tf.float32)
```

## 2. Discrete Uniform Distribution

The discrete uniform distribution assigns equal probability to each value in a finite set. We'll assume it's supported on integers $\{1, 2, \ldots, n\}$.

### Mathematical Definition
We denote this as:
$$X \sim U(n)$$

The CDF is:
$$F(x) = \begin{cases} 0 & x < 1, \\ \frac{k}{n} & k \le x < k+1 \textrm{ with } 1 \le k < n, \\ 1 & x \ge n \end{cases}$$

### Implementation

First, let's visualize the PMF:

```python
n = 5
d2l.plt.stem([i+1 for i in range(n)], n*[1 / n], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Now, plot the CDF:

```python
# For MXNet
x = np.arange(-1, 6, 0.01)
def F(x):
    return 0 if x < 1 else 1 if x > n else np.floor(x) / n
d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')

# For PyTorch
x = torch.arange(-1, 6, 0.01)
def F(x):
    return 0 if x < 1 else 1 if x > n else torch.floor(x) / n
d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')

# For TensorFlow
x = tf.range(-1, 6, 0.01)
def F(x):
    return 0 if x < 1 else 1 if x > n else tf.floor(x) / n
d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

### Key Properties
- Mean: $\mu_X = \frac{1+n}{2}$
- Variance: $\sigma_X^2 = \frac{n^2-1}{12}$

### Sampling

```python
# MXNet
np.random.randint(1, n, size=(10, 10))

# PyTorch
torch.randint(1, n, size=(10, 10))

# TensorFlow
tf.random.uniform((10, 10), 1, n, dtype=tf.int32)
```

## 3. Continuous Uniform Distribution

The continuous uniform distribution models selecting any value within an interval $[a, b]$ with equal probability.

### Mathematical Definition
We denote this as:
$$X \sim U(a, b)$$

The probability density function (PDF) is:
$$p(x) = \begin{cases} \frac{1}{b-a} & x \in [a, b], \\ 0 & x \not\in [a, b] \end{cases}$$

The CDF is:
$$F(x) = \begin{cases} 0 & x < a, \\ \frac{x-a}{b-a} & x \in [a, b], \\ 1 & x \ge b \end{cases}$$

### Implementation

First, plot the PDF:

```python
a, b = 1, 3

# For MXNet
x = np.arange(0, 4, 0.01)
p = (x > a)*(x < b)/(b - a)
d2l.plot(x, p, 'x', 'p.d.f.')

# For PyTorch
x = torch.arange(0, 4, 0.01)
p = (x > a).type(torch.float32)*(x < b).type(torch.float32)/(b-a)
d2l.plot(x, p, 'x', 'p.d.f.')

# For TensorFlow
x = tf.range(0, 4, 0.01)
p = tf.cast(x > a, tf.float32) * tf.cast(x < b, tf.float32) / (b - a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

Now, plot the CDF:

```python
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

# For MXNet
d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')

# For PyTorch
d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')

# For TensorFlow
d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

### Key Properties
- Mean: $\mu_X = \frac{a+b}{2}$
- Variance: $\sigma_X^2 = \frac{(b-a)^2}{12}$

### Sampling

```python
# MXNet
(b - a) * np.random.rand(10, 10) + a

# PyTorch
(b - a) * torch.rand(10, 10) + a

# TensorFlow
(b - a) * tf.random.uniform((10, 10)) + a
```

## 4. Binomial Distribution

The binomial distribution models the number of successes in `n` independent Bernoulli trials, each with success probability `p`.

### Mathematical Definition
We denote this as:
$$X \sim \textrm{Binomial}(n, p)$$

The CDF is:
$$F(x) = \begin{cases} 0 & x < 0, \\ \sum_{m \le k} \binom{n}{m} p^m(1-p)^{n-m} & k \le x < k+1 \textrm{ with } 0 \le k < n, \\ 1 & x \ge n \end{cases}$$

### Implementation

First, let's compute and visualize the PMF:

```python
n, p = 10, 0.2

# Helper function for binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

# For MXNet
pmf = np.array([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])
d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()

# For PyTorch
pmf = d2l.tensor([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])
d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()

# For TensorFlow
pmf = tf.constant([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])
d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Now, plot the CDF:

```python
# For MXNet
x = np.arange(-1, 11, 0.01)
cmf = np.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]
d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')

# For PyTorch
x = torch.arange(-1, 11, 0.01)
cmf = torch.cumsum(pmf, dim=0)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]
d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')

# For TensorFlow
x = tf.range(-1, 11, 0.01)
cmf = tf.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]
d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

### Key Properties
- Mean: $\mu_X = np$
- Variance: $\sigma_X^2 = np(1-p)$

### Sampling

```python
# MXNet
np.random.binomial(n, p, size=(10, 10))

# PyTorch
m = torch.distributions.binomial.Binomial(n, p)
m.sample(sample_shape=(10, 10))

# TensorFlow
m = tfp.distributions.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

## 5. Poisson Distribution

The Poisson distribution models the number of rare events occurring in a fixed interval of time or space.

### Mathematical Definition
We denote this as:
$$X \sim \textrm{Poisson}(\lambda)$$

The PMF is:
$$p_k = \frac{\lambda^ke^{-\lambda}}{k!}$$

The CDF is:
$$F(x) = \begin{cases} 0 & x < 0, \\ e^{-\lambda}\sum_{m = 0}^k \frac{\lambda^m}{m!} & k \le x < k+1 \textrm{ with } 0 \le k \end{cases}$$

### Implementation

First, visualize the PMF:

```python
lam = 5.0
xs = [i for i in range(20)]

# For MXNet
pmf = np.array([np.exp(-lam) * lam**k / factorial(k) for k in xs])
d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()

# For PyTorch
pmf = torch.tensor([torch.exp(torch.tensor(-lam)) * lam**k / factorial(k) for k in xs])
d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()

# For TensorFlow
pmf = tf.constant([tf.exp(tf.constant(-lam)).numpy() * lam**k / factorial(k) for k in xs])
d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Now, plot the CDF:

```python
# For MXNet
x = np.arange(-1, 21, 0.01)
cmf = np.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]
d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')

# For PyTorch
x = torch.arange(-1, 21, 0.01)
cmf = torch.cumsum(pmf, dim=0)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]
d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')

# For TensorFlow
x = tf.range(-1, 21, 0.01)
cmf = tf.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]
d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

### Key Properties
- Mean: $\mu_X = \lambda$
- Variance: $\sigma_X^2 = \lambda$

### Sampling

```python
# MXNet
np.random.poisson(lam, size=(10, 10))

# PyTorch
m = torch.distributions.poisson.Poisson(lam)
m.sample((10, 10))

# TensorFlow
m = tfp.distributions.Poisson(lam)
m.sample((10, 10))
```

## 6. Gaussian (Normal) Distribution

The Gaussian distribution is fundamental in probability and statistics, often arising from sums of many independent random variables (Central Limit Theorem).

### Mathematical Definition
We denote this as:
$$X \sim \mathcal{N}(\mu, \sigma^2)$$

The PDF is:
$$p_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

### Implementation

First, plot the PDF:

```python
mu, sigma = 0, 1

# For MXNet
x = np.arange(-3, 3, 0.01)
p = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))
d2l.plot(x, p, 'x', 'p.d.f.')

# For PyTorch
x = torch.arange(-3, 3, 0.01)
p = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(-(x - mu)**2 / (2 * sigma**2))
d2l.plot(x, p, 'x', 'p.d.f.')

# For TensorFlow
x = tf.range(-3, 3, 0.01)
p = 1 / tf.sqrt(2 * tf.pi * sigma**2) * tf.exp(-(x - mu)**2 / (2 * sigma**2))
d2l.plot(x, p, 'x', 'p.d.f.')
```

The Gaussian CDF doesn't have a closed-form solution, but we can compute it numerically using the error function:

```python
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2.0

# For MXNet
d2l.plot(x, np.array([phi(y) for y in x.tolist()]), 'x', 'c.d