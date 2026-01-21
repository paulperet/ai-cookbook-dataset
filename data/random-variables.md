# Continuous Random Variables: A Practical Guide

## Introduction

In our previous exploration of probability, we focused on discrete random variables—those that take on a finite set of values or integers. This guide will help you understand **continuous random variables**, which can take on any real value. The transition from discrete to continuous probability is analogous to moving from summing lists of numbers to integrating functions, requiring some new conceptual tools.

## 1. Understanding Probability Density Functions

### The Conceptual Leap
Imagine you're throwing darts at a board and want to know the probability of hitting exactly 2 cm from the center. If you measure with single-centimeter bins, you might find 20% of darts land in the "2 cm" bin. But this bin actually contains all darts between 1.5 cm and 2.5 cm—not the exact equality we wanted.

As we measure more precisely (1.9 cm, 2.0 cm, 2.1 cm), we might find only 3% of darts in the 2.0 cm bucket. But we've just pushed the problem further—now we're dealing with 2.00 cm versus 2.01 cm.

The key insight: each additional digit of accuracy decreases the matching probability by a factor of 10. For a tiny interval of length ε around point x, the probability is approximately:

$$
P(\text{distance in } \epsilon\text{-sized interval around } x) \approx \epsilon \cdot p(x)
$$

This function $p(x)$ is the **probability density function** (PDF), encoding the relative likelihood of the random variable being near different points.

### Visualizing a Probability Density Function
Let's examine what a PDF looks like in practice. We'll plot a mixture of two normal distributions:

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.arange(-5, 5, 0.01)

# Define a mixture density: 20% N(3,1) + 80% N(-1,1)
p = 0.2 * np.exp(-(x - 3)**2 / 2) / np.sqrt(2 * np.pi) + \
    0.8 * np.exp(-(x + 1)**2 / 2) / np.sqrt(2 * np.pi)

# Plot the density function
plt.figure(figsize=(10, 6))
plt.plot(x, p, 'b-', linewidth=2)
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Probability Density Function')
plt.grid(True)
plt.show()
```

The peaks at x = -1 and x = 3 indicate regions where we're more likely to find our random variable. The low portions show unlikely regions.

## 2. Properties of Probability Density Functions

### Formal Definition
For a continuous random variable X with density function $p(x)$, the probability that X falls in a tiny interval around x is:

$$
P(X \text{ is in an } \epsilon \text{-sized interval around } x) \approx \epsilon \cdot p(x)
$$

### Essential Properties
1. **Non-negativity**: $p(x) \ge 0$ for all x (probabilities are never negative)
2. **Normalization**: $\int_{-\infty}^{\infty} p(x) \; dx = 1$ (the total probability must be 1)

### Calculating Probabilities from Densities
The probability that X falls between a and b is given by:

$$
P(X \in (a, b]) = \int_{a}^{b} p(x) \; dx
$$

Let's verify this with a numerical example using our mixture density:

```python
# Calculate probability between -1 and 1
epsilon = 0.01
x = np.arange(-5, 5, 0.01)
p = 0.2 * np.exp(-(x - 3)**2 / 2) / np.sqrt(2 * np.pi) + \
    0.8 * np.exp(-(x + 1)**2 / 2) / np.sqrt(2 * np.pi)

# Find indices corresponding to interval [-1, 1]
indices = np.where((x >= -1) & (x <= 1))[0]

# Approximate the probability using numerical integration
probability = np.sum(epsilon * p[indices])
print(f"Approximate probability X ∈ [-1, 1]: {probability:.4f}")

# Visualize the region
plt.figure(figsize=(10, 6))
plt.plot(x, p, 'b-', linewidth=2)
plt.fill_between(x[indices], p[indices], alpha=0.3, color='blue')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Probability as Area Under the Curve')
plt.grid(True)
plt.show()
```

## 3. Cumulative Distribution Functions

### Why We Need CDFs
While PDFs tell us about density, their values aren't probabilities themselves (a density can be >1). The **cumulative distribution function** (CDF) gives actual probabilities:

$$
F(x) = \int_{-\infty}^{x} p(t) \; dt = P(X \le x)
$$

### Key Properties of CDFs
1. $F(x) \rightarrow 0$ as $x \rightarrow -\infty$
2. $F(x) \rightarrow 1$ as $x \rightarrow \infty$
3. $F(x)$ is non-decreasing
4. $F(x)$ is continuous for continuous random variables

The CDF provides a unified framework for both discrete and continuous variables, making it particularly valuable in practice.

## 4. Summary Statistics for Continuous Variables

### The Mean (Expected Value)
For a continuous random variable X with density $p(x)$, the mean is:

$$
\mu_X = E[X] = \int_{-\infty}^{\infty} x p(x) \; dx
$$

The mean represents the "center of mass" or average location of the random variable.

### The Variance
Variance measures how much the random variable fluctuates around its mean:

$$
\sigma_X^2 = \textrm{Var}(X) = E[(X - \mu_X)^2] = \int_{-\infty}^{\infty} (x - \mu_X)^2 p(x) \; dx
$$

An alternative computational formula is:

$$
\sigma_X^2 = E[X^2] - \mu_X^2
$$

### The Standard Deviation
Since variance is in squared units, we often use the standard deviation for interpretation:

$$
\sigma_X = \sqrt{\textrm{Var}(X)}
$$

The standard deviation is in the same units as the original variable and represents the typical range of variation.

### Example: Uniform Distribution
Consider a random variable uniformly distributed on [0, 1] with density:

$$
p(x) = \begin{cases}
1 & x \in [0, 1] \\
0 & \text{otherwise}
\end{cases}
$$

Let's compute its mean and variance:

```python
# For uniform distribution on [0, 1]
# Mean: ∫x·1 dx from 0 to 1 = [x²/2] from 0 to 1 = 1/2
mean_uniform = 0.5

# Variance: E[X²] - μ² = ∫x²·1 dx - (1/2)² = [x³/3] - 1/4 = 1/3 - 1/4 = 1/12
variance_uniform = 1/12
std_uniform = np.sqrt(variance_uniform)

print(f"Uniform [0,1] distribution:")
print(f"  Mean: {mean_uniform}")
print(f"  Variance: {variance_uniform:.4f}")
print(f"  Standard deviation: {std_uniform:.4f}")
```

## 5. Chebyshev's Inequality: Putting Standard Deviation to Work

### The Inequality
Chebyshev's inequality gives us a powerful way to interpret standard deviations:

$$
P\left(X \notin [\mu_X - \alpha\sigma_X, \mu_X + \alpha\sigma_X]\right) \le \frac{1}{\alpha^2}
$$

For example, with α = 10, at least 99% of samples fall within 10 standard deviations of the mean.

### Visualizing Chebyshev's Inequality
Let's examine a simple discrete example to understand how sharp this inequality is:

```python
def plot_chebyshev_example(p):
    """
    Plot a simple distribution and Chebyshev interval.
    Random variable takes values: a-2 with prob p, a with prob 1-2p, a+2 with prob p
    """
    a = 0  # Center at 0
    values = [a-2, a, a+2]
    probabilities = [p, 1-2*p, p]
    
    # Calculate statistics
    mean = a  # As derived earlier
    variance = 8 * p  # As derived earlier
    std = np.sqrt(variance)
    
    # Chebyshev interval with α=2
    alpha = 2
    lower = mean - alpha * std
    upper = mean + alpha * std
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.stem(values, probabilities, linefmt='b-', markerfmt='bo', basefmt=' ')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot Chebyshev interval
    plt.axhspan(-0.05, 0.05, xmin=(lower+4)/8, xmax=(upper+4)/8, 
                alpha=0.3, color='red', label=f'Chebyshev interval (α={alpha})')
    
    plt.xlim([-4, 4])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.title(f'Distribution with p={p:.3f}, σ={std:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate actual probability outside interval
    outside_prob = 0
    for val, prob in zip(values, probabilities):
        if val < lower or val > upper:
            outside_prob += prob
    
    print(f"Probability outside interval: {outside_prob:.3f}")
    print(f"Chebyshev bound: {1/alpha**2:.3f}")
    print(f"Inequality holds: {outside_prob <= 1/alpha**2}")

# Test with different p values
for p_val in [0.2, 0.125, 0.05]:
    plot_chebyshev_example(p_val)
```

Notice that at p = 1/8, the interval exactly touches the outer points, showing the inequality is sharp (cannot be improved without additional assumptions).

## 6. Working with Multiple Random Variables

### Joint Density Functions
When dealing with correlated variables (like pixel colors in an image or stock prices over time), we need joint distributions. For two continuous random variables X and Y, the joint density $p(x, y)$ satisfies:

$$
P(X \in [x, x+\epsilon] \text{ and } Y \in [y, y+\epsilon]) \approx \epsilon^2 p(x, y)
$$

Properties:
1. $p(x, y) \ge 0$
2. $\int_{\mathbb{R}^2} p(x, y) \; dx \; dy = 1$
3. $P((X, Y) \in D) = \int_D p(x, y) \; dx \; dy$

### Marginal Distributions
To find the distribution of X alone from a joint density, we integrate out Y:

$$
p_X(x) = \int_{-\infty}^{\infty} p_{X,Y}(x, y) \; dy
$$

This process is called **marginalization**.

### Covariance: Measuring Linear Relationships
Covariance quantifies how two variables change together:

For discrete variables:
$$
\textrm{Cov}(X, Y) = \sum_{i,j} (x_i - \mu_X)(y_j - \mu_Y) p_{ij}
$$

For continuous variables:
$$
\textrm{Cov}(X, Y) = \int_{\mathbb{R}^2} (x - \mu_X)(y - \mu_Y) p(x, y) \; dx \; dy
$$

A computational formula that's often easier:
$$
\textrm{Cov}(X, Y) = E[XY] - E[X]E[Y]
$$

### Correlation: A Unit-Free Measure
Since covariance depends on units, we often use the correlation coefficient:

$$
\rho(X, Y) = \frac{\textrm{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

Properties:
- $-1 \le \rho \le 1$
- $\rho = 1$: perfect positive linear relationship
- $\rho = -1$: perfect negative linear relationship
- $\rho = 0$: no linear relationship (but could have nonlinear relationship!)

### Visualizing Correlation
Let's generate and plot variables with different correlations:

```python
def generate_correlated_data(correlation, n=500):
    """Generate X and Y with specified correlation."""
    X = np.random.normal(0, 1, n)
    
    # Generate Y with specified correlation to X
    Y = correlation * X + np.sqrt(1 - correlation**2) * np.random.normal(0, 1, n)
    
    return X, Y

# Plot different correlations
correlations = [-0.9, 0.0, 0.7]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, corr in zip(axes, correlations):
    X, Y = generate_correlated_data(corr)
    
    # Calculate actual correlation
    actual_corr = np.corrcoef(X, Y)[0, 1]
    
    ax.scatter(X, Y, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Target ρ = {corr:.1f}\nActual ρ = {actual_corr:.2f}')
    ax.grid(True, alpha=0.3)
    
    # Add regression line
    if corr != 0:
        m, b = np.polyfit(X, Y, 1)
        ax.plot(X, m*X + b, 'r-', linewidth=2)

plt.tight_layout()
plt.show()
```

## 7. Important Relationships and Formulas

### Variance of a Sum
For any two random variables (not necessarily independent):

$$
\textrm{Var}(X + Y) = \textrm{Var}(X) + \textrm{Var}(Y) + 2\textrm{Cov}(X, Y)
$$

Only when X and Y are independent does this simplify to $\textrm{Var}(X) + \textrm{Var}(Y)$.

### Properties Recap
**Mean:**
- $\mu_{aX+b} = a\mu_X + b$
- $\mu_{X+Y} = \mu_X + \mu_Y$

**Variance:**
- $\textrm{Var}(aX+b) = a^2\textrm{Var}(X)$
- $\textrm{Var}(X) \ge 0$, with equality iff X is constant

**Covariance:**
- $\textrm{Cov}(X, X) = \textrm{Var}(X)$
- $\textrm{Cov}(aX+b, Y) = a\textrm{Cov}(X, Y)$
- If X and Y independent, $\textrm{Cov}(X, Y) = 0$

**Correlation:**
- $\rho(X, X) = 1$
- $\rho(aX+b, Y) = \rho(X, Y)$ for a > 0
- $\rho(X, Y) = 0$ for independent variables with non-zero variance

## 8. Cautionary Example: The Cauchy Distribution

Not all distributions have well-defined means and variances. Consider the Cauchy distribution:

$$
p(x) = \frac{1}{\pi(1 + x^2)}
$$

```python
# Plot Cauchy distribution
x = np.linspace(-10, 10, 1000)
cauchy_pdf = 1 / (np.pi * (1 + x**2))

plt.figure(figsize=(10, 6))
plt.plot(x, cauchy_pdf, 'b-', linewidth=2)
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Cauchy Distribution (Heavy-Tailed)')
plt.grid(True, alpha=0.3)
plt.show()

# Show that integrals diverge
print("Attempting to compute mean (this integral diverges):")
print("∫ x/(1+x²) dx from -∞ to ∞ = ∞")
print("\nAttempting to compute variance (this integral also diverges):")
print("∫ x²/(1+x²) dx from -∞ to ∞ = ∞")
```

The Cauchy distribution has "heavy tails" that cause both the mean and variance to be undefined. In practice, most distributions we work with in machine learning have finite means and variances, but it's important to know pathological cases exist.

## Summary

1. **Continuous random variables** can take any real value and require probability density functions (PDFs) rather than probability mass functions.

2. **PDFs** encode relative likelihood through density, with probabilities obtained by integration:
   - $p(x) \ge 0$ for all x
   - $\int_{-\infty}^{\infty} p(x) dx = 1$
   - $P(a < X \le b) = \int_a^b p(x) dx$

3. **Cumulative distribution functions (CDFs)** provide actual probabilities and unify discrete and continuous cases:
   - $F(x) = P(X \le x) = \int_{-\infty}^x p(t) dt$

4. **Key summary statistics**:
   - Mean (μ): Center of the distribution
   - Variance (σ²): Expected squared deviation from mean
   - Standard deviation (σ): Typical range of variation

5. **Chebyshev's inequality** makes standard deviation interpretable, guaranteeing most probability lies within k standard deviations of the mean.

6. **Multiple variables** require joint densities, with:
   - Marginal distributions obtained by integration
   - Covariance measuring linear relationship
   - Correlation providing unit-free measure of linear dependence

7. **Important relationships** like $\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y) + 2\textrm{Cov}(X, Y)$ extend our ability to work with correlated variables.

## Exercises

1. For a random variable with density $p(x) = 1/x^2$ for $x \ge 1$ (0 otherwise), compute $P(X > 2)$.

2. For the Laplace distribution $p(x) = \frac{1}{2}e^{-|x|}$, find the mean and standard deviation.

3. If someone claims their random variable has