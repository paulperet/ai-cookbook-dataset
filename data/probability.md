# Probability and Statistics Fundamentals for Machine Learning

Machine learning fundamentally deals with uncertainty. Whether predicting a target from features, detecting anomalies, or making decisions in reinforcement learning, we need tools to quantify and reason about the unknown. This guide introduces the core concepts of probability and statistics that underpin modern machine learning.

## Setup and Prerequisites

First, let's import the necessary libraries. We'll use a framework-agnostic approach, but the core concepts apply universally.

```python
%matplotlib inline
import random
import numpy as np
import matplotlib.pyplot as plt
```

## 1. A Simple Example: Tossing Coins

Let's start with a classic example: coin tossing. This will help us understand fundamental concepts like probability, statistics, and convergence.

### 1.1 Defining Probability

For a fair coin, both outcomes (heads and tails) are equally likely. We say the probability of heads, denoted $P(\textrm{heads})$, is 0.5. Probabilities range from 0 (impossible) to 1 (certain).

### 1.2 Simulating Coin Tosses

We can simulate coin tosses using random number generation. Let's start with 100 tosses:

```python
num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])
```

For multiple draws from distributions with finite outcomes, we can use multinomial sampling. Here's how to simulate 100 tosses of a fair coin:

```python
fair_probs = [0.5, 0.5]
counts = np.random.multinomial(100, fair_probs)
print("Counts (heads, tails):", counts)
print("Frequencies:", counts / 100)
```

Each time you run this, you'll get slightly different results due to randomness. The frequencies approximate the underlying probabilities but aren't identical to them.

### 1.3 Law of Large Numbers

Let's see what happens with more tosses. The law of large numbers tells us that as we increase sample size, our estimates should converge to the true probabilities.

```python
# Simulate 10,000 tosses
counts = np.random.multinomial(10000, fair_probs).astype(np.float32)
print("Frequencies after 10,000 tosses:", counts / 10000)
```

### 1.4 Visualizing Convergence

Let's track how our probability estimates evolve as we collect more data:

```python
# Simulate 10,000 individual tosses
counts = np.random.multinomial(1, fair_probs, size=10000).astype(np.float32)

# Calculate cumulative estimates
cum_counts = counts.cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)

# Plot the convergence
plt.figure(figsize=(4.5, 3.5))
plt.plot(estimates[:, 0], label="P(coin=heads)")
plt.plot(estimates[:, 1], label="P(coin=tails)")
plt.axhline(y=0.5, color='black', linestyle='dashed')
plt.xlabel('Samples')
plt.ylabel('Estimated probability')
plt.legend()
plt.show()
```

The plot shows both estimates converging toward 0.5 as we collect more data, demonstrating the law of large numbers in action.

## 2. Formal Probability Theory

Now let's establish a more rigorous foundation.

### 2.1 Basic Definitions

- **Sample space ($\mathcal{S}$)**: The set of all possible outcomes
- **Event**: A subset of the sample space
- **Probability function**: Maps events to values between 0 and 1

The probability function $P$ must satisfy three axioms:
1. Non-negativity: $P(\mathcal{A}) \geq 0$ for all events $\mathcal{A}$
2. Normalization: $P(\mathcal{S}) = 1$
3. Additivity: For mutually exclusive events, $P(\bigcup_i \mathcal{A}_i) = \sum_i P(\mathcal{A}_i)$

### 2.2 Random Variables

Random variables map outcomes to values. They can be:
- **Discrete**: Taking countable values (like coin tosses)
- **Continuous**: Taking values in a continuum (like height)

For continuous random variables, we work with probability densities rather than probabilities for exact values.

## 3. Multiple Random Variables

Most machine learning problems involve multiple interacting random variables.

### 3.1 Joint and Conditional Probability

- **Joint probability**: $P(A=a, B=b)$ - probability that $A=a$ AND $B=b$
- **Marginal probability**: $P(A=a) = \sum_b P(A=a, B=b)$
- **Conditional probability**: $P(B=b \mid A=a) = \frac{P(A=a, B=b)}{P(A=a)}$

### 3.2 Bayes' Theorem

One of the most important results in probability:

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}$$

This allows us to "reverse" conditional probabilities. When we don't know $P(B)$, we can use:

$$P(A \mid B) \propto P(B \mid A) P(A)$$

and then normalize.

### 3.3 Independence

Two variables are **independent** if $P(A,B) = P(A)P(B)$. This implies that knowing $A$ doesn't change our beliefs about $B$.

**Conditional independence**: $A$ and $B$ are conditionally independent given $C$ if $P(A,B \mid C) = P(A \mid C)P(B \mid C)$.

## 4. Practical Example: Medical Diagnosis

Let's apply these concepts to a realistic scenario: HIV testing.

### 4.1 Single Test Analysis

Consider an HIV test with:
- 1% false positive rate: $P(D_1=1 \mid H=0) = 0.01$
- 0% false negative rate: $P(D_1=1 \mid H=1) = 1$
- Disease prevalence: $P(H=1) = 0.0015$

What's the probability of having HIV given a positive test?

```python
# Calculate using Bayes' theorem
P_H1 = 0.0015  # Prior: P(H=1)
P_D1_given_H1 = 1.0  # P(D1=1|H=1)
P_D1_given_H0 = 0.01  # P(D1=1|H=0)

# Marginal probability of positive test
P_D1 = P_D1_given_H1 * P_H1 + P_D1_given_H0 * (1 - P_H1)

# Posterior probability
P_H1_given_D1 = (P_D1_given_H1 * P_H1) / P_D1
print(f"Probability of HIV given positive test: {P_H1_given_D1:.4f} ({P_H1_given_D1*100:.2f}%)")
```

Despite the test's accuracy, the posterior probability is only about 13% due to the low prior probability.

### 4.2 Multiple Tests

Now suppose a second, less accurate test:
- False positive: 3% ($P(D_2=1 \mid H=0) = 0.03$)
- False negative: 2% ($P(D_2=1 \mid H=1) = 0.98$)

Assuming conditional independence between tests given health status:

```python
# Second test probabilities
P_D2_given_H1 = 0.98
P_D2_given_H0 = 0.03

# Joint probabilities given H
P_both_given_H0 = P_D1_given_H0 * P_D2_given_H0
P_both_given_H1 = P_D1_given_H1 * P_D2_given_H1

# Marginal probability of both tests positive
P_both = P_both_given_H1 * P_H1 + P_both_given_H0 * (1 - P_H1)

# Updated posterior
P_H1_given_both = (P_both_given_H1 * P_H1) / P_both
print(f"Probability of HIV given both tests positive: {P_H1_given_both:.4f} ({P_H1_given_both*100:.2f}%)")
```

The second test increases our confidence to about 83%, demonstrating how multiple pieces of evidence combine.

## 5. Expectations and Variances

Beyond probabilities, we often need summary statistics.

### 5.1 Expectation (Mean)

The expected value of a random variable $X$ is:

$$E[X] = \sum_x x P(X=x) \quad \text{(discrete)}$$
$$E[X] = \int x p(x) dx \quad \text{(continuous)}$$

For functions: $E[f(X)] = \sum_x f(x) P(X=x)$

### 5.2 Variance and Standard Deviation

Variance measures spread around the mean:

$$\textrm{Var}[X] = E[(X - E[X])^2] = E[X^2] - E[X]^2$$

Standard deviation is $\sigma = \sqrt{\textrm{Var}[X]}$.

### 5.3 Example: Investment Returns

Consider an investment with:
- 50% chance: total loss (return 0×)
- 40% chance: 2× return
- 10% chance: 10× return

```python
# Calculate expected return
returns = [0, 2, 10]
probs = [0.5, 0.4, 0.1]

expected_return = sum(r * p for r, p in zip(returns, probs))
print(f"Expected return: {expected_return:.1f}×")

# Calculate variance
variance = sum((r - expected_return)**2 * p for r, p in zip(returns, probs))
std_dev = np.sqrt(variance)
print(f"Variance: {variance:.2f}")
print(f"Standard deviation: {std_dev:.2f}")
```

### 5.4 Covariance and Correlation

For vector-valued random variables $\mathbf{x}$, the covariance matrix is:

$$\boldsymbol{\Sigma} = E[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top]$$

where $\boldsymbol{\mu} = E[\mathbf{x}]$.

## 6. Types of Uncertainty

In machine learning, we distinguish:

1. **Aleatoric uncertainty**: Inherent randomness in the data
2. **Epistemic uncertainty**: Uncertainty due to limited knowledge (reducible with more data)

For example, even if we know a coin is fair (epistemic certainty), the outcome of any single toss remains uncertain (aleatoric uncertainty).

## 7. Key Takeaways

1. **Probability** provides the theoretical foundation for reasoning under uncertainty
2. **Statistics** helps us infer properties from observed data
3. **Bayes' theorem** is crucial for updating beliefs with new evidence
4. **Expectations and variances** summarize important distribution properties
5. **Multiple data sources** combine through conditional probabilities to reduce uncertainty

These concepts form the bedrock of machine learning, from simple classifiers to deep neural networks. Understanding them is essential for designing, implementing, and interpreting ML systems.

## Next Steps

To deepen your understanding:
1. Experiment with the code examples, changing parameters to see their effects
2. Work through the exercises at the end of the original chapter
3. Explore how these concepts apply to specific ML algorithms like Naive Bayes classifiers
4. Study the central limit theorem and its implications for statistical inference

Remember: probability and statistics are not just mathematical abstractions—they're practical tools for making better decisions with data.