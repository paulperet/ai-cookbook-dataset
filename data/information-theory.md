# Information Theory: A Practical Guide

## Introduction

Information theory provides the mathematical foundation for understanding how information is measured, encoded, and transmitted. In machine learning, concepts from information theory appear everywhere—from loss functions like cross-entropy to evaluation metrics and model interpretability. This guide will walk you through the core concepts with practical implementations.

## Prerequisites

We'll implement all concepts from scratch using your preferred deep learning framework. First, let's set up our environment.

```python
# For MXNet users
from mxnet import np
from mxnet.metric import NegativeLogLikelihood
from mxnet.ndarray import nansum
import random

# For PyTorch users
import torch
from torch.nn import NLLLoss

def nansum(x):
    return x[~torch.isnan(x)].sum()

# For TensorFlow users
import tensorflow as tf

def log2(x):
    return tf.math.log(x) / tf.math.log(2.)

def nansum(x):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=-1)
```

## 1. Self-Information: Measuring Surprise

Self-information quantifies how "surprising" a particular event is. Rare events carry more information than common ones.

### Mathematical Definition
For an event $X$ with probability $p$, the self-information is:
$$I(X) = -\log_2(p)$$

### Implementation
Let's implement this in a framework-agnostic way:

```python
def self_information(p):
    """Calculate self-information in bits."""
    return -np.log2(p)  # MXNet
    # return -torch.log2(torch.tensor(p)).item()  # PyTorch
    # return -log2(tf.constant(p)).numpy()  # TensorFlow

# Example: Information in a 1/64 probability event
print(f"Self-information: {self_information(1/64):.2f} bits")
```

**Output:**
```
Self-information: 6.00 bits
```

**Interpretation:** An event with probability 1/64 provides 6 bits of information. This aligns with our intuition—you need 6 binary questions to identify one outcome among 64 equally likely possibilities.

## 2. Entropy: Average Information Content

While self-information measures individual events, entropy measures the average information content of a random variable.

### Mathematical Definition
For a discrete random variable $X$ with probability distribution $P$:
$$H(X) = -\sum_i p_i \log_2 p_i$$

### Implementation

```python
def entropy(p):
    """Calculate Shannon entropy in bits."""
    entropy_vals = -p * np.log2(p)
    return nansum(entropy_vals)  # Handle any NaN values

# Example: Calculate entropy of a distribution
distribution = np.array([0.1, 0.5, 0.1, 0.3])
print(f"Entropy: {entropy(distribution):.3f} bits")
```

**Output:**
```
Entropy: 1.685 bits
```

### Properties of Entropy
1. **Non-negativity:** $H(X) \geq 0$ for discrete variables
2. **Maximum entropy:** For $k$ outcomes, maximum entropy is $\log_2(k)$ (uniform distribution)
3. **Additivity:** For independent variables, $H(X,Y) = H(X) + H(Y)$

## 3. Joint and Conditional Entropy

### 3.1 Joint Entropy
Joint entropy measures the total information in a pair of random variables.

```python
def joint_entropy(p_xy):
    """Calculate joint entropy H(X,Y)."""
    joint_ent = -p_xy * np.log2(p_xy)
    return nansum(joint_ent)

# Example joint distribution
joint_dist = np.array([[0.1, 0.5], [0.1, 0.3]])
print(f"Joint entropy: {joint_entropy(joint_dist):.3f} bits")
```

### 3.2 Conditional Entropy
Conditional entropy measures the remaining uncertainty in $Y$ after knowing $X$.

```python
def conditional_entropy(p_xy, p_x):
    """Calculate conditional entropy H(Y|X)."""
    p_y_given_x = p_xy / p_x
    cond_ent = -p_xy * np.log2(p_y_given_x)
    return nansum(cond_ent)

# Example
p_xy = np.array([[0.1, 0.5], [0.2, 0.3]])
p_x = np.array([0.2, 0.8])  # Marginal distribution of X
print(f"Conditional entropy: {conditional_entropy(p_xy, p_x):.3f} bits")
```

**Key Relationship:** $H(Y|X) = H(X,Y) - H(X)$

## 4. Mutual Information: Shared Information

Mutual information quantifies how much information two variables share.

### Mathematical Definition
$$I(X,Y) = \sum_{x,y} p(x,y) \log_2 \frac{p(x,y)}{p(x)p(y)}$$

### Implementation

```python
def mutual_information(p_xy, p_x, p_y):
    """Calculate mutual information I(X,Y)."""
    p = p_xy / (p_x * p_y)
    mutual = p_xy * np.log2(p)
    return nansum(mutual)

# Example
p_xy = np.array([[0.1, 0.5], [0.1, 0.3]])
p_x = np.array([0.2, 0.8])
p_y = np.array([[0.75, 0.25]])
print(f"Mutual information: {mutual_information(p_xy, p_x, p_y):.3f} bits")
```

### Properties of Mutual Information
1. **Symmetry:** $I(X,Y) = I(Y,X)$
2. **Non-negativity:** $I(X,Y) \geq 0$
3. **Independence test:** $I(X,Y) = 0$ if and only if $X$ and $Y$ are independent
4. **Relationship to entropy:** $I(X,Y) = H(X) + H(Y) - H(X,Y)$

## 5. KL Divergence: Measuring Distribution Difference

Kullback-Leibler (KL) divergence measures how one probability distribution differs from another.

### Mathematical Definition
$$D_{\textrm{KL}}(P\|Q) = \sum_x p(x) \log_2 \frac{p(x)}{q(x)}$$

### Implementation

```python
def kl_divergence(p, q):
    """Calculate KL divergence D_KL(P||Q)."""
    kl = p * np.log2(p / q)
    return abs(nansum(kl))  # Take absolute value for consistency

# Example with normal distributions
tensor_len = 10000

# Generate distributions
p = np.random.normal(loc=0, scale=1, size=(tensor_len,))
q1 = np.random.normal(loc=-1, scale=1, size=(tensor_len,))
q2 = np.random.normal(loc=1, scale=1, size=(tensor_len,))

# Sort for stable calculation
p = np.array(sorted(p))
q1 = np.array(sorted(q1))
q2 = np.array(sorted(q2))

# Calculate divergences
kl_pq1 = kl_divergence(p, q1)
kl_pq2 = kl_divergence(p, q2)
print(f"D_KL(P||Q1): {kl_pq1:.3f}")
print(f"D_KL(P||Q2): {kl_pq2:.3f}")

# Note: KL divergence is not symmetric
kl_q2p = kl_divergence(q2, p)
print(f"D_KL(Q2||P): {kl_q2p:.3f} (not equal to D_KL(P||Q2))")
```

### Properties of KL Divergence
1. **Non-symmetric:** $D_{\textrm{KL}}(P\|Q) \neq D_{\textrm{KL}}(Q\|P)$
2. **Non-negative:** $D_{\textrm{KL}}(P\|Q) \geq 0$
3. **Zero divergence:** $D_{\textrm{KL}}(P\|Q) = 0$ if and only if $P = Q$

## 6. Cross-Entropy: From Theory to Practice

Cross-entropy is a fundamental loss function in machine learning that connects information theory with practical optimization.

### Mathematical Definition
$$\textrm{CE}(P,Q) = -\sum_x p(x) \log_2 q(x)$$

### Relationship to KL Divergence
$$\textrm{CE}(P,Q) = H(P) + D_{\textrm{KL}}(P\|Q)$$

### Implementation for Classification

```python
def cross_entropy(y_hat, y):
    """Calculate cross-entropy loss for classification."""
    ce = -np.log(y_hat[range(len(y_hat)), y])
    return ce.mean()

# Example: 3-class classification
labels = np.array([0, 2])  # True class indices
preds = np.array([[0.3, 0.6, 0.1],  # Prediction probabilities
                  [0.2, 0.3, 0.5]])

loss = cross_entropy(preds, labels)
print(f"Cross-entropy loss: {loss:.3f}")
```

### Why Cross-Entropy Works for Classification
Minimizing cross-entropy is equivalent to:
1. Maximizing the log-likelihood of the data
2. Minimizing the KL divergence between true and predicted distributions
3. Making predictions that match the true distribution as closely as possible

## Practical Applications

### 1. Model Evaluation
Cross-entropy loss naturally penalizes confident wrong predictions more than uncertain ones, making it ideal for classification tasks.

### 2. Feature Selection
Mutual information can identify which features share the most information with the target variable, helping with feature selection.

### 3. Regularization
KL divergence appears in variational autoencoders and Bayesian methods as a regularization term.

### 4. Text Processing
Pointwise mutual information helps identify collocations and resolve ambiguities in natural language processing.

## Summary

In this guide, we've covered the essential concepts of information theory and their implementations:

1. **Self-information** measures surprise in individual events
2. **Entropy** quantifies average uncertainty in distributions
3. **Joint and conditional entropy** describe multi-variable relationships
4. **Mutual information** captures shared information between variables
5. **KL divergence** measures differences between distributions
6. **Cross-entropy** provides a practical loss function for machine learning

These concepts form the theoretical foundation for many machine learning algorithms and provide valuable tools for understanding and improving models.

## Exercises

1. Verify that a fair coin flip has 1 bit of entropy.
2. Prove that KL divergence is always non-negative using Jensen's inequality.
3. Calculate the entropy of English text under different assumptions (random characters vs. words vs. language model).
4. Show that $I(X,Y) = H(X) - H(X|Y)$ by expressing both sides as expectations.
5. Derive the KL divergence between two Gaussian distributions.

By mastering these concepts, you'll gain deeper insights into how information flows through your machine learning models and how to optimize them effectively.