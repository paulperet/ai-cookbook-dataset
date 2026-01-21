# Multilayer Perceptrons (MLPs)

In the previous chapter on softmax regression, we learned how to build a linear classifier for recognizing clothing items from images. We covered data handling, probability distributions, loss functions, and parameter optimization. Now, we'll expand our toolkit by introducing **Multilayer Perceptrons (MLPs)**, the foundational architecture of deep neural networks.

## 1. The Need for Nonlinearity

Linear models, like our softmax regression, make a strong assumption: the relationship between inputs and outputs is affine (linear transformation plus bias). This assumption fails for many real-world problems.

### 1.1 Limitations of Linearity
- **Monotonicity Assumption:** Linear models imply that increasing a feature always increases or always decreases the output. This doesn't hold for many scenarios (e.g., body temperature vs. health risk).
- **Pixel Context:** In image classification, the importance of a pixel depends on surrounding pixels—a relationship linear models can't capture.
- **Inversion Invariance:** Flipping an image preserves its category, but a linear model based purely on pixel brightness would fail.

While we could manually engineer features to handle some nonlinearities, deep learning automates this by **learning both feature representations and predictions simultaneously**.

## 2. Introducing Hidden Layers

We overcome linear limitations by adding **hidden layers** between input and output. Stacking multiple fully-connected layers creates a Multilayer Perceptron (MLP).

### 2.1 MLP Architecture
Consider an MLP with:
- **Input layer:** 4 features
- **Hidden layer:** 5 neurons with activation functions
- **Output layer:** 3 classes

```python
# Conceptual representation
Input (4) → Hidden (5) → Output (3)
```

Mathematically, for a single hidden layer:
- **Input:** $\mathbf{X} \in \mathbb{R}^{n \times d}$ (n examples, d features)
- **Hidden weights:** $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$, biases $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$
- **Output weights:** $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$, biases $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$

Without activation functions:
```python
H = X @ W1 + b1  # Linear transformation
O = H @ W2 + b2   # Another linear transformation
```

But this is still just a linear model! We can collapse it to a single layer: $\mathbf{O} = \mathbf{X}\mathbf{W} + \mathbf{b}$ where $\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$.

## 3. Activation Functions: Adding Nonlinearity

The key ingredient is a **nonlinear activation function** $\sigma$ applied to each hidden unit:

```python
H = σ(X @ W1 + b1)  # Nonlinear transformation!
O = H @ W2 + b2
```

Now the model cannot be collapsed into a single linear layer. Let's explore common activation functions.

### 3.1 ReLU (Rectified Linear Unit)
The most popular activation function:

```python
ReLU(x) = max(0, x)
```

**Properties:**
- Retains positive values, sets negatives to zero
- Simple and computationally efficient
- Mitigates vanishing gradient problem

**Implementation across frameworks:**

```python
# PyTorch
import torch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)

# TensorFlow
import tensorflow as tf
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)

# JAX
import jax.numpy as jnp
from jax import grad, vmap
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.relu(x)
```

**Derivative:**
- 0 for x < 0
- 1 for x > 0
- Undefined at x = 0 (but we use 0 in practice)

### 3.2 Sigmoid Function
Squashes inputs to range (0, 1):

```python
sigmoid(x) = 1 / (1 + exp(-x))
```

**Use cases:**
- Output layer for binary classification (probabilities)
- Historical importance in early neural networks
- Less common in hidden layers now due to optimization challenges

```python
# Implementation examples
y = torch.sigmoid(x)  # PyTorch
y = tf.nn.sigmoid(x)  # TensorFlow
y = jax.nn.sigmoid(x) # JAX
```

**Derivative:** `sigmoid(x) * (1 - sigmoid(x))`
- Maximum slope of 0.25 at x = 0
- Vanishes for large |x| (causes vanishing gradients)

### 3.3 Tanh (Hyperbolic Tangent)
Squashes inputs to range (-1, 1):

```python
tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
```

**Properties:**
- Point symmetric about origin
- Similar to sigmoid but centered at zero
- Sometimes preferred over sigmoid for hidden layers

```python
# Implementation examples
y = torch.tanh(x)  # PyTorch
y = tf.nn.tanh(x)  # TensorFlow
y = jax.nn.tanh(x) # JAX
```

**Derivative:** `1 - tanh²(x)`
- Maximum slope of 1 at x = 0
- Also suffers from vanishing gradients for large |x|

## 4. Universal Approximation

A key theoretical result: **MLPs with a single hidden layer and nonlinear activation can approximate any continuous function** given enough neurons. However:
- This doesn't mean single-layer networks are practical
- Deeper networks often learn more compact representations
- Learning the right weights remains challenging

## 5. Modern Activation Functions

While ReLU dominates, researchers continue developing alternatives:

- **Parametric ReLU (pReLU):** `max(0, x) + α * min(0, x)` (learnable α)
- **GELU:** `x * Φ(x)` where Φ is Gaussian CDF
- **Swish:** `x * sigmoid(βx)` (β may be learned)

## 6. Practical Implementation Setup

Before building MLPs, ensure you have the necessary imports:

```python
# For PyTorch users
%matplotlib inline
from d2l import torch as d2l
import torch

# For TensorFlow users
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

# For JAX users  
%matplotlib inline
from d2l import jax as d2l
import jax
import jax.numpy as jnp
from jax import grad, vmap
```

## 7. Summary

You've learned:
1. **Why linear models fail** for complex patterns
2. **How hidden layers** increase model capacity
3. **Why activation functions** are essential for nonlinearity
4. **Common activation functions:** ReLU, sigmoid, tanh
5. **Theoretical foundations:** Universal approximation theorem

With this foundation, you're ready to build and train multilayer neural networks. In practice, you'll use deep learning frameworks that handle the complexities of backpropagation through these nonlinear layers automatically.

## Exercises

1. Prove that adding linear layers (without activation) cannot increase network expressivity.
2. Compute the derivative of pReLU activation.
3. Compute the derivative of Swish activation: `x * sigmoid(βx)`.
4. Show that ReLU-based MLPs create continuous piecewise linear functions.
5. Explore relationships between sigmoid and tanh functions.
6. Consider challenges with batch-wise nonlinearities.
7. Find examples where sigmoid gradients vanish.

*Note: This guide focuses on conceptual understanding and cross-framework implementations. In practice, you'll use high-level APIs that handle most of these details automatically.*