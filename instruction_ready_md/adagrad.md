# Adagrad Optimization Algorithm

This guide provides a practical introduction to the Adagrad optimization algorithm. You will learn the intuition behind Adagrad, implement it from scratch, and apply it using high-level frameworks.

## Prerequisites

Ensure you have the necessary libraries installed. This guide uses the `d2l` library for plotting and data utilities, along with your preferred deep learning framework.

```bash
pip install d2l
```

## 1. Understanding the Problem: Sparse Features and Learning Rates

In many machine learning problems, especially in natural language processing and recommendation systems, features can be sparse. This means some features appear frequently while others appear rarely.

A standard stochastic gradient descent (SGD) approach uses a global learning rate that decays over time (e.g., `η = η₀ / √(t + c)`). This can be problematic for sparse data:

*   **Frequent features:** Their parameters converge quickly.
*   **Infrequent features:** Their parameters receive updates too rarely before the learning rate becomes too small.

Adagrad addresses this by adapting the learning rate **per parameter**, based on the historical sum of squared gradients for that parameter. Parameters with large past gradients (typically frequent features) get a smaller learning rate, while parameters with small past gradients get a relatively larger one.

## 2. Core Algorithm

The Adagrad update rule maintains a state variable **s** for each parameter, which accumulates the squares of its past gradients.

For each time step *t*:

1.  Compute the gradient **gₜ** with respect to the parameters.
2.  Accumulate the squared gradient into the state: **sₜ = sₜ₋₁ + gₜ²** (element-wise).
3.  Update the parameters: **wₜ = wₜ₋₁ - (η / √(sₜ + ε)) · gₜ**.

Here, **η** is the initial learning rate, and **ε** (e.g., 1e-6) is a small constant to prevent division by zero.

## 3. Visualizing Adagrad on a Convex Problem

Let's first see Adagrad in action on a simple quadratic convex function to understand its behavior.

### 3.1. Setup and Helper Function

We'll define our objective function `f(x1, x2) = 0.1 * x₁² + 2 * x₂²` and the Adagrad update logic for a 2D case.

```python
import math

def adagrad_2d(x1, x2, s1, s2, eta=0.4):
    """Adagrad update for a 2D parameter vector."""
    eps = 1e-6
    # Gradients of our objective function
    g1, g2 = 0.2 * x1, 4 * x2
    # Accumulate squared gradients
    s1 += g1 ** 2
    s2 += g2 ** 2
    # Update parameters with per-coordinate learning rates
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    """Objective function."""
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
```

### 3.2. Experiment with Different Learning Rates

Now, let's trace the optimization path starting from the point (-5, -2). We'll use the `d2l` library for visualization.

First, with a learning rate of **η = 0.4**:

```python
# Assuming d2l and framework imports are done
eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

You will observe that the trajectory is smooth. However, because the state variable **s** grows monotonically, the effective learning rate `η / √(s)` decays aggressively. The movement of the variables becomes very small in later iterations.

Let's try a larger learning rate, **η = 2**:

```python
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

With `η=2`, the behavior is much better. The initial larger step helps, and the per-coordinate adaptation still manages the ill-conditioned curvature (the `x2` direction has a much larger gradient). This shows that the base learning rate in Adagrad needs careful tuning, as the built-in decay can be very strong.

## 4. Implementing Adagrad from Scratch

Let's implement a general-purpose Adagrad optimizer that can work with a model of any dimension.

### 4.1. Framework-Agnostic State Initialization

We need to initialize state variables (one for each parameter tensor) to accumulate squared gradients.

```python
# For demonstration, we show a PyTorch-like structure.
# The logic is similar for MXNet/TensorFlow with framework-specific variable types.

def init_adagrad_states(feature_dim):
    """Initialize state variables for Adagrad."""
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

### 4.2. The Adagrad Update Function

This function performs one step of Adagrad optimization.

```python
def adagrad(params, states, hyperparams):
    """Update parameters using the Adagrad algorithm."""
    eps = 1e-6
    for p, s in zip(params, states):
        # Accumulate squared gradients
        s[:] += p.grad ** 2
        # Update parameter with per-element learning rate
        p[:] -= hyperparams['lr'] * p.grad / (s + eps).sqrt()
        # In a training loop, remember to zero the gradients for the next step
        # p.grad.data.zero_() # Typically done outside this function
```

### 4.3. Training a Model

We can now use our custom Adagrad to train a simple linear regression model on a synthetic dataset.

```python
# Get a data iterator and the feature dimension
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)

# Train the model using our custom Adagrad implementation
d2l.train_ch11(adagrad,
               init_adagrad_states(feature_dim),
               {'lr': 0.1},
               data_iter,
               feature_dim)
```

The training function will output the loss over epochs. You should see the loss decreasing, demonstrating that our Adagrad implementation works.

## 5. Using Built-in Adagrad Optimizers

For practical work, you should use the optimized Adagrad implementation provided by your deep learning framework. It's more efficient and battle-tested.

### 5.1. PyTorch

```python
import torch

trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

### 5.2. TensorFlow / Keras

```python
import tensorflow as tf

trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate': 0.1}, data_iter)
```

### 5.3. MXNet / Gluon

```python
# Using the d2l wrapper for Gluon's Trainer API
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

## 6. Summary and Key Takeaways

*   **Per-Parameter Adaptation:** Adagrad adapts the learning rate for each parameter individually, which is crucial for problems with sparse features or uneven curvature.
*   **Automatic Learning Rate Decay:** It uses the sum of squared historical gradients as a proxy for curvature, automatically reducing the learning rate for parameters with large past updates.
*   **Advantages:** Excellent for sparse data problems (e.g., NLP, recommender systems) and convex optimization.
*   **Limitation:** The aggressive, monotonic learning rate decay can be detrimental in non-convex settings like deep neural networks, potentially causing learning to stop too early. Variants like RMSProp and Adam were developed to address this.
*   **Implementation:** While understanding the from-scratch implementation is valuable, always prefer the built-in optimizer in production for performance and stability.

## 7. Exercises for Further Practice

1.  **Rotation Invariance:** Prove that for an orthogonal matrix **U**, the Euclidean norm is preserved: `||c - δ||₂ = ||Uc - Uδ||₂`. This shows that the *scale* of perturbations doesn't change under rotation, but an optimizer's *path* might.
2.  **Test on a Rotated Function:** Apply your Adagrad implementation to the function `f(x) = 0.1*(x₁ + x₂)² + 2*(x₁ - x₂)²` (a 45-degree rotation of our original function). Observe if the convergence behavior differs from the axis-aligned case.
3.  **Apply to a Real Network:** Use the built-in Adagrad optimizer to train a convolutional neural network (like LeNet from a previous chapter) on the Fashion-MNIST dataset. Monitor how the training loss progresses compared to SGD.
4.  **Think About Decay:** How might you modify the core Adagrad algorithm to have a less aggressive decay of the learning rate over time? (Hint: Consider introducing a forgetting factor or a moving average, as seen in RMSProp).