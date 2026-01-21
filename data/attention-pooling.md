# Attention Pooling with Nadaraya-Watson Kernel Regression

This guide introduces a classical machine learning technique, Nadaraya-Watson kernel regression, which is a direct precursor to modern attention mechanisms. You will implement several similarity kernels and use them to perform non-parametric regression, visualizing how each kernel shapes the attention weights.

## Prerequisites

Ensure you have the necessary libraries installed. This tutorial supports multiple frameworks. Choose your preferred one by setting the `tab` selector.

```python
# For PyTorch
!pip install torch torchvision numpy matplotlib

# For TensorFlow
!pip install tensorflow numpy matplotlib

# For JAX (with Flax)
!pip install jax jaxlib flax numpy matplotlib

# For MXNet
!pip install mxnet numpy matplotlib
```

## Step 1: Import Libraries and Set Up the Environment

First, import the required libraries and configure the display for plots.

```python
import numpy as np
import matplotlib.pyplot as plt

# Framework-specific imports and setup
# Select your framework by uncommenting the appropriate block

# --- PyTorch ---
# import torch
# from torch import nn
# import torch.nn.functional as F
# d2l = ... # Assume d2l library is available

# --- TensorFlow ---
# import tensorflow as tf
# d2l = ...

# --- JAX ---
# import jax
# import jax.numpy as jnp
# from flax import linen as nn
# d2l = ...

# --- MXNet ---
# from mxnet import autograd, gluon, np, npx
# from mxnet.gluon import nn
# npx.set_np()
# d2l = ...

# Use SVG for better figure quality
d2l.use_svg_display()
```

## Step 2: Define the Kernel Functions

Kernels measure similarity between a query `q` and a key `k`. We'll implement four common kernels: Gaussian, Boxcar, Constant, and Epanechikov. These are translation and rotation invariant.

```python
def gaussian(x):
    """Gaussian kernel: exp(-x^2 / 2)"""
    return d2l.exp(-x**2 / 2)

def boxcar(x):
    """Boxcar kernel: 1 if |x| < 1, else 0"""
    return d2l.abs(x) < 1.0

def constant(x):
    """Constant kernel: always returns 1"""
    return 1.0 + 0 * x

def epanechikov(x):
    """Epanechikov kernel: max(0, 1 - |x|)"""
    # Framework-specific implementation
    if framework == 'pytorch':
        return torch.max(1 - d2l.abs(x), torch.zeros_like(x))
    elif framework == 'mxnet':
        return np.maximum(1 - d2l.abs(x), 0)
    elif framework == 'tensorflow':
        return tf.maximum(1 - d2l.abs(x), 0)
    elif framework == 'jax':
        return jnp.maximum(1 - d2l.abs(x), 0)
```

## Step 3: Visualize the Kernels

Let's plot each kernel to understand their shape and smoothness.

```python
kernels = [gaussian, boxcar, constant, epanechikov]
names = ['Gaussian', 'Boxcar', 'Constant', 'Epanechikov']

x = d2l.arange(-2.5, 2.5, 0.1)
fig, axes = plt.subplots(1, 4, sharey=True, figsize=(12, 3))

for kernel, name, ax in zip(kernels, names, axes):
    ax.plot(x, kernel(x))
    ax.set_xlabel(name)
    ax.grid(True)

plt.tight_layout()
plt.show()
```

You'll observe:
- **Gaussian**: Smooth, exponentially decaying.
- **Boxcar**: Sharp cutoff at |x| = 1.
- **Constant**: Flat, gives equal weight to all points.
- **Epanechikov**: Linear decay to zero at |x| = 1.

## Step 4: Generate Synthetic Training Data

Create a simple regression problem with added noise.

```python
def f(x):
    """True underlying function: 2*sin(x) + x"""
    return 2 * d2l.sin(x) + x

# Generate 40 training examples
n = 40
if framework == 'pytorch':
    x_train, _ = torch.sort(d2l.rand(n) * 5)
    y_train = f(x_train) + d2l.randn(n)
elif framework == 'mxnet':
    x_train = np.sort(d2l.rand(n) * 5, axis=None)
    y_train = f(x_train) + d2l.randn(n)
elif framework == 'tensorflow':
    x_train = tf.sort(d2l.rand((n, 1)) * 5, 0)
    y_train = f(x_train) + d2l.normal((n, 1))
elif framework == 'jax':
    x_train = jnp.sort(jax.random.uniform(d2l.get_key(), (n,)) * 5)
    y_train = f(x_train) + jax.random.normal(d2l.get_key(), (n,))

# Validation points for plotting
x_val = d2l.arange(0, 5, 0.1)
y_val = f(x_val)  # True function values
```

## Step 5: Implement Nadaraya-Watson Regression

The core function computes the kernel regression estimate. It calculates attention weights by normalizing kernel similarities between training and validation points.

```python
def nadaraya_watson(x_train, y_train, x_val, kernel):
    """
    Compute Nadaraya-Watson kernel regression estimates.
    
    Args:
        x_train: Training features (keys)
        y_train: Training labels (values)
        x_val: Validation features (queries)
        kernel: Similarity kernel function
    
    Returns:
        y_hat: Predictions at x_val
        attention_w: Attention weight matrix
    """
    # Compute pairwise distances: shape (len(x_train), len(x_val))
    dists = d2l.reshape(x_train, (-1, 1)) - d2l.reshape(x_val, (1, -1))
    
    # Apply kernel to distances
    k = d2l.astype(kernel(dists), d2l.float32)
    
    # Normalize: attention weights sum to 1 for each query
    attention_w = k / d2l.reduce_sum(k, 0)
    
    # Weighted sum of training labels
    if framework in ['pytorch', 'jax']:
        y_hat = y_train @ attention_w
    elif framework == 'mxnet':
        y_hat = np.dot(y_train, attention_w)
    elif framework == 'tensorflow':
        y_hat = d2l.transpose(d2l.transpose(y_train) @ attention_w)
    
    return y_hat, attention_w
```

## Step 6: Evaluate Different Kernels

Now, apply each kernel to our data and visualize the results.

```python
def plot_predictions(x_train, y_train, x_val, y_val, kernels, names):
    """Plot regression estimates for multiple kernels."""
    fig, axes = plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, _ = nadaraya_watson(x_train, y_train, x_val, kernel)
        
        ax.plot(x_val, y_hat, label='Prediction')
        ax.plot(x_val, y_val, 'm--', label='True function')
        ax.plot(x_train, y_train, 'o', alpha=0.5, label='Training data')
        
        ax.set_xlabel(name)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# Use first three kernels (skip constant for better visualization)
plot_predictions(x_train, y_train, x_val, y_val, 
                 kernels[:3], names[:3])
```

Observations:
- **Gaussian**: Produces smooth, reasonable estimates.
- **Boxcar**: Piecewise constant estimates due to hard cutoff.
- **Constant**: Simply predicts the global mean (not useful).

## Step 7: Inspect the Attention Weights

The attention weight matrix shows how each training point influences each validation point.

```python
def plot_attention_weights(x_train, y_train, x_val, y_val, kernels, names):
    """Visualize attention weight matrices as heatmaps."""
    fig, axes = plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    
    for kernel, name, ax in zip(kernels, names, axes):
        _, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        
        # Convert to numpy for plotting if needed
        if framework != 'jax':
            attention_w = d2l.numpy(attention_w)
        
        im = ax.imshow(attention_w, cmap='Reds', aspect='auto')
        ax.set_xlabel(name)
        ax.set_ylabel('Training index')
    
    fig.colorbar(im, ax=axes, shrink=0.7)
    plt.tight_layout()
    plt.show()

plot_attention_weights(x_train, y_train, x_val, y_val, kernels, names)
```

The heatmaps reveal:
- **Gaussian/Epanechikov**: Smooth, localized attention around the diagonal.
- **Boxcar**: Sharp, block-like attention patterns.
- **Constant**: Uniform attention across all training points.

## Step 8: Experiment with Kernel Width

The Gaussian kernel's width parameter `σ` controls its smoothness. Let's explore different values.

```python
def gaussian_with_width(sigma):
    """Create Gaussian kernel with specified width."""
    return lambda x: d2l.exp(-x**2 / (2 * sigma**2))

sigmas = (0.1, 0.2, 0.5, 1)
width_kernels = [gaussian_with_width(sigma) for sigma in sigmas]
width_names = [f'σ = {sigma}' for sigma in sigmas]

plot_predictions(x_train, y_train, x_val, y_val, 
                 width_kernels, width_names)
```

Narrower kernels (small σ) adapt to local variations but become noisy. Wider kernels (large σ) are smoother but may oversmooth important details.

## Step 9: Visualize Width-Dependent Attention

```python
plot_attention_weights(x_train, y_train, x_val, y_val, 
                       width_kernels, width_names)
```

Notice how narrower kernels produce more concentrated, diagonal attention patterns, while wider kernels spread attention more broadly.

## Summary

In this tutorial, you:

1. **Implemented four classic kernels** (Gaussian, Boxcar, Constant, Epanechikov) for similarity measurement.
2. **Applied Nadaraya-Watson regression**, a non-parametric method that uses these kernels as attention mechanisms.
3. **Visualized predictions and attention weights**, observing how kernel choice affects model behavior.
4. **Experimented with kernel width**, seeing the trade-off between localization and smoothness.

Key insights:
- Nadaraya-Watson regression is a direct precursor to modern attention mechanisms.
- The kernel function acts as an attention scorer, determining how much each training point influences each prediction.
- Hand-crafted kernels have limitations; learned attention mechanisms (as in transformers) typically perform better.

This foundation prepares you for understanding more sophisticated attention mechanisms where query and key representations are learned rather than hand-designed.

## Exercises

1. **Parzen Windows Connection**: Show that for binary classification, Parzen windows density estimates lead to the same decision boundary as Nadaraya-Watson classification.

2. **Learning Kernel Width**:
   - Implement gradient descent to optimize the Gaussian kernel width σ.
   - Why can't you directly minimize (f(x_i) - y_i)²? (Hint: y_i appears in f(x_i))
   - Try leave-one-out cross-validation by excluding (x_i, y_i) when computing f(x_i).

3. **Unit Sphere Simplification**: If all vectors lie on the unit sphere (‖x‖ = 1), show that ‖x - x_i‖² = 2 - 2x·x_i. This relates to dot-product attention.

4. **Consistency Rate**: As you get more data, how quickly should you reduce the kernel scale? Does the optimal rate depend on data dimensionality?