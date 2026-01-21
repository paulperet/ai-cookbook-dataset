# Concise Implementation of Softmax Regression

This tutorial guides you through implementing softmax regression using high-level deep learning frameworks. You will build a model to classify Fashion-MNIST images, focusing on a clean, production-ready implementation that handles numerical stability automatically.

## Prerequisites

First, ensure you have the necessary libraries installed. The code is designed to work with multiple frameworks. Choose your preferred one by setting the `tab` selector.

```python
# Load the framework selector
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

Now, import the required modules for your chosen framework.

```python
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```python
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```python
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```python
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from functools import partial
import jax
from jax import numpy as jnp
import optax
```

## Step 1: Define the Model

You will create a `SoftmaxRegression` class that inherits from a base `Classifier`. This model consists of a flattening operation (to convert 2D images into 1D vectors) followed by a fully connected (dense) layer that outputs scores (logits) for each class.

The implementation varies slightly by framework to accommodate their APIs, but the core logic remains the same.

### PyTorch Implementation

```python
%%tab pytorch
class SoftmaxRegression(d2l.Classifier):
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)
```

### MXNet and TensorFlow Implementation

```python
%%tab mxnet, tensorflow
class SoftmaxRegression(d2l.Classifier):
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(num_outputs)
            self.net.initialize()
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            self.net.add(tf.keras.layers.Flatten())
            self.net.add(tf.keras.layers.Dense(num_outputs))

    def forward(self, X):
        return self.net(X)
```

### JAX Implementation

```python
%%tab jax
class SoftmaxRegression(d2l.Classifier):
    num_outputs: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_outputs)(X)
        return X
```

**Key Points:**
- The `Flatten` layer (or reshape operation in JAX) converts the input from a 4D tensor `(batch_size, height, width, channels)` to a 2D tensor `(batch_size, height*width*channels)`.
- The `Dense` (or `Linear`) layer produces raw scores (logits) for each of the `num_outputs` classes.
- The base `Classifier` class provides training infrastructure.

## Step 2: Understand the Loss Function (Numerical Stability)

In softmax regression, you compute probabilities from logits using the softmax function:  
$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$.

Direct computation can lead to numerical issues:
- **Overflow:** If $o_k$ is very large, $\exp(o_k)$ might exceed the maximum representable number.
- **Underflow:** If $o_k$ is very negative, $\exp(o_k)$ might round to zero.

A stable solution combines the softmax and cross-entropy loss into a single operation. This avoids computing intermediate probabilities explicitly and uses the **LogSumExp trick** for stability. The loss is computed directly from logits.

### Implementing the Loss

The loss function is added to the base `Classifier` class. It reshapes the inputs and uses the framework's built-in, numerically stable cross-entropy function.

```python
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(d2l.Classifier)
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    if tab.selected('mxnet'):
        fn = gluon.loss.SoftmaxCrossEntropyLoss()
        l = fn(Y_hat, Y)
        return l.mean() if averaged else l
    if tab.selected('pytorch'):
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return fn(Y, Y_hat)
```

```python
%%tab jax
@d2l.add_to_class(d2l.Classifier)
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    Y_hat = state.apply_fn({'params': params}, *X,
                           mutable=False, rngs=None)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    return (fn(Y_hat, Y).mean(), {}) if averaged else (fn(Y_hat, Y), {})
```

**Why this works:** The built-in loss functions compute $\log \hat{y}_j = o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o})$ internally, where $\bar{o} = \max_k o_k$. This formulation is numerically stable.

## Step 3: Train the Model

Now, you will train the model on the Fashion-MNIST dataset. The data is automatically flattened into 784-dimensional vectors.

```python
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

**Expected Output:**  
The training process will display loss and accuracy metrics for each epoch. The model should achieve reasonably high accuracy on the validation set, demonstrating that softmax regression is effective for this multiclass classification task.

## Summary

In this tutorial, you implemented softmax regression concisely using high-level framework APIs. Key takeaways:

1. **Model Definition:** You created a simple neural network with a flattening layer and a dense output layer.
2. **Numerical Stability:** You used built-in loss functions that combine softmax and cross-entropy to avoid numerical underflow/overflow.
3. **Training:** You trained the model on Fashion-MNIST, achieving good performance with minimal code.

High-level APIs abstract away complex details, making deep learning accessible. However, understanding the underlying principles (like numerical stability) remains crucial for debugging and extending models.

## Exercises

Test your understanding by trying these challenges:

1. **Numerical Formats:** Deep learning uses various number formats (FP64, FP32, BFLOAT16, etc.). Compute the smallest and largest argument for `exp()` that avoids underflow/overflow in single precision (FP32).
2. **INT8 Dynamic Range:** The INT8 format represents integers from 1 to 255. How could you extend its dynamic range without using more bits? Do standard multiplication and addition still work?
3. **Overfitting:** Increase the number of training epochs. Why might validation accuracy eventually decrease? How could you fix this?
4. **Learning Rate Tuning:** Experiment with different learning rates (e.g., 0.01, 0.1, 1.0). Plot the loss curves. Which rate works best? Why?

---
*For framework-specific discussions, refer to the links in the original notebook.*