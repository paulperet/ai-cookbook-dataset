# Building a Base Classification Model

In this guide, you will learn how to construct a foundational `Classifier` class that simplifies the implementation of classification models across different deep learning frameworks. This base class handles common tasks like calculating accuracy and managing the validation step, allowing you to focus on model architecture.

## Prerequisites

Ensure you have the necessary imports for your chosen framework. The code is designed to work with **MXNet**, **PyTorch**, **TensorFlow**, or **JAX**.

```python
# Framework-specific imports
# MXNet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()

# PyTorch
from d2l import torch as d2l
import torch

# TensorFlow
from d2l import tensorflow as d2l
import tensorflow as tf

# JAX
from d2l import jax as d2l
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## Step 1: Define the Base `Classifier` Class

The `Classifier` class extends a base `Module` and provides a structured way to handle validation. During the validation step, it computes and records both the loss and accuracy for a batch of data.

### For PyTorch, MXNet, and TensorFlow

```python
class Classifier(d2l.Module):
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
```

### For JAX

JAX requires a slightly different implementation because models (especially those with batch normalization) return auxiliary data alongside the loss. We also redefine the `training_step` to handle this.

```python
class Classifier(d2l.Module):
    """The base class of classification models."""
    def training_step(self, params, batch, state):
        # `value` is a tuple (loss, auxiliary_data)
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot("loss", l, train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        # Discard auxiliary data; we only need the loss for validation
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)
        self.plot('acc', self.accuracy(params, batch[:-1], batch[-1], state),
                  train=False)
```

## Step 2: Configure the Optimizer

By default, we use Stochastic Gradient Descent (SGD) as the optimizer. The method `configure_optimizers` sets this up.

```python
# MXNet
@d2l.add_to_class(d2l.Module)
def configure_optimizers(self):
    params = self.parameters()
    if isinstance(params, list):
        return d2l.SGD(params, self.lr)
    return gluon.Trainer(params, 'sgd', {'learning_rate': self.lr})

# PyTorch
@d2l.add_to_class(d2l.Module)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)

# TensorFlow
@d2l.add_to_class(d2l.Module)
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)

# JAX
@d2l.add_to_class(d2l.Module)
def configure_optimizers(self):
    return optax.sgd(self.lr)
```

## Step 3: Implement Accuracy Calculation

Accuracy is a critical metric for classification tasks. It measures the fraction of predictions where the predicted class matches the true label.

The method works as follows:
1.  Reshape the prediction matrix `Y_hat` so each row corresponds to a sample.
2.  Use `argmax` to get the predicted class (the index with the highest score).
3.  Compare predictions with ground truth labels.
4.  Convert the boolean comparison to floats (0 or 1) and compute the mean.

### For PyTorch, MXNet, and TensorFlow

```python
@d2l.add_to_class(Classifier)
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

### For JAX

In JAX, we need to pass the model parameters and state explicitly, and the function is JIT-compiled for performance.

```python
@d2l.add_to_class(Classifier)
@partial(jax.jit, static_argnums=(0, 5))
def accuracy(self, params, X, Y, state, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = state.apply_fn({'params': params,
                            'batch_stats': state.batch_stats},  # For BatchNorm
                           *X)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

## Step 4: (MXNet Specific) Parameter Collection

For MXNet, we need helper methods to collect model parameters from nested modules and NumPy arrays.

```python
@d2l.add_to_class(d2l.Module)
def get_scratch_params(self):
    params = []
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            params.append(a)
        if isinstance(a, d2l.Module):
            params.extend(a.get_scratch_params())
    return params

@d2l.add_to_class(d2l.Module)
def parameters(self):
    params = self.collect_params()
    return params if isinstance(params, gluon.parameter.ParameterDict) and len(
        params.keys()) else self.get_scratch_params()
```

## Summary

You have now built a reusable `Classifier` base class that:
*   Standardizes the validation step to track loss and accuracy.
*   Configures an SGD optimizer by default.
*   Provides a method to compute classification accuracy, a key performance metric.

This class abstracts away common boilerplate, making it easier to implement and experiment with new classification models. While models are trained by optimizing a specific loss function, having a convenient way to measure accuracy is essential for practical evaluation.

## Exercises

Test your understanding with these conceptual exercises:

1.  Let \( L_v \) be the true validation loss, \( L_v^q \) be the quick estimate computed by averaging batch losses, and \( l_v^b \) be the loss on the very last minibatch. Express \( L_v \) in terms of \( L_v^q \), \( l_v^b \), and the sample and minibatch sizes.
2.  Prove that the quick estimate \( L_v^q \) is an unbiased estimator of \( L_v \) (i.e., \( E[L_v] = E[L_v^q] \)). Discuss why you might still prefer to compute \( L_v \) directly in some scenarios.
3.  Given a multiclass loss function \( l(y, y') \) and a conditional probability distribution \( p(y \mid x) \), derive the decision rule for choosing the optimal prediction \( y' \) that minimizes the expected loss.