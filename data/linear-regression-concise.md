# Concise Implementation of Linear Regression with High-Level APIs

In this guide, you will implement linear regression using the high-level APIs of modern deep learning frameworks. This approach automates much of the repetitive work involved in defining models, loss functions, and optimizers, allowing you to focus on the core architecture.

## Prerequisites

First, ensure you have the necessary libraries installed. The code below imports the required modules for each supported framework.

```python
# Framework-specific imports
# For MXNet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

# For PyTorch
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn

# For TensorFlow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf

# For JAX
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## Step 1: Define the Model

When building models with high-level APIs, you can use predefined layers, which simplifies the process significantly. For linear regression, you need a single fully connected (dense) layer that outputs one scalar value.

The framework handles input shape inference automatically when data is first passed through the model.

```python
class LinearRegression(d2l.Module):
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        # MXNet: Use Dense layer with normal initialization
        if d2l.FRAMEWORK == 'mxnet':
            self.net = nn.Dense(1)
            self.net.initialize(init.Normal(sigma=0.01))
        # TensorFlow: Use Dense layer with random normal initializer
        elif d2l.FRAMEWORK == 'tensorflow':
            initializer = tf.initializers.RandomNormal(stddev=0.01)
            self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        # PyTorch: Use LazyLinear layer (infers input size) with normal initialization
        elif d2l.FRAMEWORK == 'pytorch':
            self.net = nn.LazyLinear(1)
            self.net.weight.data.normal_(0, 0.01)
            self.net.bias.data.fill_(0)
        # JAX: Use Dense layer with normal kernel initialization
        elif d2l.FRAMEWORK == 'jax':
            self.net = nn.Dense(1, kernel_init=nn.initializers.normal(0.01))
```

Next, define the forward pass. This simply calls the layer to compute the output.

```python
@d2l.add_to_class(LinearRegression)
def forward(self, X):
    return self.net(X)
```

## Step 2: Define the Loss Function

Instead of manually implementing mean squared error, use the built-in loss functions provided by each framework for efficiency and reliability.

```python
@d2l.add_to_class(LinearRegression)
def loss(self, y_hat, y):
    if d2l.FRAMEWORK == 'mxnet':
        fn = gluon.loss.L2Loss()
        return fn(y_hat, y).mean()
    elif d2l.FRAMEWORK == 'pytorch':
        fn = nn.MSELoss()
        return fn(y_hat, y)
    elif d2l.FRAMEWORK == 'tensorflow':
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)
    elif d2l.FRAMEWORK == 'jax':
        # Note: JAX loss uses params and state; shown here for completeness
        # The actual call differs slightly in the training loop
        return d2l.reduce_mean(optax.l2_loss(y_hat, y))
```

## Step 3: Define the Optimization Algorithm

Minibatch Stochastic Gradient Descent (SGD) is the standard optimizer for this task. Each framework provides an SGD implementation that you can configure with the model parameters and learning rate.

```python
@d2l.add_to_class(LinearRegression)
def configure_optimizers(self):
    if d2l.FRAMEWORK == 'mxnet':
        return gluon.Trainer(self.collect_params(),
                             'sgd', {'learning_rate': self.lr})
    elif d2l.FRAMEWORK == 'pytorch':
        return torch.optim.SGD(self.parameters(), self.lr)
    elif d2l.FRAMEWORK == 'tensorflow':
        return tf.keras.optimizers.SGD(self.lr)
    elif d2l.FRAMEWORK == 'jax':
        return optax.sgd(self.lr)
```

## Step 4: Training the Model

With the model, loss, and optimizer defined, you can now train the model using synthetic data. The training loop is encapsulated in the `fit` method from the `d2l` library.

```python
# Instantiate the model with a learning rate of 0.03
model = LinearRegression(lr=0.03)

# Generate synthetic regression data with true weights w=[2, -3.4] and bias b=4.2
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)

# Create a trainer and fit the model for 3 epochs
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

## Step 5: Evaluating the Learned Parameters

After training, compare the learned parameters to the true values used to generate the data. This verifies that the model has converged correctly.

First, define a helper method to extract the weights and bias from the model.

```python
@d2l.add_to_class(LinearRegression)
def get_w_b(self):
    if d2l.FRAMEWORK == 'mxnet':
        return (self.net.weight.data(), self.net.bias.data())
    elif d2l.FRAMEWORK == 'pytorch':
        return (self.net.weight.data, self.net.bias.data)
    elif d2l.FRAMEWORK == 'tensorflow':
        return (self.get_weights()[0], self.get_weights()[1])
    elif d2l.FRAMEWORK == 'jax':
        # For JAX, parameters are accessed via the trainer state
        net = trainer.state.params['net']
        return net['kernel'], net['bias']
```

Now, retrieve the parameters and compute the error.

```python
w, b = model.get_w_b()
print(f'error in estimating w: {data.w - d2l.reshape(w, data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```

**Expected Output:**
The errors should be small, indicating that the learned parameters are close to the true values `w=[2, -3.4]` and `b=4.2`.

## Summary

In this tutorial, you implemented linear regression using high-level framework APIs, which significantly reduced the amount of boilerplate code. You learned to:

1. **Define a model** using predefined layers (e.g., `Dense` or `Linear`).
2. **Use built-in loss functions** (e.g., `L2Loss` or `MSELoss`).
3. **Configure an optimizer** (e.g., SGD) with a few lines of code.
4. **Train the model** using a streamlined training loop.

Leveraging these high-level components allows you to build and experiment with models quickly, while still benefiting from optimized and well-tested implementations.

## Exercises

To deepen your understanding, try the following:

1. **Learning Rate Adjustment**: If you change the loss aggregation from a sum to an average over the minibatch, how should you adjust the learning rate?
2. **Alternative Loss Functions**: Replace the squared loss with Huber’s robust loss function. Consult your framework’s documentation for available loss functions.
3. **Accessing Gradients**: Learn how to access the gradients of the model weights after a backward pass.
4. **Hyperparameter Effects**: Experiment with different learning rates and numbers of training epochs. Observe how these changes affect the solution quality.
5. **Data Quantity Analysis**:
   - Systematically increase the amount of training data (e.g., 5, 10, 20, 50, …, 10,000 samples) and plot the estimation error for both weights and bias.
   - Why is a logarithmic increase in data size more informative than a linear one for this analysis?

By completing these exercises, you will gain practical insights into model training and the impact of various hyperparameters and data characteristics.