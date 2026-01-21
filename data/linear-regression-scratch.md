# Implementing Linear Regression from Scratch

In this guide, you will build a linear regression model entirely from scratch. This foundational exercise will help you understand the core components of a machine learning system: the model, loss function, optimizer, and training loop. By the end, you'll have a working implementation trained on synthetic data.

## Prerequisites and Setup

First, ensure you have the necessary libraries installed. This tutorial supports multiple frameworks. Choose your preferred one and run the corresponding import block.

```python
# For PyTorch
%matplotlib inline
from d2l import torch as d2l
import torch

# For MXNet
# %matplotlib inline
# from d2l import mxnet as d2l
# from mxnet import autograd, np, npx
# npx.set_np()

# For TensorFlow
# %matplotlib inline
# from d2l import tensorflow as d2l
# import tensorflow as tf

# For JAX
# %matplotlib inline
# from d2l import jax as d2l
# from flax import linen as nn
# import jax
# from jax import numpy as jnp
# import optax
```

## Step 1: Defining the Model

We start by defining the linear regression model. The model parameters are the weights (`w`) and bias (`b`). We initialize the weights randomly from a normal distribution and set the bias to zero.

```python
class LinearRegressionScratch(d2l.Module):
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if framework == 'pytorch':
            self.w = d2l.normal(0, sigma, (num_inputs, 1), requires_grad=True)
            self.b = d2l.zeros(1, requires_grad=True)
        elif framework == 'mxnet':
            self.w = d2l.normal(0, sigma, (num_inputs, 1))
            self.b = d2l.zeros(1)
            self.w.attach_grad()
            self.b.attach_grad()
        elif framework == 'tensorflow':
            w = tf.random.normal((num_inputs, 1), mean=0, stddev=0.01)
            b = tf.zeros(1)
            self.w = tf.Variable(w, trainable=True)
            self.b = tf.Variable(b, trainable=True)
        # JAX version uses a different, Flax-based parameter definition.
```

The forward pass computes the prediction `y_hat` using the linear equation `y_hat = X @ w + b`.

```python
@d2l.add_to_class(LinearRegressionScratch)
def forward(self, X):
    return d2l.matmul(X, self.w) + self.b
```

## Step 2: Defining the Loss Function

We use the mean squared error (MSE) loss. This measures the average squared difference between predictions and true values.

```python
@d2l.add_to_class(LinearRegressionScratch)
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return d2l.reduce_mean(l)
```

## Step 3: Defining the Optimization Algorithm

We implement a simple Stochastic Gradient Descent (SGD) optimizer. It updates parameters by moving them in the opposite direction of their gradient, scaled by the learning rate.

```python
class SGD(d2l.HyperParameters):
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
```

We then configure the model to use this optimizer.

```python
@d2l.add_to_class(LinearRegressionScratch)
def configure_optimizers(self):
    return SGD([self.w, self.b], self.lr)
```

## Step 4: The Training Loop

The core training logic iterates over the dataset for multiple epochs. In each iteration, it:
1.  Computes the loss on a minibatch.
2.  Calculates gradients via backpropagation.
3.  Updates the parameters using the optimizer.

The following method is added to the `Trainer` class to handle one epoch of training.

```python
@d2l.add_to_class(d2l.Trainer)
def fit_epoch(self):
    self.model.train()
    for batch in self.train_dataloader:
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.train_batch_idx += 1
    # Validation step omitted for brevity
```

## Step 5: Training on Synthetic Data

Let's generate a synthetic dataset with known parameters and train our model.

```python
# Instantiate the model, data, and trainer
model = LinearRegressionScratch(num_inputs=2, lr=0.03)
# Assume SyntheticRegressionData generates data with true weights w=[2, -3.4] and bias b=4.2
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)

# Train the model
trainer.fit(model, data)
```

## Step 6: Evaluating the Learned Parameters

After training, we can compare the learned parameters to the true underlying values used to generate the data.

```python
print(f'Error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
print(f'Error in estimating b: {data.b - model.b}')
```

The output should show small errors, indicating the model successfully approximated the true function.

## Summary

You have successfully implemented and trained a linear regression model from the ground up. You defined the model architecture, a loss function, an optimization algorithm, and a training loop. This hands-on process demystifies the core components that power more complex deep learning models.

## Exercises

1.  **Weight Initialization:** What happens if you initialize all weights to zero? Does training still work? What if you initialize them with a very large variance (e.g., 1000)?
2.  **Ohm's Law:** Imagine you are trying to model the relationship between voltage and current (Ohm's Law: V = I * R). Could you use automatic differentiation to learn the resistance parameter `R` from data?
3.  **Planck's Law:** Could you use a similar approach to determine an object's temperature by fitting its spectral energy density to Planck's Law?
4.  **Second Derivatives:** What challenges arise if you need to compute second derivatives (Hessians) of the loss? How might you address them?
5.  **Loss Function Reshaping:** Why is the `.reshape()` operation necessary inside the `loss` function?
6.  **Learning Rate Tuning:** Experiment with different learning rates. How does the rate affect how quickly the loss decreases? Can you achieve lower error by training for more epochs?
7.  **Batch Size Edge Case:** What happens in the final batch of an epoch if the total number of examples isn't perfectly divisible by the batch size?
8.  **Alternative Loss:** Implement the mean absolute error loss: `(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()`.
    *   How does it behave on regular data?
    *   What happens if you corrupt a single label (e.g., set `y[5] = 10000`)?
    *   Can you design a loss function that combines the stability of squared loss with the robustness of absolute loss to large errors?
9.  **Dataset Shuffling:** Why is it important to shuffle the training dataset? Can you construct an example where not shuffling would cause the optimization to fail?