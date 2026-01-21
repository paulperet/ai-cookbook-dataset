# Implementing Softmax Regression from Scratch

In this tutorial, you will implement softmax regression from scratch. This fundamental classification algorithm serves as a building block for more complex neural networks. By building it yourself, you'll gain deeper insight into how classification models work under the hood.

## Prerequisites

First, let's set up our environment by importing the necessary libraries. The code supports multiple deep learning frameworks - choose the one you're most comfortable with.

```python
# Framework selection (choose one)
import d2l  # Deep Learning 2 Library

# MXNet version
from mxnet import autograd, np, npx, gluon
npx.set_np()

# PyTorch version  
import torch

# TensorFlow version
import tensorflow as tf

# JAX version
from flax import linen as nn
import jax
from jax import numpy as jnp
from functools import partial
```

## Step 1: Understanding the Softmax Operation

The softmax function converts raw scores (logits) into probabilities. Given an input matrix `X`, softmax ensures that:
1. Each output is non-negative
2. Each row sums to 1 (forming a valid probability distribution)

The mathematical definition is:

$$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}$$

Let's first examine how to sum along different dimensions of a tensor, which we'll need for the normalization step:

```python
# Create a sample tensor
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Sum along columns (axis 0)
column_sum = d2l.reduce_sum(X, 0, keepdims=True)

# Sum along rows (axis 1)  
row_sum = d2l.reduce_sum(X, 1, keepdims=True)

print("Column sum:", column_sum)
print("Row sum:", row_sum)
```

Now, implement the softmax function:

```python
def softmax(X):
    """Convert logits to probabilities using softmax."""
    # Step 1: Exponentiate each element
    X_exp = d2l.exp(X)
    
    # Step 2: Compute normalization constant for each row
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    
    # Step 3: Normalize (broadcasting applies division row-wise)
    return X_exp / partition
```

Let's test our implementation:

```python
# Create random input
if tab.selected('mxnet'):
    X = d2l.rand(2, 5)
else:
    X = d2l.rand((2, 5))

# Apply softmax
X_prob = softmax(X)

print("Probabilities:\n", X_prob)
print("Row sums:", d2l.reduce_sum(X_prob, 1))
```

**Note**: This implementation is for educational purposes. In practice, you should use framework-built softmax functions that include numerical stability protections.

## Step 2: Building the Model Class

Now we'll create a `SoftmaxRegressionScratch` class that extends a base classifier. Our model will:
- Flatten 28×28 pixel images into 784-dimensional vectors
- Apply a linear transformation (weights × inputs + bias)
- Convert outputs to probabilities using softmax

Here's the model initialization for different frameworks:

```python
# MXNet implementation
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = np.random.normal(0, sigma, (num_inputs, num_outputs))
        self.b = np.zeros(num_outputs)
        self.W.attach_grad()
        self.b.attach_grad()

    def collect_params(self):
        return [self.W, self.b]

# PyTorch implementation  
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]

# TensorFlow implementation
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = tf.random.normal((num_inputs, num_outputs), 0, sigma)
        self.b = tf.zeros(num_outputs)
        self.W = tf.Variable(self.W)
        self.b = tf.Variable(self.b)

# JAX implementation
class SoftmaxRegressionScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W = self.param('W', nn.initializers.normal(self.sigma),
                            (self.num_inputs, self.num_outputs))
        self.b = self.param('b', nn.initializers.zeros, self.num_outputs)
```

## Step 3: Implementing the Forward Pass

The forward pass flattens the input images and applies the linear transformation followed by softmax:

```python
@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    # Flatten each 28x28 image to a 784-dimensional vector
    X = d2l.reshape(X, (-1, self.W.shape[0]))
    
    # Linear transformation: XW + b
    logits = d2l.matmul(X, self.W) + self.b
    
    # Convert to probabilities
    return softmax(logits)
```

## Step 4: Implementing Cross-Entropy Loss

Cross-entropy loss measures how well our predicted probabilities match the true labels. For efficiency, we use indexing rather than Python loops.

First, let's understand how to select the predicted probabilities for the true labels:

```python
# Create sample predictions and labels
y = d2l.tensor([0, 2])  # True labels: class 0 and class 2
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])  # Predicted probabilities

# Select probabilities for true labels
if tab.selected('mxnet', 'pytorch', 'jax'):
    selected_probs = y_hat[[0, 1], y]
else:  # TensorFlow
    selected_probs = tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))

print("Selected probabilities:", selected_probs)
```

Now implement the cross-entropy loss function:

```python
def cross_entropy(y_hat, y):
    """Compute cross-entropy loss between predictions and true labels."""
    if tab.selected('mxnet', 'pytorch', 'jax'):
        # Select predicted probabilities for true labels
        selected = y_hat[list(range(len(y_hat))), y]
        # Compute negative log likelihood and average
        return -d2l.reduce_mean(d2l.log(selected))
    else:  # TensorFlow
        mask = tf.one_hot(y, depth=y_hat.shape[-1])
        selected = tf.boolean_mask(y_hat, mask)
        return -tf.reduce_mean(tf.math.log(selected))

# Test the loss function
loss_value = cross_entropy(y_hat, y)
print("Cross-entropy loss:", loss_value)
```

Add the loss method to our model class:

```python
@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)
```

**Note for JAX users**: The JAX implementation requires a slightly different signature for JIT compilation compatibility.

## Step 5: Training the Model

We'll train our model on the Fashion-MNIST dataset for 10 epochs. The training loop is reused from our linear regression implementation.

```python
# Load data
data = d2l.FashionMNIST(batch_size=256)

# Initialize model
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)

# Create trainer
trainer = d2l.Trainer(max_epochs=10)

# Train the model
trainer.fit(model, data)
```

During training, you'll see the validation loss and accuracy improve over epochs. These metrics are computed on what we treat as the validation set (the test split of Fashion-MNIST).

## Step 6: Making Predictions

After training, let's see how our model performs on new images:

```python
# Get a batch of validation data
X, y = next(iter(data.val_dataloader()))

# Make predictions
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = d2l.argmax(model(X), axis=1)
else:  # JAX
    preds = d2l.argmax(model.apply({'params': trainer.state.params}, X), axis=1)

print("Predictions shape:", preds.shape)
```

## Step 7: Analyzing Mistakes

Let's examine where our model makes errors by visualizing incorrectly classified images:

```python
# Identify incorrect predictions
wrong = d2l.astype(preds, y.dtype) != y

# Filter to only incorrect examples
X_wrong, y_wrong, preds_wrong = X[wrong], y[wrong], preds[wrong]

# Create labels showing both true and predicted classes
labels = []
for true_label, pred_label in zip(y_wrong, preds_wrong):
    true_text = data.text_labels(true_label)
    pred_text = data.text_labels(pred_label)
    labels.append(f"{true_text}\n{pred_text}")

# Visualize the mistakes
data.visualize([X_wrong, y_wrong], labels=labels)
```

## Summary

Congratulations! You've successfully implemented softmax regression from scratch. You've learned how to:

1. Convert logits to probabilities using the softmax function
2. Build a classification model with learnable weights and biases
3. Implement the cross-entropy loss function for training
4. Train the model on image data
5. Evaluate predictions and analyze errors

This implementation represents the state of the art in statistical modeling from the 1960s-1970s. In practice, you would use deep learning framework implementations for better efficiency and numerical stability, but understanding the underlying mechanics is invaluable.

## Exercises

Test your understanding with these challenges:

1. **Numerical Stability**: Test the `softmax` function with extreme inputs:
   - What happens with an input value of 100?
   - What happens when the largest input is smaller than -100?
   - Implement a fix by subtracting the maximum value before exponentiation.

2. **Alternative Loss Implementation**: Implement cross-entropy using the definition $\sum_i y_i \log \hat{y}_i$:
   - Why might this run more slowly than our indexing approach?
   - When would this formulation be necessary?

3. **Practical Considerations**: Is returning the most likely label always appropriate? Consider medical diagnosis scenarios - how might you adjust confidence thresholds?

4. **Vocabulary Size**: If using softmax for next-word prediction with a large vocabulary (e.g., 50,000 words), what computational challenges arise?

5. **Hyperparameter Exploration**:
   - Plot validation loss vs. learning rate
   - Experiment with different batch sizes - how small/large can you go before affecting performance?

Try implementing these exercises to deepen your understanding of softmax regression and its practical considerations.