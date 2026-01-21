# Implementing Multilayer Perceptrons (MLPs)

This guide walks you through implementing a Multilayer Perceptron (MLP) from scratch and then using a high-level framework. We'll build a classifier for the Fashion-MNIST dataset, which has 10 classes and 28x28 grayscale images.

## Prerequisites

First, ensure you have the necessary libraries installed. The code below imports the required modules for your chosen deep learning framework.

```python
# Install d2l if needed: !pip install d2l
# Framework-specific imports are handled in the code blocks below.
```

## 1. Implementing an MLP from Scratch

We'll start by building an MLP with one hidden layer manually, which helps solidify your understanding of the underlying mechanics.

### 1.1 Initializing Model Parameters

Our MLP will have 784 input features (flattened 28x28 image), one hidden layer with 256 units, and 10 output classes (one per clothing category). We need to initialize weights and biases for both layers.

**Why 256 hidden units?** It's a common choice that's computationally efficient (a power of 2) and provides sufficient capacity for this task. You can adjust this as a hyperparameter.

Below is the model initialization for each framework. Notice how each framework handles parameter registration and gradient tracking differently.

```python
# MXNet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = np.random.randn(num_inputs, num_hiddens) * sigma
        self.b1 = np.zeros(num_hiddens)
        self.W2 = np.random.randn(num_hiddens, num_outputs) * sigma
        self.b2 = np.zeros(num_outputs)
        for param in self.get_scratch_params():
            param.attach_grad()
```

```python
# PyTorch
from d2l import torch as d2l
import torch
from torch import nn

class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
```

```python
# TensorFlow
from d2l import tensorflow as d2l
import tensorflow as tf

class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = tf.Variable(tf.random.normal((num_inputs, num_hiddens)) * sigma)
        self.b1 = tf.Variable(tf.zeros(num_hiddens))
        self.W2 = tf.Variable(tf.random.normal((num_hiddens, num_outputs)) * sigma)
        self.b2 = tf.Variable(tf.zeros(num_outputs))
```

```python
# JAX
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp

class MLPScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    num_hiddens: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W1 = self.param('W1', nn.initializers.normal(self.sigma),
                             (self.num_inputs, self.num_hiddens))
        self.b1 = self.param('b1', nn.initializers.zeros, self.num_hiddens)
        self.W2 = self.param('W2', nn.initializers.normal(self.sigma),
                             (self.num_hiddens, self.num_outputs))
        self.b2 = self.param('b2', nn.initializers.zeros, self.num_outputs)
```

### 1.2 Defining the Activation Function

We'll use the ReLU (Rectified Linear Unit) activation function for the hidden layer. ReLU introduces non-linearity, allowing the network to learn complex patterns. Let's implement it manually.

```python
# MXNet
def relu(X):
    return np.maximum(X, 0)
```

```python
# PyTorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```python
# TensorFlow
def relu(X):
    return tf.math.maximum(X, 0)
```

```python
# JAX
def relu(X):
    return jnp.maximum(X, 0)
```

### 1.3 Implementing the Forward Pass

Now, define the forward propagation logic. We'll flatten the input image, compute the hidden layer output (with ReLU), then compute the final logits.

```python
# All frameworks
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))  # Flatten to (batch_size, 784)
    H = relu(d2l.matmul(X, self.W1) + self.b1) # Hidden layer with ReLU
    return d2l.matmul(H, self.W2) + self.b2    # Output layer (logits)
```

### 1.4 Training the Model

The training loop is identical to the one used for softmax regression, thanks to the modular design of our `Classifier` base class. We'll instantiate the model, load the Fashion-MNIST data, and train for 10 epochs.

```python
# All frameworks
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

You should see the training loss decrease and accuracy improve over epochs.

## 2. Concise Implementation Using High-Level APIs

Building models from scratch is educational but cumbersome for production. Now, let's implement the same MLP using each framework's high-level API, which reduces boilerplate and minimizes errors.

### 2.1 Defining the Model with `Sequential`

We'll define a two-layer network using framework-specific `Sequential` containers. This abstracts away manual parameter management and forward pass definition.

```python
# MXNet
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens, activation='relu'),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```python
# PyTorch
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))
```

```python
# TensorFlow
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, activation='relu'),
            tf.keras.layers.Dense(num_outputs)])
```

```python
# JAX
class MLP(d2l.Classifier):
    num_outputs: int
    num_hiddens: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_hiddens)(X)
        X = nn.relu(X)
        X = nn.Dense(self.num_outputs)(X)
        return X
```

**Note:** The `MLP` class inherits a `forward` method that simply calls `self.net(X)`. The `Sequential` container (or Flax's `@nn.compact`) handles the layer-wise transformations automatically.

### 2.2 Training the Concise Model

Training is exactly the same as before—demonstrating the benefit of modular design.

```python
# All frameworks
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
```

## Summary

You've successfully implemented a Multilayer Perceptron both from scratch and using high-level APIs. Key takeaways:

- **From Scratch:** You manually initialized parameters, implemented ReLU, and defined the forward pass. This deepens understanding but is error-prone and hard to scale.
- **High-Level API:** Using `Sequential` containers, you built the same model with fewer lines, improving readability and maintainability.
- **Modular Training:** The training loop remained unchanged, highlighting how well-designed abstractions separate model architecture from training logic.

While fully connected networks like this MLP were state-of-the-art in the 1980s, modern deep learning often uses convolutional layers for image data—a topic we'll explore next.

## Exercises

Test your understanding by experimenting with the model:

1. **Hidden Units:** Vary `num_hiddens` (e.g., 64, 128, 512). Plot accuracy vs. hidden units. What's the optimal value?
2. **Additional Hidden Layer:** Add a second hidden layer. How does it affect accuracy and training time?
3. **Single-Neuron Hidden Layer:** Try a hidden layer with just one neuron. Why does performance collapse?
4. **Learning Rate:** Adjust the learning rate (e.g., 0.01, 0.1, 0.5). Which gives the best results? How does it interact with the number of epochs?
5. **Hyperparameter Tuning:** Jointly optimize learning rate, epochs, number of hidden layers, and hidden units per layer.
   - What's the best accuracy you can achieve?
   - Why is joint optimization challenging?
   - Propose a strategy (e.g., grid search, random search) for efficient multi-parameter tuning.
6. **Speed Comparison:** Compare training speed between the from-scratch and high-level API implementations. How does the gap change with network complexity?
7. **Matrix Multiplication Alignment:** Time matrix multiplications for dimensions 1024, 1025, 1026, 1028, 1032.
   - Compare CPU vs. GPU performance.
   - Research your hardware's memory bus width.
8. **Activation Functions:** Swap ReLU for sigmoid, tanh, or LeakyReLU. Which performs best on Fashion-MNIST?
9. **Weight Initialization:** Experiment with different initializations (e.g., Xavier, He). Does it impact convergence speed or final accuracy?