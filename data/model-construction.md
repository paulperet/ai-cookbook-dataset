# Building Neural Networks with Modules and Layers

## Introduction

When constructing neural networks, we often think in terms of individual neurons, layers, and entire models. However, as architectures grow more complex—like ResNet-152 with hundreds of layers—we need better abstractions. This guide introduces the **module** concept, which allows us to encapsulate anything from a single layer to entire model components, enabling cleaner, more maintainable code.

In this tutorial, you'll learn:
- How to use built-in sequential containers
- How to create custom modules from scratch
- How to implement flexible forward propagation with custom logic
- How to combine modules in creative ways

## Prerequisites

First, let's import the necessary libraries for your chosen framework.

```python
# For MXNet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```python
# For PyTorch
import torch
from torch import nn
from torch.nn import functional as F
```

```python
# For TensorFlow
import tensorflow as tf
```

```python
# For JAX
from typing import List
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## 1. Using Sequential Modules

The simplest way to build a model is by stacking layers sequentially. Most frameworks provide a `Sequential` class for this purpose.

Let's create a simple MLP with one hidden layer (256 units, ReLU activation) and an output layer (10 units).

```python
# MXNet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
print(net(X).shape)
```

```python
# PyTorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

X = torch.rand(2, 20)
print(net(X).shape)
```

```python
# TensorFlow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
print(net(X).shape)
```

```python
# JAX
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])

X = jax.random.uniform(d2l.get_key(), (2, 20))
params = net.init(d2l.get_key(), X)
print(net.apply(params, X).shape)
```

**How it works:** `Sequential` maintains an ordered list of modules. When you call the network with input `X`, it passes the data through each module in sequence. The syntax `net(X)` is actually shorthand for calling the forward propagation method (`forward` in PyTorch/MXNet, `call` in TensorFlow, `__call__` in JAX).

## 2. Creating Custom Modules

While `Sequential` is convenient, you'll often need more flexibility. Let's implement the same MLP as a custom module.

Every module must:
1. Ingest input data in its forward propagation method
2. Generate output (potentially with different shape than input)
3. Calculate gradients (handled automatically via autograd)
4. Store necessary parameters
5. Initialize parameters appropriately

Here's our custom `MLP` class:

```python
# MXNet
class MLP(nn.Block):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Dense(256, activation='relu')
        self.out = nn.Dense(10)

    def forward(self, X):
        return self.out(self.hidden(X))
```

```python
# PyTorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

```python
# TensorFlow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, X):
        return self.out(self.hidden((X)))
```

```python
# JAX
class MLP(nn.Module):
    def setup(self):
        self.hidden = nn.Dense(256)
        self.out = nn.Dense(10)

    def __call__(self, X):
        return self.out(nn.relu(self.hidden(X)))
```

**Key points:**
- We call `super().__init__()` to inherit parent class functionality
- We define layers as instance variables in `__init__` (or `setup` in JAX)
- The forward method defines how data flows through these layers

Let's test our custom module:

```python
# For MXNet, PyTorch, TensorFlow
net = MLP()
if framework == 'mxnet':  # MXNet requires explicit initialization
    net.initialize()
X = create_test_input(framework)  # Framework-specific input
print(net(X).shape)
```

```python
# JAX
net = MLP()
params = net.init(d2l.get_key(), X)
print(net.apply(params, X).shape)
```

## 3. Implementing Your Own Sequential Class

To understand how `Sequential` works internally, let's build our own `MySequential` class.

```python
# MXNet
class MySequential(nn.Block):
    def add(self, block):
        self._children[block.name] = block

    def forward(self, X):
        for block in self._children.values():
            X = block(X)
        return X
```

```python
# PyTorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, X):
        for module in self.children():            
            X = module(X)
        return X
```

```python
# TensorFlow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = args

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

```python
# JAX
class MySequential(nn.Module):
    modules: List

    def __call__(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

Now let's use our custom sequential container:

```python
# MXNet
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
print(net(X).shape)
```

```python
# PyTorch
net = MySequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
print(net(X).shape)
```

```python
# TensorFlow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
print(net(X).shape)
```

```python
# JAX
net = MySequential([nn.Dense(256), nn.relu, nn.Dense(10)])
params = net.init(d2l.get_key(), X)
print(net.apply(params, X).shape)
```

## 4. Adding Custom Logic to Forward Propagation

Sometimes you need more than just layer stacking. You might want to include:
- Constant (non-trainable) parameters
- Python control flow
- Arbitrary mathematical operations

Here's an example `FixedHiddenMLP` that includes constant parameters and a while-loop:

```python
# MXNet
class FixedHiddenMLP(nn.Block):
    def __init__(self):
        super().__init__()
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        X = self.dense(X)
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```python
# PyTorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20))
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)        
        X = F.relu(X @ self.rand_weight + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```python
# TensorFlow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        X = self.dense(X)
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

```python
# JAX
class FixedHiddenMLP(nn.Module):
    rand_weight: jnp.array = jax.random.uniform(d2l.get_key(), (20, 20))

    def setup(self):
        self.dense = nn.Dense(20)

    def __call__(self, X):
        X = self.dense(X)
        X = nn.relu(X @ self.rand_weight + 1)
        X = self.dense(X)
        while jnp.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

Test the module:

```python
# For MXNet, PyTorch, TensorFlow
net = FixedHiddenMLP()
if framework == 'mxnet':
    net.initialize()
print(net(X))
```

```python
# JAX
net = FixedHiddenMLP()
params = net.init(d2l.get_key(), X)
print(net.apply(params, X))
```

## 5. Mixing and Matching Modules

You can create complex architectures by nesting modules. Here's an example that combines multiple module types:

```python
# MXNet
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())
chimera.initialize()
print(chimera(X))
```

```python
# PyTorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
                                 nn.LazyLinear(32), nn.ReLU())
        self.linear = nn.LazyLinear(16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.LazyLinear(20), FixedHiddenMLP())
print(chimera(X))
```

```python
# TensorFlow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
print(chimera(X))
```

```python
# JAX
class NestMLP(nn.Module):
    def setup(self):
        self.net = nn.Sequential([nn.Dense(64), nn.relu,
                                  nn.Dense(32), nn.relu])
        self.dense = nn.Dense(16)

    def __call__(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential([NestMLP(), nn.Dense(20), FixedHiddenMLP()])
params = chimera.init(d2l.get_key(), X)
print(chimera.apply(params, X))
```

## Summary

In this tutorial, you've learned:

1. **Modules are building blocks** that can represent individual layers, groups of layers, or entire models
2. **Sequential containers** provide a simple way to stack modules linearly
3. **Custom modules** allow you to define arbitrary forward propagation logic, including control flow and custom operations
4. **Module composition** enables you to create complex architectures by nesting modules within modules

The module abstraction is powerful because it lets you:
- Reuse code effectively
- Maintain clean, organized architectures
- Implement complex patterns (like ResNet's residual blocks) without repetitive code

## Exercises

1. What issues might arise if you store modules in a regular Python list instead of the framework-specific container in `MySequential`?
2. Implement a "parallel module" that takes two networks as arguments and returns the concatenated output of both in forward propagation.
3. Create a factory function that generates multiple instances of the same module and combines them into a larger network.

*Hint for exercise 1: Consider parameter initialization and gradient computation.*