# Parameter Initialization in Deep Learning Frameworks

## Introduction

Proper parameter initialization is crucial for training neural networks effectively. In this guide, you'll learn how to initialize model parameters using both built-in and custom initialization methods across four popular deep learning frameworks: MXNet, PyTorch, TensorFlow, and JAX.

## Prerequisites

First, ensure you have the necessary imports for your chosen framework:

```python
# For MXNet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

# For PyTorch
import torch
from torch import nn

# For TensorFlow
import tensorflow as tf

# For JAX
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Default Initialization Behavior

Each framework has its own default initialization strategy:

- **MXNet**: Weight parameters are randomly drawn from a uniform distribution `U(-0.07, 0.07)`, while bias parameters are set to zero.
- **PyTorch**: Weight and bias matrices are initialized uniformly based on input and output dimensions.
- **TensorFlow**: Weight matrices are initialized uniformly based on input and output dimensions, with bias parameters set to zero.
- **JAX**: Weights are initialized using `lecun_normal` (truncated normal distribution), and biases are set to zero.

## Creating a Simple Network

Let's start by creating a simple neural network to demonstrate initialization techniques:

```python
# MXNet
net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Apply default initialization

# PyTorch
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))

# TensorFlow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

# JAX
net = nn.Sequential([nn.Dense(8), nn.relu, nn.Dense(1)])
```

## Using Built-in Initializers

### Gaussian Initialization

Initialize all weight parameters as Gaussian random variables with standard deviation 0.01, while clearing bias parameters to zero:

```python
# MXNet
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
print(net[0].weight.data()[0])

# PyTorch
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])

# TensorFlow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

print(net.weights[0], net.weights[1])

# JAX
weight_init = nn.initializers.normal(0.01)
bias_init = nn.initializers.zeros

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
print(layer_0['kernel'][:, 0], layer_0['bias'][0])
```

### Constant Initialization

Initialize all parameters to a given constant value (e.g., 1):

```python
# MXNet
net.initialize(init=init.Constant(1), force_reinit=True)
print(net[0].weight.data()[0])

# PyTorch
def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)

net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])

# TensorFlow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

print(net.weights[0], net.weights[1])

# JAX
weight_init = nn.initializers.constant(1)

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
print(layer_0['kernel'][:, 0], layer_0['bias'][0])
```

### Mixed Initialization Strategies

Apply different initializers to different layers. For example, initialize the first layer with Xavier initialization and the second layer to a constant value of 42:

```python
# MXNet
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())

# PyTorch
def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

# TensorFlow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(42)),
])

print(net.layers[1].weights[0])
print(net.layers[2].weights[0])

# JAX
net = nn.Sequential([nn.Dense(8, kernel_init=nn.initializers.xavier_uniform(),
                              bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=nn.initializers.constant(42),
                              bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
print(params['params']['layers_0']['kernel'][:, 0], params['params']['layers_2']['kernel'])
```

## Creating Custom Initializers

Sometimes you need initialization methods not provided by the framework. Let's create a custom initializer that follows this distribution:

```
w ∼ {
    U(5, 10) with probability 1/4
    0 with probability 1/2
    U(-10, -5) with probability 1/4
}
```

### Custom Initializer Implementation

```python
# MXNet
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
print(net[0].weight.data()[:2])

# PyTorch
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
print(net[0].weight[:2])

# TensorFlow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data = tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor = (tf.abs(data) >= 5)
        factor = tf.cast(factor, tf.float32)
        return data * factor

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

print(net.layers[1].weights[0])

# JAX
def my_init(key, shape, dtype=jnp.float_):
    data = jax.random.uniform(key, shape, minval=-10, maxval=10)
    return data * (jnp.abs(data) >= 5)

net = nn.Sequential([nn.Dense(8, kernel_init=my_init), nn.relu, nn.Dense(1)])
params = net.init(d2l.get_key(), X)
print(params['params']['layers_0']['kernel'][:, :2])
```

## Direct Parameter Manipulation

You can also directly modify parameter values after initialization:

```python
# MXNet
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
print(net[0].weight.data()[0])

# PyTorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
print(net[0].weight.data[0])

# TensorFlow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
print(net.layers[1].weights[0])

# Note: In JAX, parameters are immutable by design. Use params.unfreeze() to make changes.
```

## Summary

In this guide, you've learned how to:
- Use default initialization methods in each framework
- Apply built-in initializers like Gaussian, constant, and Xavier initialization
- Create custom initialization strategies
- Directly manipulate parameter values (where supported)

Proper initialization is a critical first step in training neural networks, and each framework provides flexible tools to control this process.

## Next Steps

Explore the online documentation for your chosen framework to discover more built-in initialization methods and advanced techniques for parameter initialization.