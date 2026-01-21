# Parameter Management in Deep Learning Frameworks

## Introduction

After selecting an architecture and hyperparameters, we enter the training loop to find parameter values that minimize our loss function. These parameters are essential for making predictions after training, saving models for deployment, or analyzing them for scientific insight. While deep learning frameworks typically handle parameter management automatically, understanding how to access and manipulate parameters becomes crucial when working with custom architectures.

In this tutorial, you'll learn how to:
- Access parameters for debugging, diagnostics, and visualizations
- Share parameters across different model components

## Prerequisites

First, let's set up our environment by importing the necessary libraries for each framework.

```python
# MXNet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```python
# PyTorch
import torch
from torch import nn
```

```python
# TensorFlow
import tensorflow as tf
```

```python
# JAX
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Step 1: Creating a Basic MLP

We'll start by creating a simple Multi-Layer Perceptron (MLP) with one hidden layer to demonstrate parameter management.

```python
# MXNet
net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X).shape
```

```python
# PyTorch
net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

X = torch.rand(size=(2, 4))
net(X).shape
```

```python
# TensorFlow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X).shape
```

```python
# JAX
net = nn.Sequential([nn.Dense(8), nn.relu, nn.Dense(1)])

X = jax.random.uniform(d2l.get_key(), (2, 4))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

## Step 2: Accessing Parameters

Now let's explore how to access parameters from our models. The approach varies slightly between frameworks.

### Accessing Layer Parameters

For frameworks using `Sequential` classes (MXNet, PyTorch, TensorFlow), you can access layers by indexing into the model as if it were a list. In JAX, parameters are stored in a separate dictionary after initialization.

Let's inspect the parameters of the second fully connected layer:

```python
# MXNet
net[1].params
```

```python
# PyTorch
net[2].state_dict()
```

```python
# TensorFlow
net.layers[2].weights
```

```python
# JAX
params['params']['layers_2']
```

You'll notice that each fully connected layer contains two parameters: weights and biases.

### Accessing Specific Parameter Values

To work with parameters, we often need to extract their underlying numerical values. Here's how to access the bias from the second layer:

```python
# MXNet
type(net[1].bias), net[1].bias.data()
```

```python
# PyTorch
type(net[2].bias), net[2].bias.data
```

```python
# TensorFlow
type(net.layers[2].weights[1]), tf.convert_to_tensor(net.layers[2].weights[1])
```

```python
# JAX
bias = params['params']['layers_2']['bias']
type(bias), bias
```

**Note:** In MXNet and PyTorch, parameters are complex objects containing values, gradients, and additional metadata, which is why we need to explicitly request the value.

### Accessing Gradients

Let's check the gradient of a parameter. Since we haven't performed backpropagation yet, gradients should be in their initial state:

```python
# MXNet
net[1].weight.grad()
```

```python
# PyTorch
net[2].weight.grad == None
```

**Note:** JAX handles gradients differently by decoupling them from parameters and using transformation functions like `grad()`.

### Accessing All Parameters

When working with complex models, accessing parameters individually becomes tedious. Here's how to access all parameters at once:

```python
# MXNet
net.collect_params()
```

```python
# PyTorch
[(name, param.shape) for name, param in net.named_parameters()]
```

```python
# TensorFlow
net.get_weights()
```

```python
# JAX
jax.tree_util.tree_map(lambda x: x.shape, params)
```

## Step 3: Sharing Parameters Across Layers

Parameter sharing is a powerful technique that allows multiple layers to use the same parameters. This is particularly useful for certain architectures or when implementing specific constraints.

### Implementing Parameter Sharing

Let's create a network where two layers share the same parameters:

```python
# MXNet
net = nn.Sequential()
# We need to give the shared layer a name so that we can refer to its parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)

# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```python
# PyTorch
# We need to give the shared layer a name so that we can refer to its parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```python
# TensorFlow
# TensorFlow Keras automatically removes duplicate layers
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# Check whether the parameters are different
print(len(net.layers) == 3)
```

```python
# JAX
# We need to give the shared layer a name so that we can refer to its parameters
shared = nn.Dense(8)
net = nn.Sequential([nn.Dense(8), nn.relu,
                     shared, nn.relu,
                     shared, nn.relu,
                     nn.Dense(1)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)

# Check whether the parameters are different
print(len(params['params']) == 3)
```

### Understanding Parameter Sharing

When parameters are tied between layers, they're not just equal—they're the same exact tensor object. Changing one parameter automatically changes the other. During backpropagation, gradients from both layers are added together since they affect the same parameter.

## Summary

In this tutorial, you've learned:

1. **Parameter Access**: How to access specific parameters, their values, and gradients across different deep learning frameworks
2. **Bulk Operations**: How to work with all parameters simultaneously
3. **Parameter Sharing**: How to share parameters across multiple layers and understand the implications for gradient computation

Parameter management is essential when moving beyond standard architectures, enabling you to implement custom models, debug effectively, and optimize memory usage through parameter sharing.

## Exercises

To reinforce your understanding:

1. Use the `NestMLP` model from the model construction section and practice accessing parameters from its various layers
2. Construct an MLP with a shared parameter layer, train it, and observe how parameters and gradients behave during training
3. Consider why parameter sharing can be beneficial (hint: think about memory efficiency, regularization, and specific architectural patterns)

## Further Discussion

Join the community discussions for your framework:
- [MXNet](https://discuss.d2l.ai/t/56)
- [PyTorch](https://discuss.d2l.ai/t/57)
- [TensorFlow](https://discuss.d2l.ai/t/269)
- [JAX](https://discuss.d2l.ai/t/17990)