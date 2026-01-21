# Guide: Loading and Saving Model Parameters

In this guide, you'll learn how to save and load both individual tensors and entire model parameters. This is essential for persisting trained models, checkpointing during long training runs, and deploying models to production.

## Prerequisites

First, ensure you have the necessary libraries installed. The code below imports the required modules for each supported deep learning framework.

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
import numpy as np
```

```python
# For JAX
from d2l import jax as d2l
import flax
from flax import linen as nn
from flax.training import checkpoints
import jax
from jax import numpy as jnp
```

## Step 1: Loading and Saving Individual Tensors

You can directly save and load individual tensors using framework-specific functions. This is useful for storing weight vectors, gradients, or any intermediate data.

### 1.1 Save a Single Tensor

Start by creating a simple tensor and saving it to a file.

```python
# MXNet
x = np.arange(4)
npx.save('x-file', x)
```

```python
# PyTorch
x = torch.arange(4)
torch.save(x, 'x-file')
```

```python
# TensorFlow
x = tf.range(4)
np.save('x-file.npy', x)
```

```python
# JAX
x = jnp.arange(4)
jnp.save('x-file.npy', x)
```

### 1.2 Load the Tensor Back into Memory

Now, read the saved tensor back to verify the operation.

```python
# MXNet
x2 = npx.load('x-file')
x2
```

```python
# PyTorch
x2 = torch.load('x-file')
x2
```

```python
# TensorFlow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

```python
# JAX
x2 = jnp.load('x-file.npy', allow_pickle=True)
x2
```

### 1.3 Save and Load Multiple Tensors

You can also store a list of tensors and retrieve them.

```python
# MXNet
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```python
# PyTorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```python
# TensorFlow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

```python
# JAX
y = jnp.zeros(4)
jnp.save('xy-files.npy', [x, y])
x2, y2 = jnp.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

### 1.4 Work with Dictionaries of Tensors

For more complex data, like model weights, you can save and load dictionaries mapping strings to tensors.

```python
# MXNet
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```python
# PyTorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```python
# TensorFlow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

```python
# JAX
mydict = {'x': x, 'y': y}
jnp.save('mydict.npy', mydict)
mydict2 = jnp.load('mydict.npy', allow_pickle=True)
mydict2
```

## Step 2: Loading and Saving Model Parameters

While saving individual tensors is helpful, you often need to persist entire model parameters. Deep learning frameworks provide built-in methods for this. Note that these methods save the model's *parameters*, not the entire model architecture. You must define the architecture separately in code before loading the parameters.

### 2.1 Define a Simple Model

Let's create a familiar Multilayer Perceptron (MLP) model.

```python
# MXNet
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```python
# PyTorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```python
# TensorFlow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

```python
# JAX
class MLP(nn.Module):
    def setup(self):
        self.hidden = nn.Dense(256)
        self.output = nn.Dense(10)

    def __call__(self, x):
        return self.output(nn.relu(self.hidden(x)))

net = MLP()
X = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (2, 20))
Y, params = net.init_with_output(jax.random.PRNGKey(d2l.get_seed()), X)
```

### 2.2 Save the Model Parameters

Save the model's parameters to a file named `mlp.params`.

```python
# MXNet
net.save_parameters('mlp.params')
```

```python
# PyTorch
torch.save(net.state_dict(), 'mlp.params')
```

```python
# TensorFlow
net.save_weights('mlp.params')
```

```python
# JAX
checkpoints.save_checkpoint('ckpt_dir', params, step=1, overwrite=True)
```

### 2.3 Load the Parameters into a New Model Instance

Create a new instance of the MLP and load the saved parameters into it.

```python
# MXNet
clone = MLP()
clone.load_parameters('mlp.params')
```

```python
# PyTorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

```python
# TensorFlow
clone = MLP()
clone.load_weights('mlp.params')
```

```python
# JAX
clone = MLP()
cloned_params = flax.core.freeze(checkpoints.restore_checkpoint('ckpt_dir',
                                                                target=None))
```

### 2.4 Verify the Loaded Model

Ensure the cloned model produces the same output as the original for the same input.

```python
# MXNet, PyTorch, TensorFlow
Y_clone = clone(X)
Y_clone == Y
```

```python
# JAX
Y_clone = clone.apply(cloned_params, X)
Y_clone == Y
```

## Summary

In this guide, you learned how to:
- Save and load individual tensors using framework-specific `save` and `load` functions.
- Store and retrieve lists and dictionaries of tensors.
- Persist entire model parameters to disk and reload them into a new model instance.

Remember, saving model parameters does not include the architecture. You must define the model structure in code before loading the parameters.

## Exercises

1. **Practical Benefits of Storing Model Parameters**: Even without deployment, storing parameters allows for resuming interrupted training, sharing models with collaborators, and benchmarking against previous versions.
2. **Reusing Parts of a Network**: To reuse the first two layers from a saved network, you would load the full parameter dictionary, extract the relevant weights, and manually assign them to the corresponding layers in your new model.
3. **Saving Architecture and Parameters**: While parameters are easily serialized, the architecture often requires code. Some frameworks offer serialization of the entire model (e.g., PyTorch's `torch.save(model)`), but this can be restrictive and less portable. A common practice is to save both the architecture definition (as source code) and the parameters separately.

For further discussion, refer to the framework-specific forums linked in the original content.