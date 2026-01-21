# Using GPUs for Deep Learning

## Introduction

In the past two decades, GPU performance has increased by a factor of 1000 every decade, offering tremendous computational power for deep learning. This guide will show you how to harness this power by using GPUs for your deep learning computations.

## Prerequisites

Before starting, ensure you have:
- At least one NVIDIA GPU installed
- The appropriate NVIDIA driver and CUDA toolkit installed
- The deep learning framework of your choice (MXNet, PyTorch, TensorFlow, or JAX)

You can check your GPU information using the `nvidia-smi` command in your terminal.

## Setup

First, let's import the necessary libraries. The code below handles imports for different frameworks:

```python
# For PyTorch
from d2l import torch as d2l
import torch
from torch import nn

# For MXNet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# For TensorFlow
from d2l import tensorflow as d2l
import tensorflow as tf

# For JAX
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## 1. Understanding Computing Devices

In deep learning frameworks, every tensor has a *context* or *device* that determines where it's stored and computed. By default, tensors are created on the CPU.

### Device Specification Functions

Let's create helper functions to work with different devices:

```python
def cpu():
    """Get the CPU device."""
    if framework == 'pytorch':
        return torch.device('cpu')
    elif framework == 'mxnet':
        return npx.cpu()
    elif framework == 'tensorflow':
        return tf.device('/CPU:0')
    elif framework == 'jax':
        return jax.devices('cpu')[0]

def gpu(i=0):
    """Get a GPU device."""
    if framework == 'pytorch':
        return torch.device(f'cuda:{i}')
    elif framework == 'mxnet':
        return npx.gpu(i)
    elif framework == 'tensorflow':
        return tf.device(f'/GPU:{i}')
    elif framework == 'jax':
        return jax.devices('gpu')[i]
```

### Checking Available GPUs

You can check how many GPUs are available:

```python
def num_gpus():
    """Get the number of available GPUs."""
    if framework == 'pytorch':
        return torch.cuda.device_count()
    elif framework == 'mxnet':
        return npx.num_gpus()
    elif framework == 'tensorflow':
        return len(tf.config.experimental.list_physical_devices('GPU'))
    elif framework == 'jax':
        try:
            return jax.device_count('gpu')
        except:
            return 0  # No GPU backend found

print(f"Number of available GPUs: {num_gpus()}")
```

### Safe Device Access Functions

These functions help you write code that works even when requested GPUs don't exist:

```python
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]

print(f"First GPU: {try_gpu()}")
print(f"Tenth GPU (falls back to CPU): {try_gpu(10)}")
print(f"All GPUs: {try_all_gpus()}")
```

## 2. Working with Tensors on GPUs

### Checking Tensor Device Location

By default, tensors are created on the CPU (except in TensorFlow and JAX, which use GPU if available):

```python
# Create a tensor
if framework == 'pytorch':
    x = torch.tensor([1, 2, 3])
    print(f"Tensor device: {x.device}")
elif framework == 'mxnet':
    x = np.array([1, 2, 3])
    print(f"Tensor context: {x.ctx}")
elif framework == 'tensorflow':
    x = tf.constant([1, 2, 3])
    print(f"Tensor device: {x.device}")
elif framework == 'jax':
    x = jnp.array([1, 2, 3])
    print(f"Tensor device: {x.device()}")
```

### Creating Tensors on Specific GPUs

You can specify which device to use when creating a tensor:

```python
# Create tensor X on first GPU
if framework == 'pytorch':
    X = torch.ones(2, 3, device=try_gpu())
elif framework == 'mxnet':
    X = np.ones((2, 3), ctx=try_gpu())
elif framework == 'tensorflow':
    with try_gpu():
        X = tf.ones((2, 3))
elif framework == 'jax':
    X = jax.device_put(jnp.ones((2, 3)), try_gpu())

print(f"Tensor X created on GPU: {X}")

# Create tensor Y on second GPU (if available)
if framework == 'pytorch':
    Y = torch.rand(2, 3, device=try_gpu(1))
elif framework == 'mxnet':
    Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
elif framework == 'tensorflow':
    with try_gpu(1):
        Y = tf.random.uniform((2, 3))
elif framework == 'jax':
    Y = jax.device_put(jax.random.uniform(jax.random.PRNGKey(0), (2, 3)),
                       try_gpu(1))

print(f"Tensor Y created on second GPU: {Y}")
```

### Copying Tensors Between Devices

To perform operations on tensors, they must be on the same device. Here's how to copy a tensor to another GPU:

```python
# Copy X to the second GPU
if framework == 'pytorch':
    Z = X.cuda(1) if num_gpus() > 1 else X
elif framework == 'mxnet':
    Z = X.copyto(try_gpu(1)) if num_gpus() > 1 else X
elif framework == 'tensorflow':
    with try_gpu(1):
        Z = X if num_gpus() > 1 else X
elif framework == 'jax':
    Z = jax.device_put(X, try_gpu(1)) if num_gpus() > 1 else X

print(f"Original X device: {X.device if hasattr(X, 'device') else X.ctx}")
print(f"Copied Z device: {Z.device if hasattr(Z, 'device') else Z.ctx}")

# Now we can add Y and Z since they're on the same device
if num_gpus() > 1:
    result = Y + Z
    print(f"Y + Z = {result}")
```

**Important Note:** Copying data between devices is slow! Frameworks are designed to avoid unnecessary copies. If you try to copy a tensor to the device it's already on, most frameworks will return the original tensor without making a copy.

## 3. Neural Networks on GPUs

### Placing Model Parameters on GPU

You can specify which device your neural network model should use:

```python
# Create a simple neural network
if framework == 'pytorch':
    net = nn.Sequential(nn.LazyLinear(1))
    net = net.to(device=try_gpu())
elif framework == 'mxnet':
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(ctx=try_gpu())
elif framework == 'tensorflow':
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1)])
elif framework == 'jax':
    net = nn.Sequential([nn.Dense(1)])
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    x = jax.random.normal(key1, (10,))  # Dummy input
    params = net.init(key2, x)  # Initialization call

# Pass data through the network
if framework in ['pytorch', 'mxnet', 'tensorflow']:
    output = net(X)
    print(f"Network output: {output}")
elif framework == 'jax':
    output = net.apply(params, x)
    print(f"Network output: {output}")
```

### Verifying Parameter Locations

Check that your model parameters are indeed on the GPU:

```python
if framework == 'pytorch':
    print(f"Weight device: {net[0].weight.data.device}")
elif framework == 'mxnet':
    print(f"Weight context: {net[0].weight.data().ctx}")
elif framework == 'tensorflow':
    print(f"Weight devices: {net.layers[0].weights[0].device}, "
          f"{net.layers[0].weights[1].device}")
elif framework == 'jax':
    print("Parameter devices:", 
          jax.tree_util.tree_map(lambda x: x.device(), params))
```

## 4. Training on GPUs

For efficient training, ensure all your data and model parameters are on the same device. Most deep learning frameworks provide utilities to handle device placement during training.

**Key Principle:** Keep data transfers between devices to a minimum. Each transfer incurs significant overhead that can slow down your training pipeline.

## Best Practices

1. **Minimize Data Transfers:** Transferring data between CPU and GPU is slow. Try to keep all computations on the GPU once data is loaded.

2. **Batch Operations:** Perform as many operations as possible in a batch rather than one at a time.

3. **Avoid Synchronization:** Operations that require synchronization between devices (like printing to console) can create bottlenecks.

4. **Monitor GPU Memory:** Use tools like `nvidia-smi` to ensure you're not exceeding GPU memory limits.

## Exercises

1. **Performance Comparison:** Try multiplying large matrices on both CPU and GPU. Compare the execution times. What do you notice for small computations?

2. **Parameter Management:** Experiment with reading and writing model parameters when they're on the GPU. How does this differ from CPU-based parameters?

3. **Logging Efficiency:** Compare two approaches:
   - Computing 1000 matrix multiplications and logging the Frobenius norm after each one
   - Computing all multiplications on GPU and only transferring the final results
   Which is more efficient and why?

4. **Multi-GPU Scaling:** If you have multiple GPUs, try performing matrix multiplications on two GPUs simultaneously versus sequentially on one GPU. Do you see linear scaling?

## Summary

- GPUs provide significant speedups for deep learning computations
- Each tensor has a device context (CPU or GPU)
- All tensors in an operation must be on the same device
- Data transfers between devices are expensive and should be minimized
- Neural network models can be placed on GPUs for faster training
- Proper device management is crucial for efficient deep learning workflows

By following these guidelines, you can effectively leverage GPU acceleration for your deep learning projects.