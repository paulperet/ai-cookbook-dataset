# Multi-GPU Training Guide

## Overview

This guide demonstrates how to implement data parallelism for training deep learning models across multiple GPUs. You'll learn the core concepts of distributing computation and synchronizing gradients while implementing a practical training workflow from scratch.

## Prerequisites

Ensure you have the necessary libraries installed. This guide provides implementations for both MXNet and PyTorch.

```bash
# Install required packages
pip install d2l  # For educational utilities
# Install either mxnet or torch based on your preference
pip install mxnet-cuXXX  # For GPU support, replace XXX with CUDA version
# or
pip install torch torchvision
```

## 1. Understanding Parallelization Strategies

Before implementing multi-GPU training, let's review the three main approaches:

1. **Network Partitioning**: Split different layers across GPUs
2. **Layer-wise Partitioning**: Distribute channels/units within layers
3. **Data Parallelism**: Distribute data batches across GPUs

Data parallelism is the most practical approach for most scenarios, as it:
- Requires minimal synchronization
- Scales well with additional GPUs
- Maintains identical model copies on each device

## 2. Setting Up the Environment

First, import the necessary libraries and configure your environment.

```python
# For MXNet users
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()

# For PyTorch users
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

## 3. Defining a Toy Network

We'll use a modified LeNet architecture to demonstrate multi-GPU training concepts. This network includes convolutional layers, pooling, and fully connected layers.

### 3.1 Initialize Model Parameters

```python
# MXNet implementation
scale = 0.01
W1 = np.random.normal(scale=scale, size=(20, 1, 3, 3))
b1 = np.zeros(20)
W2 = np.random.normal(scale=scale, size=(50, 20, 5, 5))
b2 = np.zeros(50)
W3 = np.random.normal(scale=scale, size=(800, 128))
b3 = np.zeros(128)
W4 = np.random.normal(scale=scale, size=(128, 10))
b4 = np.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# PyTorch implementation
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]
```

### 3.2 Define the Network Architecture

```python
# MXNet implementation
def lenet(X, params):
    h1_conv = npx.convolution(data=X, weight=params[0], bias=params[1],
                              kernel=(3, 3), num_filter=20)
    h1_activation = npx.relu(h1_conv)
    h1 = npx.pooling(data=h1_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2_conv = npx.convolution(data=h1, weight=params[2], bias=params[3],
                              kernel=(5, 5), num_filter=50)
    h2_activation = npx.relu(h2_conv)
    h2 = npx.pooling(data=h2_activation, pool_type='avg', kernel=(2, 2),
                     stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = np.dot(h2, params[4]) + params[5]
    h3 = npx.relu(h3_linear)
    y_hat = np.dot(h3, params[6]) + params[7]
    return y_hat

# PyTorch implementation
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat
```

### 3.3 Define the Loss Function

```python
# MXNet
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# PyTorch
loss = nn.CrossEntropyLoss(reduction='none')
```

## 4. Implementing Core Multi-GPU Operations

For efficient multi-GPU training, we need two fundamental operations: parameter distribution and gradient aggregation.

### 4.1 Parameter Distribution Function

This function copies parameters to a specified device and enables gradient computation.

```python
# MXNet implementation
def get_params(params, device):
    new_params = [p.copyto(device) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params

# PyTorch implementation
def get_params(params, device):
    new_params = [p.to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params
```

Test the parameter distribution:

```python
new_params = get_params(params, d2l.try_gpu(0))
print('b1 weight:', new_params[1])
print('b1 grad:', new_params[1].grad)
```

### 4.2 Gradient Aggregation Function

The `allreduce` function sums gradients from all GPUs and broadcasts the result back to each device.

```python
# MXNet implementation
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].ctx)
    for i in range(1, len(data)):
        data[0].copyto(data[i])

# PyTorch implementation
def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i][:] = data[0].to(data[i].device)
```

Test gradient aggregation:

```python
# Create test data on different devices
data = [np.ones((1, 2), ctx=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
print('before allreduce:\n', data[0], '\n', data[1])
allreduce(data)
print('after allreduce:\n', data[0], '\n', data[1])
```

### 4.3 Data Distribution Function

We need to split each minibatch evenly across available GPUs.

```python
# MXNet implementation
def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (gluon.utils.split_and_load(X, devices),
            gluon.utils.split_and_load(y, devices))

# PyTorch implementation
def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
```

## 5. Implementing Multi-GPU Training

### 5.1 Single Minibatch Training Function

This function processes one minibatch across all available GPUs.

```python
# MXNet implementation
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    with autograd.record():  # Loss is calculated separately on each GPU
        ls = [loss(lenet(X_shard, device_W), y_shard)
              for X_shard, y_shard, device_W in zip(
                  X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    for i in range(len(device_params[0])):
        allreduce([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0])  # Here, we use a full-size batch

# PyTorch implementation
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # Loss is calculated separately on each GPU
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    for l in ls:  # Backpropagation is performed separately on each GPU
        l.backward()
    # Sum all gradients from each GPU and broadcast them to all GPUs
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # The model parameters are updated separately on each GPU
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # Here, we use a full-size batch
```

### 5.2 Complete Training Function

Now let's implement the full training loop that coordinates multi-GPU training.

```python
# MXNet implementation
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            train_batch(X, y, device_params, devices, lr)
            npx.waitall()
        timer.stop()
        # Evaluate the model on GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')

# PyTorch implementation
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # Copy model parameters to `num_gpus` GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # Perform multi-GPU training for a single minibatch
            train_batch(X, y, device_params, devices, lr)
            torch.cuda.synchronize()
        timer.stop()
        # Evaluate the model on GPU 0
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

## 6. Running Experiments

### 6.1 Single GPU Baseline

Let's first establish a baseline with a single GPU.

```python
train(num_gpus=1, batch_size=256, lr=0.2)
```

### 6.2 Scaling to Multiple GPUs

Now let's scale to two GPUs while keeping the same hyperparameters.

```python
train(num_gpus=2, batch_size=256, lr=0.2)
```

## Key Takeaways

1. **Data parallelism** is the most practical approach for multi-GPU training, where each GPU processes a subset of the data and gradients are synchronized.

2. The effective minibatch size increases with more GPUs. When using `k` GPUs, you're effectively training with a batch size of `k × original_batch_size`.

3. Gradient synchronization requires careful implementation to ensure all GPUs have consistent parameter updates.

4. For optimal performance, consider adjusting the learning rate when increasing the effective batch size.

## Exercises

1. **Batch Size Scaling**: Modify the training function to automatically scale the batch size with the number of GPUs. When training on `k` GPUs, the effective batch size should be `k × b`.

2. **Learning Rate Sensitivity**: Experiment with different learning rates when using multiple GPUs. How does the optimal learning rate change as you add more GPUs?

3. **Efficient AllReduce**: Implement a more efficient `allreduce` function that aggregates different parameters simultaneously rather than sequentially. Consider using framework-specific collective operations.

4. **Multi-GPU Evaluation**: Extend the accuracy computation to use all GPUs during evaluation, not just GPU 0.

## Next Steps

This implementation demonstrates the fundamental concepts of data parallelism. For production use, consider:

- Using framework-provided distributed training utilities
- Implementing gradient accumulation for very large models
- Exploring model parallelism for models that don't fit on a single GPU
- Optimizing communication patterns for your specific hardware configuration

Remember that the benefits of multi-GPU training become more apparent with larger models and datasets. For small networks like LeNet on Fashion-MNIST, the overhead of synchronization may outweigh the computational benefits.