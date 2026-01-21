# Multi-GPU Training with High-Level Frameworks

## Introduction

Training deep learning models on multiple GPUs can significantly accelerate the process, but implementing parallelism from scratch is complex and time-consuming. Fortunately, modern deep learning frameworks provide high-level APIs that abstract away much of this complexity. This guide demonstrates how to leverage these APIs for efficient multi-GPU training using a modified ResNet-18 model.

**Prerequisites:** You will need at least two GPUs to run the multi-GPU examples in this tutorial.

## Setup

First, install the necessary libraries and import the required modules. This tutorial provides code for both MXNet and PyTorch.

```python
# For MXNet
# !pip install d2l
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```python
# For PyTorch
# !pip install d2l
from d2l import torch as d2l
import torch
from torch import nn
```

## Step 1: Define a Toy Network

We'll use a slightly modified version of ResNet-18, which is more meaningful than a simple LeNet but still quick to train. The modifications include a smaller initial convolution kernel, stride, and padding, and the removal of the max-pooling layer to suit smaller input images (28x28, as from Fashion-MNIST).

### MXNet Implementation

```python
def resnet18(num_classes):
    """A slightly modified ResNet-18 model for MXNet."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

### PyTorch Implementation

```python
def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model for PyTorch."""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(out_channels, use_1x1conv=True, 
                                        strides=2))
            else:
                blk.append(d2l.Residual(out_channels))
        return nn.Sequential(*blk)

    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

## Step 2: Initialize the Network Across Devices

### MXNet: Initialization on Multiple GPUs

In MXNet, the `initialize` function allows you to initialize network parameters on specific devices, including multiple GPUs simultaneously.

```python
net = resnet18(10)
devices = d2l.try_all_gpus()
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

To verify the initialization, you can split a batch of data across GPUs and perform a forward pass. The network automatically uses the appropriate GPU for each data shard.

```python
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

Parameters are initialized only on the devices where data passes through. Attempting to access them on the CPU will raise an error.

```python
weight = net[0].params.get('weight')
try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

### PyTorch: Initialization During Training

In PyTorch, network initialization typically occurs inside the training loop. We'll define the network and get the list of available GPUs.

```python
net = resnet18(10)
devices = d2l.try_all_gpus()
```

## Step 3: Evaluate Accuracy in Parallel

For multi-GPU evaluation, we need a function that splits each minibatch across devices, performs forward passes in parallel, and aggregates the results.

### MXNet Implementation

```python
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Compute the accuracy for a model on a dataset using multiple GPUs."""
    devices = list(net.collect_params().values())[0].list_ctx()
    metric = d2l.Accumulator(2)  # Correct predictions, total predictions
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(pred_shards, y_shards)), 
                   labels.size)
    return metric[0] / metric[1]
```

## Step 4: Implement the Training Loop

The training process must handle several tasks in parallel:
1. Initialize network parameters across all devices.
2. Split each minibatch across devices.
3. Compute loss and gradients in parallel.
4. Aggregate gradients and update parameters.
5. Compute accuracy in parallel to monitor performance.

### MXNet Training Function

```python
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```

### PyTorch Training Function

```python
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(module):
        if type(module) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

## Step 5: Run the Training

### Warm-up: Training on a Single GPU

Start by training the network on a single GPU to establish a baseline.

```python
# MXNet
train(num_gpus=1, batch_size=256, lr=0.1)
```

```python
# PyTorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

### Scaling Up: Training on Two GPUs

Now, train the same model using two GPUs. With the more complex ResNet-18 architecture, the computational time is significantly larger than the synchronization overhead, making parallelization highly beneficial.

```python
# MXNet
train(num_gpus=2, batch_size=512, lr=0.2)
```

```python
# PyTorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## Summary

- **MXNet** provides primitives like `initialize(ctx=devices)` to easily initialize model parameters across multiple GPUs.
- Data is automatically evaluated on the device where it resides.
- **Crucial:** Ensure the network is initialized on each device before accessing parameters on that device, or you will encounter errors.
- Optimization algorithms automatically handle gradient aggregation across multiple GPUs.

## Exercises

1. Experiment with different hyperparameters: epochs, batch sizes, and learning rates. Try using more GPUs (e.g., 16 on an AWS p2.16xlarge instance). Observe how performance scales.
2. Consider heterogeneous computing environments with GPUs of different powers or a mix of GPUs and CPUs. How would you divide the work? Is the complexity worth the potential gain?
3. **MXNet-specific:** What is the effect of removing `npx.waitall()`? How could you modify training to allow overlap of up to two steps for increased parallelism?

For further discussion, refer to the MXNet [forum](https://discuss.d2l.ai/t/365) or PyTorch [forum](https://discuss.d2l.ai/t/1403).