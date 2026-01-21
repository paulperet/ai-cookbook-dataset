# Learning Rate Scheduling: A Practical Guide

## Introduction

In deep learning, the learning rate is a critical hyperparameter that controls how much we adjust our model's weights with respect to the loss gradient. While optimization algorithms like SGD, Adam, or RMSprop determine *how* to update weights, the learning rate determines the *magnitude* of these updates.

Choosing the right learning rate schedule can significantly impact:
- **Training speed**: Too small a rate slows convergence; too large causes divergence.
- **Final accuracy**: A well-tuned schedule helps the model settle into a better minimum.
- **Generalization**: Proper decay can reduce overfitting.

In this guide, you'll learn how to implement and experiment with various learning rate schedules using a convolutional neural network on the Fashion-MNIST dataset.

## Prerequisites

Ensure you have the necessary libraries installed. This guide provides code for three popular frameworks: MXNet, PyTorch, and TensorFlow.

```bash
# Install d2l library which provides common utilities
pip install d2l
```

Depending on your chosen framework, you'll also need:
- `mxnet` and `gluon` for MXNet
- `torch` and `torchvision` for PyTorch
- `tensorflow` for TensorFlow

## Step 1: Setup and Model Definition

First, let's import the necessary modules and define our model. We'll use a modernized LeNet architecture with ReLU activations and MaxPooling.

### MXNet Implementation

```python
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()
```

### PyTorch Implementation

```python
from d2l import torch as d2l
import torch
from torch import nn

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))
    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()
```

### TensorFlow Implementation

```python
from d2l import tensorflow as d2l
import tensorflow as tf

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])
```

## Step 2: Data Loading and Training Function

Load the Fashion-MNIST dataset and define a training function that supports learning rate scheduling.

```python
# Common for all frameworks
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

The training function varies slightly by framework but follows the same structure: it iterates through epochs, computes gradients, updates weights, and tracks metrics. The key addition is the `scheduler` parameter that adjusts the learning rate at each step or epoch.

## Step 3: Baseline - Constant Learning Rate

Before exploring schedules, let's establish a baseline with a constant learning rate of 0.3 over 30 epochs.

### MXNet

```python
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

### PyTorch

```python
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

### TensorFlow

```python
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

**Observation**: With a constant rate, training accuracy improves but test accuracy plateaus early, indicating overfitting. The gap between train and test curves suggests we need to regularize training, which a learning rate schedule can help with.

## Step 4: Manual Learning Rate Adjustment

The simplest form of scheduling is manually adjusting the rate during training. Most optimizers provide a way to update the learning rate.

### MXNet

```python
trainer.set_learning_rate(0.1)
print(f'learning rate is now {trainer.learning_rate:.2f}')
```

### PyTorch

```python
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

### TensorFlow

```python
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate is now {dummy_model.optimizer.lr.numpy()}')
```

While manual adjustment works, it's tedious. Let's automate this with schedulers.

## Step 5: Implementing a Custom Scheduler

A scheduler is a callable that takes the current step or epoch number and returns the appropriate learning rate. Let's implement a square root decay schedule: `η = η₀ × (t + 1)⁻⁰·⁵`.

```python
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

Visualize the schedule over 30 epochs:

```python
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Now train with this scheduler:

### MXNet

```python
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

### PyTorch

```python
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

### TensorFlow

```python
from tensorflow.keras.callbacks import LearningRateScheduler
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

**Result**: The square root scheduler produces smoother training curves and reduces overfitting compared to the constant baseline. The test accuracy improves as the decaying rate allows the model to settle into a better minimum.

## Step 6: Exploring Common Scheduling Policies

### 6.1 Factor Scheduler

This multiplies the learning rate by a factor `α` at each step until a minimum rate is reached: `ηₜ₊₁ = max(η_min, ηₜ × α)`.

```python
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

### 6.2 Multi-Factor Scheduler

This decreases the rate by a factor at predefined steps (e.g., halve at epochs 15 and 30). It's useful when you want to reduce the rate after validation accuracy plateaus.

#### MXNet Built-in

```python
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

#### PyTorch Built-in

```python
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)
```

#### TensorFlow Custom

```python
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr

    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
```

Training with a multi-factor scheduler often yields better solutions by aggressively reducing the rate only when necessary.

### 6.3 Cosine Scheduler

Proposed by Loshchilov & Hutter (2016), this schedule uses a cosine annealing pattern:

`ηₜ = η_T + (η₀ - η_T)/2 × (1 + cos(π × t/T))`

where `η₀` is the initial rate, `η_T` is the target rate at step `T`, and `t` is the current step.

```python
import math

class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Cosine schedules work particularly well for computer vision tasks, though results can vary.

### 6.4 Warmup

A warmup period gradually increases the learning rate from a small value to the initial maximum over a few epochs. This stabilizes training early on, especially for deep networks.

```python
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

Training with warmup often shows better initial convergence and overall stability.

## Summary

In this guide, you've learned:

1. **Why scheduling matters**: Learning rate decay improves accuracy and reduces overfitting by allowing the model to settle into sharper minima.
2. **How to implement schedulers**: From simple square root decay to sophisticated cosine annealing with warmup.
3. **Framework-specific nuances**: Each deep learning library provides built-in schedulers, but custom implementations are straightforward.
4. **Practical insights**: 
   - Multi-factor schedules are intuitive and effective.
   - Cosine schedules can yield better results for vision tasks.
   - Warmup periods prevent early divergence in deep networks.

## Next Steps

1. **Experiment with fixed rates**: Find the best constant rate for your problem as a baseline.
2. **Try polynomial decay**: Implement `η = η₀ × (t + 1)^{-β}` and vary `β`.
3. **Apply to larger problems**: Test cosine schedules on ImageNet or your own datasets.
4. **Tune warmup duration**: Systematically vary warmup length to see its impact.
5. **Explore connections to sampling**: Investigate Stochastic Gradient Langevin Dynamics, which links optimization to Bayesian sampling.

Learning rate scheduling is both an art and a science. While theory provides guidance, empirical testing on your specific problem is essential. The right schedule can be the difference between a good model and a great one.