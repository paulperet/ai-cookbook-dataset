# Object-Oriented Design for Deep Learning Implementation

## Overview

This guide introduces a modular, object-oriented design pattern for implementing deep learning workflows. Inspired by frameworks like PyTorch Lightning, we structure code into three core components: **Models**, **Data**, and **Training**. This design promotes code reuse, clarity, and scalability across projects.

## Prerequisites

First, ensure you have the necessary imports. The code supports multiple backends (MXNet, PyTorch, TensorFlow, JAX). Select your framework by setting the `tab` variable appropriately.

```python
import time
import numpy as np

# Framework-specific imports
# For PyTorch:
from d2l import torch as d2l
import torch
from torch import nn

# For other frameworks, import the corresponding d2l module and libraries.
```

## Step 1: Core Utilities

We begin with utility functions and classes that enable flexible class design, especially useful in interactive environments like Jupyter notebooks.

### 1.1 Dynamically Adding Methods to Classes

The `add_to_class` decorator allows you to register a function as a method of a class *after* the class has been instantiated.

```python
def add_to_class(Class):
    """Register functions as methods in a created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

**Example Usage:**

```python
class A:
    def __init__(self):
        self.b = 1

a = A()  # Create an instance

# Define a function and add it to class A
@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

# Now the instance can use the new method
a.do()
```

### 1.2 Managing Hyperparameters

The `HyperParameters` base class (fully implemented in the D2L library) automatically saves constructor arguments as instance attributes, simplifying class initialization.

```python
# Using the D2L library's implementation
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
```

### 1.3 Tracking Training Progress

The `ProgressBoard` class provides a simple interface for plotting metrics during training. Its full implementation is in the D2L library.

```python
class ProgressBoard(d2l.HyperParameters):
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented  # Implementation in D2L
```

**Example Usage:**

```python
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```

## Step 2: Defining the Model (`Module`)

The `Module` class is the foundation for all neural network models. It handles the model definition, loss computation, and optimizer configuration.

### 2.1 Base Module Structure

Here is the base `Module` class, adapted for different frameworks. It inherits from the framework's native module class (e.g., `nn.Module` in PyTorch) and `HyperParameters`.

```python
class Module(d2l.nn_Module, d2l.HyperParameters):
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / self.trainer.num_train_batches
            n = self.trainer.num_train_batches / self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / self.plot_valid_per_epoch
        self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

**Key Components:**
-   `loss`: Must be implemented by subclasses to compute the loss.
-   `forward`: Defines the forward pass of the network.
-   `training_step`/`validation_step`: Called by the trainer for each batch.
-   `configure_optimizers`: Returns the optimizer(s) for the model.
-   `plot`: Helper to send metrics to the `ProgressBoard`.

**Framework Notes:**
-   In **PyTorch** and **MXNet**, `Module` subclasses `nn.Module` and `nn.Block` respectively, gaining their functionality.
-   In **TensorFlow**, it subclasses `tf.keras.Model`.
-   In **JAX/Flax**, it uses a dataclass structure and handles parameter gradients explicitly in `training_step`.

## Step 3: Managing Data (`DataModule`)

The `DataModule` class standardizes data loading for training and validation.

```python
class DataModule(d2l.HyperParameters):
    """The base class of data."""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```

**How to Use:**
1.  Subclass `DataModule`.
2.  Implement `get_dataloader` to return a data loader (a Python generator yielding batches) for either the training or validation set.
3.  The `train_dataloader` and `val_dataloader` methods will provide the correct loader to the trainer.

## Step 4: The Training Loop (`Trainer`)

The `Trainer` class orchestrates the training process, connecting a `Module` with a `DataModule`.

```python
class Trainer(d2l.HyperParameters):
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
```

**Workflow:**
1.  **Initialization**: Set maximum epochs and hardware configuration.
2.  `prepare_data`: Fetches data loaders from the `DataModule`.
3.  `prepare_model`: Links the model to this trainer and sets up plotting.
4.  `fit`: The main training loop. It calls `configure_optimizers` from the model and iterates for `max_epochs`.
5.  `fit_epoch`: This method must be implemented to define what happens in a single epoch (e.g., iterating over batches, computing loss, applying gradients).

**JAX/Flax Note:** The JAX version has an extended `fit` method to handle explicit parameter states and random key management, which is idiomatic in JAX.

## Summary

This object-oriented design provides a clean separation of concerns:
-   **`Module`** encapsulates the model, loss, and optimizer logic.
-   **`DataModule`** handles data loading and preparation.
-   **`Trainer`** orchestrates the training loop.

This modular approach, implemented in the [D2L library](https://github.com/d2l-ai/d2l-en), promotes code reuse and makes it easy to swap components (e.g., try a different optimizer or dataset) with minimal changes. As you progress, you will enrich these base classes to build complex deep learning systems.

## Exercises

1.  Explore the full implementations of `HyperParameters`, `ProgressBoard`, and `Trainer.fit_epoch` in the D2L library. Understanding these details will deepen your grasp of the framework's mechanics.
2.  In the `B` class example, remove the `self.save_hyperparameters(ignore=['c'])` line. What happens when you try to print `self.a`? Investigate the `HyperParameters` source code in D2L to understand how `save_hyperparameters` works.