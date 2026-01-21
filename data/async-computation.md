# Asynchronous Computation in Deep Learning Frameworks

## Introduction

Modern computers are highly parallel systems with multiple CPU cores, GPU processing elements, and often multiple GPUs per device. While Python is single-threaded and not ideal for parallel programming, deep learning frameworks like MXNet and PyTorch implement asynchronous programming models to improve performance. Understanding this model helps you write more efficient programs by reducing computational dependencies and improving processor utilization.

## Prerequisites

First, let's set up our environment with the necessary imports.

```python
# For MXNet users
from d2l import mxnet as d2l
import numpy as np
import os
import subprocess
from mxnet import np, npx
npx.set_np()

# For PyTorch users
from d2l import torch as d2l
import numpy as np
import torch
```

## Understanding Asynchrony via Backend

### Benchmarking Synchronous vs. Asynchronous Execution

Let's start with a simple benchmark to understand the performance difference between synchronous (NumPy) and asynchronous (framework) execution.

```python
# For MXNet
with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)

with d2l.Benchmark('mxnet.np'):
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
```

```python
# For PyTorch
device = d2l.try_gpu()
# Warmup for GPU computation
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
```

You'll notice the framework-based operations are significantly faster. This is because operations are enqueued to the device and executed asynchronously, allowing the Python frontend to continue while the backend processes computations.

### Forcing Synchronization

To see what's happening behind the scenes, let's force the backend to complete all operations before returning control.

```python
# For MXNet
with d2l.Benchmark():
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
    npx.waitall()
```

```python
# For PyTorch
with d2l.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)
```

The frameworks have a frontend (Python) that interacts with users and a backend (C++) that executes computations. Operations from the frontend are queued in the backend, which manages threads to execute tasks while tracking dependencies in the computational graph.

## Understanding Dependency Graphs

Let's examine how the backend tracks dependencies between operations.

```python
# For MXNet
x = np.ones((1, 2))
y = np.ones((1, 2))
z = x * y + 2
print(z)
```

```python
# For PyTorch
x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
print(z)
```

When you execute these statements, the Python frontend queues each operation (`x`, `y`, and the computation of `z`) to the backend. Only when you need to print `z` does the frontend wait for the backend to complete the computation. This design minimizes the performance impact of Python's single-threaded nature.

## Barriers and Blockers

Certain operations force Python to wait for completion, which can impact performance if used carelessly.

### Explicit Barriers

```python
# For MXNet only
with d2l.Benchmark('waitall'):
    b = np.dot(a, a)
    npx.waitall()

with d2l.Benchmark('wait_to_read'):
    b = np.dot(a, a)
    b.wait_to_read()
```

Both operations take similar time, but `wait_to_read()` is more specific, blocking only until a particular variable is available while allowing other computations to continue.

### Implicit Blockers

Several common operations are implicit blockers:
- Printing variables
- Converting to NumPy via `asnumpy()` (MXNet) or `numpy()` (PyTorch)
- Converting to scalars via `item()`

```python
# For MXNet
with d2l.Benchmark('numpy conversion'):
    b = np.dot(a, a)
    b.asnumpy()

with d2l.Benchmark('scalar conversion'):
    b = np.dot(a, a)
    b.sum().item()
```

Frequent data copying between framework and NumPy can destroy performance, as each conversion requires evaluating all intermediate results in the computational graph.

## Improving Computation with Asynchrony

Let's demonstrate the benefits of asynchronous execution by incrementing a variable multiple times.

```python
# For MXNet
with d2l.Benchmark('synchronous'):
    for _ in range(10000):
        y = x + 1
        y.wait_to_read()

with d2l.Benchmark('asynchronous'):
    for _ in range(10000):
        y = x + 1
    npx.waitall()
```

The asynchronous version is significantly faster because:
1. The frontend queues all 10,000 operations quickly
2. The backend processes them in parallel
3. Only one final synchronization is needed

### Performance Analysis

Consider three stages of computation:
1. **Frontend queuing** ($t_1$): Time to insert task into queue
2. **Backend execution** ($t_2$): Time to perform actual computation
3. **Result return** ($t_3$): Time to return results to frontend

- **Synchronous**: Total time ≈ $10000 (t_1 + t_2 + t_3)$
- **Asynchronous**: Total time ≈ $t_1 + 10000 t_2 + t_3$ (assuming $10000 t_2 > 9999 t_1$)

The asynchronous approach eliminates most of the frontend waiting time, leading to better performance when you have many independent operations.

## Summary

1. Deep learning frameworks decouple Python frontends from execution backends, enabling fast asynchronous command insertion and parallelism.
2. Asynchrony creates a responsive frontend, but avoid overfilling the task queue to prevent excessive memory consumption. Synchronize periodically (e.g., per minibatch) to keep frontend and backend aligned.
3. Be mindful of implicit blockers like `print()`, `asnumpy()`, and `item()` that force synchronization and can impact performance.
4. Use vendor performance analysis tools for fine-grained efficiency insights.

## Exercises

### For MXNet Users
1. Why do we assume $10000 t_2 > 9999 t_1$ in the asynchronous performance analysis?
2. Measure the practical difference between `waitall()` and `wait_to_read()` by performing multiple operations with intermediate synchronization.

### For PyTorch Users
1. Benchmark the same matrix multiplication operations on CPU. Can you still observe backend asynchrony?

## Further Discussion

- MXNet: [Discuss on D2L](https://discuss.d2l.ai/t/361)
- PyTorch: [Discuss on D2L](https://discuss.d2l.ai/t/2564)