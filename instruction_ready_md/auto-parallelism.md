# Guide: Understanding and Leveraging Automatic Parallelism

## Overview

This guide explores how deep learning frameworks like MXNet and PyTorch automatically parallelize computation. By leveraging computational graphs, these systems identify and execute independent tasks concurrently, improving performance—especially across multiple devices like CPUs and GPUs.

**Prerequisites:**
- Basic familiarity with deep learning concepts
- At least two GPUs to run the experiments
- MXNet or PyTorch installed

## Setup

First, import the necessary libraries. The code is shown for both MXNet and PyTorch; use the version corresponding to your framework.

```python
# For MXNet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```python
# For PyTorch
from d2l import torch as d2l
import torch
```

## 1. Parallel Computation on GPUs

We'll start by defining a benchmark workload: a function that performs 50 matrix-matrix multiplications. This helps us observe how the framework handles parallel execution.

### Step 1: Define the Workload and Data

Create a function `run` and allocate large matrices on two separate GPUs.

```python
# Get available GPUs
devices = d2l.try_all_gpus()

# Define the workload
def run(x):
    return [x.dot(x) for _ in range(50)]  # MXNet
    # return [x.mm(x) for _ in range(50)]  # PyTorch

# Allocate data on two different GPUs
x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])  # MXNet
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])

# For PyTorch:
# x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
# x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```

### Step 2: Warm Up and Benchmark Sequential Execution

To ensure accurate timing, warm up the GPUs by running the function once on each device. Then, benchmark the execution time for each GPU sequentially.

```python
# Warm-up
run(x_gpu1)
run(x_gpu2)

# Synchronize devices (prevents overlapping execution)
npx.waitall()  # MXNet
# torch.cuda.synchronize(devices[0])  # PyTorch
# torch.cuda.synchronize(devices[1])

# Benchmark GPU1
with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    npx.waitall()  # MXNet
    # torch.cuda.synchronize(devices[0])  # PyTorch

# Benchmark GPU2
with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    npx.waitall()  # MXNet
    # torch.cuda.synchronize(devices[1])  # PyTorch
```

### Step 3: Enable Automatic Parallelism

Now, remove the synchronization between the two tasks. This allows the framework to schedule both computations concurrently.

```python
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()  # MXNet
    # torch.cuda.synchronize()  # PyTorch
```

You should observe that the total execution time is less than the sum of the individual GPU times, demonstrating automatic parallelization.

## 2. Parallel Computation and Communication

In distributed scenarios, data often needs to move between devices (e.g., from GPU to CPU). We can overlap computation and communication to improve efficiency.

### Step 1: Define a Data Transfer Function

Create a helper function to copy data from GPU to CPU.

```python
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]  # MXNet
    # return [y.to('cpu', non_blocking=False) for y in x]  # PyTorch
```

### Step 2: Benchmark Sequential Computation and Transfer

First, run the computation on GPU1, then copy the results to CPU.

```python
with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    npx.waitall()  # MXNet
    # torch.cuda.synchronize()  # PyTorch

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()  # MXNet
    # torch.cuda.synchronize()  # PyTorch
```

### Step 3: Overlap Computation and Communication

By removing synchronization (or using non-blocking transfers in PyTorch), the system can start copying data while the GPU is still computing.

```python
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)  # MXNet
    # y_cpu = copy_to_cpu(y, non_blocking=True)  # PyTorch
    npx.waitall()  # MXNet
    # torch.cuda.synchronize()  # PyTorch
```

The total time should be reduced, as the framework overlaps computation on the GPU with data transfer over the PCI-Express bus.

## Key Insights

- **Automatic Parallelization:** Deep learning frameworks automatically identify and execute independent tasks in parallel across multiple GPUs.
- **Overlapping Operations:** Computation and communication can be overlapped to hide latency, especially when transferring data between CPU and GPU.
- **Resource Utilization:** Efficient use of multiple devices (CPUs, GPUs) and interconnects (PCIe) is crucial for maximizing performance in complex workflows like distributed training.

## Exercises

1. **Explore Operator-Level Parallelism:** The `run` function performs eight independent matrix multiplications. Modify the experiment to see if the framework executes them in parallel on a single GPU.
2. **Fine-Grained Parallelism:** Design a workload with many small, independent operations to test if parallelization benefits even a single CPU or GPU.
3. **Multi-Device Workflow:** Create a task that uses both CPUs and GPUs simultaneously, including data transfers between them.
4. **Profiling:** Use a tool like NVIDIA Nsight to profile your code and verify that parallelism and overlaps are occurring as expected.
5. **Complex Dependencies:** Implement a computation with more complex data dependencies (e.g., a small neural network) and experiment with manual vs. automatic scheduling.

## Further Reading

- MXNet Discussion: [https://discuss.d2l.ai/t/362](https://discuss.d2l.ai/t/362)
- PyTorch Discussion: [https://discuss.d2l.ai/t/1681](https://discuss.d2l.ai/t/1681)