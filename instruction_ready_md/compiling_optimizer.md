# Optimizing PyTorch Training: Compiling the Optimizer with `torch.compile`

**Author:** [Michael Lazos](https://github.com/mlazos)

The optimizer is a core component of any deep learning training loop, responsible for updating every model parameter. For large models, the optimizer step can become a significant performance bottleneck. This guide demonstrates how to use `torch.compile` to accelerate your optimizer, yielding measurable GPU performance gains.

> **Note:** This tutorial requires PyTorch 2.2.0 or later.

## 1. Prerequisites and Setup

First, ensure you have the necessary imports. We'll use a simple model for benchmarking, as optimizer performance scales with the number of parameters, not the model architecture.

```python
import torch
import torch.utils.benchmark as benchmark
```

## 2. Define a Simple Model

We'll create a sequence of 10 linear layers. This model has enough parameters to make optimizer performance measurable.

```python
# Create a model with a substantial number of parameters
model = torch.nn.Sequential(
    *[torch.nn.Linear(1024, 1024, False, device="cuda") for _ in range(10)]
)

# Create a dummy input and perform a forward/backward pass to generate gradients
input = torch.rand(1024, device="cuda")
output = model(input)
output.sum().backward()
```

## 3. Verify Hardware Compatibility

`torch.compile` requires a CUDA device with compute capability 7.0 or higher (e.g., Volta architecture or newer). Let's add a check to exit gracefully if this condition isn't met.

```python
# Exit cleanly if torch.compile is not supported on this device
if torch.cuda.get_device_capability() < (7, 0):
    print("Exiting because torch.compile is not supported on this device.")
    import sys
    sys.exit(0)
```

## 4. Initialize the Optimizer and Compile Its Step

We'll use the Adam optimizer. The key step is wrapping the optimizer's `step()` function with `torch.compile`.

```python
# Initialize the Adam optimizer
opt = torch.optim.Adam(model.parameters(), lr=0.01)

# Compile the optimizer step function
@torch.compile(fullgraph=False)
def compiled_step():
    opt.step()
```

## 5. Create a Benchmarking Utility

To accurately compare performance, we define a helper function that measures the execution time of a callable in microseconds.

```python
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    """Benchmark a function and return the average runtime in microseconds."""
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6
```

## 6. Warm Up and Run the Benchmark

Compilation happens on the first few runs. We execute several warm-up calls to ensure the compiled function is ready, then benchmark both the eager (standard) and compiled versions.

```python
# Warm-up runs to trigger compilation
for _ in range(5):
    compiled_step()

# Benchmark the standard (eager) optimizer step
eager_runtime = benchmark_torch_function_in_microseconds(opt.step)

# Benchmark the compiled optimizer step
compiled_runtime = benchmark_torch_function_in_microseconds(compiled_step)

# Verify the compiled version is faster
assert eager_runtime > compiled_runtime

print(f"Eager runtime: {eager_runtime:.2f} µs")
print(f"Compiled runtime: {compiled_runtime:.2f} µs")
```

## 7. Expected Results

Your exact timings will vary based on your hardware, but you should observe a significant speedup. For example, on a compatible GPU, you might see results like:

*   **Eager runtime:** 747.24 µs
*   **Compiled runtime:** 392.07 µs

This represents a performance improvement of nearly 2x for the optimizer step.

## Summary

By compiling the optimizer with `torch.compile`, you can reduce the time spent updating model parameters, which is especially beneficial for training large models. The process involves:
1.  Ensuring hardware compatibility.
2.  Wrapping the optimizer's `step()` method in a `@torch.compile` decorator.
3.  Performing warm-up runs to allow PyTorch to compile the computation graph.
4.  Profiting from reduced step time throughout the rest of training.

## Further Reading

For an in-depth technical overview of how PyTorch compiles the optimizer, see the article [Compiling the optimizer with PT2](https://dev-discuss.pytorch.org/t/compiling-the-optimizer-with-pt2/1669).