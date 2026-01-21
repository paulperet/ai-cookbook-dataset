# Debugging Distributed Training with PyTorch's `CommDebugMode`

**Author**: [Anshul Sinha](https://github.com/sinhaanshul)

This guide walks you through using `CommDebugMode`, a powerful debugging tool for PyTorch's DistributedTensor (DTensor). You'll learn how to track and visualize collective communication operations in distributed training workflows.

## Prerequisites

-   Python 3.8 - 3.11
-   PyTorch 2.2 or later

## Understanding `CommDebugMode`

As models grow larger, developers combine various parallel strategies (like Tensor Parallelism) for distributed training. PyTorch's **DistributedTensor (DTensor)** provides a unified abstraction to manage tensor communication across these strategies, simplifying the user experience.

However, when integrating DTensor with existing code or developing new parallel solutions, it can be difficult to see *when* and *why* collective operations (like `all_reduce`) happen under the hood. This lack of transparency makes debugging performance issues or errors challenging.

`CommDebugMode` solves this. It's a Python context manager that acts as a debugger for DTensors, allowing you to trace and log every collective communication operation during execution.

## Tutorial: Using `CommDebugMode`

Follow these steps to instrument your DTensor-based code.

### Step 1: Import and Initialize

First, ensure you have the necessary import. The mode is typically used within a distributed training script.

```python
from torch.distributed.tensor.debug import CommDebugMode
```

### Step 2: Wrap Your Model Execution

Instantiate `CommDebugMode` and use it as a context manager around the part of your code you want to debug (e.g., a forward pass).

```python
# Assume `model` is your DTensor-based module and `inp` is your input
comm_mode = CommDebugMode()

with comm_mode:
    output = model(inp)
```

While inside the `with` block, `CommDebugMode` silently records all communication operations.

### Step 3: Analyze the Results

After execution, you can inspect the collected data in several ways, controlling verbosity with the `noise_level` parameter.

**Option A: Print a Summary Table to Console**
Print a hierarchical view of communication counts per module.

```python
print(comm_mode.generate_comm_debug_tracing_table(noise_level=0))
```

**Option B: Log Detailed Information to a File**
Write more detailed operation logs, including DTensor operations and module sharding info, to a text file.

```python
comm_mode.log_comm_debug_tracing_table_to_file(
    noise_level=1,
    file_name="transformer_operation_log.txt"
)
```

**Option C: Generate a JSON Dump for Visualization**
Create a JSON file that can be loaded into an interactive visualizer (see Step 4).

```python
comm_mode.generate_json_dump(noise_level=2)
```

### Understanding Noise Levels

The `noise_level` argument filters the amount of detail:

| Noise Level | Output Includes                                                                 |
| :---------- | :------------------------------------------------------------------------------ |
| **0**       | Module-level collective operation counts.                                       |
| **1**       | DTensor operations (excluding trivial ops) and module sharding information.     |
| **2**       | Tensor operations (excluding trivial ops).                                      |
| **3**       | **All** operations.                                                            |

### Step 4: Interpret the Output

When you run the print command with `noise_level=0` on a simple MLP model using Tensor Parallelism, you might see output like this:

```
Global
  FORWARD PASS
    *c10d_functional.all_reduce: 1
    MLPModule
      FORWARD PASS
        *c10d_functional.all_reduce: 1
        MLPModule.net1
        MLPModule.relu
        MLPModule.net2
          FORWARD PASS
            *c10d_functional.all_reduce: 1
```

This tells you:
-   An `all_reduce` collective happened once in the global forward pass.
-   Drilling down, that collective occurred within the `MLPModule`.
-   Further inspection shows the collective specifically happened in the `MLPModule.net2` (the second linear layer).

This hierarchical pinpointing is the core strength of `CommDebugMode`.

### Step 5: Visualize with the Interactive Browser

For complex models, the JSON dump (`noise_level=2`) can be loaded into an interactive HTML visualizer to navigate the module tree and communication events graphically. Refer to the official PyTorch examples for the visualizer script and usage.

## Conclusion

You've learned how to use `CommDebugMode` to gain visibility into the collective communication operations within your PyTorch DTensor applications. This is essential for debugging performance, verifying correctness, and understanding the behavior of distributed training setups.

**Next Steps:**
-   Experiment with different `noise_level` settings on your own models.
-   Generate JSON dumps and explore the interactive visualization.
-   Review the [detailed example file](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/examples/comm_mode_features_example.py) for advanced use cases.