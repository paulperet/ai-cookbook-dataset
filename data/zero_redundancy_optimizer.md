# PyTorch Cookbook: Shard Optimizer States with ZeroRedundancyOptimizer

## Overview
This guide demonstrates how to use PyTorch's `ZeroRedundancyOptimizer` (ZeRO-1) to reduce memory consumption in distributed training by sharding optimizer states across processes. You'll learn the core concept and implement a working example comparing memory usage with and without this optimization.

## Prerequisites
- PyTorch 1.8 or later
- Basic familiarity with Distributed Data Parallel (DDP) training
- A system with multiple GPUs (or CPU for demonstration)

## Understanding ZeroRedundancyOptimizer
Traditional Distributed Data Parallel (DDP) training maintains a complete copy of the optimizer and its states on every process. For optimizers like Adam, which store per-parameter statistics (`exp_avg` and `exp_avg_sq`), this doubles the memory footprint beyond the model parameters themselves.

`ZeroRedundancyOptimizer` solves this by:
1. Sharding optimizer states across DDP processes
2. Having each process update only its assigned parameter shard
3. Broadcasting updated parameters to keep all model replicas synchronized

This approach can significantly reduce per-process memory consumption while maintaining identical model states across processes.

## Implementation Guide

### Step 1: Import Required Libraries
```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
```

### Step 2: Create Memory Monitoring Utility
Add this helper function to track GPU memory usage:

```python
def print_peak_memory(prefix, device):
    """Print peak memory usage for the specified device."""
    if device == 0:  # Only print from rank 0 to avoid clutter
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB")
```

### Step 3: Define the Training Example Function
This function sets up distributed training with an option to use `ZeroRedundancyOptimizer`:

```python
def example(rank, world_size, use_zero):
    """Run a single training step with optional ZeroRedundancyOptimizer."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # Initialize distributed process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # Create a simple model (20 linear layers)
    model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(20)])
    print_peak_memory("Max memory allocated after creating local model", rank)
    
    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[rank])
    print_peak_memory("Max memory allocated after creating DDP", rank)
    
    # Define loss function
    loss_fn = nn.MSELoss()
    
    # Choose optimizer based on use_zero flag
    if use_zero:
        optimizer = ZeroRedundancyOptimizer(
            ddp_model.parameters(),
            optimizer_class=torch.optim.Adam,
            lr=0.01
        )
    else:
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.01)
    
    # Forward pass
    outputs = ddp_model(torch.randn(20, 2000).to(rank))
    labels = torch.randn(20, 2000).to(rank)
    
    # Backward pass
    loss_fn(outputs, labels).backward()
    
    # Update parameters and measure memory
    print_peak_memory("Max memory allocated before optimizer step()", rank)
    optimizer.step()
    print_peak_memory("Max memory allocated after optimizer step()", rank)
    
    # Verify parameter consistency
    print(f"params sum is: {sum(model.parameters()).sum()}")
```

### Step 4: Create Main Execution Function
This function runs the example twice: once with `ZeroRedundancyOptimizer` and once without:

```python
def main():
    world_size = 2  # Using 2 processes for demonstration
    
    print("=== Using ZeroRedundancyOptimizer ===")
    mp.spawn(
        example,
        args=(world_size, True),
        nprocs=world_size,
        join=True
    )
    
    print("\n=== Not Using ZeroRedundancyOptimizer ===")
    mp.spawn(
        example,
        args=(world_size, False),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
```

## Expected Output
When you run this code, you should see output similar to:

```
=== Using ZeroRedundancyOptimizer ===
Max memory allocated after creating local model: 335.0MB
Max memory allocated after creating DDP: 656.0MB
Max memory allocated before optimizer step(): 992.0MB
Max memory allocated after optimizer step(): 1361.0MB
params sum is: -3453.6123046875
params sum is: -3453.6123046875

=== Not Using ZeroRedundancyOptimizer ===
Max memory allocated after creating local model: 335.0MB
Max memory allocated after creating DDP: 656.0MB
Max memory allocated before optimizer step(): 992.0MB
Max memory allocated after optimizer step(): 1697.0MB
params sum is: -3453.6123046875
params sum is: -3453.6123046875
```

## Key Observations

1. **Memory Reduction**: With `ZeroRedundancyOptimizer`, the peak memory after `optimizer.step()` is 1361MB compared to 1697MB without it - approximately a 20% reduction with 2 processes.

2. **Mathematical Equivalence**: Both approaches produce identical parameter sums (-3453.6123046875), confirming that `ZeroRedundancyOptimizer` maintains numerical correctness while reducing memory.

3. **Scalability**: The memory savings increase with more processes, as optimizer states are sharded across all participating ranks.

## Best Practices

1. **Process Group Compatibility**: Ensure your distributed setup is properly configured before initializing `ZeroRedundancyOptimizer`.

2. **Optimizer Selection**: `ZeroRedundancyOptimizer` works with any standard PyTorch optimizer via the `optimizer_class` parameter.

3. **Mixed Precision Training**: Combine with techniques like AMP (Automatic Mixed Precision) for additional memory savings.

4. **Monitoring**: Always verify that model parameters remain synchronized across processes when using sharded optimizers.

## Troubleshooting

- **Connection Issues**: If you encounter "Connection refused" errors, ensure no other process is using port 29500.
- **CUDA Errors**: For GPU training, replace `"gloo"` with `"nccl"` in `init_process_group()`.
- **Memory Not Reduced**: Verify that your optimizer has significant states (like Adam) to benefit from sharding.

## Next Steps
Explore more advanced memory optimization techniques:
- ZeRO-2 (gradient sharding)
- ZeRO-3 (parameter sharding)
- Activation checkpointing
- Model parallelism for extremely large models

This implementation provides a foundation for memory-efficient distributed training that scales to larger models and datasets.