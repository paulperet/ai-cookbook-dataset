# PyTorch Distributed: A Practical Guide

## Introduction

Welcome to this practical guide on PyTorch's distributed training capabilities. Whether you're scaling up from single-GPU training or building large-scale distributed systems, this guide will help you navigate PyTorch's distributed ecosystem and choose the right tools for your needs.

PyTorch Distributed provides a comprehensive suite of tools for parallel training across multiple GPUs and nodes. The library includes:
- **Parallelism APIs** for different scaling strategies
- **Sharding primitives** for fine-grained tensor distribution
- **Communication APIs** for low-level control
- **Launch utilities** for managing distributed processes

## Core Components

### 1. Parallelism Modules

These high-level APIs integrate seamlessly with your existing PyTorch models:

| Module | Purpose | Best For |
|--------|---------|----------|
| **Distributed Data-Parallel (DDP)** | Replicates model across GPUs, synchronizes gradients | Models that fit on a single GPU |
| **Fully Sharded Data-Parallel (FSDP2)** | Shards model parameters, gradients, and optimizer states | Models too large for a single GPU |
| **Tensor Parallel (TP)** | Splits individual tensor operations across devices | Very large models with specific layer types |
| **Pipeline Parallel (PP)** | Splits model layers across devices in sequence | Models with sequential dependencies |

### 2. Sharding Primitives

For building custom parallelism strategies:

- **DTensor**: Represents tensors that are sharded and/or replicated across devices. Automatically handles communication for tensor operations.
- **DeviceMesh**: Manages multi-dimensional process groups for collective communications in complex parallel setups.

### 3. Communication Layer (C10D)

The underlying communication engine provides both collective and point-to-point operations:

```python
# Collective operations
torch.distributed.all_reduce(tensor)  # Sum across all processes
torch.distributed.all_gather(tensor_list, tensor)  # Gather from all processes

# Point-to-point operations
torch.distributed.send(tensor, dst)  # Blocking send
torch.distributed.isend(tensor, dst)  # Non-blocking send
```

### 4. Process Launcher

**torchrun** is the recommended launcher for distributed PyTorch applications. It handles process spawning across local and remote machines.

## Choosing Your Parallelism Strategy

Follow this decision tree to select the right approach for your model:

### Step 1: Assess Your Model Size

**If your model fits in a single GPU:**
- Use **Distributed Data-Parallel (DDP)** for straightforward multi-GPU scaling
- Launch with `torchrun` for multi-node setups
- Works seamlessly with Automatic Mixed Precision (AMP)

**Example DDP setup:**
```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')
model = DDP(model)
```

### Step 2: Handle Large Models

**If your model doesn't fit in a single GPU:**
- Use **Fully Sharded Data-Parallel (FSDP2)** as your first choice
- FSDP2 shards parameters, gradients, and optimizer states across devices
- Provides near-linear scaling for very large models

**Example FSDP2 setup:**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
)
```

### Step 3: Scale Beyond FSDP2

**If you reach scaling limits with FSDP2:**
- Combine **Tensor Parallel (TP)** for intra-layer parallelism
- Add **Pipeline Parallel (PP)** for inter-layer parallelism
- Use **3D Parallelism** (DDP + TP + PP) for maximum scaling

**Example 3D parallelism pattern:**
```python
# This is a conceptual example - actual implementation
# depends on your specific model architecture

# 1. Use TP for attention layers within each device
# 2. Use PP to split transformer blocks across devices
# 3. Use DDP for data parallelism across model replicas
```

## Practical Recommendations

### Getting Started
1. **Start with DDP** if your model fits on one GPU
2. **Move to FSDP2** when you encounter memory limitations
3. **Explore TP/PP** only when necessary for extreme scaling

### Development Tips
- Use `torchrun` for all distributed launches
- Test with small batch sizes first
- Monitor GPU memory utilization
- Profile communication overhead

### Common Pitfalls
- **Synchronization issues**: Ensure all processes reach collective operations
- **Memory fragmentation**: FSDP2 can help reduce peak memory usage
- **Communication bottlenecks**: Profile and optimize data transfer

## Next Steps

Ready to implement? Here are practical tutorials to follow:

1. **Beginner**: [Getting Started with Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
2. **Intermediate**: [Getting Started with FSDP2](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
3. **Advanced**: [Tensor Parallelism Tutorial](https://pytorch.org/tutorials/intermediate/TP_tutorial.html)
4. **Production**: [TorchTitan end-to-end 3D parallelism example](https://github.com/pytorch/torchtitan)

## Contributing

Interested in contributing to PyTorch Distributed? Check out the [Developer Guide](https://github.com/pytorch/pytorch/blob/master/torch/distributed/CONTRIBUTING.md) for contribution guidelines and development practices.

---

Remember: The best parallelism strategy depends on your specific model architecture, hardware configuration, and scaling requirements. Start simple, profile thoroughly, and scale incrementally. Happy distributed training!