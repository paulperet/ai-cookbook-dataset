# Context Parallel Tutorial: Scaling LLM Training to Long Sequences

## Overview

Context Parallel is a distributed training technique that enables training large language models with extremely long input sequences by sharding activations across multiple devices. This tutorial demonstrates how to use PyTorch's Context Parallel APIs to parallelize Scaled Dot Product Attention (SDPA) computation along the sequence dimension.

### What You'll Learn
- How to use Context Parallel APIs to shard tensors and enable Ring Attention
- How to choose between different Ring Attention implementations
- How to verify numerical correctness in distributed settings

### Prerequisites
- PyTorch 2.7 or later
- CUDA-capable GPUs with NCCL support
- Basic understanding of distributed training in PyTorch

## Understanding Context Parallel

Context Parallel addresses the memory bottleneck in training Transformers with long sequences by distributing the sequence dimension across multiple devices. The key innovation is Ring Attention, which comes in two variants:

1. **All-Gather Based Pass-KV**: Used in Llama3 training, this approach overlaps attention computation with KV tensor gathering
2. **All-to-All Based Pass-KV**: Uses interleaved all-to-all collectives to shuffle KV shards between devices

Both approaches allow you to train models with sequences much longer than what fits on a single GPU's memory.

## Setup and Imports

First, let's set up the basic imports and verify our environment:

```python
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import (
    context_parallel_unshard,
    set_rotate_method
)
from torch.nn.attention import sdpa_kernel, SDPBackend
```

## Step 1: Baseline Single-GPU SDPA

Let's start with a simple single-GPU SDPA implementation to establish our baseline:

```python
def sdpa_example():
    """Baseline SDPA implementation on a single GPU."""
    assert torch.cuda.is_available()
    torch.cuda.set_device("cuda:0")
    torch.cuda.manual_seed(0)

    # Configuration
    batch = 8
    nheads = 8
    qkv_len = 8192
    dim = 32
    backend = SDPBackend.FLASH_ATTENTION
    
    # Select appropriate dtype for the backend
    dtype = (
        torch.bfloat16
        if backend == SDPBackend.FLASH_ATTENTION
        or backend == SDPBackend.CUDNN_ATTENTION
        else torch.float32
    )

    # Create query, key, value tensors
    qkv = [
        torch.rand(
            (batch, nheads, qkv_len, dim),
            dtype=dtype,
            requires_grad=True,
            device='cuda',
        )
        for _ in range(3)
    ]
    
    # Execute SDPA with specified backend
    with sdpa_kernel(backend):
        out = F.scaled_dot_product_attention(*qkv, is_causal=True)
    
    return out, qkv
```

This function creates random query, key, and value tensors and computes attention using the Flash Attention backend. We'll use this as our reference for numerical correctness.

## Step 2: Distributed Setup with Context Parallel

Now, let's adapt this to a distributed setting with Context Parallel. The key changes are:
1. Initialize distributed process group
2. Create a device mesh for context parallel dimension
3. Use `context_parallel()` to shard tensors and enable Ring Attention

```python
def context_parallel_sdpa_example(world_size: int, rank: int):
    """Distributed SDPA implementation using Context Parallel."""
    assert torch.cuda.is_available()
    assert dist.is_nccl_available()
    
    # Set up device and random seed
    torch.cuda.set_device(f"cuda:{rank}")
    torch.cuda.manual_seed(0)
    
    # Initialize distributed process group
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    
    # Create device mesh for context parallel dimension
    device_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("cp",)  # "cp" stands for context parallel
    )
    
    # Configuration (using smaller sequence for demonstration)
    batch = 8
    nheads = 8
    qkv_len = 64  # Smaller sequence for demonstration
    dim = 32
    backend = SDPBackend.FLASH_ATTENTION
    
    # Select appropriate dtype
    dtype = (
        torch.bfloat16
        if backend == SDPBackend.FLASH_ATTENTION
        or backend == SDPBackend.CUDNN_ATTENTION
        else torch.float32
    )
    
    # Create query, key, value tensors
    qkv = [
        torch.rand(
            (batch, nheads, qkv_len, dim),
            dtype=dtype,
            requires_grad=True,
            device='cuda',
        )
        for _ in range(3)
    ]
    
    # Step 1: Compute baseline SDPA on each rank
    with sdpa_kernel(backend):
        out = F.scaled_dot_product_attention(*qkv, is_causal=True)
    
    # Step 2: Create copies for Context Parallel execution
    cp_qkv = [t.detach().clone() for t in qkv]
    
    # Step 3: Execute with Context Parallel
    with sdpa_kernel(backend):
        # The context_parallel() context manager performs two actions:
        # 1. Shards the tensors in `buffers` along dimensions specified in `buffer_seq_dims`
        # 2. Replaces SDPA execution with Ring Attention
        with context_parallel(
            device_mesh,
            buffers=tuple(cp_qkv),
            buffer_seq_dims=(2, 2, 2)  # Shard along sequence dimension (dimension 2)
        ):
            cp_out = F.scaled_dot_product_attention(*cp_qkv, is_causal=True)
        
        # The output is still sharded - unshard it for comparison
        (cp_out,) = context_parallel_unshard(device_mesh, [cp_out], [2])
    
    # Verify numerical correctness
    # Note: We use relaxed tolerance for bfloat16 due to precision differences
    tolerance = 1e-08 if dtype == torch.float32 else 1e-03 * world_size
    assert torch.allclose(cp_out, out, atol=tolerance), \
        "Context Parallel output doesn't match baseline!"
    
    print(f"Rank {rank}: Context Parallel verification passed!")
    
    return cp_out
```

## Step 3: Choosing the Rotation Method

Context Parallel supports two different Ring Attention implementations. You can choose between them using `set_rotate_method()`:

```python
def context_parallel_with_rotation_method(world_size: int, rank: int, method: str = "alltoall"):
    """Demonstrate different rotation methods for Ring Attention."""
    # ... (same setup as previous function)
    
    # Set the desired rotation method
    set_rotate_method(method)  # Options: "alltoall" or "allgather" (default)
    
    # Create copies for Context Parallel execution
    cp_qkv = [t.detach().clone() for t in qkv]
    
    # Execute with chosen rotation method
    with sdpa_kernel(backend):
        with context_parallel(
            device_mesh,
            buffers=tuple(cp_qkv),
            buffer_seq_dims=(2, 2, 2)
        ):
            cp_out = F.scaled_dot_product_attention(*cp_qkv, is_causal=True)
        
        # Unshard the output
        (cp_out,) = context_parallel_unshard(device_mesh, [cp_out], [2])
    
    return cp_out
```

**Rotation Method Options:**
- `"allgather"` (default): All-gather based pass-KV, used in Llama3 training
- `"alltoall"`: All-to-all based pass-KV, uses interleaved collectives

## Step 4: Complete Example Script

Here's the complete script that you can run with `torchrun`:

```python
# file: cp_sdpa_example.py
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import (
    context_parallel_unshard,
    set_rotate_method
)
from torch.nn.attention import sdpa_kernel, SDPBackend


def main():
    """Main function to demonstrate Context Parallel SDPA."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    try:
        # Example 1: Default all-gather based approach
        print(f"Rank {rank}: Running with default all-gather rotation...")
        output1 = context_parallel_sdpa_example(world_size, rank)
        
        # Example 2: All-to-all based approach
        print(f"Rank {rank}: Running with all-to-all rotation...")
        output2 = context_parallel_with_rotation_method(world_size, rank, "alltoall")
        
    finally:
        # Clean up distributed process group
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

## Running the Example

To run this example on 4 GPUs, use the following command:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=4 cp_sdpa_example.py
```

This will:
1. Initialize a distributed process group across 4 GPUs
2. Run SDPA with Context Parallel using the default all-gather method
3. Run SDPA with Context Parallel using the all-to-all method
4. Verify numerical correctness against single-GPU execution

## Key Concepts to Remember

1. **Buffer Sharding**: When using `context_parallel()`, you must specify which tensors to shard and along which dimensions. Typically, you'll shard along the sequence dimension (dimension 2 for tensors shaped as `[batch, heads, sequence, dim]`).

2. **Automatic SDPA Replacement**: Inside the `context_parallel()` context, `F.scaled_dot_product_attention` is automatically replaced with Ring Attention implementation.

3. **Output Unsharding**: The output of Context Parallel SDPA remains sharded. Use `context_parallel_unshard()` to gather the full tensor if needed.

4. **Numerical Precision**: When using bfloat16, expect slightly different results due to floating-point non-associativity in distributed computation. The tolerance scales with world size.

## Best Practices

1. **Include All Sequence-Dependent Tensors**: Make sure to include all tensors that compute along the sequence dimension in the `buffers` argument. For example, in Llama3 training, missing `freq_cis` would cause incorrect rotary embeddings.

2. **Choose Appropriate Rotation Method**:
   - Use `"allgather"` for compatibility with existing training setups (like Llama3)
   - Use `"alltoall"` if you want to experiment with potentially better communication overlap

3. **Verify with Small Sequences First**: Always test with small sequences to verify numerical correctness before scaling up.

## Next Steps

For more advanced usage, including:
- End-to-end training examples with TorchTitan
- Performance analysis and tuning
- Integration with other parallelism techniques (tensor, pipeline, sequence parallelism)

Check out the [PyTorch native long-context training blog post](https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082) for comprehensive examples and benchmarks.

## Conclusion

Context Parallel provides a powerful way to scale LLM training to extremely long sequences by distributing the sequence dimension across multiple devices. With PyTorch's simple API, you can easily integrate this technique into your existing training pipelines and choose between different Ring Attention implementations based on your specific needs.