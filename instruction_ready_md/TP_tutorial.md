# Large-Scale Transformer Model Training with Tensor Parallel (TP)

## Overview

This tutorial demonstrates how to train large Transformer models across hundreds to thousands of GPUs using PyTorch's native Tensor Parallel (TP) and Fully Sharded Data Parallel (FSDP). You will learn to apply TP to different model components without modifying the underlying model code.

## Prerequisites

- PyTorch 2.3.0 or later with CUDA/Linux
- Basic understanding of:
  - [Tensor Parallel APIs](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
  - [DeviceMesh](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)
  - [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

## Understanding Tensor Parallel

Tensor Parallel (TP) is a model parallelism technique originally proposed in the [Megatron-LM paper](https://arxiv.org/abs/1909.08053). It shards model parameters and computations across multiple GPUs to enable training of models that don't fit on a single GPU.

**Key Concepts:**
- **Column-wise Parallelism**: Shards linear layers along the column dimension
- **Row-wise Parallelism**: Shards linear layers along the row dimension  
- **Sequence Parallelism**: Variant that shards on the sequence dimension for normalization layers to save activation memory

### When to Use Tensor Parallel

Combine TP with FSDP when:

1. **FSDP world size becomes too large** (exceeding 128-256 GPUs), causing communication latency to dominate
2. **Hitting data parallelism limits** where global batch size can't increase due to convergence or memory constraints
3. **Optimizing matrix multiplication shapes** for better FLOPS efficiency with smaller local batch sizes

## Step 1: Setup and Initialization

First, initialize the distributed environment. Tensor Parallel typically works within each host, so we'll create a DeviceMesh connecting 8 GPUs:

```python
from torch.distributed.device_mesh import init_device_mesh

# Initialize a 1D DeviceMesh for 8 GPUs within a host
tp_mesh = init_device_mesh("cuda", (8,))
```

## Step 2: Understanding the Model Architecture

We'll use a Llama2-style Transformer model as our example. The core component is the `TransformerBlock`, which consists of:
- An `Attention` layer with query, key, value, and output projections
- A `FeedForward` layer with three linear layers in a SwiGLU configuration

### FeedForward Layer Parallelization

The FeedForward layer performs: `w2(F.silu(w1(x)) * w3(x))`

We can apply TP by:
- Sharding `w1` and `w3` column-wise
- Sharding `w2` row-wise
- This requires only one `allreduce` communication at the end

```python
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

# TP plan for FeedForward layer
feedforward_tp_plan = {
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
}
```

### Attention Layer Parallelization

The Attention layer has:
- `wq`, `wk`, `wv` for query/key/value projections (shard column-wise)
- `wo` for output projection (shard row-wise)

```python
attention_tp_plan = {
    "attention.wq": ColwiseParallel(use_local_output=False),
    "attention.wk": ColwiseParallel(use_local_output=False),
    "attention.wv": ColwiseParallel(use_local_output=False),
    "attention.wo": RowwiseParallel(),
}
```

## Step 3: Complete TransformerBlock TP Plan

Combine the plans for a complete `TransformerBlock`:

```python
layer_tp_plan = {
    # Attention layer components
    "attention.wq": ColwiseParallel(use_local_output=False),
    "attention.wk": ColwiseParallel(use_local_output=False),
    "attention.wv": ColwiseParallel(use_local_output=False),
    "attention.wo": RowwiseParallel(),
    
    # FeedForward layer components
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(),
    "feed_forward.w3": ColwiseParallel(),
}
```

**Note**: We use `use_local_output=False` for column-wise layers to ensure outputs remain as DTensors, which automatically handle dimension changes (like `num_heads`) during view operations.

## Step 4: Apply Tensor Parallel to the Model

Now apply the TP plan to each `TransformerBlock`:

```python
for layer_id, transformer_block in enumerate(model.layers):
    parallelize_module(
        module=transformer_block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_tp_plan,
    )
```

For the embedding and output layers:

```python
from torch.distributed.tensor.parallel import Replicate

model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
        ),
        "output": ColwiseParallel(
            output_layouts=Replicate(),
        ),
    }
)
```

**Tip**: If your model is too large for CPU memory, use meta device initialization or parallelize layers during model initialization.

## Step 5: Add Sequence Parallelism

Sequence Parallelism saves activation memory by keeping tensors sharded on the sequence dimension. Apply it to normalization layers:

```python
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    SequenceParallel,
)
from torch.distributed.tensor.parallel import Shard

layer_tp_plan_with_sp = {
    # Sequence Parallel for normalization layers
    "attention_norm": SequenceParallel(),
    "ffn_norm": SequenceParallel(),
    
    # Prepare inputs for attention and feedforward
    "attention": PrepareModuleInput(
        input_layouts=(Shard(1), Replicate()),
        desired_input_layouts=(Replicate(), Replicate()),
    ),
    "feed_forward": PrepareModuleInput(
        input_layouts=(Shard(1),),
        desired_input_layouts=(Replicate(),),
    ),
    
    # Attention components
    "attention.wq": ColwiseParallel(use_local_output=False),
    "attention.wk": ColwiseParallel(use_local_output=False),
    "attention.wv": ColwiseParallel(use_local_output=False),
    "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
    
    # FeedForward components
    "feed_forward.w1": ColwiseParallel(),
    "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
    "feed_forward.w3": ColwiseParallel(),
}
```

Update the model-level plan for sequence-parallel inputs/outputs:

```python
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate()
        ),
    }
)
```

## Step 6: Enable Loss Parallel

Loss Parallel computes cross-entropy loss efficiently when model outputs are sharded on the vocabulary dimension:

```python
model = parallelize_module(
    model,
    tp_mesh,
    {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "norm": SequenceParallel(),
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            use_local_output=False,  # Keep as DTensor for loss parallel
        ),
    }
)
```

Use the loss parallel context manager:

```python
import torch.nn.functional as F
from torch.distributed.tensor.parallel import loss_parallel

pred = model(input_ids)
with loss_parallel():
    # Flatten batch and sequence dimensions
    loss = F.cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))
    loss.backward()
```

## Step 7: Combine Tensor Parallel with FSDP

For large-scale training, combine TP (intra-host) with FSDP (inter-host):

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.fsdp import fully_shard

# Create a 2D mesh: [data_parallel, tensor_parallel]
# Example: 64 GPUs with 8-way DP and 8-way TP
mesh_2d = init_device_mesh("cuda", (8, 8))
tp_mesh = mesh_2d["tp"]  # Intra-host submesh
dp_mesh = mesh_2d["dp"]  # Inter-host submesh

# Apply Tensor Parallel
model_tp = parallelize_module(model, tp_mesh, tp_plan)

# Apply FSDP on top
model_2d = fully_shard(model_tp, mesh=dp_mesh)
```

This combination allows you to scale both model size and number of GPUs efficiently.

## Complete Example

For a complete end-to-end implementation, refer to the [Tensor Parallel examples](https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/fsdp_tp_example.py) in the PyTorch examples repository.

## Key Takeaways

1. **No Code Changes**: Apply Tensor Parallel without modifying your model code
2. **Flexible Configuration**: Use `ParallelStyle` primitives to configure sharding per layer
3. **Memory Efficiency**: Combine TP with Sequence Parallel and Loss Parallel for optimal memory usage
4. **Scalability**: Use TP with FSDP for training on hundreds to thousands of GPUs

Tensor Parallel is an essential technique for large-scale Transformer training, enabling you to overcome memory and communication bottlenecks while maintaining training efficiency.