# PyTorch FSDP2: A Practical Guide to Fully Sharded Data Parallel Training

## Introduction

Fully Sharded Data Parallel (FSDP2) is PyTorch's next-generation distributed training strategy that significantly reduces GPU memory footprint by sharding model parameters, gradients, and optimizer states across multiple devices. This enables training models that are too large to fit on a single GPU.

In this guide, you'll learn how to implement FSDP2 in your training workflows, from basic setup to advanced features like mixed precision and checkpointing.

## Prerequisites

Before starting, ensure you have:

- PyTorch 2.4 or later
- Multiple GPUs (or a multi-GPU environment)
- Basic familiarity with distributed training concepts

## How FSDP2 Works

Unlike DistributedDataParallel (DDP) where each rank maintains a full model replica, FSDP2 shards parameters across ranks:

- **Outside forward/backward**: Parameters remain fully sharded
- **Before forward/backward**: Sharded parameters are all-gathered into unsharded parameters
- **During backward**: Local unsharded gradients are reduce-scattered into sharded gradients
- **Optimizer step**: Sharded parameters are updated with sharded gradients

This decomposition of DDP's all-reduce into reduce-scatter and all-gather operations enables significant memory savings.

## Getting Started with FSDP2

### 1. Basic Model Setup

First, let's set up a simple training script with FSDP2. Create a file called `train.py`:

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard, FSDPModule
from torch.distributed.tensor import DTensor, Shard

# Define a simple transformer model (simplified for example)
class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        # Simplified forward pass
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size=10000, dim=512, num_layers=3):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(num_layers)])
        self.output = nn.Linear(dim, vocab_size)
    
    def forward(self, x):
        x = self.tok_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
```

### 2. Applying FSDP2 to Your Model

Now, let's apply FSDP2 to shard the model parameters:

```python
def setup_model():
    # Initialize the model
    model = Transformer()
    
    # Apply fully_shard to each transformer block (submodule)
    for layer in model.layers:
        fully_shard(layer)
    
    # Apply fully_shard to the root model
    fully_shard(model)
    
    # Verify the model is now an FSDPModule
    assert isinstance(model, Transformer)
    assert isinstance(model, FSDPModule)
    
    # Inspect the wrapped model structure
    print(model)
    # Output: FSDPTransformer(...)
    
    return model

# Parameters are now DTensors sharded across ranks
def inspect_parameters(model):
    for param in model.parameters():
        assert isinstance(param, DTensor)
        assert param.placements == (Shard(0),)
        # Access local shard with param.to_local()
        local_shard = param.to_local()
        print(f"Parameter shape: {local_shard.shape}")
```

### 3. Training Loop with FSDP2

Here's the basic training loop with FSDP2:

```python
def train(model, device, vocab_size=10000, batch_size=32, seq_len=512, epochs=10):
    # Initialize optimizer AFTER applying fully_shard
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    for epoch in range(epochs):
        # Generate dummy data
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Forward pass
        loss = model(x).sum()
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optim.step()
        optim.zero_grad()
        
        if torch.distributed.get_rank() == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
```

### 4. Running the Training Script

To run the training script with 2 GPUs:

```bash
torchrun --nproc_per_node 2 train.py
```

## Advanced FSDP2 Features

### Implicit Prefetching (Default)

FSDP2 automatically overlaps all-gather operations with computation using implicit prefetching. This happens automatically in the training loop above:

- CPU thread issues all-gather for layer i+1 while layer i computes
- All-gathers are queued in their own CUDA stream
- For non-CPU-bound workloads, this provides good overlap

### Explicit Prefetching (Advanced Control)

For more control over prefetching, use explicit prefetching:

```python
def setup_explicit_prefetching(model):
    # Forward prefetching: prefetch 2 layers ahead
    num_to_forward_prefetch = 2
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)
    
    # Backward prefetching: prefetch 2 layers ahead in backward
    num_to_backward_prefetch = 2
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)

def train_with_explicit_prefetch(model, device):
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    for epoch in range(10):
        # Trigger first all-gather earlier to overlap with data loading
        model.unshard()
        
        x = torch.randint(0, 10000, (32, 512), device=device)
        loss = model(x).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()
```

Run with explicit prefetching:

```bash
torchrun --nproc_per_node 2 train.py --explicit-prefetching
```

### Mixed Precision Training

FSDP2 provides flexible mixed precision policies:

```python
from torch.distributed.fsdp import MixedPrecisionPolicy

def setup_mixed_precision():
    model = Transformer()
    
    # Configure mixed precision policy
    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,    # Cast to bfloat16 for computation
            reduce_dtype=torch.float32,    # Reduce gradients in float32
        )
    }
    
    # Apply with mixed precision
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)
    
    # Sharded parameters remain in float32
    for param in model.parameters():
        assert param.dtype == torch.float32
    
    # Unsharded parameters become bfloat16
    model.unshard()
    for param in model.parameters(recurse=False):
        assert param.dtype == torch.bfloat16
    model.reshard()
    
    return model
```

Run with mixed precision:

```bash
torchrun --nproc_per_node 2 train.py --mixed-precision
```

### Gradient Clipping with DTensor

Gradient clipping works seamlessly with DTensor parameters:

```python
def train_with_gradient_clipping(model, device, max_norm=1.0):
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    for epoch in range(10):
        x = torch.randint(0, 10000, (32, 512), device=device)
        loss = model(x).sum()
        loss.backward()
        
        # Gradient clipping works with DTensor parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        optim.step()
        optim.zero_grad()
```

## Checkpointing with FSDP2

### Using DTensor APIs for State Dicts

#### Loading Checkpoints

```python
from torch.distributed.tensor import distribute_tensor

def load_checkpoint_dtensor(model, checkpoint_path):
    # Load full state dict with memory mapping
    full_sd = torch.load(
        checkpoint_path,
        mmap=True,
        weights_only=True,
        map_location='cpu',
    )
    
    # Get meta sharded state dict from model
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    
    # Convert full tensors to sharded DTensors
    for param_name, full_tensor in full_sd.items():
        sharded_meta_param = meta_sharded_sd.get(param_name)
        sharded_tensor = distribute_tensor(
            full_tensor,
            sharded_meta_param.device_mesh,
            sharded_meta_param.placements,
        )
        sharded_sd[param_name] = nn.Parameter(sharded_tensor)
    
    # Load into model
    model.load_state_dict(sharded_sd, assign=True)
```

#### Saving Checkpoints

```python
def save_checkpoint_dtensor(model, checkpoint_path):
    sharded_sd = model.state_dict()
    cpu_state_dict = {}
    
    # Convert DTensors to full tensors and save on rank 0
    for param_name, sharded_param in sharded_sd.items():
        full_param = sharded_param.full_tensor()
        if torch.distributed.get_rank() == 0:
            cpu_state_dict[param_name] = full_param.cpu()
        else:
            del full_param
    
    # Save on rank 0
    if torch.distributed.get_rank() == 0:
        torch.save(cpu_state_dict, checkpoint_path)
```

### Using DCP APIs (Recommended)

#### Loading with DCP

```python
from torch.distributed.checkpoint.state_dict import set_model_state_dict, StateDictOptions

def load_checkpoint_dcp(model, full_sd):
    set_model_state_dict(
        model=model,
        model_state_dict=full_sd,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,  # Load only on rank 0
        ),
    )
```

#### Saving with DCP

```python
from torch.distributed.checkpoint.state_dict import get_model_state_dict

def save_checkpoint_dcp(model, checkpoint_path):
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,  # Automatically offload to CPU
        )
    )
    
    if torch.distributed.get_rank() == 0:
        torch.save(model_state_dict, checkpoint_path)
```

Run with DCP APIs:

```bash
torchrun --nproc_per_node 2 train.py --dcp-api
```

## Migration from FSDP1 to FSDP2

### Key Changes

1. **Import changes**: Replace `FullyShardedDataParallel` with `fully_shard`
2. **Wrapping strategy**: Apply `fully_shard` to submodules explicitly
3. **Parameter initialization**: Use `reset_parameters()` instead of `param_init_fn`

### Migration Example

#### FSDP1 Code

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

with torch.device("meta"):
    model = Transformer()

policy = ModuleWrapPolicy({TransformerBlock})
model = FSDP(model, auto_wrap_policy=policy)

def param_init_fn(module: nn.Module) -> None:
    # Custom initialization
    pass

model = FSDP(model, auto_wrap_policy=policy, param_init_fn=param_init_fn)
```

#### FSDP2 Equivalent

```python
with torch.device("meta"):
    model = Transformer()

# Apply fully_shard to submodules explicitly
for module in model.modules():
    if isinstance(module, TransformerBlock):
        fully_shard(module)

# Apply to root model
fully_shard(model)

# Verify all tensors are on meta device
for tensor in itertools.chain(model.parameters(), model.buffers()):
    assert tensor.device == torch.device("meta")

# Initialize model after sharding
model.to_empty(device="cuda")
model.reset_parameters()  # Replaces param_init_fn
```

### Parameter Mapping Guide

| FSDP1 Parameter | FSDP2 Equivalent |
|----------------|------------------|
| `sharding_strategy=FULL_SHARD` | `reshard_after_forward=True` (default) |
| `sharding_strategy=SHARD_GRAD_OP` | `reshard_after_forward=False` |
| `cpu_offload=CPUOffload(offload_params=True)` | `offload_policy=CPUOffloadPolicy()` |
| `backward_prefetch=BACKWARD_PRE` | Always used |
| `use_orig_params=True` | Always used (no flat parameter) |
| `ignored_params` | `ignored_params` parameter in `fully_shard` |
| `no_sync()` | `set_requires_gradient_sync(False)` |

## Best Practices

1. **Apply `fully_shard` to submodules first**, then to the root model
2. **Initialize optimizer after** applying `fully_shard`
3. **Use DCP APIs** for checkpointing in production
4. **Start with implicit prefetching**, then optimize with explicit prefetching if needed
5. **Consider mixed precision** for memory savings and speed
6. **Monitor GPU memory** to understand sharding benefits

## Troubleshooting

- **Memory issues**: Ensure you're applying `fully_shard` to appropriate submodules
- **Performance issues**: Try explicit prefetching for CPU-bound workloads
- **Checkpoint loading**: Use DCP APIs for simplicity and robustness
- **Numerical issues**: Consider float32 gradient reduction in mixed precision

## Conclusion

FSDP2 provides a powerful, flexible framework for distributed training of large models. By sharding parameters, gradients, and optimizer states, it enables training models that would otherwise be impossible to fit on single GPUs. The DTensor-based implementation offers clean abstractions and seamless integration with existing PyTorch workflows.

Start with the basic setup, then gradually incorporate advanced features like mixed precision and explicit prefetching as needed for your specific use case.

## Additional Resources

- [PyTorch FSDP2 Documentation](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
- [Example Code on GitHub](https://github.com/pytorch/examples/tree/main/distributed/FSDP2)
- [DTensor Documentation](https://docs.pytorch.org/docs/stable/distributed.tensor.html)
- [Distributed Checkpoint (DCP) Documentation](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)