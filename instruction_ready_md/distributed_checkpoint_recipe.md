# Distributed Checkpointing (DCP) for FSDP Models: A Practical Guide

## Overview
PyTorch Distributed Checkpointing (DCP) enables efficient checkpointing of models during distributed training, particularly when using Fully Sharded Data Parallel (FSDP). Unlike traditional `torch.save`, DCP handles parameter sharding across multiple GPUs and supports resharding when resuming with different cluster configurations.

## Prerequisites

Ensure you have the required PyTorch version installed:
```bash
pip install torch>=2.0.0
```

## Key Concepts

### How DCP Differs from `torch.save`
- **Multiple Files**: Creates at least one checkpoint file per rank
- **In-Place Operations**: Uses pre-allocated model storage instead of creating new tensors
- **Stateful Objects**: Automatically calls `state_dict()` and `load_state_dict()` on objects implementing the `Stateful` protocol
- **Distributed Support**: Designed for multi-GPU environments with flexible resharding

## Implementation Guide

### 1. Setup and Model Definition

First, let's define our model and the necessary helper classes:

```python
import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

CHECKPOINT_DIR = "checkpoint"

class AppState(Stateful):
    """
    Wrapper for checkpointing application state.
    Implements the Stateful protocol for automatic state_dict handling.
    """
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # Automatically manages FSDP FQNs and sets default state dict type
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # Applies loaded state dicts to model and optimizer
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )

class ToyModel(nn.Module):
    """Simple model for demonstration purposes."""
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))
```

### 2. Distributed Setup Utilities

```python
def setup(rank, world_size):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed process group."""
    dist.destroy_process_group()
```

### 3. Saving a Checkpoint with DCP

Now let's create a function to save our FSDP-wrapped model:

```python
def run_fsdp_checkpoint_save_example(rank, world_size):
    """Demonstrates saving an FSDP model using DCP."""
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")
    setup(rank, world_size)

    # Create and wrap model with FSDP
    model = ToyModel().to(rank)
    model = fully_shard(model)

    # Setup optimizer and perform a training step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    optimizer.zero_grad()
    model(torch.rand(8, 16, device="cuda")).sum().backward()
    optimizer.step()

    # Save checkpoint using DCP
    state_dict = {"app": AppState(model, optimizer)}
    dcp.save(state_dict, checkpoint_id=CHECKPOINT_DIR)

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running FSDP checkpoint example on {world_size} devices.")
    mp.spawn(
        run_fsdp_checkpoint_save_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
```

After running this script, you'll find checkpoint files in the `checkpoint/` directory. With 8 GPUs, you should see 8 files (one per rank).

### 4. Loading a Checkpoint with DCP

Loading follows a similar pattern but requires the model's `state_dict` to be passed to DCP for in-place loading:

```python
def run_fsdp_checkpoint_load_example(rank, world_size):
    """Demonstrates loading an FSDP model using DCP."""
    print(f"Running basic FSDP checkpoint loading example on rank {rank}.")
    setup(rank, world_size)

    # Create a fresh model (same architecture as saved)
    model = ToyModel().to(rank)
    model = fully_shard(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Load checkpoint into the model
    state_dict = {"app": AppState(model, optimizer)}
    dcp.load(
        state_dict=state_dict,
        checkpoint_id=CHECKPOINT_DIR,
    )

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running FSDP checkpoint example on {world_size} devices.")
    mp.spawn(
        run_fsdp_checkpoint_load_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
```

**Important**: DCP requires the model's `state_dict` before loading because:
1. It uses pre-allocated storage from the model
2. It needs sharding information to support resharding across different cluster topologies

### 5. Loading for Inference (Non-Distributed)

You can also load DCP checkpoints in a non-distributed environment for inference:

```python
def run_checkpoint_load_example():
    """Loads a DCP checkpoint in a non-distributed setting."""
    # Create a non-FSDP wrapped model
    model = ToyModel()
    state_dict = {
        "model": model.state_dict(),
    }

    # DCP automatically disables collectives when no process group is initialized
    dcp.load(
        state_dict=state_dict,
        checkpoint_id=CHECKPOINT_DIR,
    )
    
    # Apply the loaded state dict
    model.load_state_dict(state_dict["model"])

if __name__ == "__main__":
    print("Running basic DCP checkpoint loading example.")
    run_checkpoint_load_example()
```

### 6. Format Conversion Utilities

DCP checkpoints use a different format than `torch.save`. Use the format utilities for conversion:

#### Command Line Interface
```bash
# Convert DCP to torch.save format
python -m torch.distributed.checkpoint.format_utils dcp_to_torch checkpoint/ torch_save_checkpoint.pth

# Convert torch.save to DCP format
python -m torch.distributed.checkpoint.format_utils torch_to_dcp torch_save_checkpoint.pth checkpoint_new/
```

#### Programmatic Conversion
```python
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save, torch_save_to_dcp

CHECKPOINT_DIR = "checkpoint"
TORCH_SAVE_CHECKPOINT_DIR = "torch_save_checkpoint.pth"

# Convert DCP to torch.save
dcp_to_torch_save(CHECKPOINT_DIR, TORCH_SAVE_CHECKPOINT_DIR)

# Convert torch.save back to DCP
torch_save_to_dcp(TORCH_SAVE_CHECKPOINT_DIR, f"{CHECKPOINT_DIR}_new")
```

## Best Practices

1. **Use the `AppState` Wrapper**: This simplifies state management and ensures proper handling of FSDP-specific FQNs
2. **Check Directory Permissions**: Ensure all ranks have write access to the checkpoint directory
3. **Monitor Storage**: DCP creates multiple files, so plan your storage accordingly
4. **Test Loading Early**: Verify checkpoint loading works before running long training jobs

## Conclusion

PyTorch Distributed Checkpointing provides a robust solution for checkpointing FSDP models in distributed environments. Key advantages include:
- Support for changing cluster topologies during resume
- Automatic handling of sharded parameters
- Flexible loading options (distributed and non-distributed)
- Format conversion utilities for compatibility

For more information, refer to:
- [Saving and Loading Models Tutorial](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [DCP API Documentation](https://pytorch.org/docs/stable/distributed.checkpoint.html)