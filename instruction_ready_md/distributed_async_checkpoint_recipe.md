# Asynchronous Saving with Distributed Checkpoint (DCP)

**Author:** Lucas Pasqualin, Iris Zhang, Rodrigo Kumpera, Chien-Chin Huang

Checkpointing is often a bottleneck in the critical path for distributed training workloads, incurring larger costs as model and world sizes grow. An effective strategy to offset this cost is to checkpoint in parallel, asynchronously. This guide expands on the basic save example to show how you can integrate `torch.distributed.checkpoint.async_save` into your workflow.

## What You Will Learn
- How to use DCP to generate checkpoints in parallel.
- Effective strategies to optimize asynchronous checkpointing performance.

## Prerequisites
- PyTorch v2.4.0 or later.
- Familiarity with the [Getting Started with Distributed Checkpoint Tutorial](https://github.com/pytorch/tutorials/blob/main/recipes_source/distributed_checkpoint_recipe.rst).

## Asynchronous Checkpointing Overview

Before you begin, understand the key differences and limitations compared to synchronous checkpointing:

1.  **Memory Requirements:** Asynchronous checkpointing works by first copying model and optimizer states into internal CPU buffers. This ensures weights are not modified during the checkpointing process, but it increases CPU memory usage by approximately `checkpoint_size_per_rank * number_of_ranks`. Be mindful of your system's memory constraints, especially regarding pinned (`page-lock`) memory, which can be scarcer than pageable memory.
2.  **Checkpoint Management:** Since checkpointing is asynchronous, you must manage concurrent checkpoint requests. You can do this by handling the `Future` object returned by `async_save`. For most use cases, we recommend limiting yourself to one asynchronous request at a time to avoid excessive memory pressure.

## Step 1: Basic Asynchronous Checkpointing Setup

First, let's set up a basic training loop with Fully Sharded Data Parallel (FSDP) and integrate asynchronous checkpointing.

### 1.1 Import Required Modules

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
```

### 1.2 Define a Stateful Application Wrapper

Create a `Stateful` class to manage the model and optimizer state dictionaries. DCP will automatically call its `state_dict` and `load_state_dict` methods.

```python
class AppState(Stateful):
    """Wrapper for checkpointing the Application State."""
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # Automatically manages FSDP FQNs and uses SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {"model": model_state_dict, "optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        # Loads the state dicts back onto the model and optimizer
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"]
        )
```

### 1.3 Define a Simple Model

```python
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))
```

### 1.4 Set Up Distributed Training

Define helper functions to initialize and clean up the distributed process group.

```python
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()
```

### 1.5 Implement the Training Loop with Async Save

This is the core function that runs on each process (rank). It performs training steps and initiates asynchronous checkpoints.

```python
def run_fsdp_checkpoint_save_example(rank, world_size):
    print(f"Running basic FSDP checkpoint saving example on rank {rank}.")
    setup(rank, world_size)

    # Create and shard the model
    model = ToyModel().to(rank)
    model = fully_shard(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    checkpoint_future = None  # Will hold the Future object for the async save

    for step in range(10):
        # Training step
        optimizer.zero_grad()
        model(torch.rand(8, 16, device="cuda")).sum().backward()
        optimizer.step()

        # Wait for the previous checkpoint to finish before starting a new one
        if checkpoint_future is not None:
            checkpoint_future.result()

        # Prepare the state dict and start an asynchronous save
        state_dict = {"app": AppState(model, optimizer)}
        checkpoint_future = dcp.async_save(
            state_dict,
            checkpoint_id=f"{CHECKPOINT_DIR}_step{step}"
        )

    cleanup()
```

### 1.6 Launch the Distributed Script

Finally, use `torch.multiprocessing.spawn` to launch the distributed processes.

```python
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Running async checkpoint example on {world_size} devices.")
    mp.spawn(
        run_fsdp_checkpoint_save_example,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
```

**Key Point:** The loop uses `checkpoint_future.result()` to wait for the previous asynchronous save to complete before starting a new one. This simple strategy ensures you don't queue multiple checkpoint requests, controlling memory usage.

## Step 2: Optimizing Performance with Pinned Memory

If the basic async save isn't fast enough, you can use a pinned memory buffer to accelerate the data copying stage. This optimization reduces the overhead of copying tensors to the checkpointing buffer by using Direct Memory Access (DMA).

> **Note:** The trade-off is that the pinned memory buffer persists between checkpointing steps. Without this optimization, checkpointing buffers are released immediately after saving. With pinned memory, the buffer remains allocated, sustaining peak memory pressure throughout the application's lifetime.

### 2.1 Modify the Setup to Use a Persistent Writer

You need to create a `StorageWriter` (like `FileSystemWriter`) with the `cache_staged_state_dict=True` flag and persist it across checkpointing steps.

Update your imports and the main training function as follows:

```python
# ... (previous imports remain the same)
from torch.distributed.checkpoint import FileSystemWriter as StorageWriter

def run_fsdp_checkpoint_save_example(rank, world_size):
    print(f"Running FSDP checkpoint saving example with pinned memory on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel().to(rank)
    model = fully_shard(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Create a persistent StorageWriter with pinned memory caching enabled
    writer = StorageWriter(cache_staged_state_dict=True, path=CHECKPOINT_DIR)
    checkpoint_future = None

    for step in range(10):
        optimizer.zero_grad()
        model(torch.rand(8, 16, device="cuda")).sum().backward()
        optimizer.step()

        state_dict = {"app": AppState(model, optimizer)}

        if checkpoint_future is not None:
            checkpoint_future.result()  # Wait for the previous save

        # Pass the persistent writer to async_save
        checkpoint_future = dcp.async_save(
            state_dict,
            storage_writer=writer,
            checkpoint_id=f"{CHECKPOINT_DIR}_step{step}"
        )

    cleanup()
```

The `__main__` block remains identical to the previous example.

**How it works:** The `StorageWriter` with `cache_staged_state_dict=True` maintains a pinned memory buffer. When `async_save` is called, data is copied into this persistent buffer much faster than into a regular pageable buffer, speeding up the staging phase of checkpointing.

## Conclusion

You have learned how to use DCP's `async_save` API to generate checkpoints off the critical training path, reducing wait times in your training loop. You've also explored the memory and concurrency considerations of this approach and implemented a performance optimization using pinned memory buffers.

### Next Steps
- [Saving and loading models tutorial](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Getting started with FullyShardedDataParallel tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)