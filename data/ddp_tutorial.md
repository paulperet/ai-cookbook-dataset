# PyTorch Distributed Data Parallel (DDP) Tutorial

## Overview

This guide provides a practical introduction to PyTorch's **DistributedDataParallel (DDP)** module. DDP enables you to parallelize model training across multiple GPUs and machines, making it essential for large-scale deep learning applications.

### Key Concepts

- **Process-Based Parallelism**: DDP uses multiple processes (one per GPU) rather than threads
- **Gradient Synchronization**: Automatically synchronizes gradients across all processes during backward passes
- **Model Parallelism Support**: Works with models that span multiple GPUs

### Prerequisites

Before starting, ensure you have:
- PyTorch installed with distributed support
- Basic understanding of PyTorch training loops
- Multiple GPUs available (for actual distributed training)

## Setup and Initialization

First, let's set up the necessary imports and process group initialization:

```python
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
```

### Process Group Setup

DDP requires proper initialization of process groups. Here's a setup function that configures the distributed environment:

```python
def setup(rank, world_size):
    """
    Initialize the distributed process group.
    
    Args:
        rank: Unique identifier for each process (0 to world_size-1)
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Automatically select the appropriate backend for the current accelerator
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed process group."""
    dist.destroy_process_group()
```

## Basic DDP Example

Let's start with a simple example to understand the DDP workflow.

### Step 1: Define a Toy Model

```python
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))
```

### Step 2: Create the DDP Training Function

```python
def demo_basic(rank, world_size):
    """Basic DDP training example for a single process."""
    print(f"Running basic DDP example on rank {rank}.")
    
    # Initialize the process group
    setup(rank, world_size)
    
    # Create model and move it to the appropriate device
    model = ToyModel().to(rank)
    
    # Wrap the model with DDP
    ddp_model = DDP(model, device_ids=[rank])
    
    # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    # Training step
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    
    # Compute loss and backpropagate
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    
    # Clean up
    cleanup()
    print(f"Finished running basic DDP example on rank {rank}.")
```

### Step 3: Launch Multiple Processes

```python
def run_demo(demo_fn, world_size):
    """
    Spawn multiple processes to run the DDP demo.
    
    Args:
        demo_fn: The function to run in each process
        world_size: Number of processes to spawn
    """
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
```

### Step 4: Execute the Basic Example

```python
if __name__ == "__main__":
    n_gpus = torch.accelerator.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    
    world_size = n_gpus
    run_demo(demo_basic, world_size)
```

**Key Points:**
- DDP automatically broadcasts initial model parameters from rank 0 to all processes
- Gradient synchronization happens automatically during backward pass
- Each process operates on its own GPU (device_id = rank)

## Checkpointing with DDP

When training with DDP, you can optimize checkpointing by saving the model from only one process and loading it on all processes.

### Step 1: Create Checkpoint Function

```python
def demo_checkpoint(rank, world_size):
    """Demonstrate checkpointing with DDP."""
    print(f"Running DDP checkpoint example on rank {rank}.")
    setup(rank, world_size)
    
    # Create and wrap model
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    # Define checkpoint path
    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    
    # Save checkpoint only from rank 0
    if rank == 0:
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)
    
    # Synchronize all processes before loading
    dist.barrier()
    
    # Load checkpoint on all processes with proper device mapping
    acc = torch.accelerator.current_accelerator()
    map_location = {f'{acc}:0': f'{acc}:{rank}'}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location, weights_only=True))
    
    # Continue training
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    
    loss_fn(outputs, labels).backward()
    optimizer.step()
    
    # Clean up checkpoint file (rank 0 only)
    if rank == 0:
        os.remove(CHECKPOINT_PATH)
    
    cleanup()
    print(f"Finished running DDP checkpoint example on rank {rank}.")
```

**Important Notes:**
- Use `dist.barrier()` to ensure all processes wait for the save to complete
- Provide proper `map_location` to load tensors to the correct device
- All processes start with identical parameters after loading

## Combining DDP with Model Parallelism

DDP can work with models that span multiple GPUs. This is useful for training very large models.

### Step 1: Define a Model-Parallel Model

```python
class ToyMpModel(nn.Module):
    def __init__(self, dev0, dev1):
        super(ToyMpModel, self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10, 10).to(dev0)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to(dev1)

    def forward(self, x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = x.to(self.dev1)
        return self.net2(x)
```

### Step 2: Create DDP with Model Parallelism

```python
def demo_model_parallel(rank, world_size):
    """Combine DDP with model parallelism."""
    print(f"Running DDP with model parallel example on rank {rank}.")
    setup(rank, world_size)
    
    # Assign two GPUs per process for model parallelism
    dev0 = rank * 2
    dev1 = rank * 2 + 1
    
    # Create model-parallel model
    mp_model = ToyMpModel(dev0, dev1)
    
    # Wrap with DDP (no device_ids specified for multi-GPU models)
    ddp_mp_model = DDP(mp_model)
    
    # Training setup
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_mp_model.parameters(), lr=0.001)
    
    # Training step
    optimizer.zero_grad()
    outputs = ddp_mp_model(torch.randn(20, 10))  # Outputs on dev1
    labels = torch.randn(20, 5).to(dev1)
    
    loss_fn(outputs, labels).backward()
    optimizer.step()
    
    cleanup()
    print(f"Finished running DDP with model parallel example on rank {rank}.")
```

**Key Configuration:**
- Don't specify `device_ids` when wrapping multi-GPU models with DDP
- The model's `forward()` method handles device placement
- Each DDP process uses model parallelism internally

## Using DDP with torchrun (Recommended)

PyTorch Elastic (`torchrun`) simplifies DDP initialization. Here's how to adapt our code:

### Step 1: Create Elastic-Compatible Script

```python
# elastic_ddp.py
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic():
    """DDP example compatible with torchrun."""
    # Set device from environment variable
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    
    # Initialize process group
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend)
    
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    
    # Create model and move to appropriate device
    device_id = rank % torch.accelerator.device_count()
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    
    # Training step
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_id)
    
    loss_fn(outputs, labels).backward()
    optimizer.step()
    
    dist.destroy_process_group()
    print(f"Finished running basic DDP example on rank {rank}.")

if __name__ == "__main__":
    demo_basic()
```

### Step 2: Launch with torchrun

Run the following command to launch distributed training:

```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29400 \
    elastic_ddp.py
```

**Command Breakdown:**
- `--nnodes=2`: Use 2 machines/nodes
- `--nproc_per_node=8`: 8 processes per node (typically 8 GPUs)
- `--rdzv_id=100`: Unique job identifier
- `--rdzv_backend=c10d`: Use PyTorch's distributed backend
- `--rdzv_endpoint`: Master node address and port

### Step 3: SLURM Integration (Optional)

For cluster environments using SLURM, create a launch script:

```bash
#!/bin/bash
# torchrun_script.sh

# Set master address from SLURM node list
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

# Launch torchrun
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29400 \
    elastic_ddp.py
```

Then submit with SLURM:
```bash
srun --nodes=2 ./torchrun_script.sh
```

## Best Practices and Considerations

### 1. Balanced Workloads
Ensure all processes have similar computation loads to avoid synchronization timeouts. DDP has synchronization points at:
- Constructor
- Forward pass
- Backward pass

### 2. Timeout Configuration
Set appropriate timeouts for process group initialization to handle network delays:

```python
dist.init_process_group(
    backend, 
    rank=rank, 
    world_size=world_size,
    timeout=datetime.timedelta(seconds=30)  # Adjust as needed
)
```

### 3. Gradient Synchronization
- Gradients are synchronized automatically during backward pass
- Synchronization overlaps with backward computation for efficiency
- After `backward()`, `param.grad` contains synchronized gradients

### 4. Device Management
- One GPU per DDP process (no sharing)
- For multi-GPU models, let the model handle device placement in `forward()`
- Use `map_location` properly when loading checkpoints

## Complete Execution Example

Here's how to run all the examples together:

```python
if __name__ == "__main__":
    # Check GPU availability
    n_gpus = torch.accelerator.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    
    # Run basic DDP example
    print("Running basic DDP example...")
    world_size = n_gpus
    run_demo(demo_basic, world_size)
    
    # Run checkpoint example
    print("\nRunning checkpoint example...")
    run_demo(demo_checkpoint, world_size)
    
    # Run model parallel example (requires even number of GPUs)
    print("\nRunning model parallel example...")
    world_size = n_gpus // 2
    run_demo(demo_model_parallel, world_size)
```

## Troubleshooting

### Common Issues:

1. **Timeout Errors**: Increase `timeout` in `init_process_group()`
2. **CUDA Out of Memory**: Reduce batch size per process
3. **Checkpoint Loading Errors**: Ensure proper `map_location` mapping
4. **Hanging Processes**: Check network connectivity between nodes

### Debug Tips:
- Set `NCCL_DEBUG=INFO` for detailed NCCL logging
- Use `torch.distributed.get_rank()` to identify process-specific issues
- Check `torch.cuda.memory_allocated()` for memory issues

## Next Steps

Now that you understand DDP basics, consider exploring:
- [TorchElastic](https://pytorch.org/elastic) for fault-tolerant training
- [FSDP (Fully Sharded Data Parallel)](https://pytorch.org/docs/stable/fsdp.html) for even larger models
- [PyTorch Lightning](https://www.pytorchlightning.ai/) for higher-level abstractions

Remember that DDP is a powerful tool for scaling PyTorch training. Start with the basic examples and gradually incorporate more advanced features as needed for your specific use case.