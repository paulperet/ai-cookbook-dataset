# Multi-GPU Training with Distributed Data Parallel (DDP)

## Overview

This guide walks you through migrating a single-GPU PyTorch training script to a multi-GPU setup using Distributed Data Parallel (DDP). You'll learn how to set up the distributed process group, modify your model and data loading for distributed training, and properly handle checkpoint saving in a multi-GPU environment.

## Prerequisites

- Basic understanding of [how DDP works](ddp_series_theory.html)
- A machine with multiple GPUs (this tutorial assumes 4 GPUs)
- PyTorch installed with CUDA support

## Setup

First, ensure you have the necessary imports:

```python
import torch
import torch.nn.functional as F
from utils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
```

## Step 1: Initialize the Distributed Process Group

Before training begins, you need to set up communication between all processes. Each process (one per GPU) must join a distributed process group.

```python
def ddp_setup(rank: int, world_size: int):
    """
    Initialize the distributed process group.
    
    Args:
        rank: Unique identifier for each process (0 to world_size-1)
        world_size: Total number of processes (typically equal to number of GPUs)
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
```

**Key points:**
- `torch.cuda.set_device(rank)` ensures each process uses a different GPU
- The `backend="nccl"` is optimized for NVIDIA GPU communication
- `MASTER_ADDR` and `MASTER_PORT` define how processes coordinate

## Step 2: Wrap Your Model with DDP

To enable synchronized gradient computation across GPUs, wrap your model with `DistributedDataParallel`:

```python
self.model = DDP(model, device_ids=[gpu_id])
```

This wrapper automatically handles gradient synchronization during the backward pass.

## Step 3: Distribute Your Training Data

In distributed training, each process should see a different subset of the data. Use `DistributedSampler` with your DataLoader:

```python
train_data = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=False,  # Don't shuffle here - the sampler handles it
    sampler=DistributedSampler(train_dataset),
)
```

**Important:** The effective batch size becomes `batch_size * world_size`. With 4 GPUs and batch_size=32, you're actually training with 128 samples per iteration.

## Step 4: Handle Epoch Shuffling Correctly

To ensure proper shuffling across epochs, call `set_epoch()` on the sampler at the beginning of each epoch:

```python
def _run_epoch(self, epoch):
    self.train_data.sampler.set_epoch(epoch)  # Critical for proper shuffling
    for source, targets in self.train_data:
        self._run_batch(source, targets)
```

Without this call, each epoch would use the same data ordering.

## Step 5: Save Checkpoints from Only One Process

Since all processes have identical model states after synchronization, you only need to save checkpoints from one process (typically rank 0):

```python
# Instead of:
# ckp = self.model.state_dict()

# Use:
ckp = self.model.module.state_dict()

# And only save from rank 0:
if self.gpu_id == 0 and epoch % self.save_every == 0:
    self._save_checkpoint(epoch)
```

**Warning:** Avoid collective calls (like `dist.all_reduce`) in checkpoint-saving code that only runs on rank 0, as this would cause hangs.

## Step 6: Launch Distributed Training

Modify your main function to handle distributed setup and use `mp.spawn` to launch multiple processes:

```python
def main(rank, world_size, total_epochs, save_every):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size=32)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, total_epochs, save_every), nprocs=world_size)
```

**Key changes from single-GPU code:**
- Added `rank` and `world_size` parameters
- Call `ddp_setup()` to initialize distributed environment
- Use `mp.spawn()` to launch one process per GPU
- Call `destroy_process_group()` to clean up after training

## Step 7: Handle BatchNorm Layers (If Needed)

If your model contains BatchNorm layers, convert them to SyncBatchNorm to synchronize running statistics across GPUs:

```python
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
```

## Complete Migration Checklist

Here's a summary of changes needed to migrate from single-GPU to multi-GPU DDP:

1. **Imports**: Add distributed training modules
2. **Process Group**: Initialize with `ddp_setup()`
3. **Model**: Wrap with `DDP()`
4. **Data Loading**: Use `DistributedSampler`
5. **Epoch Shuffling**: Call `sampler.set_epoch(epoch)`
6. **Checkpoint Saving**: Save only from rank 0 using `model.module.state_dict()`
7. **Launch**: Use `mp.spawn()` instead of direct function call
8. **Cleanup**: Call `destroy_process_group()` after training

## Running Your Distributed Training

Execute your script with:
```bash
python multigpu.py 50 10
```

Where `50` is the total number of epochs and `10` is the checkpoint saving frequency (every 10 epochs).

## Next Steps

- Learn about [Fault Tolerant distributed training](ddp_series_fault_tolerance.html) for handling failures
- Explore [multi-node training](../intermediate/ddp_series_multinode.html) for scaling beyond a single machine
- Check the official PyTorch [DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for more examples

By following these steps, you've successfully converted a single-GPU training script to leverage multiple GPUs with DDP, enabling faster training through parallel computation while maintaining model consistency across all devices.