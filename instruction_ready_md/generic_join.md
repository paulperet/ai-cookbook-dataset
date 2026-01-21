# Distributed Training with Uneven Inputs Using PyTorch's Join Context Manager

## Overview

This guide demonstrates how to use PyTorch's `Join` context manager to handle distributed training with uneven inputs across processes. You'll learn to prevent hangs or errors when some ranks process fewer data batches than others during synchronous collective communications.

## Prerequisites

- PyTorch 1.10 or later
- Basic familiarity with [Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- Understanding of [ZeroRedundancyOptimizer](https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html)

## Understanding the Join Context Manager

When using distributed training with synchronous collective communications (like all-reduces in DDP), all ranks in the process group must participate. If one rank exhausts its inputs early, other ranks will hang or error. The `Join` context manager solves this by allowing early-joining ranks to "shadow" the collective communications of still-active ranks.

## Setup

First, let's set up our basic imports and configuration:

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP

BACKEND = "nccl"
WORLD_SIZE = 2
NUM_INPUTS = 5
```

## Basic Usage with DistributedDataParallel

Let's start with a simple example using only `DistributedDataParallel`:

```python
def worker(rank):
    # Initialize process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(BACKEND, rank=rank, world_size=WORLD_SIZE)
    
    # Create model with DDP
    model = DDP(torch.nn.Linear(1, 1).to(rank), device_ids=[rank])
    
    # Create uneven inputs: rank 1 gets one more input than rank 0
    inputs = [torch.tensor([1]).float() for _ in range(NUM_INPUTS + rank)]
    
    num_inputs = 0
    # Wrap training loop with Join context manager
    with Join([model]):
        for input in inputs:
            num_inputs += 1
            loss = model(input).sum()
            loss.backward()
    
    print(f"Rank {rank} has exhausted all {num_inputs} of its inputs!")

def main():
    mp.spawn(worker, nprocs=WORLD_SIZE, join=True)

if __name__ == "__main__":
    main()
```

When you run this code, you'll see output similar to:

```
Rank 0 has exhausted all 5 of its inputs!
Rank 1 has exhausted all 6 of its inputs!
```

**Note:** DDP provides its own `join()` context manager, but the generic `Join` context manager supports multiple participating classes simultaneously.

## Combining DDP with ZeroRedundancyOptimizer

Now let's extend our example to use both `DistributedDataParallel` and `ZeroRedundancyOptimizer`:

```python
from torch.distributed.optim import ZeroRedundancyOptimizer as ZeRO
from torch.optim import Adam

def worker(rank):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(BACKEND, rank=rank, world_size=WORLD_SIZE)
    
    model = DDP(torch.nn.Linear(1, 1).to(rank), device_ids=[rank])
    optim = ZeRO(model.parameters(), Adam, lr=0.01)
    
    # Rank 1 gets one more input than rank 0
    inputs = [torch.tensor([1]).float() for _ in range(NUM_INPUTS + rank)]
    
    num_inputs = 0
    # Pass both model and optimizer to Join
    with Join([model, optim]):
        for input in inputs:
            num_inputs += 1
            loss = model(input).sum()
            loss.backward()
            optim.step()
    
    print(f"Rank {rank} has exhausted all {num_inputs} of its inputs!")
```

The key change is passing both the model and optimizer instances to the `Join` context manager.

## Passing Keyword Arguments

Some classes accept keyword arguments that modify their behavior within the context manager. For example, `DistributedDataParallel` accepts `divide_by_initial_world_size`:

```python
with Join([model, optim], divide_by_initial_world_size=False):
    for input in inputs:
        # Training loop
        pass
```

**Warning:** Keyword arguments are shared across all participating classes. Ensure all classes support the same arguments.

## Understanding How Join Works

To effectively use the `Join` context manager, it helps to understand its internal components:

### Joinable Interface
Classes compatible with `Join` must inherit from `Joinable` and implement:
- `join_hook(self, **kwargs) -> JoinHook`: Returns the hook that defines how joined processes shadow communications
- `join_device(self) -> torch.device`: Returns the device for collective communications
- `join_process_group(self) -> ProcessGroup`: Returns the process group for communications

### JoinHook Class
A `JoinHook` provides two entry points:
- `main_hook(self)`: Called repeatedly by joined ranks to shadow per-iteration communications
- `post_hook(self, is_last_joiner: bool)`: Called once all ranks have joined

### Join Context Manager
The `Join` class:
- Takes a list of `Joinable` instances
- Optionally accepts `enable` (default `True`) and `throw_on_early_termination` (default `False`) parameters
- Manages the main and post hooks for all participating classes

**Important:** The context manager requires a heartbeat from non-joined processes. Each `Joinable` class should call `Join.notify_join_context()` before its per-iteration collective communications.

## Creating a Custom Joinable Class

Let's create a toy `Counter` class that demonstrates how to make a custom class compatible with `Join`:

```python
from torch.distributed.algorithms.join import Joinable, JoinHook

class CounterJoinHook(JoinHook):
    def __init__(self, counter, sync_max_count):
        self.counter = counter
        self.sync_max_count = sync_max_count
    
    def main_hook(self):
        """Shadow the counter's all-reduce by all-reducing a zero tensor."""
        t = torch.zeros(1, device=self.counter.device)
        dist.all_reduce(t)
    
    def post_hook(self, is_last_joiner: bool):
        """Synchronize max count across all counters if sync_max_count=True."""
        if not self.sync_max_count:
            return
        rank = dist.get_rank(self.counter.process_group)
        common_rank = self.counter.find_common_rank(rank, is_last_joiner)
        if rank == common_rank:
            self.counter.max_count = self.counter.count.detach().clone()
        dist.broadcast(self.counter.max_count, src=common_rank)

class Counter(Joinable):
    def __init__(self, device, process_group):
        super(Counter, self).__init__()
        self.device = device
        self.process_group = process_group
        self.count = torch.tensor([0], device=device).float()
        self.max_count = torch.tensor([0], device=device).float()
    
    def __call__(self):
        """Count inputs processed across all ranks."""
        Join.notify_join_context(self)
        t = torch.ones(1, device=self.device).float()
        dist.all_reduce(t)
        self.count += t
    
    def join_hook(self, **kwargs) -> JoinHook:
        sync_max_count = kwargs.get("sync_max_count", False)
        return CounterJoinHook(self, sync_max_count)
    
    @property
    def join_device(self) -> torch.device:
        return self.device
    
    @property
    def join_process_group(self):
        return self.process_group
    
    def find_common_rank(self, rank, to_consider):
        """Find max rank among those to consider."""
        common_rank = torch.tensor([rank if to_consider else -1], device=self.device)
        dist.all_reduce(common_rank, op=dist.ReduceOp.MAX, group=self.process_group)
        return common_rank.item()
```

Now let's use our custom `Counter` class:

```python
def worker(rank):
    assert torch.cuda.device_count() >= WORLD_SIZE
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(BACKEND, rank=rank, world_size=WORLD_SIZE)
    
    counter = Counter(torch.device(f"cuda:{rank}"), dist.group.WORLD)
    inputs = [torch.tensor([1]).float() for _ in range(NUM_INPUTS + rank)]
    
    with Join([counter], sync_max_count=True):
        for _ in inputs:
            counter()
    
    print(f"{int(counter.count.item())} inputs processed before rank {rank} joined!")
    print(f"{int(counter.max_count.item())} inputs processed across all ranks!")
```

When you run this, you'll see:

```
10 inputs processed before rank 0 joined!
11 inputs processed across all ranks!
11 inputs processed before rank 1 joined!
11 inputs processed across all ranks!
```

## Key Takeaways

1. The `Join` context manager enables distributed training with uneven inputs by allowing early-joining ranks to shadow collective communications.
2. Multiple classes (like DDP and ZeRO) can participate simultaneously in the same context manager.
3. Custom classes can be made compatible by inheriting from `Joinable` and implementing the required methods.
4. Always call `Join.notify_join_context()` before per-iteration collective communications in your `Joinable` classes.
5. Use `throw_on_early_termination=True` when dealing with complex scenarios involving interleaved collective communications from different classes.

The `Join` context manager provides a robust solution for handling uneven inputs in distributed training, ensuring all ranks can complete their training loops without hanging or erroring.