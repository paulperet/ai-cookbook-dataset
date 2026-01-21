# Combining Distributed DataParallel with Distributed RPC Framework: A Hybrid Parallelism Tutorial

**Authors**: [Pritam Damania](https://github.com/pritamdamania87) and [Yi Wang](https://github.com/wayi1)

This tutorial demonstrates how to combine **Distributed Data Parallel (DDP)** with the **Distributed RPC Framework** to implement hybrid parallelism. You will learn to train a model where a sparse embedding table is hosted on a parameter server (model parallelism via RPC) while dense fully-connected layers are replicated and synchronized across multiple trainers (data parallelism via DDP).

## Prerequisites

This guide assumes you are familiar with:
- Basic PyTorch concepts
- Distributed Data Parallel (DDP)
- The Distributed RPC Framework

## Tutorial Overview

We will set up a system with four processes:
1. **Master** (Rank 2): Orchestrates training and creates the embedding table on the parameter server.
2. **Parameter Server** (Rank 3): Hosts the embedding table in memory.
3. **Trainers** (Ranks 0 & 1): Hold the replicated FC layer (via DDP) and execute the training loop.

The workflow is as follows:
1. The master creates a remote embedding table on the parameter server.
2. It initiates the training loop on both trainers.
3. Each trainer performs an embedding lookup via RPC, then passes the result through its local FC layer (synchronized via DDP).
4. Backward pass uses Distributed Autograd to compute gradients for both the FC layer (via DDP's allreduce) and the remote embedding table (via RPC to the parameter server).
5. A Distributed Optimizer updates all parameters.

## Step 1: Environment Setup

First, ensure you have PyTorch installed with distributed support. Then, import the necessary modules.

```python
import os
import sys
import threading
from datetime import datetime
import argparse
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed import rpc
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.nn import RemoteModule
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
import torch.distributed.autograd as dist_autograd
```

## Step 2: Define the Hybrid Model

The `HybridModel` integrates a remote embedding module (on the parameter server) with a local fully-connected layer (wrapped in DDP).

```python
class HybridModel(nn.Module):
    def __init__(self, remote_emb_module, device):
        super(HybridModel, self).__init__()
        self.remote_emb_module = remote_emb_module
        self.fc = DDP(nn.Linear(16, 8).to(device), device_ids=[device])
        self.device = device

    def forward(self, indices, offsets):
        emb_lookup = self.remote_emb_module.forward(indices, offsets)
        return self.fc(emb_lookup.to(self.device))
```

**Explanation:**
- `remote_emb_module`: A `RemoteModule` that performs embedding lookups on the parameter server.
- `fc`: A linear layer wrapped in `DDP` to synchronize gradients across trainers.
- The `forward` method sends indices/offsets to the remote embedding table, retrieves embeddings, and passes them through the local FC layer.

## Step 3: Trainer Setup Function

Each trainer initializes the model, retrieves parameters for optimization, and sets up the loss function and distributed optimizer.

```python
def _setup_trainer(remote_emb_module, rank):
    # Setup the process group for DDP (world_size = 2 for two trainers)
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://localhost:29501",
        world_size=2,
        rank=rank
    )

    # Create the hybrid model
    model = HybridModel(remote_emb_module, rank)

    # Retrieve parameters for optimization
    # Get remote parameters from the embedding table
    model_parameter_rrefs = list(remote_emb_module.remote_parameters())
    # Add local FC layer parameters as RRefs
    for param in model.fc.parameters():
        model_parameter_rrefs.append(RRef(param))

    # Setup distributed optimizer and loss function
    opt = DistributedOptimizer(
        optim.SGD,
        model_parameter_rrefs,
        lr=0.05,
    )
    criterion = nn.CrossEntropyLoss()

    return model, opt, criterion
```

**Key Points:**
- DDP is initialized with a `gloo` backend (TCP). Note the port (`29501`) differs from the RPC port to avoid conflicts.
- Parameters are collected as a list of `RRef`s: remote parameters from the embedding table and local parameters from the FC layer.
- `DistributedOptimizer` will update parameters across both the trainers and the parameter server.

## Step 4: Training Loop

The training loop runs on each trainer. It uses Distributed Autograd for the backward pass—**essential when combining DDP and RPC**.

```python
def _run_trainer(remote_emb_module, rank):
    model, opt, criterion = _setup_trainer(remote_emb_module, rank)

    # Training loop for 10 epochs
    for epoch in range(10):
        # Generate a dummy batch
        indices = torch.LongTensor(16).random_(0, 10).cuda(rank)
        offsets = torch.LongTensor([0, 4, 8, 12, 16]).cuda(rank)
        target = torch.LongTensor([0, 1, 2, 3, 4]).cuda(rank)

        # Start distributed autograd context
        with dist_autograd.context() as context_id:
            output = model(indices, offsets)
            loss = criterion(output, target)

            # Run distributed backward pass
            dist_autograd.backward(context_id, [loss])

            # Step the distributed optimizer
            opt.step(context_id)

        print(f"Trainer {rank} - Epoch {epoch} - Loss: {loss.item()}")
```

**Step-by-Step Breakdown:**
1. **Generate Batch**: Create dummy input indices, offsets, and targets.
2. **Distributed Autograd Context**: Encapsulates the entire forward/backward pass to track gradients across RPC boundaries.
3. **Forward Pass**: The model performs a remote embedding lookup, then passes the result through the local FC layer.
4. **Loss Calculation**: Compute cross-entropy loss.
5. **Distributed Backward**: `dist_autograd.backward` propagates gradients through both the FC layer (via DDP) and the remote embedding table (via RPC).
6. **Optimizer Step**: `DistributedOptimizer.step` updates all parameters (local and remote).

## Step 5: Worker Initialization and Orchestration

The master process orchestrates the entire setup. It creates the remote embedding module on the parameter server and launches the trainers.

```python
def run_worker(rank, world_size):
    # Common RPC initialization for all workers
    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method="tcp://localhost:29500",
            # Set device mappings if using CUDA
            # device_maps={...}
        )
    )

    if rank == 2:  # Master
        # Create remote embedding module on the parameter server (rank 3)
        remote_emb_module = RemoteModule(
            remote_device="worker3",
            module_cls=nn.EmbeddingBag,
            args=(10, 16),
            kwargs={"mode": "sum"},
        )

        # Launch training on both trainers (ranks 0 and 1)
        futs = []
        for trainer_rank in [0, 1]:
            fut = rpc.rpc_async(
                f"worker{trainer_rank}",
                _run_trainer,
                args=(remote_emb_module, trainer_rank)
            )
            futs.append(fut)

        # Wait for both trainers to complete
        torch.futures.wait_all(futs)
        print("Training complete!")

    elif rank == 3:  # Parameter Server
        pass  # Just waits for RPCs

    else:  # Trainers (ranks 0 and 1)
        pass  # Wait for RPC from master

    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=4)
    args = parser.parse_args()

    run_worker(args.rank, args.world_size)
```

**Process Roles:**
- **Master (Rank 2)**: Initializes the `RemoteModule` on the parameter server (rank 3), then asynchronously calls `_run_trainer` on both trainers.
- **Parameter Server (Rank 3)**: Hosts the embedding table; simply initializes RPC and waits for requests.
- **Trainers (Ranks 0 & 1)**: Initialize DDP and RPC, then wait for the master's RPC to start training.

## Step 6: Running the Example

To run this hybrid training setup, launch four separate processes. Below is an example using the `torchrun` launcher (or you can use `mp.spawn`).

```bash
# On a single machine, launch four processes:
torchrun --nproc_per_node=4 --nnodes=1 hybrid_training.py
```

**Expected Output:**
You should see loss values printed from each trainer for each epoch, demonstrating that both DDP (FC layer) and RPC (embedding table) are functioning together.

```
Trainer 0 - Epoch 0 - Loss: 2.312
Trainer 1 - Epoch 0 - Loss: 2.298
Trainer 0 - Epoch 1 - Loss: 2.285
...
Training complete!
```

## Important Considerations

1. **Distributed Autograd is Mandatory**: Always use `dist_autograd` for the backward pass when combining DDP and RPC. It coordinates gradient computation across both frameworks.
2. **Port Management**: Ensure different ports are used for DDP's `init_process_group` and RPC's `init_rpc` to prevent conflicts.
3. **Parameter Collection**: When using `DistributedOptimizer`, collect all parameters (local and remote) as `RRef`s. Use `RemoteModule.remote_parameters()` for remote parameters and manually create `RRef`s for local ones.
4. **Device Placement**: If using CUDA, ensure tensors are on the correct devices and specify `device_maps` in `TensorPipeRpcBackendOptions` for cross-device RPC.

## Conclusion

You have successfully implemented a hybrid parallelism training scheme using PyTorch's Distributed DataParallel and RPC frameworks. This pattern is powerful for models with heterogeneous components—like large embedding tables combined with dense layers—enabling efficient use of multiple machines and GPUs.

For the complete runnable example, refer to the [source code](https://github.com/pytorch/examples/tree/master/distributed/rpc/ddp_rpc).