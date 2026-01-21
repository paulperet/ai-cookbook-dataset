# PyTorch DeviceMesh Tutorial: Simplifying Multi-Dimensional Parallelism

**Author**: [Iris Zhang](https://github.com/wz337), [Wanchao Liang](https://github.com/wanchaol)

## Overview

Managing distributed communication for complex, multi-dimensional parallel training (e.g., 2D or 3D parallelism) is challenging. This tutorial introduces `DeviceMesh`, a high-level abstraction that simplifies the creation and management of process groups (like NCCL communicators) across nodes and devices. You'll learn how to replace verbose, error-prone manual setup with concise, declarative code.

## Prerequisites

*   Python 3.8 - 3.11
*   PyTorch 2.2 or later
*   Basic familiarity with [`torch.distributed`](https://pytorch.org/docs/stable/distributed.html)

## 1. Understanding the Problem: Manual 2D Parallel Setup

Before `DeviceMesh`, setting up a hybrid sharding pattern (a form of 2D parallelism) required manually calculating rank groups. This process is complex and prone to errors.

Let's examine a typical manual setup. Create a file named `2d_setup.py` with the following content:

```python
import os
import torch
import torch.distributed as dist

# Initialize the distributed environment
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
print(f"Running example on {rank=} in a world with {world_size=}")

dist.init_process_group("nccl")
torch.cuda.set_device(rank)

# Configuration: Simulating 2D parallelism on a single node with 8 GPUs
num_node_devices = torch.cuda.device_count()  # Typically 8 for this example
shard_factor = num_node_devices // 2  # Split into two shard groups

# Step 1: Create shard groups (e.g., (0,1,2,3) and (4,5,6,7))
shard_rank_lists = (
    list(range(0, shard_factor)),
    list(range(shard_factor, num_node_devices))
)
shard_groups = (
    dist.new_group(shard_rank_lists[0]),
    dist.new_group(shard_rank_lists[1]),
)
# Assign the correct shard group to the current rank
current_shard_group = (
    shard_groups[0] if rank in shard_rank_lists[0] else shard_groups[1]
)

# Step 2: Create replicate groups (e.g., (0,4), (1,5), (2,6), (3,7))
current_replicate_group = None
for i in range(shard_factor):
    replicate_group_ranks = list(range(i, num_node_devices, shard_factor))
    replicate_group = dist.new_group(replicate_group_ranks)
    if rank in replicate_group_ranks:
        current_replicate_group = replicate_group

print(f"Rank {rank}: Shard Group assigned. Replicate Group assigned.")
```

Run this script using `torchrun` to simulate an 8-GPU setup:

```bash
torchrun --nproc_per_node=8 --rdzv_id=100 --rdzv_endpoint=localhost:29400 2d_setup.py
```

> **Note:** This example uses a single node for simplicity, but the same logic applies to multi-host setups.

## 2. Introducing DeviceMesh

`DeviceMesh` abstracts this complexity. It manages the underlying `ProcessGroup` objects, allowing you to define a multi-dimensional grid of devices (a *mesh*) declaratively.

### 2.1 Simplified 2D Setup with DeviceMesh

The entire manual setup from the previous section can be replaced with just a few lines. Create a new file, `2d_setup_with_device_mesh.py`:

```python
from torch.distributed.device_mesh import init_device_mesh

# Create a 2D device mesh with shape (2, 4).
# The first dimension ('replicate') is for data replication across nodes/hosts.
# The second dimension ('shard') is for model sharding within a node.
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("replicate", "shard"))

# You can still access the underlying process groups if needed.
replicate_group = mesh_2d.get_group(mesh_dim="replicate")
shard_group = mesh_2d.get_group(mesh_dim="shard")
print(f"Mesh created. Process groups are accessible.")
```

Run it with:

```bash
torchrun --nproc_per_node=8 2d_setup_with_device_mesh.py
```

The `init_device_mesh` function automatically handles rank assignment and `ProcessGroup` creation for both the `replicate` and `shard` dimensions.

## 3. Practical Application: Hybrid Sharding Data Parallel (HSDP)

HSDP is a 2D strategy combining Fully Sharded Data Parallel (FSDP) within a host and Distributed Data Parallel (DDP) across hosts. `DeviceMesh` integrates seamlessly with PyTorch's `FSDP`.

### 3.1 Applying HSDP to a Model

Let's wrap a simple model with FSDP using a 2D mesh. Create a file named `hsdp.py`:

```python
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

# Step 1: Initialize the 2D device mesh for HSDP.
# Mesh shape (2, 4): 2 replicate groups, 4 shard groups.
mesh_2d = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp_replicate", "dp_shard"))

# Step 2: Create the model and apply FSDP with the mesh.
# The `device_mesh` argument tells FSDP about the parallelism layout.
model = ToyModel().cuda()
model = FSDP(model, device_mesh=mesh_2d)
print("Model wrapped with FSDP using a 2D DeviceMesh for HSDP.")
```

Run the example:

```bash
torchrun --nproc_per_node=8 hsdp.py
```

The `FSDP` wrapper uses the provided `device_mesh` to manage sharding (`dp_shard` dimension) and replication (`dp_replicate` dimension) communication internally.

## 4. Advanced Usage: Slicing Meshes for Custom Parallelism

For complex 3D parallel strategies, you can create a parent mesh and slice it into sub-meshes for different parallelism techniques (e.g., combining HSDP with Tensor Parallelism). `DeviceMesh` efficiently reuses the underlying NCCL communicators.

### 4.1 Creating and Slicing a 3D Mesh

Create a script to demonstrate mesh slicing:

```python
from torch.distributed.device_mesh import init_device_mesh

# Step 1: Initialize a 3D device mesh.
# Dimensions: (replicate, shard, tensor_parallel)
mesh_3d = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("replicate", "shard", "tp"))

# Step 2: Slice out sub-meshes for specific parallelisms.
# Create a 2D mesh for HSDP by selecting the 'replicate' and 'shard' dimensions.
hsdp_mesh = mesh_3d["replicate", "shard"]
# Create a 1D mesh for Tensor Parallelism by selecting the 'tp' dimension.
tp_mesh = mesh_3d["tp"]

print(f"Parent 3D Mesh Shape: {mesh_3d.shape}")
print(f"HSDP (2D) Sub-Mesh Shape: {hsdp_mesh.shape}")
print(f"TP (1D) Sub-Mesh Shape: {tp_mesh.shape}")

# Step 3: Access the process groups for each sub-mesh.
replicate_group = hsdp_mesh["replicate"].get_group()
shard_group = hsdp_mesh["shard"].get_group()
tp_group = tp_mesh.get_group()
print("Process groups for each parallelism dimension are ready.")
```

This approach allows clean separation of concerns. You can pass `hsdp_mesh` to FSDP and `tp_mesh` to a tensor parallelism library, each managing its own dimension of the overall parallel strategy.

## Conclusion

In this tutorial, you learned how `DeviceMesh` and `init_device_mesh` provide a declarative API to manage the layout of devices across a cluster. This abstraction significantly reduces the boilerplate and complexity of setting up multi-dimensional parallel training.

### Key Takeaways:
1.  **Simplification:** Replaces dozens of lines of manual `ProcessGroup` creation with a single `init_device_mesh` call.
2.  **Integration:** Works seamlessly with PyTorch distributed features like `FSDP`.
3.  **Composability:** Enables complex parallelism strategies through easy mesh slicing.

### Further Reading
*   **[Example: 2D Parallel Combining FSDP with Tensor Parallelism](https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/fsdp_tp_example.py)**
*   **[Slides: Composable PyTorch Distributed with PT2](https://static.sched.com/hosted_files/pytorch2023/d1/%5BPTC%2023%5D%20Composable%20PyTorch%20Distributed%20with%20PT2.pdf)**