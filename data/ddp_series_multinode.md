# Multi-Node Distributed Training with PyTorch

## Overview

This guide walks you through the process of scaling your PyTorch training jobs from a single machine with multiple GPUs to multiple machines (nodes) across a network. You will learn the minimal code changes required and how to launch jobs using `torchrun` and workload managers like SLURM.

### Prerequisites

-   Familiarity with single-node multi-GPU training using PyTorch's Distributed Data Parallel (DDP).
-   Two or more GPU-equipped machines that can communicate over TCP (this tutorial uses AWS p3.2xlarge instances).
-   PyTorch with CUDA installed on all machines.

### What You Will Learn

-   Launching multi-node training jobs with `torchrun`.
-   Understanding the differences between local and global ranks.
-   Key considerations and troubleshooting steps for multi-node setups.

---

## 1. Understanding Ranks in a Multi-Node Context

In a single-node setup, you typically track each GPU by an ID. In a distributed setting managed by `torchrun`, two key environment variables are provided:

*   **`LOCAL_RANK`**: Uniquely identifies each GPU process **within a single node**.
*   **`RANK`** (Global Rank): Uniquely identifies each process **across all nodes** in the entire job.

**Important:** Do not rely on `RANK` for critical logic in fault-tolerant jobs. When `torchrun` restarts processes after a failure, there is no guarantee that a process will retain its original `LOCAL_RANK` or `RANK`.

## 2. Code Modifications for Multi-Node Training

The transition from single-node to multi-node training requires minimal code changes. The primary adjustment is in how you initialize the distributed process group. Here is the standard pattern:

```python
import torch
import torch.distributed as dist
import os

def setup(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()
```

In practice, you will use the environment variables provided by `torchrun`:

```python
import os

local_rank = int(os.environ['LOCAL_RANK'])
global_rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])

# Use the global_rank for process group initialization
setup(global_rank, world_size)

# Use the local_rank to set the device for this process
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

# Your model, optimizer, and dataloader setup goes here
# Remember to wrap your model with DDP:
# model = DDP(model, device_ids=[local_rank])
```

The rest of your training loop (forward/backward pass, optimizer step) remains identical to the single-node multi-GPU case.

## 3. Launching Jobs with `torchrun`

You can launch a multi-node job by running an identical `torchrun` command on each machine. The key is the `--rdzv_endpoint` argument, which points to a "rendezvous" server (typically the first node).

### Step-by-Step Launch

1.  **On the first node (Node 0, IP: `192.168.1.1`)**, run:
    ```bash
    torchrun \
        --nnodes=2 \
        --nproc_per_node=4 \
        --rdzv_id=123 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=192.168.1.1:29500 \
        YOUR_TRAINING_SCRIPT.py
    ```
    *   `--nnodes=2`: Total number of nodes.
    *   `--nproc_per_node=4`: Processes (GPUs) per node.
    *   `--rdzv_id=123`: A unique ID for this job session.
    *   `--rdzv_endpoint`: The address of the rendezvous server (this node).

2.  **On the second node (Node 1)**, run the **exact same command**. `torchrun` will use the provided endpoint to discover and join the job.

## 4. Heterogeneous Scaling

`torchrun` supports heterogeneous scaling, meaning each node in your cluster can use a different number of GPUs. For example, you could have:
*   Node 0: `--nproc_per_node=4`
*   Node 1: `--nproc_per_node=2`

The total `world_size` would be 6. Your training script does not need modification for this; `torchrun` handles the process orchestration.

## 5. Troubleshooting Common Issues

Multi-node training introduces network dependencies. Use these tips to diagnose problems:

*   **Network Connectivity**: Ensure all nodes can reach each other over TCP on the specified `MASTER_PORT` (or `rdzv_endpoint` port). Use tools like `ping` or `telnet`.
*   **Verbose Logging**: Set `export NCCL_DEBUG=INFO` before running your script to print detailed NCCL communication logs.
*   **Network Interface**: In some environments (e.g., AWS), you may need to explicitly specify the network interface:
    ```bash
    export NCCL_SOCKET_IFNAME=eth0
    ```
*   **Firewalls/Security Groups**: Verify that security group rules allow traffic between nodes on the necessary ports.

## 6. Using a Cluster Manager (SLURM)

On HPC clusters managed by SLURM, you typically use `srun` instead of manually running `torchrun` on each node. Your SLURM submission script (`job.slurm`) would look like this:

```bash
#!/bin/bash
#SBATCH --job-name=pt_multinode
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4

# Set master address to the first node
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

# Launch the training job
srun python YOUR_TRAINING_SCRIPT.py
```

The cluster manager handles node allocation and process launching, but the core PyTorch distributed code in your script remains the same.

## Key Takeaways

1.  Moving from single-node to multi-node training requires only minor changes to your initialization code.
2.  Use `torchrun` environment variables (`LOCAL_RANK`, `RANK`, `WORLD_SIZE`) to configure your training script.
3.  Multi-node performance is often bottlenecked by inter-node network latency. Training on 4 GPUs in one node is usually faster than on 4 nodes with 1 GPU each.
4.  Always test network connectivity and use `NCCL_DEBUG` logs to troubleshoot launch issues.

### Next Steps

*   Experiment with the [complete example script](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multinode.py) on GitHub.
*   Learn to train a real model by following the [minGPT with DDP](ddp_series_minGPT.html) tutorial.
*   For production clusters, review the guide on [setting up a cluster on AWS](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/slurm/setup_pcluster_slurm.md).