# Exploring TorchRec Sharding: A Practical Guide

This tutorial explores the sharding schemes available for embedding tables in TorchRec using the `EmbeddingPlanner` and `DistributedModelParallel` APIs. You'll learn how to explicitly configure different sharding strategies and understand their benefits for distributed training scenarios.

## Prerequisites

Before starting, ensure you have:
- Python ≥ 3.7
- CUDA ≥ 11.0 (recommended for GPU acceleration)

## Setup

First, let's install the necessary packages. We'll use Miniconda to manage the PyTorch installation with CUDA 11.3 support.

```bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh
chmod +x Miniconda3-py37_4.9.2-Linux-x86_64.sh
bash ./Miniconda3-py37_4.9.2-Linux-x86_64.sh -b -f -p /usr/local
```

```bash
# Install PyTorch with CUDA 11.3
conda install pytorch cudatoolkit=11.3 -c pytorch-nightly -y
```

Now install TorchRec, which includes FBGEMM (a collection of CUDA kernels and GPU-enabled operations):

```bash
pip install torchrec-nightly
```

For multi-processing support within this environment:

```bash
pip install multiprocess
```

**Important for Colab Runtime**: Copy shared libraries to the expected location and restart your runtime after installation:

```bash
sudo cp /usr/local/lib/lib* /usr/lib/
```

After restarting, update the Python path:

```python
import sys
sys.path = ['', '/env/python', '/usr/local/lib/python37.zip', '/usr/local/lib/python3.7', '/usr/local/lib/python3.7/lib-dynload', '/usr/local/lib/python3.7/site-packages', './.local/lib/python3.7/site-packages']
```

## Distributed Setup Configuration

Since we're working in a notebook environment, we'll use multiprocessing to simulate a distributed setup. In production, you would use an SPMD launcher.

Configure the distributed environment variables:

```python
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
```

## Step 1: Constructing the Embedding Model

We'll use TorchRec's `EmbeddingBagCollection` to create a model with four embedding bags. We'll have two types of tables:
- **Large tables**: 4096 rows, 64-dimensional embeddings
- **Small tables**: 1024 rows, 64-dimensional embeddings

First, let's define helper functions to generate table configurations and constraints:

```python
import torch
import torchrec
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.types import ShardingType
from typing import Dict

# Configuration for large and small tables
large_table_cnt = 2
small_table_cnt = 2

large_tables = [
    torchrec.EmbeddingBagConfig(
        name="large_table_" + str(i),
        embedding_dim=64,
        num_embeddings=4096,
        feature_names=["large_table_feature_" + str(i)],
        pooling=torchrec.PoolingType.SUM,
    ) for i in range(large_table_cnt)
]

small_tables = [
    torchrec.EmbeddingBagConfig(
        name="small_table_" + str(i),
        embedding_dim=64,
        num_embeddings=1024,
        feature_names=["small_table_feature_" + str(i)],
        pooling=torchrec.PoolingType.SUM,
    ) for i in range(small_table_cnt)
]

def gen_constraints(sharding_type: ShardingType = ShardingType.TABLE_WISE) -> Dict[str, ParameterConstraints]:
    """Generate parameter constraints for all tables with a specific sharding type."""
    large_table_constraints = {
        "large_table_" + str(i): ParameterConstraints(
            sharding_types=[sharding_type.value],
        ) for i in range(large_table_cnt)
    }
    
    small_table_constraints = {
        "small_table_" + str(i): ParameterConstraints(
            sharding_types=[sharding_type.value],
        ) for i in range(small_table_cnt)
    }
    
    constraints = {**large_table_constraints, **small_table_constraints}
    return constraints
```

Now create the EmbeddingBagCollection. Note that we initially allocate it on the "cuda" device:

```python
ebc = torchrec.EmbeddingBagCollection(
    device="cuda",
    tables=large_tables + small_tables
)
```

## Step 2: Distributed Execution Function

We'll define a function that simulates a single rank's work in a distributed setup. This function:
1. Initializes the distributed process group
2. Uses the planner to create a sharding plan
3. Wraps the model with `DistributedModelParallel` to create the sharded version

```python
def single_rank_execution(
    rank: int,
    world_size: int,
    constraints: Dict[str, ParameterConstraints],
    module: torch.nn.Module,
    backend: str,
) -> None:
    """Execute a single rank's work in distributed training."""
    import os
    import torch
    import torch.distributed as dist
    from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
    from torchrec.distributed.model_parallel import DistributedModelParallel
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.types import ModuleSharder, ShardingEnv
    from typing import cast

    def init_distributed_single_host(
        rank: int,
        world_size: int,
        backend: str,
    ) -> dist.ProcessGroup:
        """Initialize distributed process group for single host."""
        os.environ["RANK"] = f"{rank}"
        os.environ["WORLD_SIZE"] = f"{world_size}"
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        return dist.group.WORLD

    # Set device based on backend
    if backend == "nccl":
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # Initialize topology and process group
    topology = Topology(world_size=world_size, compute_device="cuda")
    pg = init_distributed_single_host(rank, world_size, backend)
    
    # Create planner and generate sharding plan
    planner = EmbeddingShardingPlanner(
        topology=topology,
        constraints=constraints,
    )
    sharders = [cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())]
    plan = planner.collective_plan(module, sharders, pg)
    
    # Create sharded model
    sharded_model = DistributedModelParallel(
        module,
        env=ShardingEnv.from_process_group(pg),
        plan=plan,
        sharders=sharders,
        device=device,
    )
    
    print(f"rank:{rank}, sharding plan: {plan}")
    return sharded_model
```

## Step 3: Multi-Process Execution Wrapper

To simulate multiple GPU ranks, we'll create a wrapper function that spawns multiple processes:

```python
import multiprocess

def spmd_sharing_simulation(
    sharding_type: ShardingType = ShardingType.TABLE_WISE,
    world_size: int = 2,
):
    """Simulate SPMD execution with multiple processes."""
    ctx = multiprocess.get_context("spawn")
    processes = []
    
    for rank in range(world_size):
        p = ctx.Process(
            target=single_rank_execution,
            args=(
                rank,
                world_size,
                gen_constraints(sharding_type),
                ebc,
                "nccl"
            ),
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        assert 0 == p.exitcode
```

## Step 4: Exploring Different Sharding Schemes

Now let's explore the different sharding schemes available in TorchRec.

### 4.1 Table-Wise Sharding

Table-wise sharding places entire tables on individual devices. This is ideal for load balancing small to medium-sized tables across devices.

```python
spmd_sharing_simulation(ShardingType.TABLE_WISE)
```

**Expected Output**: Each rank will print its sharding plan. You'll see that the planner balances the tables across devices - each GPU gets one large table and one small table.

### 4.2 Row-Wise Sharding

Row-wise sharding splits tables along the row dimension and distributes shards across devices. This is useful for very large tables that don't fit on a single device.

```python
spmd_sharing_simulation(ShardingType.ROW_WISE)
```

**Expected Output**: In the sharding plan, you'll see `shard_sizes` showing tables split by row dimension (e.g., `[2048, 64]` for large tables, indicating half the rows on each device).

### 4.3 Column-Wise Sharding

Column-wise sharding splits tables along the embedding dimension. This addresses load imbalance for tables with large embedding dimensions.

```python
spmd_sharing_simulation(ShardingType.COLUMN_WISE)
```

**Expected Output**: The `shard_sizes` will show tables split by column dimension (e.g., `[4096, 32]` for large tables, indicating half the embedding dimensions on each device).

### 4.4 Data Parallel Sharding

Data parallel sharding replicates tables across all devices. This is the traditional data parallelism approach.

```python
spmd_sharing_simulation(ShardingType.DATA_PARALLEL)
```

**Expected Output**: The sharding plan shows `sharding_type='data_parallel'` with `ranks=[0, 1]`, indicating replication across both devices.

## Key Takeaways

1. **Table-wise sharding** is ideal for load balancing multiple small to medium tables across devices.
2. **Row-wise sharding** addresses memory constraints for very large tables by splitting them across devices.
3. **Column-wise sharding** helps with tables that have large embedding dimensions.
4. **Data parallel sharding** replicates tables across all devices (traditional approach).
5. **Table-row-wise sharding** (not demonstrated here) is optimized for multi-host setups with fast intra-machine interconnects like NVLink.

## Next Steps

For production deployment, you would:
1. Use a proper SPMD launcher instead of multiprocessing
2. Configure your actual hardware topology in the `Topology` object
3. Add data loading and training loops around the sharded model

This tutorial has shown you how to configure and explore different sharding schemes in TorchRec. Experiment with different table sizes and world sizes to see how the planner adapts to your specific configuration.