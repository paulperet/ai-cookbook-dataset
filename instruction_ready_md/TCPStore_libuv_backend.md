# PyTorch Distributed Training: Using the New Libuv TCPStore Backend

## Introduction

In PyTorch 2.4, we introduced a new TCPStore server backend built on [libuv](https://github.com/libuv/libuv), a high-performance asynchronous I/O library. This change addresses scalability and robustness challenges in large-scale distributed training jobs, particularly those with thousands of ranks.

The libuv backend significantly improves store initialization time while maintaining comparable I/O performance. As a result, it has become the default TCPStore backend in PyTorch 2.4. This tutorial will guide you through understanding the benefits, benchmarking results, and how to work with (or revert from) this new backend.

## Prerequisites

- PyTorch 2.4 or later
- Basic understanding of [TCPStore API](https://pytorch.org/docs/main/distributed.html#torch.distributed.TCPStore)

## Performance Benchmark

Let's examine the performance improvements through two key benchmarks.

### 1. TCPStore Initialization Time

First, we'll measure how quickly TCPStore initializes across different job sizes:

```python
import logging
import os
from time import perf_counter

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Environment variables are preset when launching the benchmark
env_rank = os.environ.get("RANK", 0)
env_world_size = os.environ.get("WORLD_SIZE", 1)
env_master_addr = os.environ.get("MASTER_ADDR", "localhost")
env_master_port = os.environ.get("MASTER_PORT", "23456")

# Measure TCPStore initialization time
start = perf_counter()
tcp_store = dist.TCPStore(
    env_master_addr,
    int(env_master_port),
    world_size=int(env_world_size),
    is_master=(int(env_rank) == 0),
)
end = perf_counter()

time_elapsed = end - start
logger.info(
    f"Complete TCPStore init with rank={env_rank}, world_size={env_world_size} in {time_elapsed} seconds."
)
```

**Key Findings:**
- The libuv backend consistently initializes faster than the legacy backend
- At 96K ranks, the legacy backend timed out (over 30 minutes) while libuv completed in ~100 seconds
- Performance improvements are most noticeable at super-large scales

### 2. Store-Based Barrier Performance

Next, let's benchmark the I/O performance using a store-based barrier operation:

```python
import logging
import os
import time
from datetime import timedelta
from time import perf_counter

import torch
import torch.distributed as dist

DistStoreError = torch._C._DistStoreError
logger = logging.getLogger(__name__)

# Custom store-based barrier implementation
def store_based_barrier(
    rank,
    store,
    group_name,
    rendezvous_count,
    timeout=dist.constants.default_pg_timeout,
    logging_interval=timedelta(seconds=10),
):
    store_key = f"store_based_barrier_key:{group_name}"
    store.add(store_key, 1)

    world_size = rendezvous_count
    worker_count = store.add(store_key, 0)

    last_worker_key = f"{store_key}:last_worker"
    if worker_count == world_size:
        store.set(last_worker_key, "1")

    start = time.time()
    while True:
        try:
            store.wait([last_worker_key], logging_interval)
            break
        except RuntimeError as e:
            worker_count = store.add(store_key, 0)
            logger.info(
                "Waiting in store based barrier to initialize process group for "
                "rank: %s, key: %s (world_size=%s, num_workers_joined=%s, timeout=%s)"
                "error: %s",
                rank,
                store_key,
                world_size,
                worker_count,
                timeout,
                e,
            )

            if timedelta(seconds=(time.time() - start)) > timeout:
                raise DistStoreError(
                    "Timed out initializing process group in store based barrier on "
                    "rank {}, for key: {} (world_size={}, num_workers_joined={}, timeout={})".format(
                        rank, store_key, world_size, worker_count, timeout
                    )
                )

    logger.info(
        "Rank %s: Completed store-based barrier for key:%s with %s nodes.",
        rank,
        store_key,
        world_size,
    )

# Environment setup
env_rank = os.environ.get("RANK", 0)
env_world_size = os.environ.get("WORLD_SIZE", 1)
env_master_addr = os.environ.get("MASTER_ADDR", "localhost")
env_master_port = os.environ.get("MASTER_PORT", "23456")

# Initialize TCPStore
tcp_store = dist.TCPStore(
    env_master_addr,
    int(env_master_port),
    world_size=int(env_world_size),
    is_master=(int(env_rank) == 0),
)

# Initial synchronization
store_based_barrier(int(env_rank), tcp_store, "tcpstore_test", int(env_world_size))

# Benchmark multiple barrier operations
number_runs = 10
start = perf_counter()
for _ in range(number_runs):
    store_based_barrier(
        int(env_rank), tcp_store, "tcpstore_test", int(env_world_size)
    )
end = perf_counter()

time_elapsed = end - start
logger.info(
    f"Complete {number_runs} TCPStore barrier runs with rank={env_rank}, world_size={env_world_size} in {time_elapsed} seconds."
)
```

**Key Findings:**
- The libuv backend maintains comparable I/O performance to the legacy backend
- Performance remains stable as the number of ranks increases
- Both backends show similar runtime characteristics for barrier operations

## Important Compatibility Note

The libuv backend currently doesn't support initialization with a `listen_fd`. If you need this functionality, you must explicitly use the legacy backend.

## Using the Legacy Backend (Exit Routes)

If you need to revert to the legacy TCPStore backend, here are three methods in order of priority:

### Exit Route 1: Direct TCPStore Initialization (Highest Priority)

Pass `use_libuv=False` when creating a TCPStore directly:

```python
import socket
import torch
import torch.distributed as dist

# Create a listening socket
listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listen_sock.bind(("localhost", 0))
addr, port, *_ = listen_sock.getsockname()
listen_fd = listen_sock.detach()

# This will fail with libuv backend
# tcpstore = dist.TCPStore(addr, port, 1, True, master_listen_fd=listen_fd)  # NotImplementedError

# Use legacy backend explicitly
tcpstore = dist.TCPStore(addr, port, 1, True, master_listen_fd=listen_fd, use_libuv=False)  # Works with legacy backend
```

### Exit Route 2: ProcessGroup Initialization

Add `use_libuv=0` to the `init_method` query string:

```python
import torch
import torch.distributed as dist

addr = "localhost"
port = 23456

dist.init_process_group(
    backend="cpu:gloo,cuda:nccl",
    rank=0,
    world_size=1,
    init_method=f"tcp://{addr}:{port}?use_libuv=0",  # Specify legacy backend
)

# Your distributed training code here

dist.destroy_process_group()
```

### Exit Route 3: Environment Variable (Lowest Priority)

Set the `USE_LIBUV` environment variable to `"0"`:

```python
import os
import torch
import torch.distributed as dist

addr = "localhost"
port = 23456

# Set environment variable before ProcessGroup initialization
os.environ["USE_LIBUV"] = "0"

dist.init_process_group(
    backend="cpu:gloo,cuda:nccl",
    rank=0,
    world_size=1,
    init_method=f"tcp://{addr}:{port}",
)

dist.destroy_process_group()
```

**Priority Order:** Route 1 > Route 2 > Route 3. For example, if you set `USE_LIBUV=1` but pass `use_libuv=0` in `init_method`, the legacy backend will be used.

## Conclusion

The new libuv TCPStore backend in PyTorch 2.4 provides significant performance improvements for large-scale distributed training initialization while maintaining comparable I/O performance. Although it introduces a minor incompatibility with `listen_fd` initialization, the benefits for most users are substantial.

For those needing the legacy backend, three straightforward exit routes are available. In the long term, we plan to eventually deprecate the legacy backend as the libuv backend becomes more mature and feature-complete.