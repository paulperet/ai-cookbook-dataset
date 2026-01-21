# Debugging Stuck Distributed Training Jobs with PyTorch Flight Recorder

## Overview

A distributed AI training job is considered **stuck** when it stops making meaningful progress for an extended period. This can happen due to various reasons like data starvation, resource constraints, network issues, software bugs, or synchronization problems like deadlocks in collective operations.

PyTorch Flight Recorder is a diagnostic tool that captures information about collective operations during distributed training. When a job gets stuck or times out, this captured data helps identify the root cause by showing which ranks failed to join collectives or experienced other issues.

## Prerequisites

- PyTorch version 2.5 or later
- Install the `tabulate` package:
```bash
pip install tabulate
```

## Enabling Flight Recorder

Flight Recorder is controlled through environment variables. Here are the essential settings:

### Required Settings

```bash
# Enable collection with a buffer of 2000 entries (recommended)
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000

# Dump diagnostic files to disk on job timeout
export TORCH_NCCL_DUMP_ON_TIMEOUT=1

# Set the dump file path prefix (one file per rank)
export TORCH_FR_DUMP_TEMP_FILE="/tmp/nccl_trace_rank_"
```

### Optional Settings

```bash
# Enable C++ stack traces for detailed code path information
export TORCH_NCCL_TRACE_CPP_STACK=1

# Enable timing information for collectives (adds some CPU overhead)
export TORCH_NCCL_ENABLE_TIMING=1

# Use faster symbolization for C++ traces (experimental)
export TORCH_SYMBOLIZE_MODE=fast
```

## Retrieving Flight Recorder Data Programmatically

You can also retrieve Flight Recorder data via API calls instead of relying on file dumps:

```python
import torch
import pickle

# Dump the trace data
trace_data = torch._C._distributed_c10d._dump_nccl_trace(
    includeCollectives=True, 
    includeStackTraces=True, 
    onlyActive=False
)

# Unpickle and view the data
t = pickle.loads(trace_data)
print(t)
```

## Understanding Flight Recorder File Format

Flight Recorder files are dumped in pickle format. When unpickled, the data structure looks like this:

```json
{
  "version": "2.5",
  "pg_config": {
    "0": {
      "name": "0",
      "desc": "default_pg",
      "ranks": "[0, 1]"
    }
  },
  "pg_status": {
    "0": {
      "last_enqueued_collective": 2,
      "last_started_collective": -1,
      "last_completed_collective": 2
    }
  },
  "entries": [
    {
      "frames": [
        {
          "name": "test_short_pickle",
          "filename": "pytorch/test/distributed/test_c10d_nccl.py",
          "line": 3647
        }
      ],
      "record_id": 0,
      "pg_id": 0,
      "process_group": ("0", "default_pg"),
      "collective_seq_id": 1,
      "p2p_seq_id": 0,
      "op_id": 1,
      "profiling_name": "nccl:all_reduce",
      "time_created_ns": 1724779239936775119,
      "input_sizes": [[3, 4]],
      "input_dtypes": ["Float"],
      "output_sizes": [[3, 4]],
      "output_dtypes": ["Float"],
      "state": "completed",
      "time_discovered_started_ns": null,
      "time_discovered_completed_ns": 1724779239975811724,
      "retired": true,
      "timeout_ms": 600000,
      "is_p2p": false
    }
  ]
}
```

## Analyzing Flight Recorder Dumps

### Step 1: Prepare Your Trace Files

First, copy all Flight Recorder dump files from each rank into a single directory.

### Step 2: Run the Analyzer Script

You have two options to run the analyzer:

**Option A: Using the script directly (if you have the PyTorch source):**
```bash
python fr_trace.py <dump_dir_containing_trace_files> [-o <output_file>]
```

**Option B: Using the installed command (PyTorch nightly or built from source with `USE_DISTRIBUTED=1`):**
```bash
torchfrtrace <dump_dir_containing_trace_files> [-o <output_file>]
```

### Step 3: Customize Your Analysis

The analyzer supports filtering by specific ranks and process groups:

```bash
# Analyze specific ranks (i, j, k) and process groups (0, 2)
torchfrtrace <dump_dir> -j --selected-ranks i j k --pg-filters 0 2
```

## End-to-End Example: Debugging a Mismatched Collective

Let's walk through a practical example where we intentionally create a stuck job by having mismatched collectives between ranks.

### Step 1: Create the Test Script

Create a file called `crash.py` with the following content:

```python
import torch
import torch.distributed as dist
import os
from datetime import timedelta

# Get rank information
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert world_size <= 8, "world size must be less than or equal to 8"

# Enable Flight Recorder
os.environ["TORCH_FR_DUMP_TEMP_FILE"] = "/tmp/trace_"
os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "2000"

device = torch.device(f"cuda:{local_rank}")
print(f"{local_rank=} {world_size=} {device=}")

# Initialize process group with short timeout for quick failure
dist.init_process_group("nccl", world_size=world_size, rank=local_rank, timeout=timedelta(seconds=1))

# Create a tensor and perform some collectives
a = torch.full((3, 4), float(local_rank), device=device)

for i in range(2):
    print(f"calling allreduce on {local_rank=}")
    f = dist.all_reduce(a)

# Rank 0 does an extra collective that other ranks don't know about
# This will cause the job to get stuck
if local_rank == 0:
    print("rank0 is doing an allreduce on tensor b, but other ranks forgot")
    b = torch.full((4, 5), float(local_rank), device=device)
    f = dist.all_reduce(b)

# Try more collectives (these will fail due to the mismatch above)
for i in range(2):
    print(f"calling allreduce on {local_rank=}")
    f = dist.all_reduce(a)

torch.cuda.synchronize(device=device)
print(f"{local_rank=} exiting")
```

### Step 2: Run the Distributed Job

Execute the script with `torchrun` using 2 processes:

```bash
torchrun --nnodes=1 --nproc_per_node=2 crash.py
```

The job will fail due to the mismatched collective. Flight Recorder will automatically dump trace files to `/tmp`.

### Step 3: Check the Generated Trace Files

Verify that trace files were created:

```bash
ls /tmp/trace_*
```

You should see two files:
```
/tmp/trace_0
/tmp/trace_1
```

### Step 4: Analyze the Trace Files

Run the analyzer to understand what went wrong:

```bash
torchfrtrace --prefix "trace_" /tmp/
```

### Step 5: Interpret the Results

The analyzer output will clearly show the problem:

```
Not all ranks joining collective 5 at entry 4
group info: 0:default_pg
collective: nccl:all_reduce
missing ranks: {1}
input sizes: [[3, 4]]
output sizes: [[3, 4]]
expected ranks: 2
collective state: scheduled
collective stack trace:
all_reduce at /home/cpio/local/pytorch/torch/distributed/distributed_c10d.py:2696
wrapper at /home/cpio/local/pytorch/torch/distributed/c10d_logger.py:83
<module> at /home/cpio/test/crash.py:44
```

The output clearly indicates that rank 1 failed to join the "all_reduce" collective, which is exactly what we programmed - rank 0 executed an extra collective that rank 1 didn't know about.

## Conclusion

Flight Recorder is a powerful tool for debugging stuck distributed training jobs. By capturing detailed information about collective operations, it helps you quickly identify synchronization issues, deadlocks, and other problems that cause jobs to get stuck.

Key takeaways:
1. Enable Flight Recorder with environment variables before starting your training job
2. The tool automatically captures data in a circular buffer
3. On timeout, data is dumped to files for analysis
4. Use the `torchfrtrace` analyzer to identify which ranks failed to join collectives
5. The analyzer provides human-readable output with stack traces to help pinpoint the exact location of the problem

For more information and the latest updates, refer to the [Flight Recorder directory](https://github.com/pytorch/pytorch/tree/main/torch/distributed/flight_recorder) in the PyTorch repository.