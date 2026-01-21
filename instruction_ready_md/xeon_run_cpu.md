# Optimizing PyTorch Inference on Intel® Xeon® with `run_cpu`

This guide provides a step-by-step tutorial for optimizing PyTorch inference performance on Intel® Xeon® Scalable Processors using the `torch.backends.xeon.run_cpu` script. You will learn how to configure thread affinity, memory allocation, and NUMA policies to achieve peak performance for both single and multi-instance workloads.

## What You Will Learn
- How to use system tools (`numactl`, `taskset`) and optimized libraries (Intel® OpenMP, TCMalloc, JeMalloc) to enhance performance.
- How to configure CPU cores and memory management to maximize PyTorch inference throughput and latency on Intel® Xeon® processors.

## Prerequisites
Ensure you have PyTorch installed. The `run_cpu` script is included with PyTorch. You may also need to install system utilities and optimized libraries depending on your configuration.

### 1. Understanding System Configuration with NUMA
Modern multi-socket CPUs use Non-Uniform Memory Access (NUMA), where memory local to a CPU socket is faster to access than remote memory. Properly binding your workload to specific cores and local memory is critical for performance.

First, examine your CPU topology using `lscpu`:

```bash
lscpu
```

Look for key information:
- `CPU(s)`: Total logical cores.
- `Core(s) per socket`: Physical cores per socket.
- `Socket(s)`: Number of CPU sockets.
- `NUMA node(s)`: Number of NUMA nodes (usually matches sockets).
- `NUMA nodeX CPU(s)`: Core IDs belonging to each NUMA node.

For deep learning workloads, you should typically bind to physical cores and avoid logical cores (Hyper-Threads) for best performance.

### 2. Installing System Tools and Optimized Libraries
The `run_cpu` script leverages several system tools and libraries. Install them if they are not already present.

#### Install `numactl` and `taskset`
These tools control NUMA policy and CPU affinity.

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install numactl util-linux
```

**On CentOS/RHEL:**
```bash
sudo yum install numactl util-linux
```

#### Install Intel® OpenMP Runtime Library
Intel's OpenMP library (`libiomp`) often provides better performance than the default GNU OpenMP on Intel platforms.

Install via `pip` or `conda`:
```bash
pip install intel-openmp
```
or
```bash
conda install mkl
```

#### Install Optimized Memory Allocators
Efficient memory allocators like TCMalloc or JeMalloc can reduce overhead and improve performance.

**Install TCMalloc:**
- Ubuntu: `sudo apt-get install google-perftools`
- CentOS: `sudo yum install gperftools`
- Conda: `conda install conda-forge::gperftools`

**Install JeMalloc:**
- Ubuntu: `sudo apt-get install libjemalloc2`
- CentOS: `sudo yum install jemalloc`
- Conda: `conda install conda-forge::jemalloc`

### 3. Using the `torch.backends.xeon.run_cpu` Script
The `run_cpu` script automates the configuration of thread affinity, memory allocation, and NUMA binding. View its help menu to see all options:

```bash
python -m torch.backends.xeon.run_cpu --help
```

#### Key Configuration Knobs
The script offers several knobs to control optimization:

**Generic Options:**
- `-m`, `--module`: Run the script as a Python module.
- `--no-python`: Execute the program directly (for non-Python scripts).
- `--log-path`: Directory for log files.
- `--log-file-prefix`: Prefix for log filenames (default: `"run"`).

**Optimization Knobs:**
- `--enable-tcmalloc`: Enable TCMalloc memory allocator.
- `--enable-jemalloc`: Enable JeMalloc memory allocator.
- `--use-default-allocator`: Use PyTorch's default allocator.
- `--disable-iomp`: Disable Intel OpenMP (use GNU OpenMP).

**Resource Allocation Knobs:**
- `--ninstances`: Number of instances (thread groups) to launch.
- `--ncores-per-instance`: Cores allocated per instance.
- `--node-id`: Specify a NUMA node (socket) to use.
- `--core-list`: Manually specify core IDs or ranges (e.g., `'0,1,2,3'` or `'0-3'`).
- `--use-logical-core`: Enable use of logical cores (Hyper-Threads).
- `--skip-cross-node-cores`: Prevent workload from spanning NUMA nodes.
- `--latency-mode`: Quick preset for latency benchmarking (uses all physical cores, 4 cores per instance).
- `--throughput-mode`: Quick preset for throughput benchmarking (uses all physical cores, one NUMA node per instance).
- `--disable-numactl`: Disable NUMA control via `numactl`.
- `--disable-taskset`: Disable CPU affinity control via `taskset`.

> **Note:** If you do not specify a memory allocator (`--enable-tcmalloc` or `--enable-jemalloc`), the script will search for them in the order TCMalloc > JeMalloc > Default, and use the first one found.

### 4. Practical Examples
Here are common usage patterns for the `run_cpu` script. Replace `<program.py> [program_args]` with your actual script and its arguments.

#### Example 1: Single-Instance Inference on One Core
Run your program using only Core #0. This is useful for testing or very lightweight workloads.

```bash
python -m torch.backends.xeon.run_cpu --ninstances 1 --ncores-per-instance 1 <program.py> [program_args]
```

#### Example 2: Single-Instance Inference on One NUMA Node
Bind your workload to all cores on NUMA node 0 (typically the first socket). This ensures all memory access is local.

```bash
python -m torch.backends.xeon.run_cpu --node-id 0 <program.py> [program_args]
```

#### Example 3: Multi-Instance Inference for High Throughput
Launch 8 instances, each using 14 cores, on a 112-core system. This is ideal for maximizing throughput with multiple concurrent model executions.

```bash
python -m torch.backends.xeon.run_cpu --ninstances 8 --ncores-per-instance 14 <program.py> [program_args]
```

#### Example 4: Throughput Mode Preset
Use the built-in throughput mode, which automatically configures one instance per NUMA node using all physical cores.

```bash
python -m torch.backends.xeon.run_cpu --throughput-mode <program.py> [program_args]
```

> **Note:** In this context, an "instance" refers to a group of threads managed by the script within a single process, not a separate cloud instance.

### 5. Understanding Environment Variables
The `run_cpu` script sets several environment variables to optimize performance. It respects any pre-existing values, so it will not overwrite them if they are already set.

Key variables include:
- `LD_PRELOAD`: Preloads optimized libraries (e.g., `libiomp5.so`, `libtcmalloc.so`).
- `KMP_AFFINITY`: Sets thread affinity for Intel OpenMP (e.g., `"granularity=fine,compact,1,0"`).
- `KMP_BLOCKTIME`: Sets time before threads sleep after completing work (set to `"1"`).
- `OMP_NUM_THREADS`: Set to the value of `--ncores-per-instance`.
- `MALLOC_CONF`: Configures JeMalloc settings for performance.

## Conclusion
You have learned how to use the `torch.backends.xeon.run_cpu` script to optimize PyTorch inference on Intel® Xeon® processors. By properly configuring thread affinity, NUMA memory access, and memory allocators, you can significantly improve the performance of your workloads. Experiment with the provided examples and knobs to find the optimal configuration for your specific application.

## Further Reading
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#cpu-specific-optimizations)
- [PyTorch Multiprocessing Best Practices](https://pytorch.org/docs/stable/notes/multiprocessing.html#cpu-in-multiprocessing)