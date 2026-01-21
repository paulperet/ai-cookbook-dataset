# PyTorch Compile Time Caching Guide

**Author:** [Oguz Ulgen](https://github.com/oulgen)

## Introduction

PyTorch Compiler provides several caching mechanisms to reduce compilation latency. This guide explains these offerings in detail to help you select the best option for your use case.

**Related Resources:**
- [Compile Time Caching Configurations](https://pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html) - How to configure these caches
- [PT CacheBench Benchmarks](https://hud.pytorch.org/benchmark/llms?repoName=pytorch%2Fpytorch&benchmarkName=TorchCache+Benchmark) - Performance benchmarks

## Prerequisites

Before starting, ensure you have:

- **Basic understanding of `torch.compile`**:
  - [torch.compiler API documentation](https://pytorch.org/docs/stable/torch.compiler.html#torch-compiler)
  - [Introduction to torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
  - [Triton language documentation](https://triton-lang.org/main/index.html)
- **PyTorch 2.4 or later**

## Caching Overview

`torch.compile` provides two main caching approaches:

1. **End-to-end caching (Mega-Cache)** - Portable cache artifacts
2. **Modular caching** - Individual component caches (TorchDynamo, TorchInductor, Triton)

**Important:** All caches validate that artifacts are used with:
- Same PyTorch version
- Same Triton version  
- Same GPU (when using CUDA)

## Step 1: Using End-to-End Caching (Mega-Cache)

Mega-Cache is ideal for portable caching solutions that can be stored in a database and used across different machines. It provides two main APIs:

### 1.1 Save Cache Artifacts

First, compile your model and save the cache artifacts:

```python
import torch

# Define a simple function to compile
@torch.compile
def fn(x, y):
    return x.sin() @ y

# Create sample tensors
dtype = torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.rand(100, 100, dtype=dtype, device=device)
b = torch.rand(100, 100, dtype=dtype, device=device)

# Compile and execute
result = fn(a, b)

# Save cache artifacts
artifacts = torch.compiler.save_cache_artifacts()
assert artifacts is not None

# Extract the portable bytes and cache info
artifact_bytes, cache_info = artifacts

# Now you can store artifact_bytes in a database
# Use cache_info for logging purposes
print(f"Cache info: {cache_info}")
```

### 1.2 Load Cache Artifacts

Later, potentially on a different machine, load the artifacts to pre-populate the cache:

```python
# Fetch artifacts from your database/storage
# Then load them to jump-start the cache
torch.compiler.load_cache_artifacts(artifact_bytes)

# Now subsequent compilations will use the pre-populated cache
```

## Step 2: Understanding Modular Caching

Mega-Cache is composed of individual modular caches that work automatically in the background. By default, PyTorch Compiler maintains local on-disk caches for:

### 2.1 Available Modular Caches

1. **FXGraphCache** - Caches graph-based IR components used during compilation
2. **TritonCache** - Stores Triton-compilation results, including generated `cubin` files
3. **InductorCache** - Bundles FXGraphCache and Triton cache
4. **AOTAutogradCache** - Caches joint graph artifacts
5. **PGO-cache** - Stores dynamic shape decisions to reduce recompilations
6. **AutotuningCache** - Caches benchmark results for selecting the fastest Triton kernels

### 2.2 Cache Location

All modular cache artifacts are written to `TORCHINDUCTOR_CACHE_DIR`, which defaults to:
```
/tmp/torchinductor_<your_username>
```

You can override this location by setting the environment variable:
```bash
export TORCHINDUCTOR_CACHE_DIR=/path/to/your/cache
```

## Step 3: Remote Caching with Redis

For team environments or distributed workflows, PyTorch supports Redis-based remote caching:

**Configuration:** Refer to the [Compile Time Caching Configurations](https://pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html) tutorial for setup instructions.

**Benefits:**
- Shared cache across team members
- Reduced compilation times in CI/CD pipelines
- Centralized cache management

## Conclusion

In this guide, you've learned:

1. **Mega-Cache** provides portable end-to-end caching that can be saved and loaded across machines
2. **Modular caches** work automatically in the background to reduce compilation latency
3. **Remote caching** with Redis enables team-wide cache sharing

PyTorch's caching mechanisms operate seamlessly, requiring minimal user intervention while significantly improving compilation performance. Choose Mega-Cache for portable scenarios and rely on modular caches for local development workflows.