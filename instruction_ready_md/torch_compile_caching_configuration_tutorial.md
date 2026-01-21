# PyTorch Compile Time Caching Configuration Guide

**Authors:** [Oguz Ulgen](https://github.com/oulgen) and [Sam Larsen](https://github.com/masnesral)

## Introduction

PyTorch Compiler (`torch.compile`) implements several caching mechanisms to reduce compilation latency. This guide explains how to configure these caches to optimize performance for your workflows.

## Prerequisites

Before starting, ensure you have:

*   A basic understanding of `torch.compile`. Recommended reading:
    *   [torch.compiler API documentation](https://pytorch.org/docs/stable/torch.compiler.html#torch-compiler)
    *   [Introduction to torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
    *   [Compile Time Caching in torch.compile](https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)
*   PyTorch 2.4 or later installed.

## Understanding Inductor Caches

Most caches are in-memory and transparent to the user. However, the **FX Graph Cache** and **AOTAutograd Cache** are special. They store compiled FX graphs on disk, allowing `torch.compile` to skip recompilation across different process runs when it encounters the same computation graph with identical tensor input shapes and configuration.

By default, these artifacts are stored in your system's temporary directory. An advanced option also supports sharing cached artifacts across a cluster using a Redis database.

## Configuring Cache Settings

You can control caching via environment variables. The following sections detail each setting.

### 1. Enable Local FX Graph Cache (`TORCHINDUCTOR_FX_GRAPH_CACHE`)

This setting enables the local on-disk cache for compiled FX graphs.

*   **Set to:** `1` to enable, any other value to disable.
*   **Default Location:** System temp directory (e.g., `/tmp/torchinductor_<your_username>`).

### 2. Enable Local AOTAutograd Cache (`TORCHINDUCTOR_AUTOGRAD_CACHE`)

This extends caching to the AOTAutograd level, which can be more efficient than the FX Graph Cache alone.

*   **Set to:** `1` to enable, any other value to disable.
*   **Requirement:** `TORCHINDUCTOR_FX_GRAPH_CACHE` must also be enabled (`1`).
*   **Storage:** Cache entries are stored under subdirectories in the shared cache directory:
    *   `{CACHE_DIR}/aotautograd` for AOTAutogradCache.
    *   `{CACHE_DIR}/fxgraph` for FXGraphCache.

### 3. Specify a Custom Cache Directory (`TORCHINDUCTOR_CACHE_DIR`)

Override the default temporary directory location for all on-disk caches.

*   **Example:** `export TORCHINDUCTOR_CACHE_DIR="/path/to/my/cache"`
*   **Note:** If `TRITON_CACHE_DIR` is not set, the Triton compiler cache will also be placed under a subdirectory in this location.

### 4. Enable Remote FX Graph Cache (`TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE`)

Enable a shared, remote cache using a Redis server. This is useful for sharing compiled artifacts across machines in a cluster.

*   **Set to:** `1` to enable, any other value to disable.
*   **Configuration:** Set the Redis server host and port.
    ```bash
    export TORCHINDUCTOR_REDIS_HOST="your.redis.host"
    export TORCHINDUCTOR_REDIS_PORT=6379
    ```
*   **Important:** When a remote cache entry is found, the artifact is downloaded and stored in the *local* on-disk cache (`TORCHINDUCTOR_CACHE_DIR`). Subsequent runs on the same machine will use this local copy.

### 5. Enable Remote AOTAutograd Cache (`TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE`)

Enable a remote cache for AOTAutograd-level artifacts.

*   **Set to:** `1` to enable, any other value to disable.
*   **Requirement:** `TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE` must also be enabled (`1`).
*   **Configuration:** Uses the same `TORCHINDUCTOR_REDIS_HOST` and `TORCHINDUCTOR_REDIS_PORT` variables. The same Redis server can store both FX Graph and AOTAutograd cache entries.

### 6. Enable Remote Autotuner Cache (`TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE`)

Enable a remote cache for Inductor's autotuner results, which can significantly speed up the kernel tuning process.

*   **Set to:** `1` to enable, any other value to disable.
*   **Configuration:** Uses the same Redis host/port environment variables as the remote graph caches.

### 7. Force Disable All Caches (`TORCHINDUCTOR_FORCE_DISABLE_CACHES`)

A master switch to disable all Inductor caching mechanisms. Useful for debugging or measuring cold-start compilation performance.

*   **Set to:** `1` to disable all caches.

## Summary

This guide covered the key environment variables for configuring PyTorch Compiler's caching. By adjusting these settings, you can optimize compilation times for local development, CI/CD pipelines, or distributed training clusters.