# Distributed Training with Parameter Servers

This guide explores the core concepts and practical considerations for scaling deep learning training across multiple GPUs and servers using the Parameter Server architecture.

## Introduction

Moving from single-GPU to multi-GPU and multi-server training introduces significant complexity. Network interconnects vary widely in bandwidth (NVLink: ~100 GB/s, PCIe 4.0: 32 GB/s, 100GbE: 10 GB/s), and modelers shouldn't need to be networking experts. The Parameter Server paradigm, introduced to abstract this complexity, allows efficient distributed training by managing gradient aggregation and parameter updates across many devices.

## Prerequisites

This is a conceptual guide. To implement these strategies, you would typically use a framework like **Horovod**, **PyTorch Distributed**, or **TensorFlow MirroredStrategy**, which handle the underlying communication.

## 1. Revisiting Data-Parallel Training

The most common approach is **data parallelism**, where the same model is replicated across all devices, and each processes a different subset of the data.

**The Standard Approach:**
In a simple multi-GPU setup, gradients are computed on each GPU, aggregated onto one designated GPU (e.g., GPU 0), the parameters are updated there, and then broadcast back to all GPUs.

**Why is this design arbitrary?**
Mathematically, gradients can be aggregated anywhere—on any GPU or even on the CPU. The choice is a **systems optimization problem** dictated by physical hardware bandwidth constraints.

## 2. Hardware Bandwidth Hierarchy

Consider a typical 4-GPU server:
*   **GPU-to-GPU (via PCIe Switch):** ~16 GB/s per link.
*   **GPU-to-CPU:** ~16 GB/s (shared across all GPUs via limited PCIe lanes).
*   **Network (Ethernet):** 0.1–1 GB/s.

**Performance Thought Experiment:**
Synchronizing 160 MB of gradients:
*   **Aggregate on one GPU:** 3 GPUs send to GPU 0 (10 ms each) = 30 ms. Broadcast back = 30 ms. **Total: ~60 ms.**
*   **Aggregate on CPU:** All 4 GPUs send to CPU (10 ms each, serialized) = 40 ms. Broadcast back = 40 ms. **Total: ~80 ms.**
*   **Aggregate in parallel:** Split gradient into 4 parts, aggregate each part on a different GPU simultaneously via the PCIe switch = ~7.5 ms. **Total: ~15 ms.**

**Conclusion:** Synchronization strategy dramatically impacts performance. The optimal strategy depends entirely on your specific hardware topology.

## 3. Advanced Synchronization: Ring All-Reduce

On modern hardware like NVIDIA DGX servers with high-speed NVLink connections, a more sophisticated pattern called **Ring All-Reduce** is optimal.

**How it works:**
1.  GPUs are arranged in a logical ring.
2.  Each GPU splits its gradient into `N` chunks (where `N` is the number of GPUs).
3.  In step `k`, GPU `i` sends its `k`-th gradient chunk to GPU `i+1` and receives the `(k-1)`-th chunk from GPU `i-1`.
4.  Each GPU accumulates the chunks it receives. After `N-1` steps, every GPU holds the fully aggregated gradient for one chunk.
5.  A second phase redistributes these aggregated chunks so every GPU has the complete, averaged gradient.

**Key Insight:** This pipeline utilizes all available links simultaneously, making aggregation time nearly constant regardless of the number of GPUs. For 8 GPUs synchronizing 160 MB over NVLink, this can take as little as **~6 ms**.

## 4. Scaling to Multiple Machines

Adding servers introduces a slower network link (Ethernet). The standard multi-machine, multi-GPU training flow is:

1.  Each machine reads a batch, computes gradients on its local GPUs.
2.  Gradients are aggregated locally within the machine (e.g., using Ring All-Reduce across its GPUs).
3.  Local aggregates are sent to a central **Parameter Server**.
4.  The server aggregates gradients from all workers, updates the model parameters.
5.  The updated parameters are broadcast back to all workers.

**The Bottleneck:** A single central server has limited network bandwidth. Synchronization time grows linearly (`O(m)`) with the number of workers `m`.

**The Solution: Shard the Parameters**
Use multiple parameter servers. Each server stores and updates only a fraction (`1/n`) of the total parameters. Now, the network load is distributed, and total update time scales as `O(m/n)`. By scaling `n` with `m`, you can achieve constant-time updates regardless of cluster size.

## 5. The Key-Value Store Abstraction

Implementing this manually is complex. The standard abstraction is a **distributed key-value store**.

*   **Key:** A unique identifier for a parameter tensor (e.g., `"layer1.weights"`).
*   **Value:** The gradient or parameter tensor itself.

The store provides two simple operations:
*   **`push(key, value)`:** A worker sends a gradient to the store, where it is summed with other pushes for the same key.
*   **`pull(key)`:** A worker retrieves the current aggregated value (e.g., the updated parameter) for a key.

This abstraction cleanly separates the **statistical model** (defining the optimization) from the **systems engineering** (handling distributed synchronization).

## Summary

*   **Strategy is hardware-dependent:** Optimal synchronization depends on your specific NVLink, PCIe, and network topology.
*   **Ring All-Reduce is efficient:** For multi-GPU nodes with fast interconnects, Ring All-Reduce minimizes synchronization time.
*   **Hierarchical scaling works:** For multi-server clusters, sharding parameters across multiple parameter servers prevents the central server from becoming a bottleneck.
*   **Use an abstraction:** Frameworks implementing a key-value store interface (like `push`/`pull`) hide the immense complexity of distributed communication.

## Exercises for Further Exploration

1.  **Bi-directional Rings:** Can you design a ring synchronization scheme where data flows both clockwise and counter-clockwise simultaneously to cut the number of steps in half?
2.  **Asynchronous Updates:** What are the trade-offs of allowing workers to push/pull parameters asynchronously, without waiting for all others? This can improve throughput but may harm convergence.
3.  **Fault Tolerance:** How would you design the system to recover from a server failure mid-training? Strategies include parameter replication, checkpointing, and consensus protocols.