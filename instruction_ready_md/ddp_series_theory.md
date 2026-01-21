# PyTorch Distributed Data Parallel (DDP) Fundamentals

## What You Will Learn
*   How DDP works under the hood.
*   The role of `DistributedSampler`.
*   How gradients are synchronized across multiple GPUs.

## Prerequisites
*   Familiarity with [basic non-distributed training](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) in PyTorch.

## Introduction to Distributed Data Parallel (DDP)

This guide introduces PyTorch's **DistributedDataParallel (DDP)**, a powerful module for data parallel training. Data parallelism is a technique to process multiple data batches across several devices (like GPUs) simultaneously, significantly improving training performance.

In a DDP setup:
1.  Your model is replicated onto each available device.
2.  A `DistributedSampler` ensures each device receives a unique, non-overlapping subset of the input data batch.
3.  Each model replica processes its data, calculates local gradients, and then synchronizes these gradients with all other replicas using an efficient **ring all-reduce algorithm**.

For a deeper, code-first dive into DDP's mechanics, refer to the [official illustrative tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html#).

## DDP vs. DataParallel (DP): Why Choose DDP?

While PyTorch's older `DataParallel` (DP) module offers single-line simplicity for data parallelism, `DistributedDataParallel` (DDP) is the modern, recommended approach due to its superior performance and scalability.

| Feature | `DataParallel` (DP) | `DistributedDataParallel` (DDP) |
| :--- | :--- | :--- |
| **Architecture** | Single-process, multi-threaded. Suffers from Python's Global Interpreter Lock (GIL) contention. | Multi-process. Avoids GIL issues, leading to better CPU utilization. |
| **Model Replication** | Replicates and destroys the model on each forward pass, creating significant overhead. | Replicates the model once per process, resulting in lower overhead. |
| **Scalability** | Limited to single-machine, multi-GPU training. | Supports scaling across multiple machines (multi-node training). |
| **Performance** | Slower due to GIL contention and replication overhead. | Faster and more efficient, designed for production-scale training. |

## Next Steps

Ready to implement DDP? Continue to the next tutorial in this series: [Multi-GPU Training with DDP](ddp_series_multigpu.html).

## Further Reading
*   [DDP API Documentation](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
*   [DDP Internal Design Notes](https://pytorch.org/docs/master/notes/ddp.html#internal-design)
*   [DDP Mechanics Tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html#)