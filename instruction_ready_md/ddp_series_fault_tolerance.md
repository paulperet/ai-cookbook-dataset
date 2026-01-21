# Fault-Tolerant Distributed Training with `torchrun`

## Overview

In distributed training, a single process failure can disrupt the entire job. Since the susceptibility for failure is higher in distributed setups, making your training script robust is critical. You might also want your training job to be *elastic*, allowing compute resources to join and leave dynamically.

PyTorch provides a utility called `torchrun` that offers fault-tolerance and elastic training. When a failure occurs, `torchrun` logs the errors and attempts to automatically restart all processes from the last saved "snapshot" of the training job. This snapshot saves more than just the model state; it can include the number of epochs run, optimizer states, or any other stateful attribute necessary for continuity.

## Why Use `torchrun`?

`torchrun` handles the minutiae of distributed training so you don't have to:

- You don't need to set environment variables or explicitly pass `rank` and `world_size`; `torchrun` assigns these along with several other [environment variables](https://pytorch.org/docs/stable/elastic/run.html#environment-variables).
- No need to call `mp.spawn` in your script; you only need a generic `main()` entry point and launch the script with `torchrun`. This allows the same script to run in non-distributed, single-node, and multi-node setups.
- Gracefully restart training from the last saved snapshot.

## Prerequisites

- High-level [overview](ddp_series_theory.html) of DDP.
- Familiarity with [DDP code](ddp_series_multigpu.html).
- A machine with multiple GPUs (this tutorial uses an AWS p3.8xlarge instance).
- PyTorch [installed](https://pytorch.org/get-started/locally/) with CUDA.

## Step 1: Understanding Graceful Restarts

For graceful restarts, structure your training script as follows:

```python
def main():
    load_snapshot(snapshot_path)
    initialize()
    train()

def train():
    for batch in iter(dataset):
        train_step(batch)

        if should_checkpoint:
            save_snapshot(snapshot_path)
```

If a failure occurs, `torchrun` terminates all processes and restarts them. Each process first loads and initializes the last saved snapshot, then continues training. This means you only lose training progress from the last saved snapshot.

In elastic training, whenever membership changes (adding or removing nodes), `torchrun` terminates and spawns processes on available devices. This structure ensures your training job continues without manual intervention.

## Step 2: Process Group Initialization

With `torchrun`, you no longer need to manually set `rank` and `world_size`. Update your `ddp_setup` function as shown below.

**Before:**
```python
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
```

**After:**
```python
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
```

`torchrun` automatically sets `RANK`, `WORLD_SIZE`, and other [environment variables](https://pytorch.org/docs/stable/elastic/run.html#environment-variables).

## Step 3: Using `torchrun`-Provided Environment Variables

Update your trainer to use the `LOCAL_RANK` environment variable provided by `torchrun`.

**Before:**
```python
self.gpu_id = gpu_id
```

**After:**
```python
self.gpu_id = int(os.environ["LOCAL_RANK"])
```

## Step 4: Saving and Loading Snapshots

Regularly saving all relevant information in snapshots allows your training job to seamlessly resume after an interruption.

Add the following methods to your trainer class:

```python
def _save_snapshot(self, epoch):
    snapshot = {}
    snapshot["MODEL_STATE"] = self.model.module.state_dict()
    snapshot["EPOCHS_RUN"] = epoch
    torch.save(snapshot, "snapshot.pt")
    print(f"Epoch {epoch} | Training snapshot saved at snapshot.pt")

def _load_snapshot(self, snapshot_path):
    snapshot = torch.load(snapshot_path)
    self.model.load_state_dict(snapshot["MODEL_STATE"])
    self.epochs_run = snapshot["EPOCHS_RUN"]
    print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
```

## Step 5: Loading a Snapshot in the Trainer Constructor

When restarting an interrupted training job, your script should first try to load a snapshot to resume training.

Update your trainer's `__init__` method:

```python
class Trainer:
    def __init__(self, snapshot_path, ...):
        ...
        if os.path.exists(snapshot_path):
            self._load_snapshot(snapshot_path)
        ...
```

## Step 6: Resuming Training

Modify your training loop to resume from the last epoch run instead of starting from scratch.

**Before:**
```python
def train(self, max_epochs: int):
    for epoch in range(max_epochs):
        self._run_epoch(epoch)
```

**After:**
```python
def train(self, max_epochs: int):
    for epoch in range(self.epochs_run, max_epochs):
        self._run_epoch(epoch)
```

## Step 7: Running the Script

Simply call your entry point function as you would for a non-multiprocessing script; `torchrun` automatically spawns the processes.

Update your main block:

**Before:**
```python
if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, total_epochs, save_every,), nprocs=world_size)
```

**After:**
```python
if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    main(save_every, total_epochs)
```

Now, launch your script with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=4 multigpu_torchrun.py 50 10
```

This command runs the script in standalone mode with 4 processes per node.

## Summary

By integrating `torchrun` into your distributed training pipeline, you gain fault tolerance and elasticity without manual intervention. The key steps are:

1. Simplify process group initialization by letting `torchrun` handle environment variables.
2. Use `torchrun`-provided variables like `LOCAL_RANK`.
3. Implement snapshot saving and loading to capture training state.
4. Modify your training loop to resume from the last saved epoch.
5. Launch your script with `torchrun` instead of manual process spawning.

This approach ensures your distributed training jobs are robust and can recover automatically from failures.

## Further Reading

- [Multi-Node Training with DDP](../intermediate/ddp_series_multinode.html) (next tutorial in this series).
- [Multi-GPU Training with DDP](ddp_series_multigpu.html) (previous tutorial in this series).
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html).
- [Torchrun Launch Options](https://github.com/pytorch/pytorch/blob/bbe803cb35948df77b46a2d38372910c96693dcd/torch/distributed/run.py#L401).
- [Migrating from torch.distributed.launch to torchrun](https://pytorch.org/docs/stable/elastic/train_script.html#elastic-train-script).