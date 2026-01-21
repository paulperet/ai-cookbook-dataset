# PyTorch FSDP Tutorial: Training Models with Fully Sharded Data Parallelism

## Introduction

Training large AI models requires significant computational resources and introduces engineering complexity. PyTorch's **Fully Sharded Data Parallel (FSDP)** API, introduced in PyTorch 1.11, helps address these challenges by enabling efficient distributed training of very large models.

This tutorial demonstrates how to use FSDP APIs with a simple MNIST model. The concepts and code patterns shown here can be extended to train larger models like HuggingFace BERT or GPT-3 models with up to 1 trillion parameters.

> **Note**: FSDP1 is deprecated. Please refer to the [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) for the latest implementation.

## How FSDP Works

### Comparison with DDP

In **DistributedDataParallel (DDP)** training:
- Each worker maintains a full replica of the model
- Processes a batch of data independently
- Uses all-reduce to synchronize gradients across workers
- Model weights and optimizer states are replicated on every worker

**FSDP** improves upon DDP by:
- Sharding model parameters, optimizer states, and gradients across DDP ranks
- Reducing GPU memory footprint per worker
- Enabling training of larger models or batch sizes
- Increasing communication volume but optimizing with techniques like overlapping communication and computation

### FSDP Workflow

**During initialization:**
- Model parameters are sharded across ranks
- Each rank keeps only its assigned shard

**During forward pass:**
1. All ranks perform `all_gather` to collect all parameter shards
2. Forward computation executes with full parameters
3. Parameter shards are discarded after computation

**During backward pass:**
1. All ranks perform `all_gather` to collect parameter shards
2. Backward computation executes
3. Gradients are synchronized via `reduce_scatter`
4. Parameters are discarded

Conceptually, FSDP decomposes DDP's gradient all-reduce into reduce-scatter (backward) and all-gather (forward) operations, allowing more efficient memory usage.

## Implementation Tutorial

### Step 1: Setup and Imports

First, ensure you have PyTorch installed. Then create a Python script called `FSDP_mnist.py` with the following imports:

```python
# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
```

> **Note**: This tutorial targets PyTorch versions 1.12+. For earlier versions, replace `size_based_auto_wrap_policy` with `default_auto_wrap_policy` and `fsdp_auto_wrap_policy` with `auto_wrap_policy`.

### Step 2: Distributed Training Setup

FSDP requires a distributed training environment. Define helper functions to initialize and clean up the process group:

```python
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
```

### Step 3: Define the Model Architecture

Create a simple CNN for MNIST digit classification:

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

### Step 4: Implement Training and Validation Functions

Define the training function that handles forward/backward passes and gradient synchronization:

```python
def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)
    
    if sampler:
        sampler.set_epoch(epoch)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(rank), target.to(rank)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))
```

Create the validation function to evaluate model performance:

```python
def test(model, rank, world_size, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(rank)
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))
```

### Step 5: Implement the Main FSDP Training Loop

Create the distributed training function that wraps the model with FSDP:

```python
def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load datasets
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)

    # Create distributed samplers
    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)

    # Configure data loaders
    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    # Set up auto-wrap policy (shard layers with >100 parameters)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    
    torch.cuda.set_device(rank)

    # Initialize timing events
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # Create and wrap model with FSDP
    model = Net().to(rank)
    model = FSDP(model)

    # Set up optimizer and scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    # Training loop
    init_start_event.record()
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        test(model, rank, world_size, test_loader)
        scheduler.step()

    init_end_event.record()

    # Print timing results
    if rank == 0:
        init_end_event.synchronize()
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")

    # Save model (requires calling state_dict on each rank)
    if args.save_model:
        dist.barrier()  # Ensure all ranks complete training
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

    cleanup()
```

### Step 6: Configure Command-Line Arguments and Launch Training

Add argument parsing and the main execution block:

```python
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Launch distributed training
    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_main,
             args=(WORLD_SIZE, args),
             nprocs=WORLD_SIZE,
             join=True)
```

## Running the Training

Execute the script with:

```bash
python FSDP_mnist.py
```

You should see output similar to:
```
CUDA event elapsed time on training loop 40.67462890625sec
```

## Advanced FSDP Features

### Auto-Wrap Policy

Without an auto-wrap policy, FSDP wraps the entire model in a single FSDP unit, which reduces memory and computation efficiency. The auto-wrap policy creates multiple FSDP units that shard parameters more granularly.

Update the FSDP initialization to use auto-wrapping:

```python
my_auto_wrap_policy = functools.partial(
    size_based_auto_wrap_policy, min_num_params=20000
)
torch.cuda.set_device(rank)
model = Net().to(rank)

model = FSDP(model, auto_wrap_policy=my_auto_wrap_policy)
```

This configuration shards layers with more than 20,000 parameters. Finding the optimal auto-wrap policy requires experimentation and profiling.

### CPU Offloading

For extremely large models that don't fit in GPU memory even with FSDP, you can enable CPU offloading:

```python
model = FSDP(model,
    auto_wrap_policy=my_auto_wrap_policy,
    cpu_offload=CPUOffload(offload_params=True))
```

This feature copies parameters and gradients to CPU memory when not needed, trading training speed for increased memory efficiency.

## Comparison with DDP

To compare with DDP, modify the model wrapping:

```python
model = Net().to(rank)
model = DDP(model)
```

Run the DDP version:

```bash
python DDP_mnist.py
```

## Performance Analysis

For the MNIST example:
- **FSDP without auto-wrap**: ~75 MB peak memory per device
- **FSDP with auto-wrap**: ~66 MB peak memory per device  
- **DDP**: Higher memory usage (full model replication per device)
- **Training time**: All three approaches show similar performance for this small model

The memory savings become more significant with larger models. FSDP's sharding strategy reduces per-device memory footprint compared to DDP's full replication.

## Conclusion

FSDP enables efficient distributed training of large models by:
1. Sharding parameters, gradients, and optimizer states across devices
2. Reducing per-device memory requirements
3. Maintaining training performance through communication optimizations

For real-world applications with large models, FSDP provides substantial memory savings that enable training of models that wouldn't fit in GPU memory with DDP. The auto-wrap policy and CPU offloading features provide additional flexibility for extreme-scale models.

> **Further Reading**: For detailed performance comparisons and advanced use cases, refer to the [PyTorch FSDP blog post](https://pytorch.medium.com/6c8da2be180d).