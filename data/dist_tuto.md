# Writing Distributed Applications with PyTorch

**Author**: [Séb Arnold](https://seba1511.com)

> **Note**: View and edit this tutorial on [GitHub](https://github.com/pytorch/tutorials/blob/main/intermediate_source/dist_tuto.rst).

**Prerequisites**:
- [PyTorch Distributed Overview](../beginner/dist_overview.html)

In this tutorial, you'll learn how to use PyTorch's distributed package (`torch.distributed`) to parallelize computations across multiple processes and machines. We'll cover setting up the distributed environment, using different communication strategies (point-to-point and collective), and delve into some advanced topics like communication backends and initialization methods.

## 1. Setup and Process Initialization

The `torch.distributed` package enables parallel computation using message passing, allowing processes to communicate data with each other. Unlike `torch.multiprocessing`, processes can use different communication backends and aren't restricted to the same machine.

To begin, you need to run multiple processes simultaneously. While clusters use tools like `pdsh` or `slurm`, for this tutorial we'll use a single machine and spawn processes with the following template.

### 1.1. Initialization Script

Create a file named `run.py` with the following content:

```python
#!/usr/bin/env python
import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    """ Distributed function to be implemented later. """
    pass

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    world_size = 2
    processes = []
    if "google.colab" in sys.modules:
        print("Running in Google Colab")
        mp.get_context("spawn")
    else:
        mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

This script spawns two processes. Each process:
1. Sets environment variables (`MASTER_ADDR` and `MASTER_PORT`) for coordination.
2. Initializes the process group via `dist.init_process_group()`.
3. Executes the `run` function.

The `init_process` function ensures all processes can coordinate through a master at the same IP and port. We use the `gloo` backend here, but others are available (see Section 5.1).

## 2. Point-to-Point Communication

Point-to-point communication transfers data directly between two processes using `send`/`recv` (blocking) or `isend`/`irecv` (non-blocking).

### 2.1. Blocking Communication

Update the `run` function in `run.py`:

```python
def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])
```

Both processes start with a zero tensor. Process 0 increments it to 1.0 and sends it to process 1. Process 1 must allocate memory to receive the data. The `send`/`recv` operations are **blocking**—execution halts until communication completes.

### 2.2. Non-blocking Communication

Modify `run` to use immediate (non-blocking) operations:

```python
def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])
```

With immediates, you **must not** modify the sent tensor or access the received tensor until `req.wait()` completes. After `wait()`, communication is guaranteed, and `tensor[0]` will be 1.0.

Point-to-point communication is useful for custom algorithms like those in [Baidu's DeepSpeech](https://github.com/baidu-research/baidu-allreduce) or [Facebook's large-scale experiments](https://research.fb.com/publications/imagenet1kin1h/).

## 3. Collective Communication

Collectives involve communication across **all processes in a group**. A group is a subset of processes, created with `dist.new_group(group)`. By default, collectives operate on the entire **world** (all processes).

### 3.1. All-Reduce Example

Update `run` to perform an all-reduce sum:

```python
def run(rank, size):
    """ Simple collective communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])
```

We use `dist.ReduceOp.SUM` to sum all tensors in the group. PyTorch supports several commutative operators:
- `dist.ReduceOp.SUM`
- `dist.ReduceOp.PRODUCT`
- `dist.ReduceOp.MAX`
- `dist.ReduceOp.MIN`
- `dist.ReduceOp.BAND`
- `dist.ReduceOp.BOR`
- `dist.ReduceOp.BXOR`
- `dist.ReduceOp.PREMUL_SUM`

See the [full list](https://pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp).

### 3.2. Supported Collectives

Key collective operations include:
- `dist.broadcast(tensor, src, group)`: Copies `tensor` from `src` to all processes.
- `dist.reduce(tensor, dst, op, group)`: Applies `op` to all tensors, stores result in `dst`.
- `dist.all_reduce(tensor, op, group)`: Like reduce, but result stored in all processes.
- `dist.scatter(tensor, scatter_list, src, group)`: Sends `scatter_list[i]` to the i-th process.
- `dist.gather(tensor, gather_list, dst, group)`: Copies `tensor` from all processes to `dst`.
- `dist.all_gather(tensor_list, tensor, group)`: Copies `tensor` from all processes to `tensor_list` on all processes.
- `dist.barrier(group)`: Blocks until all processes in the group reach this call.
- `dist.all_to_all(output_tensor_list, input_tensor_list, group)`: Scatters input tensors to all, gathers results into output list.

Refer to the [documentation](https://pytorch.org/docs/stable/distributed.html) for the complete list.

## 4. Distributed Training

Let's implement a distributed version of stochastic gradient descent (SGD) to replicate `DistributedDataParallel` functionality. This is a didactic example; for production, use the official [DistributedDataParallel](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel).

We'll partition the dataset so each process computes gradients on its subset, then averages gradients across all processes.

### 4.1. Dataset Partitioning

First, define helper classes to partition data:

```python
""" Dataset partitioning helper """
from random import Random

class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])
```

### 4.2. Partition MNIST

Now, partition the MNIST dataset:

```python
import torch
import torch.distributed as dist
from torchvision import datasets, transforms
from math import ceil

def partition_dataset():
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = 128 // size
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         shuffle=True)
    return train_set, bsz
```

With 2 processes, each gets 30,000 samples (60,000 / 2). The batch size per process is 64 (128 / 2), maintaining the overall batch size.

### 4.3. Define the Model and Training Loop

Define a simple neural network and the training loop:

```python
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)
```

### 4.4. Gradient Averaging

Implement `average_gradients` to average gradients across all processes:

```python
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size
```

This performs an all-reduce sum on each parameter's gradients, then divides by the world size to compute the average.

You've now implemented distributed synchronous SGD! For production, use the optimized `DistributedDataParallel`.

### 4.5. Custom Ring-Allreduce (Optional Challenge)

As an advanced exercise, implement a ring-allreduce like DeepSpeech's using point-to-point collectives:

```python
def allreduce(send, recv):
   rank = dist.get_rank()
   size = dist.get_world_size()
   send_buff = send.clone()
   recv_buff = send.clone()
   accum = send.clone()

   left = ((rank - 1) + size) % size
   right = (rank + 1) % size

   for i in range(size - 1):
       if i % 2 == 0:
           # Send send_buff
           send_req = dist.isend(send_buff, right)
           dist.recv(recv_buff, left)
           accum[:] += recv_buff[:]
       else:
           # Send recv_buff
           send_req = dist.isend(recv_buff, right)
           dist.recv(send_buff, left)
           accum[:] += send_buff[:]
       send_req.wait()
   recv[:] = accum[:]
```

This function stores the sum of all `send` tensors in `recv`. For optimal bandwidth usage, real implementations split tensors into chunks using `torch.chunk`.

## 5. Advanced Topics

### 5.1. Communication Backends

`torch.distributed` abstracts over different backends. Choose based on your hardware and needs. Compare supported functions [here](https://pytorch.org/docs/stable/distributed.html#module-torch.distributed).

#### Gloo Backend
We've used Gloo. It works on Linux and macOS, supports CPU point-to-point and collectives, and GPU collectives (though less optimized than NCCL). To use multiple GPUs, modify the training script:

1. Use Accelerator API: `device_type = torch.accelerator.current_accelerator()`
2. Set device: `device = torch.device(f"{device_type}:{rank}")`
3. Move model: `model = Net().to(device)`
4. Move data: `data, target = data.to(device), target.to(device)`

#### MPI Backend
MPI is optimized for high-performance clusters. PyTorch binaries don't include MPI; you must compile from source:

1. Set up PyTorch from source but don't install yet.
2. Install an MPI implementation (e.g., Open-MPI): `conda install -c conda-forge openmpi`
3. Install PyTorch: `python setup.py install`

To test, replace the main block in `run.py` with:
```python
if __name__ == '__main__':
    init_process(0, 0, run, backend='mpi')
```
Run with: `mpirun -n 4 python run.py`

MPI spawns its own processes, making `rank` and `size` arguments redundant.

#### NCCL Backend
NCCL provides optimized collective operations for CUDA tensors. Use it for best GPU performance. It's included in PyTorch binaries with CUDA support.

#### XCCL Backend
XCCL offers optimized collectives for XPU tensors. Use it for Intel GPU workloads. Included in binaries with XPU support.

### 5.2. Initialization Methods

The `dist.init_process_group(backend, init_method)` function supports different initialization schemes.

#### Environment Variable (Default)
Set these environment variables on all machines:
- `MASTER_PORT`: Free port on the rank 0 machine.
- `MASTER_ADDR`: IP address of the rank 0 machine.
- `WORLD_SIZE`: Total number of processes.
- `RANK`: Rank of each process (0 for master).

#### Shared File System
All processes must access a shared file system with `fcntl` locking support:

```python
dist.init_process_group(
    init_method='file:///mnt/nfs/sharedfile',
    rank=args.rank,
    world_size=4)
```

#### TCP
Provide the IP and port of the rank 0 process:

```python
dist.init_process_group(
    init_method='tcp://10.1.1.20:23456',
    rank=args.rank,
    world_size=4)
```

## Acknowledgments

Thanks to the PyTorch developers for their excellent implementation, documentation, and tests. Special thanks to Soumith Chintala, Adam Paszke, and Natalia Gimelshein for their insights and feedback on early drafts.