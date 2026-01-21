# Distributed Parameter Server with PyTorch RPC: A Step-by-Step Guide

## Overview
This guide demonstrates how to implement a parameter server training strategy using PyTorch's Distributed RPC Framework. You'll learn to build a system where multiple trainers fetch and update model parameters from a centralized server, enabling distributed training across multiple machines.

## Prerequisites

Ensure you have the following installed:
```bash
pip install torch torchvision
```

## 1. Network Architecture

First, let's define the neural network we'll train on the MNIST dataset. This ConvNet is adapted from PyTorch's examples repository.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_gpus=0):
        super(Net, self).__init__()
        print(f"Using {num_gpus} GPUs to train")
        self.num_gpus = num_gpus
        
        # Determine initial device
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.num_gpus > 0 else "cpu")
        print(f"Putting first 2 convs on {str(device)}")
        
        # Place first convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1).to(device)
        self.conv2 = nn.Conv2d(32, 64, 3, 1).to(device)
        
        # Move to second GPU if available
        if "cuda" in str(device) and num_gpus > 1:
            device = torch.device("cuda:1")
        
        print(f"Putting rest of layers on {str(device)}")
        # Place remaining layers
        self.dropout1 = nn.Dropout2d(0.25).to(device)
        self.dropout2 = nn.Dropout2d(0.5).to(device)
        self.fc1 = nn.Linear(9216, 128).to(device)
        self.fc2 = nn.Linear(128, 10).to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        
        # Move tensor to next device if necessary
        next_device = next(self.fc1.parameters()).device
        x = x.to(next_device)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
```

## 2. RPC Helper Functions

These helper functions simplify remote method invocation using RPC:

```python
import torch.distributed.rpc as rpc

def call_method(method, rref, *args, **kwargs):
    """Call a method on the local value of an RRef"""
    return method(rref.local_value(), *args, **kwargs)

def remote_method(method, rref, *args, **kwargs):
    """Invoke a method on a remote node via RPC"""
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)
```

## 3. Parameter Server Implementation

The parameter server stores the model and handles requests from trainers:

```python
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer

class ParameterServer(nn.Module):
    def __init__(self, num_gpus=0):
        super().__init__()
        self.model = Net(num_gpus=num_gpus)
        self.input_device = torch.device(
            "cuda:0" if torch.cuda.is_available() and num_gpus > 0 else "cpu")

    def forward(self, inp):
        """Forward pass - moves output to CPU for RPC compatibility"""
        inp = inp.to(self.input_device)
        out = self.model(inp)
        out = out.to("cpu")  # RPC only supports CPU tensors
        return out

    def get_dist_gradients(self, cid):
        """Retrieve gradients computed via distributed autograd"""
        grads = dist_autograd.get_gradients(cid)
        cpu_grads = {}
        for k, v in grads.items():
            k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
            cpu_grads[k_cpu] = v_cpu
        return cpu_grads

    def get_param_rrefs(self):
        """Wrap local parameters as RRefs for DistributedOptimizer"""
        return [rpc.RRef(param) for param in self.model.parameters()]

# Global parameter server instance
param_server = None
global_lock = Lock()

def get_parameter_server(num_gpus=0):
    """Singleton accessor for the parameter server"""
    global param_server
    with global_lock:
        if not param_server:
            param_server = ParameterServer(num_gpus=num_gpus)
        return param_server

def run_parameter_server(rank, world_size):
    """Main loop for the parameter server process"""
    print("PS master initializing RPC")
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
    print("RPC initialized! Running parameter server...")
    rpc.shutdown()
    print("RPC shutdown on parameter server.")
```

## 4. Trainer Implementation

Trainers fetch parameters from the server and perform local training:

```python
class TrainerNet(nn.Module):
    def __init__(self, num_gpus=0):
        super().__init__()
        self.num_gpus = num_gpus
        # Get remote reference to parameter server
        self.param_server_rref = rpc.remote(
            "parameter_server", get_parameter_server, args=(num_gpus,))

    def get_global_param_rrefs(self):
        """Get RRefs to all parameters for DistributedOptimizer"""
        return remote_method(
            ParameterServer.get_param_rrefs,
            self.param_server_rref)

    def forward(self, x):
        """Forward pass via RPC to parameter server"""
        return remote_method(
            ParameterServer.forward, self.param_server_rref, x)
```

## 5. Training Loop

Here's the distributed training loop that coordinates with the parameter server:

```python
from torch import optim

def run_training_loop(rank, num_gpus, train_loader, test_loader):
    """Main training loop for trainer processes"""
    net = TrainerNet(num_gpus=num_gpus)
    
    # Setup distributed optimizer
    param_rrefs = net.get_global_param_rrefs()
    opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.03)
    
    for i, (data, target) in enumerate(train_loader):
        with dist_autograd.context() as cid:
            # Forward pass
            model_output = net(data)
            target = target.to(model_output.device)
            loss = F.nll_loss(model_output, target)
            
            if i % 5 == 0:
                print(f"Rank {rank} training batch {i} loss {loss.item()}")
            
            # Distributed backward pass
            dist_autograd.backward(cid, [loss])
            
            # Verify gradients were computed
            assert remote_method(
                ParameterServer.get_dist_gradients,
                net.param_server_rref,
                cid) != {}
            
            # Optimizer step
            opt.step(cid)
    
    print("Training complete!")
    print("Getting accuracy....")
    get_accuracy(test_loader, net)

def get_accuracy(test_loader, model):
    """Calculate model accuracy"""
    model.eval()
    correct_sum = 0
    device = torch.device("cuda:0" if model.num_gpus > 0 
                          and torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        for data, target in test_loader:
            out = model(data)
            pred = out.argmax(dim=1, keepdim=True)
            pred, target = pred.to(device), target.to(device)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum += correct
    
    print(f"Accuracy {correct_sum / len(test_loader.dataset)}")

def run_worker(rank, world_size, num_gpus, train_loader, test_loader):
    """Main loop for trainer processes"""
    print(f"Worker rank {rank} initializing RPC")
    rpc.init_rpc(
        name=f"trainer_{rank}",
        rank=rank,
        world_size=world_size)
    
    print(f"Worker {rank} done initializing RPC")
    run_training_loop(rank, num_gpus, train_loader, test_loader)
    rpc.shutdown()
```

## 6. Launch Script

Finally, create the main script to launch the parameter server and trainers:

```python
import argparse
import os
import torch.multiprocessing as mp
from torchvision import datasets, transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parameter-Server RPC based training")
    parser.add_argument("--world_size", type=int, default=4,
                       help="Total number of participating processes")
    parser.add_argument("--rank", type=int, default=None,
                       help="Global rank of this process (0 for master)")
    parser.add_argument("--num_gpus", type=int, default=0,
                       help="Number of GPUs to use (0-2 supported)")
    parser.add_argument("--master_addr", type=str, default="localhost",
                       help="Address of master node")
    parser.add_argument("--master_port", type=str, default="29500",
                       help="Port master is listening on")
    
    args = parser.parse_args()
    assert args.rank is not None, "must provide rank argument."
    assert args.num_gpus <= 2, f"Only 0-2 GPUs supported (got {args.num_gpus})."
    
    # Set environment variables for process discovery
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    
    processes = []
    world_size = args.world_size
    
    if args.rank == 0:
        # Launch parameter server
        p = mp.Process(target=run_parameter_server, args=(0, world_size))
        p.start()
        processes.append(p)
    else:
        # Prepare MNIST data loaders
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True, transform=transform),
            batch_size=32, shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transform),
            batch_size=32, shuffle=True)
        
        # Launch trainer
        p = mp.Process(
            target=run_worker,
            args=(args.rank, world_size, args.num_gpus, train_loader, test_loader))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
```

## 7. Running the Example

To run this distributed training system, execute the following commands in separate terminal windows:

**Parameter Server (rank 0):**
```bash
python rpc_parameter_server.py --world_size=2 --rank=0
```

**Trainer (rank 1):**
```bash
python rpc_parameter_server.py --world_size=2 --rank=1
```

For multi-machine training, specify the master address:
```bash
python rpc_parameter_server.py --world_size=3 --rank=2 --master_addr=192.168.1.100
```

## Key Concepts Explained

1. **RRef (Remote Reference)**: Acts as a distributed pointer to objects on remote nodes
2. **Distributed Autograd**: Tracks operations across multiple nodes for gradient computation
3. **Distributed Optimizer**: Coordinates optimizer steps across distributed parameters
4. **Parameter Server Pattern**: Centralized storage of model parameters with distributed trainers

This implementation demonstrates how PyTorch's RPC framework enables sophisticated distributed training patterns while maintaining a familiar PyTorch API surface.