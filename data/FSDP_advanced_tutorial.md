# Advanced Model Training with Fully Sharded Data Parallel (FSDP)

**Author**: Hamid Shojanazeri, Less Wright, Rohan Varma, Yanli Zhao

## What You Will Learn
- How to use PyTorch's Fully Sharded Data Parallel (FSDP) module to shard model parameters across data parallel workers.
- Advanced FSDP features including Transformer Auto Wrap Policy, Mixed Precision, and Backward Prefetch.
- How to fine-tune a large HuggingFace T5 model for text summarization using FSDP.

## Prerequisites
- PyTorch 1.12 or later
- Basic understanding of the [FSDP API](https://pytorch.org/docs/main/fsdp.html)
- Familiarity with the [FSDP getting started tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

## Introduction
This tutorial demonstrates advanced features of Fully Sharded Data Parallel (FSDP) released in PyTorch 1.12. We'll fine-tune a HuggingFace T5 model for text summarization using the WikiHow dataset as a practical example.

FSDP reduces memory footprint on each GPU, enabling training of larger models or increasing batch sizes compared to traditional Distributed Data Parallel (DDP). It achieves this by sharding model parameters, gradients, and optimizer states across data parallel workers, with computation and communication overlap for efficient training.

## FSDP Features Covered
- Transformer Auto Wrap Policy
- Mixed Precision
- Initializing FSDP Model on Device
- Sharding Strategy
- Backward Prefetch
- Model Checkpoint Saving via Streaming to CPU

## How FSDP Works
At a high level, FSDP operates as follows:

1. **In the constructor**: Shard model parameters so each rank only keeps its own shard
2. **In the forward pass**: 
   - Run `all_gather` to collect all shards from all ranks to recover full parameters
   - Run forward computation
   - Discard non-owned parameter shards to free memory
3. **In the backward pass**:
   - Run `all_gather` to collect all shards for backward computation
   - Discard non-owned parameters
   - Run `reduce_scatter` to synchronize gradients

## Setup

### 1. Install Dependencies
```bash
pip3 install torch torchvision torchaudio
```

### 2. Prepare Dataset
Create a `data` folder and download the WikiHow dataset:
- [wikihowAll.csv](https://ucsb.app.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358)
- [wikihowSep.csv](https://ucsb.app.box.com/s/7yq601ijl1lzvlfu4rjdbbxforzd2oag)

Place both files in the `data` folder.

### 3. Create Training Script
Create a Python script named `T5_training.py` with the following code.

## Implementation

### Step 1: Import Required Packages
```python
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import T5Tokenizer, T5ForConditionalGeneration
import functools
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers.models.t5.modeling_t5 import T5Block

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from summarization_dataset import *
from typing import Type
import time
import tqdm
from datetime import datetime
```

### Step 2: Set Up Distributed Training
```python
def setup():
    """Initialize the process group for distributed training."""
    dist.init_process_group("nccl")

def cleanup():
    """Clean up the process group after training."""
    dist.destroy_process_group()
```

### Step 3: Load Model and Tokenizer
```python
def setup_model(model_name):
    """Load pre-trained T5 model and tokenizer."""
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer
```

### Step 4: Define Helper Functions
```python
def get_date_of_run():
    """Create date and time string for file save uniqueness."""
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run

def format_metrics_to_gb(item):
    """Format numbers to gigabytes with 4-digit precision."""
    g_gigabyte = 1024**3
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num
```

### Step 5: Define Training Function
```python
def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    """Training loop for one epoch."""
    model.train()
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).to(local_rank)

    if sampler:
        sampler.set_epoch(epoch)
    
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )
    
    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(local_rank)
        
        optimizer.zero_grad()
        output = model(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=batch["target_ids"]
        )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        
        fsdp_loss[0] += loss.item()
        fsdp_loss[1] += len(batch)
        
        if rank == 0:
            inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = fsdp_loss[0] / fsdp_loss[1]

    if rank == 0:
        inner_pbar.close()
        print(f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}")
    
    return train_accuracy
```

### Step 6: Define Validation Function
```python
def validation(model, rank, world_size, val_loader):
    """Validation loop."""
    model.eval()
    local_rank = int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(3).to(local_rank)
    
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
    
    with torch.no_grad():
        for batch in val_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            
            output = model(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                labels=batch["target_ids"]
            )
            fsdp_loss[0] += output["loss"].item()
            fsdp_loss[1] += len(batch)
            
            if rank == 0:
                inner_pbar.update(1)

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    
    if rank == 0:
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    
    return val_loss
```

### Step 7: Main FSDP Training Function
```python
def fsdp_main(args):
    """Main training function with FSDP setup."""
    # Load model and tokenizer
    model, tokenizer = setup_model("t5-base")
    
    # Get distributed training info
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # Load dataset
    dataset = load_dataset('wikihow', 'all', data_dir='data/')
    print(f"Size of train dataset: {dataset['train'].shape}")
    print(f"Size of Validation dataset: {dataset['validation'].shape}")
    
    # Create datasets
    train_dataset = wikihow(tokenizer, 'train', 1500, 512, 150, False)
    val_dataset = wikihow(tokenizer, 'validation', 300, 512, 150, False)
    
    # Create distributed samplers
    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)
    
    # Initialize distributed training
    setup()
    
    # Create data loaders
    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)
    
    # Define Transformer Auto Wrap Policy
    t5_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={T5Block},
    )
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    # Check BFloat16 support
    bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and LooseVersion(torch.version.cuda) >= "11.0"
        and dist.is_nccl_available()
    )
    
    # Define mixed precision policy
    if bf16_ready:
        bfSixteen = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        mp_policy = bfSixteen
    else:
        mp_policy = None  # defaults to fp32
    
    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device()
    )
    
    # Define optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    # Training variables
    best_val_loss = float("inf")
    curr_val_loss = float("inf")
    file_save_name = "T5-model-"
    
    if rank == 0:
        time_of_run = get_date_of_run()
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        training_start_time = time.time()
    
    if rank == 0 and args.track_memory:
        mem_alloc_tracker = []
        mem_reserved_tracker = []
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # Train for one epoch
        train_accuracy = train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        
        # Run validation if enabled
        if args.run_validation:
            curr_val_loss = validation(model, rank, world_size, val_loader)
        
        scheduler.step()
        
        # Rank 0: Log metrics
        if rank == 0:
            print(f"--> epoch {epoch} completed...entering save and stats zone")
            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())
            
            if args.run_validation:
                val_acc_tracking.append(curr_val_loss.item())
            
            if args.track_memory:
                mem_alloc_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_allocated())
                )
                mem_reserved_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_reserved())
                )
            
            print(f"completed save and stats zone...")
        
        # Save model if validation loss improved
        if args.save_model and curr_val_loss < best_val_loss:
            if rank == 0:
                print(f"--> entering save model state")
            
            # Save with CPU offloading
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state = model.state_dict()
            
            if rank == 0:
                print(f"--> saving model ...")
                currEpoch = f"-{epoch}-{round(curr_val_loss.item(), 4)}.pt"
                print(f"--> attempting to save model prefix {currEpoch}")
                save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
                print(f"--> saving as model name {save_name}")
                torch.save(cpu_state, save_name)
        
        # Update best validation loss
        if curr_val_loss < best_val_loss:
            best_val_loss = curr_val_loss
            if rank == 0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")
    
    # Cleanup
    dist.barrier()
    cleanup()
```

### Step 8: Parse Arguments and Run Main Function
```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch T5 FSDP Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=.002, metavar='LR',
                        help='learning rate (default: .002)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--track_memory', action='store_false', default=True,
                        help='track the gpu memory')
    parser.add_argument('--run_validation', action='store_false', default=True,
                        help='running the validation')
    parser.add_argument('--save-model', action='store_false', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    fsdp_main(args)
```

## Running the Training
To run the training on 4 GPUs:
```bash
torchrun --nnodes 1 --nproc_per_node 4 T5_training.py
```

## Advanced FSDP Features

### 1. Transformer Auto Wrap Policy
For transformer models with shared components (like embedding tables), use the transformer auto wrap policy to ensure efficient sharding:

```python
t5_auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={T5Block},
)

model = FSDP(
    model,
    auto_wrap_policy=t5_auto_wrap_policy
)
```

### 2. Mixed Precision
FSDP supports flexible mixed precision policies. Here are examples for different precision levels:

```python
# BFloat16 policy (for Ampere+ GPUs)
bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# FP16 policy
fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

# Gradient-only BFloat16 (reduces communication overhead)
grad_bf16 = MixedPrecision(reduce_dtype=torch.bfloat16)
```

### 3. Initializing FSDP Model on Device
Initialize the model directly on GPU to avoid CPU memory issues:

```python
model = FSDP(
    model,
    auto_wrap_policy=t5_auto_wrap_policy,
    mixed_precision=bfSixteen,
    device_id=torch.cuda.current_device()
)
```

### 4. Sharding Strategy
Choose between Zero2 (SHARD_GRAD_OP) and Zero3 (FULL_SHARD) sharding:

```python
# Zero2: Shard gradients and optimizer states only
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP
)

# Zero3: Full sharding (default)
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD
)
```

### 5. Backward Prefetch
Overlap communication and computation for faster training:

```python
model = FSDP(
    model,
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE
)
```

### 6. Model Checkpoint Saving with CPU Offloading
Save large models by streaming to CPU memory:

```python
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    cpu_state = model.state_dict()

