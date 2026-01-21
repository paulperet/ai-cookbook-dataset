# Distributed Training with PyTorch DDP: A Practical Guide to Training minGPT

## Overview

In this tutorial, you'll learn how to implement distributed training for real-world models using PyTorch's Distributed Data Parallel (DDP). We'll walk through training a GPT model across multiple nodes and GPUs, covering best practices, cloud storage integration, and understanding DDP's limitations.

## Prerequisites

Before starting, ensure you have:

- PyTorch installed with CUDA on all machines
- Familiarity with [multi-GPU training](../beginner/ddp_series_multigpu.html) and [torchrun](../beginner/ddp_series_fault_tolerance.html)
- Optional: Experience with [multinode training](ddp_series_multinode.html)
- 2 or more TCP-reachable GPU machines for multi-node training (this tutorial uses AWS p3.2xlarge instances)

## What You'll Learn

- Best practices for writing distributed training scripts
- How to save and load artifacts from cloud storage
- When DDP might not be suitable for your use case

## Project Structure

We'll be working with a modified version of the [minGPT repository](https://github.com/karpathy/minGPT). The key files are:

1. **trainer.py** - Contains the Trainer class for distributed training iterations
2. **model.py** - Defines the GPT model architecture
3. **char_dataset.py** - Dataset class for character-level training data
4. **gpt2_train_cfg.yaml** - Configuration file for data, model, optimizer, and training parameters
5. **main.py** - Entry point that sets up DDP and runs the training job

## Step 1: Clone and Prepare the Repository

First, clone the minGPT repository and refactor it for distributed training:

```bash
git clone https://github.com/pytorch/examples.git
cd examples/distributed/minGPT-ddp
```

## Step 2: Understand the Configuration System

We use [Hydra](https://hydra.cc/) to manage all training configurations centrally. This allows us to easily switch between different training setups without modifying code.

The configuration file `gpt2_train_cfg.yaml` contains:

```yaml
# Example configuration structure
data:
  dataset: "shakespeare"
  batch_size: 64
  
model:
  n_layer: 12
  n_head: 12
  n_embd: 768
  
training:
  max_epochs: 10
  learning_rate: 0.0003
  save_dir: "s3://my-bucket/training-checkpoints/"
```

## Step 3: Set Up Distributed Training

The `main.py` file handles DDP initialization. Here's the key setup process:

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    """Initialize the distributed process group"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup():
    """Clean up distributed resources"""
    dist.destroy_process_group()

def main(rank, world_size, config):
    """Main training function for each process"""
    setup(rank, world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)
    
    # Create model and move to device
    model = GPT(config.model).to(rank)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Create dataset and dataloader
    dataset = CharDataset(config.data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=config.data.batch_size, sampler=sampler)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    
    # Create trainer and run training
    trainer = Trainer(model, dataloader, optimizer, config)
    trainer.train()
    
    cleanup()
```

## Step 4: Implement Cloud Storage Integration

One of the advantages of distributed training is the ability to save checkpoints directly to cloud storage, allowing training to resume from any node with access to the bucket:

```python
import boto3
import torch
from io import BytesIO

class CloudCheckpointSaver:
    def __init__(self, bucket_name, prefix="checkpoints"):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
        self.prefix = prefix
    
    def save_checkpoint(self, epoch, model_state, optimizer_state, filename):
        """Save checkpoint to S3"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
        }
        
        # Save to buffer
        buffer = BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)
        
        # Upload to S3
        key = f"{self.prefix}/{filename}"
        self.s3.upload_fileobj(buffer, self.bucket_name, key)
        print(f"Checkpoint saved to s3://{self.bucket_name}/{key}")
    
    def load_checkpoint(self, filename):
        """Load checkpoint from S3"""
        key = f"{self.prefix}/{filename}"
        buffer = BytesIO()
        self.s3.download_fileobj(self.bucket_name, key, buffer)
        buffer.seek(0)
        checkpoint = torch.load(buffer)
        return checkpoint
```

## Step 5: Run Single-Node Multi-GPU Training

Test your setup with single-node training first:

```bash
# Run on a single node with 4 GPUs
torchrun --nproc_per_node=4 main.py --config-name=gpt2_train_cfg.yaml
```

## Step 6: Run Multi-Node Training

For multi-node training, you'll need to specify all nodes:

```bash
# Node 0 (master)
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=123 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=node0_ip:29500 \
    main.py --config-name=gpt2_train_cfg.yaml

# Node 1 (worker)
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id=123 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=node0_ip:29500 \
    main.py --config-name=gpt2_train_cfg.yaml
```

## Step 7: Implement Mixed Precision Training (Optional)

To speed up training, you can implement mixed precision:

```python
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, dataloader, optimizer, config):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.config = config
        self.scaler = GradScaler()  # For mixed precision
    
    def train_step(self, batch):
        inputs, targets = batch
        inputs, targets = inputs.cuda(), targets.cuda()
        
        # Mixed precision context
        with autocast():
            outputs = self.model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        # Scale loss and backward pass
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return loss.item()
```

## Understanding DDP Limitations

DDP replicates the entire model on each GPU, which means it only works when GPUs have sufficient memory for:
- Model weights
- Activations
- Gradients
- Input batches
- Optimizer states

For larger models, consider these alternatives:

### 1. Activation Checkpointing
Instead of saving all intermediate activations, recompute them during backward pass:

```python
from torch.utils.checkpoint import checkpoint_sequential

# Use checkpointing for memory efficiency
def forward_with_checkpointing(self, x):
    segments = [segment for segment in self.model_segments]
    return checkpoint_sequential(segments, len(segments), x)
```

### 2. Fully-Sharded Data Parallel (FSDP)
Shard the model across GPUs instead of replicating it:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

# Wrap model with FSDP
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    cpu_offload=True
)
```

## Best Practices Summary

1. **Always test single-node first**: Ensure your code works locally before scaling to multiple nodes
2. **Use cloud storage for checkpoints**: This enables flexible training resumption
3. **Implement proper logging**: Each process should log to separate files or a centralized service
4. **Monitor memory usage**: Keep an eye on GPU memory to prevent OOM errors
5. **Use mixed precision when possible**: This can significantly speed up training with minimal accuracy loss

## Troubleshooting Common Issues

1. **Connection errors**: Ensure all nodes can reach each other on the specified ports
2. **CUDA out of memory**: Reduce batch size or implement gradient accumulation
3. **Slow training**: Check network bandwidth between nodes and consider using mixed precision
4. **Checkpoint loading failures**: Verify cloud storage permissions and file paths

## Next Steps

Now that you've successfully implemented distributed training with DDP, you can:

1. Experiment with different model architectures
2. Try training on larger datasets
3. Implement more advanced techniques like gradient accumulation
4. Explore FSDP for even larger models

## Further Reading

- [Multi-Node training with DDP](ddp_series_multinode.html) (previous tutorial in this series)
- [Mixed Precision training](https://pytorch.org/docs/stable/amp.html)
- [Fully-Sharded Data Parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Training a 1T parameter model with FSDP](https://medium.com/pytorch/training-a-1-trillion-parameter-model-with-pytorch-fully-sharded-data-parallel-on-aws-3ac13aa96cff)

Remember that distributed training is an iterative process. Start small, validate each step, and gradually scale up as you gain confidence in your implementation.