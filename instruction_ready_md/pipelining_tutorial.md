# Distributed Pipeline Parallelism with PyTorch: A Step-by-Step Guide

## Introduction

This tutorial demonstrates how to implement distributed pipeline parallelism using PyTorch's `torch.distributed.pipelining` APIs. We'll use a GPT-style transformer model to show how to partition a model across multiple GPUs and schedule computation on micro-batches for efficient distributed training.

### What You'll Learn
- How to use `torch.distributed.pipelining` APIs
- How to apply pipeline parallelism to a transformer model
- How to utilize different schedules on a set of microbatches

### Prerequisites
- Familiarity with basic distributed training in PyTorch
- Basic understanding of transformer architectures

## Setup and Initialization

First, let's set up our environment and define the model architecture.

### 1. Define the Transformer Model

We'll create a simplified transformer decoder model with configurable parameters:

```python
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 10000

class Transformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        
        # Using a ModuleDict lets us delete layers without affecting names,
        # ensuring checkpoints will correctly save and load.
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = nn.TransformerDecoderLayer(model_args.dim, model_args.n_heads)
        
        self.norm = nn.LayerNorm(model_args.dim)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size)
    
    def forward(self, tokens: torch.Tensor):
        # Handling layers being 'None' at runtime enables easy pipeline splitting
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens
        
        for layer in self.layers.values():
            h = layer(h, h)
        
        h = self.norm(h) if self.norm else h
        output = self.output(h).clone() if self.output else h
        return output
```

### 2. Initialize Distributed Training

Now, let's set up the distributed environment. This function initializes the process group and sets up pipeline-specific variables:

```python
import os
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, PipelineStage, ScheduleGPipe

global rank, device, pp_group, stage_index, num_stages

def init_distributed():
    global rank, device, pp_group, stage_index, num_stages
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    dist.init_process_group()
    
    # This group can be a sub-group in the N-D parallel case
    pp_group = dist.new_group()
    stage_index = rank
    num_stages = world_size
```

The key pipeline-specific variables are:
- `pp_group`: The process group for pipeline communication
- `stage_index`: The index of the current pipeline stage (equal to rank in this example)
- `num_stages`: Total number of pipeline stages (equal to world_size)

## Step 1: Partition the Transformer Model

There are two approaches to partitioning the model: manual splitting and tracer-based splitting.

### Option 1: Manual Model Splitting

For a 2-stage pipeline, we split the model in half by deleting specific layers from each stage:

```python
def manual_model_split(model) -> PipelineStage:
    if stage_index == 0:
        # Prepare the first stage model
        for i in range(4, 8):
            del model.layers[str(i)]
        model.norm = None
        model.output = None
    
    elif stage_index == 1:
        # Prepare the second stage model
        for i in range(4):
            del model.layers[str(i)]
        model.tok_embeddings = None
    
    stage = PipelineStage(
        model,
        stage_index,
        num_stages,
        device,
    )
    return stage
```

In this split:
- **Stage 0**: Contains the first 4 transformer layers and the embedding layer
- **Stage 1**: Contains the last 4 transformer layers, layer norm, and output layer

### Option 2: Tracer-Based Model Splitting

The tracer-based approach automatically splits the model based on a specification:

```python
def tracer_model_split(model, example_input_microbatch) -> PipelineStage:
    pipe = pipeline(
        module=model,
        mb_args=(example_input_microbatch,),
        split_spec={
            "layers.4": SplitPoint.BEGINNING,
        }
    )
    stage = pipe.build_stage(stage_index, device, pp_group)
    return stage
```

This splits the model before the 4th transformer layer, mirroring the manual split above.

## Step 2: Define the Main Execution

Now let's create the main execution logic that sets up the pipeline schedule:

```python
if __name__ == "__main__":
    init_distributed()
    num_microbatches = 4
    model_args = ModelArgs()
    model = Transformer(model_args)
    
    # Create dummy data
    x = torch.ones(32, 500, dtype=torch.long)
    y = torch.randint(0, model_args.vocab_size, (32, 500), dtype=torch.long)
    example_input_microbatch = x.chunk(num_microbatches)[0]
    
    # Option 1: Manual model splitting
    stage = manual_model_split(model)
    
    # Option 2: Tracer model splitting (uncomment to use)
    # stage = tracer_model_split(model, example_input_microbatch)
    
    # Move data to device
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    
    # Define loss function
    def tokenwise_loss_fn(outputs, targets):
        loss_fn = nn.CrossEntropyLoss()
        outputs = outputs.reshape(-1, model_args.vocab_size)
        targets = targets.reshape(-1)
        return loss_fn(outputs, targets)
    
    # Create pipeline schedule
    schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=tokenwise_loss_fn)
    
    # Execute pipeline
    if rank == 0:
        schedule.step(x)
    elif rank == 1:
        losses = []
        output = schedule.step(target=y, losses=losses)
        print(f"losses: {losses}")
    
    dist.destroy_process_group()
```

Key components:
1. **Data Preparation**: We create dummy input and target tensors
2. **Model Splitting**: Choose either manual or tracer-based splitting
3. **Loss Function**: Define a token-wise cross-entropy loss
4. **Schedule Creation**: Use `ScheduleGPipe` for the pipeline schedule
5. **Execution**: Different ranks handle different parts of the pipeline

The `.step()` function automatically splits the minibatch into microbatches and processes them according to the GPipe schedule (all forwards followed by all backwards).

## Step 3: Launch the Distributed Processes

To run the distributed pipeline, use `torchrun` with the appropriate configuration:

```bash
torchrun --nnodes 1 --nproc_per_node 2 pipelining_tutorial.py
```

This command launches 2 processes on a single node, with each process handling one pipeline stage.

## Conclusion

In this tutorial, you've learned how to implement distributed pipeline parallelism using PyTorch's `torch.distributed.pipelining` APIs. We covered:

1. Setting up the distributed environment for pipeline parallelism
2. Defining a transformer model suitable for partitioning
3. Two methods for splitting the model across pipeline stages
4. Creating and executing a pipeline schedule with micro-batches
5. Launching the distributed training process

Pipeline parallelism is a powerful technique for scaling large models across multiple GPUs, and PyTorch's pipelining APIs provide flexible tools for implementing it efficiently.

## Additional Resources

For production-ready implementations and examples of pipeline parallelism combined with other distributed techniques, check out:

- [TorchTitan repository](https://github.com/pytorch/torchtitan): A clean, minimal codebase for large-scale LLM training
- [TorchTitan 3D parallelism example](https://github.com/pytorch/torchtitan): End-to-end example combining pipeline, tensor, and data parallelism