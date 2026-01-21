# Distributed Training with PyTorch RPC: A Practical Guide

## Overview

This guide demonstrates how to use PyTorch's Distributed RPC Framework to build distributed training applications. We'll explore two common scenarios:
1. **Distributed Reinforcement Learning** - Using RPC and RRef to coordinate multiple observers with a central agent
2. **Distributed Model Parallel Training** - Using distributed autograd and optimizer to split models across workers

## Prerequisites

Before starting, ensure you have:
- PyTorch installed (v1.4 or later)
- Basic understanding of distributed training concepts
- Familiarity with reinforcement learning and RNNs (for the examples)

## Part 1: Distributed Reinforcement Learning with RPC and RRef

### 1.1 Setup and Imports

First, let's import the necessary modules and set up argument parsing:

```python
import argparse
import os
from itertools import count

import gym
import numpy as np
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_async, remote
from torch.distributions import Categorical

parser = argparse.ArgumentParser(
    description="RPC Reinforcement Learning Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument('--world_size', default=2, type=int, metavar='W',
                    help='number of workers')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='how much to value future rewards')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed for reproducibility')
args = parser.parse_args()
```

### 1.2 Define the Policy Network

We'll use a simple neural network policy for our CartPole agent:

```python
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
```

### 1.3 Implement the Observer

Each observer runs in its own environment and communicates with the agent via RPC:

```python
class Observer:
    def __init__(self):
        self.id = rpc.get_worker_info().id
        self.env = gym.make('CartPole-v1')
        self.env.seed(args.seed)

    def run_episode(self, agent_rref):
        state, ep_reward = self.env.reset(), 0
        for _ in range(10000):
            # Send state to agent and get action back
            action = agent_rref.rpc_sync().select_action(self.id, state)
            
            # Apply action to environment
            state, reward, done, _ = self.env.step(action)
            
            # Report reward to agent for training
            agent_rref.rpc_sync().report_reward(self.id, reward)
            
            if done:
                break
```

### 1.4 Implement the Agent

The agent coordinates all observers and performs training:

```python
class Agent:
    def __init__(self, world_size):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.saved_log_probs = {}
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.running_reward = 0
        self.reward_threshold = gym.make('CartPole-v1').spec.reward_threshold
        
        # Initialize remote observers
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(remote(ob_info, Observer))
            self.rewards[ob_info.id] = []
            self.saved_log_probs[ob_info.id] = []
    
    def select_action(self, ob_id, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs[ob_id].append(m.log_prob(action))
        return action.item()
    
    def report_reward(self, ob_id, reward):
        self.rewards[ob_id].append(reward)
    
    def run_episode(self):
        futs = []
        for ob_rref in self.ob_rrefs:
            # Kick off episodes on all observers asynchronously
            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    ob_rref.rpc_sync().run_episode,
                    args=(self.agent_rref,)
                )
            )
        
        # Wait for all episodes to complete
        for fut in futs:
            fut.wait()
    
    def finish_episode(self):
        # Combine data from all observers
        R, probs, rewards = 0, [], []
        for ob_id in self.rewards:
            probs.extend(self.saved_log_probs[ob_id])
            rewards.extend(self.rewards[ob_id])
        
        # Calculate running reward
        min_reward = min([sum(self.rewards[ob_id]) for ob_id in self.rewards])
        self.running_reward = 0.05 * min_reward + (1 - 0.05) * self.running_reward
        
        # Clear stored data
        for ob_id in self.rewards:
            self.rewards[ob_id] = []
            self.saved_log_probs[ob_id] = []
        
        # Compute policy loss
        policy_loss, returns = [], []
        for r in rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        
        for log_prob, R in zip(probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        return min_reward
```

### 1.5 Launch Distributed Workers

Now let's set up the distributed execution:

```python
AGENT_NAME = "agent"
OBSERVER_NAME = "obs{}"

def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    if rank == 0:
        # Rank 0 is the agent
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)
        
        agent = Agent(world_size)
        print(f"This will run until reward threshold of {agent.reward_threshold} is reached.")
        
        for i_episode in count(1):
            agent.run_episode()
            last_reward = agent.finish_episode()
            
            if i_episode % args.log_interval == 0:
                print(f"Episode {i_episode}\tLast reward: {last_reward:.2f}\t"
                      f"Average reward: {agent.running_reward:.2f}")
            
            if agent.running_reward > agent.reward_threshold:
                print(f"Solved! Running reward is now {agent.running_reward}!")
                break
    else:
        # Other ranks are observers
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)
        # Observers wait passively for agent commands
    
    rpc.shutdown()

# Start the distributed training
mp.spawn(
    run_worker,
    args=(args.world_size,),
    nprocs=args.world_size,
    join=True
)
```

### 1.6 Expected Output

When running with `--world_size=2`, you should see output similar to:

```
This will run until reward threshold of 475.0 is reached.
Episode 10      Last reward: 26.00      Average reward: 10.01
Episode 20      Last reward: 16.00      Average reward: 11.27
...
Episode 290     Last reward: 500.00     Average reward: 464.65
Solved! Running reward is now 475.32!
```

## Part 2: Distributed Model Parallel Training

### 2.1 Setup Distributed Modules

For distributed model parallel training, we'll split an RNN model across workers:

```python
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import torch.optim as optim

class EmbeddingTable(nn.Module):
    def __init__(self, ntoken, ninp, dropout):
        super(EmbeddingTable, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp).cuda()
        self.encoder.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, input):
        return self.drop(self.encoder(input.cuda()).cpu())

class Decoder(nn.Module):
    def __init__(self, ntoken, nhid, dropout):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, output):
        return self.decoder(self.drop(output))
```

### 2.2 Create Distributed RNN Model

Now let's create the main model that uses RPC to coordinate between components:

```python
def _remote_method(method_rref, *args, **kwargs):
    return method_rref.local_value()(*args, **kwargs)

def _parameter_rrefs(module):
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs

class RNNModel(nn.Module):
    def __init__(self, ps, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        
        # Create embedding table on parameter server
        self.emb_table_rref = rpc.remote(ps, EmbeddingTable, 
                                         args=(ntoken, ninp, dropout))
        
        # Create LSTM locally
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        
        # Create decoder on parameter server
        self.decoder_rref = rpc.remote(ps, Decoder, 
                                       args=(ntoken, nhid, dropout))
    
    def forward(self, input, hidden):
        # Remote embedding lookup
        emb = _remote_method(EmbeddingTable.forward, 
                            self.emb_table_rref, input)
        
        # Local LSTM processing
        output, hidden = self.rnn(emb, hidden)
        
        # Remote decoding
        decoded = _remote_method(Decoder.forward, 
                                self.decoder_rref, output)
        
        return decoded, hidden
    
    def parameter_rrefs(self):
        remote_params = []
        # Get embedding table parameters
        remote_params.extend(_remote_method(_parameter_rrefs, 
                                           self.emb_table_rref))
        # Get local LSTM parameters
        remote_params.extend(_parameter_rrefs(self.rnn))
        # Get decoder parameters
        remote_params.extend(_remote_method(_parameter_rrefs, 
                                           self.decoder_rref))
        return remote_params
```

### 2.3 Implement Distributed Training Loop

Here's how to train the distributed model:

```python
def run_trainer():
    # Model configuration
    batch = 5
    ntoken = 10
    ninp = 2
    nhid = 3
    nindices = 3
    nlayers = 4
    
    # Initialize hidden state
    hidden = (
        torch.randn(nlayers, nindices, nhid),
        torch.randn(nlayers, nindices, nhid)
    )
    
    # Create distributed model
    model = RNNModel('ps', ntoken, ninp, nhid, nlayers)
    
    # Setup distributed optimizer
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    def get_next_batch():
        for _ in range(5):
            data = torch.LongTensor(batch, nindices) % ntoken
            target = torch.LongTensor(batch, ntoken) % nindices
            yield data, target
    
    # Training loop
    for epoch in range(10):
        for data, target in get_next_batch():
            # Create distributed autograd context
            with dist_autograd.context() as context_id:
                # Detach hidden state
                hidden = (hidden[0].detach(), hidden[1].detach())
                
                # Forward pass
                output, hidden = model(data, hidden)
                loss = criterion(output, target)
                
                # Distributed backward pass
                dist_autograd.backward(context_id, [loss])
                
                # Distributed optimizer step
                opt.step(context_id)
        
        print(f"Training epoch {epoch}")
```

### 2.4 Launch Distributed Workers

Finally, let's set up the parameter server and trainer:

```python
def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    if rank == 1:
        rpc.init_rpc("trainer", rank=rank, world_size=world_size)
        run_trainer()
    else:
        rpc.init_rpc("ps", rank=rank, world_size=world_size)
        # Parameter server waits for requests
    
    rpc.shutdown()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
```

## Key Takeaways

1. **RPC enables flexible communication** between distributed components without being tied to a specific training paradigm
2. **RRef provides transparent remote object references** that handle lifetime management automatically
3. **Distributed autograd** allows automatic gradient computation across multiple workers
4. **Distributed optimizer** coordinates parameter updates across different machines
5. **Model parallel training** becomes straightforward by splitting sub-modules across workers

The PyTorch RPC framework provides a powerful foundation for building custom distributed training systems beyond the standard data-parallel approach. By understanding these building blocks, you can implement complex distributed architectures tailored to your specific needs.