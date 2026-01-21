# Batch RPC Processing with Asynchronous Execution

## Overview

This guide demonstrates how to build batch-processing RPC applications using PyTorch's `@rpc.functions.async_execution` decorator. This approach reduces blocked RPC threads and consolidates CUDA operations, significantly improving training performance.

## Prerequisites

- PyTorch v1.6.0 or higher
- Basic understanding of PyTorch's Distributed RPC Framework
- Familiarity with parameter server architectures

## Core Concepts

### The Problem with Traditional RPC
In PyTorch v1.5 and earlier, each RPC request blocks one thread on the callee until the function returns. This causes inefficiency when functions block on I/O or nested RPC calls.

### The Solution: Asynchronous Execution
PyTorch v1.6+ introduces two key features:
1. **`torch.futures.Future`**: Encapsulates asynchronous execution with callback support
2. **`@rpc.functions.async_execution`**: Decorator indicating a function returns a Future and can yield multiple times

These tools allow breaking functions into smaller pieces, chaining them as callbacks, and freeing RPC threads during waits.

---

## Part 1: Batch-Updating Parameter Server

### Implementation

#### 1.1 Parameter Server Setup

```python
import threading
import torchvision
import torch
import torch.distributed.rpc as rpc
from torch import optim

num_classes, batch_update_size = 30, 5

class BatchUpdateParameterServer:
    def __init__(self, batch_update_size=batch_update_size):
        self.model = torchvision.models.resnet50(num_classes=num_classes)
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.batch_update_size = batch_update_size
        self.curr_update_size = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
        # Initialize gradients
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

    def get_model(self):
        return self.model
```

#### 1.2 Asynchronous Update Method

```python
    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads):
        # Retrieve the local PS instance using the RRef
        self = ps_rref.local_value()
        
        with self.lock:
            self.curr_update_size += 1
            
            # Accumulate gradients
            for p, g in zip(self.model.parameters(), grads):
                p.grad += g

            # Save current future_model to ensure thread safety
            fut = self.future_model

            # Check if we've reached batch size
            if self.curr_update_size >= self.batch_update_size:
                # Average gradients and update model
                for p in self.model.parameters():
                    p.grad /= self.batch_update_size
                
                self.curr_update_size = 0
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Notify all waiting trainers
                fut.set_result(self.model)
                self.future_model = torch.futures.Future()

        return fut
```

**How it works:**
- Trainers call `update_and_fetch_model` with their gradients
- Early arrivals accumulate gradients and return immediately
- The last trainer triggers the optimizer step
- All trainers receive the updated model simultaneously via the Future object

#### 1.3 Trainer Implementation

```python
batch_size, image_w, image_h = 20, 64, 64

class Trainer:
    def __init__(self, ps_rref):
        self.ps_rref = ps_rref
        self.loss_fn = torch.nn.MSELoss()
        self.one_hot_indices = torch.LongTensor(batch_size) \
                                    .random_(0, num_classes) \
                                    .view(batch_size, 1)

    def get_next_batch(self):
        for _ in range(6):
            inputs = torch.randn(batch_size, 3, image_w, image_h)
            labels = torch.zeros(batch_size, num_classes) \
                        .scatter_(1, self.one_hot_indices, 1)
            yield inputs.cuda(), labels.cuda()

    def train(self):
        name = rpc.get_worker_info().name
        
        # Get initial model
        m = self.ps_rref.rpc_sync().get_model().cuda()
        
        # Training loop
        for inputs, labels in self.get_next_batch():
            # Forward and backward pass
            self.loss_fn(m(inputs), labels).backward()
            
            # Send gradients and get updated model
            m = rpc.rpc_sync(
                self.ps_rref.owner(),
                BatchUpdateParameterServer.update_and_fetch_model,
                args=(self.ps_rref, [p.grad for p in m.cpu().parameters()]),
            ).cuda()
```

**Key Points:**
- Trainers don't need to know about the async decorator
- The `rpc_sync` call blocks until the updated model is ready
- All trainers receive synchronized model updates

---

## Part 2: Batch-Processing Reinforcement Learning

### 2.1 Policy Network with Batch Support

```python
import argparse
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch RPC Batch RL example')
parser.add_argument('--gamma', type=float, default=1.0, metavar='G',
                    help='discount factor (default: 1.0)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--num-episode', type=int, default=10, metavar='E',
                    help='number of episodes (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self, batch=True):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)
        self.dim = 2 if batch else 1  # Batch dimension control

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=self.dim)
```

### 2.2 Observer with Batch Support

```python
import gym
import torch.distributed.rpc as rpc

class Observer:
    def __init__(self, batch=True):
        self.id = rpc.get_worker_info().id - 1
        self.env = gym.make('CartPole-v1')
        self.env.seed(args.seed)
        self.select_action = Agent.select_action_batch if batch else Agent.select_action

    def run_episode(self, agent_rref, n_steps):
        state, ep_reward = self.env.reset(), n_steps
        rewards = torch.zeros(n_steps)
        start_step = 0
        
        for step in range(n_steps):
            state = torch.from_numpy(state).float().unsqueeze(0)
            
            # Get action from agent
            action = rpc.rpc_sync(
                agent_rref.owner(),
                self.select_action,
                args=(agent_rref, self.id, state)
            )

            # Apply action
            state, reward, done, _ = self.env.step(action)
            rewards[step] = reward

            # Handle episode completion
            if done or step + 1 >= n_steps:
                curr_rewards = rewards[start_step:(step + 1)]
                R = 0
                # Calculate discounted rewards
                for i in range(curr_rewards.numel() - 1, -1, -1):
                    R = curr_rewards[i] + args.gamma * R
                    curr_rewards[i] = R
                
                state = self.env.reset()
                if start_step == 0:
                    ep_reward = min(ep_reward, step - start_step + 1)
                start_step = step + 1

        return [rewards, ep_reward]
```

### 2.3 Agent with Batch Processing

```python
import threading
from torch.distributed.rpc import RRef
from torch.distributions import Categorical

class Agent:
    def __init__(self, world_size, batch=True):
        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.policy = Policy(batch).cuda()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.running_reward = 0

        # Initialize observers
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(f'OBSERVER_{ob_rank}')
            self.ob_rrefs.append(rpc.remote(ob_info, Observer, args=(batch,)))
            self.rewards[ob_info.id] = []

        # Batch-specific initialization
        self.states = torch.zeros(len(self.ob_rrefs), 1, 4)
        self.batch = batch
        self.saved_log_probs = [] if batch else {k: [] for k in range(len(self.ob_rrefs))}
        self.future_actions = torch.futures.Future()
        self.lock = threading.Lock()
        self.pending_states = len(self.ob_rrefs)
```

### 2.4 Non-Batch Action Selection

```python
    @staticmethod
    def select_action(agent_rref, ob_id, state):
        self = agent_rref.local_value()
        probs = self.policy(state.cuda())
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs[ob_id].append(m.log_prob(action))
        return action.item()
```

### 2.5 Batch Action Selection with Async Execution

```python
    @staticmethod
    @rpc.functions.async_execution
    def select_action_batch(agent_rref, ob_id, state):
        self = agent_rref.local_value()
        
        # Store state in batch tensor
        self.states[ob_id].copy_(state)
        
        # Chain callback for this specific observer
        future_action = self.future_actions.then(
            lambda future_actions: future_actions.wait()[ob_id].item()
        )

        with self.lock:
            self.pending_states -= 1
            
            # Last observer triggers batch processing
            if self.pending_states == 0:
                self.pending_states = len(self.ob_rrefs)
                
                # Process all states in one batch
                probs = self.policy(self.states.cuda())
                m = Categorical(probs)
                actions = m.sample()
                self.saved_log_probs.append(m.log_prob(actions).t()[0])
                
                # Notify all waiting observers
                future_actions = self.future_actions
                self.future_actions = torch.futures.Future()
                future_actions.set_result(actions.cpu())
                
        return future_action
```

### 2.6 Episode Execution

```python
    def run_episode(self, n_steps=0):
        # Launch episodes on all observers
        futs = []
        for ob_rref in self.ob_rrefs:
            futs.append(ob_rref.rpc_async().run_episode(self.agent_rref, n_steps))

        # Wait for all observers
        rets = torch.futures.wait_all(futs)
        rewards = torch.stack([ret[0] for ret in rets]).cuda().t()
        ep_rewards = sum([ret[1] for ret in rets]) / len(rets)

        # Prepare probabilities for backpropagation
        if self.batch:
            probs = torch.stack(self.saved_log_probs)
        else:
            probs = [torch.stack(self.saved_log_probs[i]) for i in range(len(rets))]
            probs = torch.stack(probs)

        # Policy gradient update
        policy_loss = -probs * rewards / len(rets)
        policy_loss.sum().backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Reset for next episode
        self.saved_log_probs = [] if self.batch else {k: [] for k in range(len(self.ob_rrefs))}
        self.states = torch.zeros(len(self.ob_rrefs), 1, 4)

        # Update running reward
        self.running_reward = 0.5 * ep_rewards + 0.5 * self.running_reward
        return ep_rewards, self.running_reward
```

### 2.7 Worker Launch Function

```python
import os
import time
import torch.multiprocessing as mp

AGENT_NAME = "agent"
OBSERVER_NAME = "observer_{}"
NUM_STEPS = 100

def run_worker(rank, world_size, n_episode, batch, print_log=True):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    if rank == 0:
        # Agent process
        rpc.init_rpc(AGENT_NAME, rank=rank, world_size=world_size)
        agent = Agent(world_size, batch)
        
        for i_episode in range(n_episode):
            last_reward, running_reward = agent.run_episode(n_steps=NUM_STEPS)
            
            if print_log:
                print(f'Episode {i_episode}\tLast reward: {last_reward:.2f}\tAverage reward: {running_reward:.2f}')
    else:
        # Observer processes
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank=rank, world_size=world_size)
        # Observers wait passively for agent commands
    
    rpc.shutdown()

def main():
    for world_size in range(2, 12):
        delays = []
        for batch in [True, False]:
            start_time = time.time()
            mp.spawn(
                run_worker,
                args=(world_size, args.num_episode, batch),
                nprocs=world_size,
                join=True
            )
            end_time = time.time()
            delays.append(end_time - start_time)
        
        print(f"{world_size}, {delays[0]}, {delays[1]}")

if __name__ == '__main__':
    main()
```

## Performance Benefits

Batch processing with `@rpc.functions.async_execution` provides:
1. **Reduced thread blocking**: RPC threads yield during waits
2. **Consolidated CUDA operations**: Batch inference reduces kernel launches
3. **Lower communication overhead**: Fewer RPC round trips
4. **Better GPU utilization**: Larger, more efficient batches

## Key Takeaways

1. Use `@rpc.functions.async_execution` for functions that can yield during execution
2. Return `torch.futures.Future` objects from async RPC functions
3. Chain callbacks using `.then()` for dependent asynchronous operations
4. Use locking to synchronize batch accumulation
5. The last arriving request typically triggers batch processing

## Further Resources

- [Batch-Updating Parameter Server Example](https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/parameter_server.py)
- [Batch-Processing CartPole Solver](https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/reinforce.py)
- [PyTorch RPC Documentation](https://pytorch.org/docs/stable/rpc.html)