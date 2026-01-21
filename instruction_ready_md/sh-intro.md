# Multi-Fidelity Hyperparameter Optimization Guide

## Introduction

Training neural networks can be expensive, even on moderate-sized datasets. Hyperparameter optimization (HPO) typically requires tens to hundreds of function evaluations to find a well-performing configuration. While parallel resources can speed up wall-clock time, they don't reduce the total computational cost.

This guide introduces **multi-fidelity hyperparameter optimization**, a technique that allocates more resources to promising configurations and stops poorly performing ones early. By evaluating configurations at different resource levels (e.g., number of epochs), we can explore more configurations within the same total budget.

## Prerequisites

Ensure you have the necessary libraries installed:

```bash
pip install numpy scipy matplotlib torch
```

Then, import the required modules:

```python
import numpy as np
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
```

## 1. Understanding the Multi-Fidelity Approach

In standard HPO, each configuration is evaluated with the same resource allocation (e.g., 100 training epochs). However, as shown in learning curves, we can often distinguish between good and bad configurations after just a few epochs.

Multi-fidelity HPO formalizes this by introducing a resource parameter `r` (e.g., number of epochs) to the objective function `f(x, r)`. The key assumptions are:
- Error `f(x, r)` decreases as `r` increases.
- Computational cost `c(x, r)` increases with `r`.

This allows us to initially evaluate many configurations with minimal resources, then progressively allocate more resources only to the most promising ones.

## 2. Implementing Successive Halving

Successive halving is a simple yet effective multi-fidelity algorithm. The process works as follows:

1. Start with `N` randomly sampled configurations, each trained for `r_min` epochs.
2. Keep the top `1/η` fraction of performers, discard the rest.
3. Train surviving configurations for `η` times more epochs.
4. Repeat until at least one configuration reaches `r_max` epochs.

### 2.1 Define the Successive Halving Scheduler

We'll create a scheduler that manages this process, compatible with our HPO framework.

```python
class SuccessiveHalvingScheduler:
    def __init__(self, searcher, eta, r_min, r_max, prefact=1):
        self.searcher = searcher
        self.eta = eta
        self.r_min = r_min
        self.r_max = r_max
        self.prefact = prefact
        
        # Compute number of rungs (K)
        self.K = int(np.log(r_max / r_min) / np.log(eta))
        
        # Define rung levels
        self.rung_levels = [r_min * eta ** k for k in range(self.K + 1)]
        if r_max not in self.rung_levels:
            self.rung_levels.append(r_max)
            self.K += 1
        
        # Bookkeeping structures
        self.observed_error_at_rungs = defaultdict(list)
        self.all_observed_error_at_rungs = defaultdict(list)
        
        # Queue for configurations to evaluate
        self.queue = []
```

### 2.2 Suggest Configurations

When the tuner asks for a new configuration to evaluate, we return one from our queue. If the queue is empty, we start a new round of successive halving.

```python
def suggest(self):
    if len(self.queue) == 0:
        # Start a new round
        n0 = int(self.prefact * self.eta ** self.K)  # Initial configurations
        for _ in range(n0):
            config = self.searcher.sample_configuration()
            config["max_epochs"] = self.r_min  # Start with minimum resource
            self.queue.append(config)
    
    # Return next configuration from queue
    return self.queue.pop()

SuccessiveHalvingScheduler.suggest = suggest
```

### 2.3 Update with Results

When we receive evaluation results, we update our records and determine if we can promote configurations to the next rung.

```python
def update(self, config, error, info=None):
    ri = int(config["max_epochs"])  # Current rung level
    
    # Update the searcher
    self.searcher.update(config, error, additional_info=info)
    
    # Record the result
    self.all_observed_error_at_rungs[ri].append((config, error))
    
    if ri < self.r_max:
        self.observed_error_at_rungs[ri].append((config, error))
        
        # Determine how many configurations should be evaluated at this rung
        ki = self.K - self.rung_levels.index(ri)
        ni = int(self.prefact * self.eta ** ki)
        
        # If we've observed all configurations at this rung, promote the best ones
        if len(self.observed_error_at_rungs[ri]) >= ni:
            kiplus1 = ki - 1
            niplus1 = int(self.prefact * self.eta ** kiplus1)
            
            # Get top-performing configurations
            best_configs = self.get_top_n_configurations(rung_level=ri, n=niplus1)
            
            # Next rung level
            riplus1 = self.rung_levels[self.K - kiplus1]
            
            # Promote configurations to next rung (insert at beginning of queue)
            self.queue = [
                dict(config, max_epochs=riplus1)
                for config in best_configs
            ] + self.queue
            
            # Reset current rung observations
            self.observed_error_at_rungs[ri] = []

SuccessiveHalvingScheduler.update = update
```

### 2.4 Helper Method to Select Top Configurations

We need a method to identify the best-performing configurations at each rung.

```python
def get_top_n_configurations(self, rung_level, n):
    rung = self.observed_error_at_rungs[rung_level]
    if not rung:
        return []
    
    # Sort by error (lower is better)
    sorted_rung = sorted(rung, key=lambda x: x[1])
    return [x[0] for x in sorted_rung[:n]]

SuccessiveHalvingScheduler.get_top_n_configurations = get_top_n_configurations
```

## 3. Applying Successive Halving to Neural Network HPO

Now let's apply our successive halving scheduler to optimize a neural network.

### 3.1 Define the Configuration Space

We'll use a simple configuration space with learning rate and batch size:

```python
# Define configuration space
config_space = {
    "learning_rate": stats.loguniform(1e-2, 1),
    "batch_size": stats.randint(32, 256),
}

# Initial configuration
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

### 3.2 Set Up Successive Halving Parameters

Choose appropriate resource levels for our multi-fidelity approach:

```python
min_number_of_epochs = 2
max_number_of_epochs = 10
eta = 2  # Halving constant
```

### 3.3 Create and Run the Tuner

We'll use a random searcher with our successive halving scheduler:

```python
# Create searcher and scheduler
searcher = RandomSearcher(config_space, initial_config=initial_config)
scheduler = SuccessiveHalvingScheduler(
    searcher=searcher,
    eta=eta,
    r_min=min_number_of_epochs,
    r_max=max_number_of_epochs,
)

# Create and run tuner
tuner = HPOTuner(
    scheduler=scheduler,
    objective=hpo_objective_lenet,  # Your objective function
)
tuner.run(number_of_trials=30)
```

## 4. Visualizing Results

Let's visualize how configurations were evaluated at different resource levels:

```python
# Plot validation errors at each rung
for rung_index, rung in scheduler.all_observed_error_at_rungs.items():
    errors = [xi[1] for xi in rung]
    plt.scatter([rung_index] * len(errors), errors)

plt.xlim(min_number_of_epochs - 0.5, max_number_of_epochs + 0.5)
plt.xticks(
    np.arange(min_number_of_epochs, max_number_of_epochs + 1),
    np.arange(min_number_of_epochs, max_number_of_epochs + 1)
)
plt.ylabel("validation error")
plt.xlabel("epochs")
plt.title("Multi-Fidelity HPO with Successive Halving")
plt.show()
```

The visualization shows that most configurations are evaluated only at lower resource levels (fewer epochs), while only the best-performing configurations progress to higher resource levels. This contrasts with standard random search, which would allocate `max_epochs` to every configuration.

## 5. Implementation Notes

Our implementation handles an important edge case: when a worker becomes available but we haven't yet collected all results for the current rung (because other workers are still busy). In this case:
1. We start a new round of successive halving to keep workers busy.
2. When the current rung completes, we insert promoted configurations at the beginning of the queue, giving them priority over configurations from the new round.

This ensures efficient resource utilization without idle workers.

## Summary

In this guide, we implemented successive halving for multi-fidelity hyperparameter optimization. Key takeaways:

1. **Multi-fidelity HPO** uses cheap approximations (e.g., validation error after few epochs) to identify promising configurations early.
2. **Successive halving** progressively allocates more resources to top-performing configurations while discarding poor ones.
3. This approach reduces total computational cost compared to standard HPO methods that allocate equal resources to all configurations.

By implementing this scheduler, you can significantly accelerate your hyperparameter optimization workflows while exploring a larger configuration space within the same computational budget.

## Next Steps

- Experiment with different values of `η` to balance exploration vs. exploitation.
- Try alternative multi-fidelity methods like Hyperband, which runs multiple successive halving brackets in parallel.
- Integrate with more advanced searchers like Bayesian optimization for better configuration sampling.