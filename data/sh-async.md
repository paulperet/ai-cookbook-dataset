# Asynchronous Successive Halving (ASHA) Tutorial

## Overview
This guide demonstrates how to implement Asynchronous Successive Halving (ASHA) for hyperparameter optimization using Syne Tune. ASHA extends the successive halving algorithm to work efficiently in distributed, asynchronous environments by eliminating synchronization bottlenecks.

## Prerequisites

First, ensure you have the required libraries installed:

```bash
pip install syne-tune[gpsearchers]==0.3.2
pip install matplotlib
```

## 1. Import Required Modules

```python
import logging
import matplotlib.pyplot as plt
from syne_tune.config_space import loguniform, randint
from syne_tune.backend.python_backend import PythonBackend
from syne_tune.optimizer.baselines import ASHA
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment

# Set up logging
logging.basicConfig(level=logging.INFO)
```

## 2. Define the Objective Function

We'll use a LeNet model trained on FashionMNIST as our optimization target. The function reports validation error after each epoch.

```python
def hpo_objective_lenet_synetune(learning_rate, batch_size, max_epochs):
    from d2l import torch as d2l
    from syne_tune import Reporter

    # Initialize model and trainer
    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=1, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)
    
    # Initialize model weights
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    
    # Set up reporting
    report = Reporter()
    
    # Training loop
    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            # Initialize trainer state
            trainer.fit(model=model, data=data)
        else:
            trainer.fit_epoch()
        
        # Report validation error
        validation_error = d2l.numpy(trainer.validation_error().cpu())
        report(epoch=epoch, validation_error=float(validation_error))
```

## 3. Configure the Search Space

Define the hyperparameter ranges and an initial configuration:

```python
min_number_of_epochs = 2
max_number_of_epochs = 10
eta = 2  # Reduction factor

config_space = {
    "learning_rate": loguniform(1e-2, 1),
    "batch_size": randint(32, 256),
    "max_epochs": max_number_of_epochs,
}

initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

## 4. Set Up Distributed Execution Parameters

Configure the number of parallel workers and total runtime:

```python
n_workers = 2  # Should be ≤ available GPUs
max_wallclock_time = 12 * 60  # 12 minutes in seconds
```

## 5. Configure the ASHA Scheduler

Create the ASHA scheduler with appropriate parameters:

```python
mode = "min"  # We want to minimize validation error
metric = "validation_error"
resource_attr = "epoch"  # The resource being allocated (epochs)

scheduler = ASHA(
    config_space,
    metric=metric,
    mode=mode,
    points_to_evaluate=[initial_config],
    max_resource_attr="max_epochs",
    resource_attr=resource_attr,
    grace_period=min_number_of_epochs,  # Minimum resource level (r_min)
    reduction_factor=eta,  # η factor for promotion
)
```

**Key Parameters:**
- `grace_period`: Minimum number of epochs before early stopping (r_min)
- `reduction_factor`: Factor by which configurations are reduced at each rung (η)
- `max_resource_attr`: Name of the parameter representing maximum resources

## 6. Run the ASHA Optimization

Set up the backend and run the tuner:

```python
# Create backend for trial execution
trial_backend = PythonBackend(
    tune_function=hpo_objective_lenet_synetune,
    config_space=config_space,
)

# Configure stopping criterion
stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

# Create and run the tuner
tuner = Tuner(
    trial_backend=trial_backend,
    scheduler=scheduler,
    stop_criterion=stop_criterion,
    n_workers=n_workers,
    print_update_interval=int(max_wallclock_time * 0.6),
)

tuner.run()
```

This will run for approximately 12 minutes. The tuner executes trials asynchronously, with workers picking up new configurations as soon as they become available.

## 7. Analyze Results

After completion, load and visualize the experiment results:

```python
# Load the experiment
experiment = load_experiment(tuner.name)

# Plot the optimization progress
experiment.plot()
```

## 8. Visualize Learning Curves

Examine individual trial performance over time:

```python
import matplotlib.pyplot as plt

results = experiment.results

# Plot learning curves for each trial
plt.figure(figsize=(6, 2.5))
for trial_id in results.trial_id.unique():
    df = results[results["trial_id"] == trial_id]
    plt.plot(
        df["st_tuner_time"],
        df["validation_error"],
        marker="o"
    )

plt.xlabel("Wall-clock Time")
plt.ylabel("Validation Error")
plt.title("ASHA Trial Progress")
plt.show()
```

## Key Observations

1. **Asynchronous Operation**: Unlike synchronous successive halving, ASHA promotes configurations immediately when enough data is available, eliminating worker idle time.

2. **Early Stopping**: Most trials are stopped early (at 1-2 epochs), focusing resources only on promising configurations.

3. **Time Efficiency**: Workers continuously process trials without waiting for stragglers, maximizing hardware utilization.

## Comparison with Synchronous Successive Halving

| Aspect | Synchronous SH | ASHA |
|--------|---------------|------|
| Worker Synchronization | Required at each rung | Eliminated |
| Worker Idle Time | High due to stragglers | Minimal |
| Promotion Timing | After all trials complete rung | Immediate when η trials complete |
| Implementation Complexity | Lower | Higher |

## Summary

Asynchronous Successive Halving provides an efficient distributed implementation of successive halving by:
- Eliminating synchronization points between workers
- Promoting configurations as soon as sufficient data is available
- Maintaining high hardware utilization
- Achieving comparable performance to synchronous versions with significantly reduced wall-clock time

The trade-off is occasional suboptimal promotions, but in practice this has minimal impact on final optimization quality while providing substantial speed improvements.