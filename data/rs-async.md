# Asynchronous Random Search for Hyperparameter Optimization

## Overview

In hyperparameter optimization (HPO), evaluating configurations can take hours or days. This guide demonstrates how to accelerate random search using asynchronous parallel execution across multiple workers, eliminating idle time and achieving near-linear speedups.

## Prerequisites

Ensure you have the required libraries installed:

```bash
pip install syne-tune[gpsearchers]==0.3.2
pip install d2l
```

## 1. Import Dependencies

```python
import logging
logging.basicConfig(level=logging.INFO)

from syne_tune.config_space import loguniform, randint
from syne_tune.backend.python_backend import PythonBackend
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune import Tuner, StoppingCriterion, Reporter
from syne_tune.experiments import load_experiment

from d2l import torch as d2l
```

## 2. Define the Objective Function

Create a training function that reports metrics back to Syne Tune. Note that all dependencies must be imported inside the function for compatibility with Syne Tune's Python backend.

```python
def hpo_objective_lenet_synetune(learning_rate, batch_size, max_epochs):
    # Import inside function for Syne Tune compatibility
    from d2l import torch as d2l
    from syne_tune import Reporter
    
    # Initialize model, trainer, and data
    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=1, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    
    # Create reporter for metric logging
    report = Reporter()
    
    # Training loop
    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            trainer.fit(model=model, data=data)  # Initial training
        else:
            trainer.fit_epoch()  # Subsequent epochs
        
        # Report validation error after each epoch
        validation_error = d2l.numpy(trainer.validation_error().cpu())
        report(epoch=epoch, validation_error=float(validation_error))
```

## 3. Configure Asynchronous Execution

Set up the parallel execution environment and optimization parameters.

```python
# Number of parallel workers (must not exceed available GPUs)
n_workers = 2

# Maximum wall-clock time for the optimization (12 minutes)
max_wallclock_time = 12 * 60

# Optimization objective
mode = "min"  # Minimize validation error
metric = "validation_error"  # Must match report() argument name
```

## 4. Define the Search Space

Specify the hyperparameter ranges and initial configuration.

```python
config_space = {
    "learning_rate": loguniform(1e-2, 1),  # Log-uniform between 0.01 and 1
    "batch_size": randint(32, 256),       # Integer between 32 and 256
    "max_epochs": 10,                     # Constant passed to training function
}

initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

## 5. Set Up the Execution Backend

Configure Syne Tune to execute trials as sub-processes on your local machine.

```python
trial_backend = PythonBackend(
    tune_function=hpo_objective_lenet_synetune,
    config_space=config_space,
)
```

## 6. Create the Scheduler

Initialize the asynchronous random search scheduler.

```python
scheduler = RandomSearch(
    config_space,
    metric=metric,
    mode=mode,
    points_to_evaluate=[initial_config],  # Start with this configuration
)
```

## 7. Configure the Tuner

The tuner manages the main experiment loop and coordinates between scheduler and backend.

```python
stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

tuner = Tuner(
    trial_backend=trial_backend,
    scheduler=scheduler,
    stop_criterion=stop_criterion,
    n_workers=n_workers,
    print_update_interval=int(max_wallclock_time * 0.6),  # Print updates periodically
)
```

## 8. Run the Optimization

Execute the asynchronous random search experiment.

```python
tuner.run()
```

The optimization will run for approximately 12 minutes, with up to 2 trials executing in parallel.

## 9. Analyze Results

After the optimization completes, load and visualize the results.

```python
# Plot the incumbent trajectory (best validation error over time)
d2l.set_figsize()
tuning_experiment = load_experiment(tuner.name)
tuning_experiment.plot()
```

## 10. Visualize Asynchronous Execution

Examine how trials progress concurrently over time.

```python
d2l.set_figsize([6, 2.5])
results = tuning_experiment.results

# Plot each trial's validation error over wall-clock time
for trial_id in results.trial_id.unique():
    df = results[results["trial_id"] == trial_id]
    d2l.plt.plot(
        df["st_tuner_time"],
        df["validation_error"],
        marker="o"
    )
    
d2l.plt.xlabel("wall-clock time")
d2l.plt.ylabel("validation error")
d2l.plt.show()
```

This visualization shows multiple trials (different colors) executing concurrently. New trials start immediately when workers become available, minimizing idle time.

## Key Concepts

- **Synchronous vs. Asynchronous Scheduling**: Synchronous scheduling waits for all trials in a batch to complete before starting the next batch, potentially causing worker idle time. Asynchronous scheduling starts new trials immediately when resources become available.

- **Linear Speedup**: With asynchronous random search, you can achieve K-times faster convergence when running K trials in parallel, as each configuration is chosen independently.

- **Worker Management**: Ensure `n_workers` does not exceed available GPUs. Each trial requires dedicated resources.

## Exercises

1. **Extend to Different Models**:
   - Implement `hpo_objective_dropoutmlp_synetune` for the `DropoutMLP` model
   - Compare random search with Bayesian optimization (`syne_tune.optimizer.baselines.BayesianOptimization`)
   - Experiment with different worker counts (1, 2, 4) to observe scaling behavior

2. **Advanced: Implement Custom Scheduler**:
   - Set up a development environment with d2lbook and syne-tune sources
   - Implement the `LocalSearcher` as a new Syne Tune scheduler
   - Compare your implementation with `RandomSearch` on the `DropoutMLP` benchmark

## Summary

Asynchronous random search provides an efficient way to distribute hyperparameter optimization across multiple workers. By eliminating synchronization points and starting new trials immediately when resources become available, you can achieve near-linear speedups compared to sequential execution. While random search is particularly easy to parallelize asynchronously, more sophisticated HPO methods require additional modifications for effective parallelization.