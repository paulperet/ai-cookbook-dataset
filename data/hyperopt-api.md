# Hyperparameter Optimization API: A Practical Guide
**Objective:** Learn to implement a modular HPO framework and run a basic random search to tune a neural network.

## 1. Setup & Imports

First, ensure you have the necessary libraries installed. We'll use PyTorch, SciPy, and the D2L library for utilities.

```bash
pip install torch scipy d2l
```

Now, import the required modules.

```python
import time
from d2l import torch as d2l
from scipy import stats
```

## 2. Core HPO Components

We structure HPO around three core classes: a **Searcher** (samples configurations), a **Scheduler** (manages trial execution), and a **Tuner** (orchestrates the process). This mirrors popular libraries like Syne Tune, Ray Tune, and Optuna.

### 2.1 The Base Searcher

The `HPOSearcher` is responsible for proposing new hyperparameter configurations. Simple methods (like random search) sample randomly, while advanced ones (like Bayesian Optimization) learn from past trials.

```python
class HPOSearcher(d2l.HyperParameters):
    """Base class for hyperparameter configuration samplers."""
    def sample_configuration(self) -> dict:
        raise NotImplementedError

    def update(self, config: dict, error: float, additional_info=None):
        """Update the searcher's internal state with trial results."""
        pass
```

### 2.2 Implementing Random Search

Let's create a `RandomSearcher` that samples configurations uniformly from a defined space. It can optionally start with a user-specified initial configuration.

```python
class RandomSearcher(HPOSearcher):
    def __init__(self, config_space: dict, initial_config=None):
        self.save_hyperparameters()
        self.config_space = config_space
        self.initial_config = initial_config

    def sample_configuration(self) -> dict:
        if self.initial_config is not None:
            result = self.initial_config
            self.initial_config = None  # Use it only once
        else:
            # Sample each hyperparameter from its defined distribution
            result = {
                name: domain.rvs()
                for name, domain in self.config_space.items()
            }
        return result
```

### 2.3 The Base Scheduler

The `HPOScheduler` decides when to start a new trial and how long to run it. It uses a `HPOSearcher` to get configuration suggestions.

```python
class HPOScheduler(d2l.HyperParameters):
    """Base class for scheduling HPO trials."""
    def suggest(self) -> dict:
        raise NotImplementedError

    def update(self, config: dict, error: float, info=None):
        raise NotImplementedError
```

### 2.4 A Basic Scheduler

For random search, we need a simple scheduler that suggests a new configuration whenever resources are free and passes results back to the searcher.

```python
class BasicScheduler(HPOScheduler):
    def __init__(self, searcher: HPOSearcher):
        self.save_hyperparameters()
        self.searcher = searcher

    def suggest(self) -> dict:
        return self.searcher.sample_configuration()

    def update(self, config: dict, error: float, info=None):
        self.searcher.update(config, error, additional_info=info)
```

### 2.5 The Tuner: Running the Optimization Loop

The `HPOTuner` sequentially executes trials, manages the optimization loop, and tracks results.

```python
class HPOTuner(d2l.HyperParameters):
    def __init__(self, scheduler: HPOScheduler, objective: callable):
        self.save_hyperparameters()
        self.scheduler = scheduler
        self.objective = objective

        # Bookkeeping attributes
        self.incumbent = None           # Best config found
        self.incumbent_error = None     # Error of best config
        self.incumbent_trajectory = []  # Historical best errors
        self.cumulative_runtime = []    # Total runtime per step
        self.current_runtime = 0
        self.records = []               # Full trial history

    def run(self, number_of_trials):
        """Execute the HPO loop for a given number of trials."""
        for i in range(number_of_trials):
            start_time = time.time()

            # 1. Get a new configuration from the scheduler
            config = self.scheduler.suggest()
            print(f"Trial {i}: config = {config}")

            # 2. Evaluate the configuration
            error = self.objective(**config)
            error = float(d2l.numpy(error.cpu()))  # Ensure it's a Python float

            # 3. Update the scheduler with the result
            self.scheduler.update(config, error)

            # 4. Record results
            runtime = time.time() - start_time
            self.bookkeeping(config, error, runtime)
            print(f"    error = {error}, runtime = {runtime}")
```

### 2.6 Performance Tracking

To compare HPO algorithms, we track the best validation error over time ("any-time performance"). This shows how quickly an optimizer finds good configurations.

```python
@d2l.add_to_class(HPOTuner)
def bookkeeping(self, config: dict, error: float, runtime: float):
    """Update records and track the incumbent (best) configuration."""
    self.records.append({"config": config, "error": error, "runtime": runtime})

    # Update incumbent if this trial is better
    if self.incumbent is None or self.incumbent_error > error:
        self.incumbent = config
        self.incumbent_error = error

    # Track the trajectory of best error over time
    self.incumbent_trajectory.append(self.incumbent_error)

    # Accumulate total runtime
    self.current_runtime += runtime
    self.cumulative_runtime.append(self.current_runtime)
```

## 3. Practical Example: Tuning LeNet on Fashion-MNIST

Let's apply our framework to optimize the **learning rate** and **batch size** for the LeNet CNN on the Fashion-MNIST dataset.

### 3.1 Define the Objective Function

The objective function takes hyperparameters, trains a model, and returns the validation error.

```python
def hpo_objective_lenet(learning_rate, batch_size, max_epochs=10):
    """Objective: Train LeNet and return validation error."""
    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=max_epochs, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)

    # Initialize model weights
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)

    # Train
    trainer.fit(model=model, data=data)

    # Return validation error
    validation_error = trainer.validation_error()
    return validation_error
```

### 3.2 Define the Search Space

We'll search over a log-uniform distribution for the learning rate and a uniform integer distribution for batch size. We also specify a sensible initial configuration.

```python
config_space = {
    "learning_rate": stats.loguniform(1e-2, 1),  # Log scale between 0.01 and 1
    "batch_size": stats.randint(32, 256),        # Integers between 32 and 255
}

initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

### 3.3 Run Random Search

Now, instantiate the components and run the optimization for 5 trials.

```python
# 1. Create the searcher
searcher = RandomSearcher(config_space, initial_config=initial_config)

# 2. Create the scheduler
scheduler = BasicScheduler(searcher=searcher)

# 3. Create the tuner
tuner = HPOTuner(scheduler=scheduler, objective=hpo_objective_lenet)

# 4. Run the optimization
tuner.run(number_of_trials=5)
```

**Example Output:**
```
Trial 0: config = {'learning_rate': 0.1, 'batch_size': 128}
    error = 0.089, runtime = 45.2
Trial 1: config = {'learning_rate': 0.034, 'batch_size': 217}
    error = 0.123, runtime = 42.8
...
```

### 3.4 Visualize the Optimization Trajectory

Plot the best validation error observed over time to assess the any-time performance.

```python
board = d2l.ProgressBoard(xlabel="time", ylabel="error")
for time_stamp, error in zip(tuner.cumulative_runtime, tuner.incumbent_trajectory):
    board.draw(time_stamp, error, "random search", every_n=1)
```

## 4. Comparing HPO Algorithms: Best Practices

When comparing HPO methods like Random Search and Bayesian Optimization, follow these guidelines:

1.  **Multiple Runs:** Each algorithm has inherent randomness. Run multiple independent repetitions (e.g., 50) with different random seeds.
2.  **Report Statistics:** Plot the **mean** (solid line) and **standard deviation** (dashed band) of the incumbent error across all runs.
3.  **Any-Time Performance:** The plot of best error vs. cumulative runtime reveals how quickly an algorithm finds good configurations.

For example, a proper comparison might show that Bayesian Optimization outperforms Random Search after ~1000 seconds by leveraging past observations to guide its search.

## 5. Summary

You've built a modular HPO framework with interchangeable Searchers and Schedulers. This structure is the foundation for implementing more advanced algorithms. Remember to always compare HPO methods with multiple runs to account for randomness.

## 6. Exercises

### 6.1 Implement a More Complex Objective
1.  Create an objective function for tuning a `DropoutMLP` (from :numref:`sec_dropout`). It should depend on hyperparameters like `num_hiddens_1`, `num_hiddens_2`, `dropout_1`, `dropout_2`, `lr`, and `batch_size`. Use `max_epochs=50`.
2.  Define a sensible `config_space` using `scipy.stats` distributions.
3.  Run random search for 20 trials, starting from the default configuration: `{'num_hiddens_1': 256, 'num_hiddens_2': 256, 'dropout_1': 0.5, 'dropout_2': 0.5, 'lr': 0.1, 'batch_size': 256}`. Plot the results.

### 6.2 Implement a Local Search Searcher
Create a `LocalSearcher` that:
-   For the first `num_init_random` trials, acts like `RandomSearcher`.
-   Afterwards, with probability `probab_local`, it takes the best configuration so far and randomly perturbs **one** of its hyperparameters. Otherwise, it acts like `RandomSearcher`.
1.  Implement this searcher, including the `update` method to track the best configuration.
2.  Test it on the exercise from 6.1, experimenting with different values for `probab_local` and `num_init_random`. For a fair comparison, you would need to run multiple repetitions.

---
*For further discussion, visit the [D2L.ai forum](https://discuss.d2l.ai/t/12092).*