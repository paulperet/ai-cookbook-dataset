# Hyperparameter Optimization: A Practical Guide

## Introduction

Hyperparameter optimization (HPO) is a critical component of machine learning workflows. While neural networks learn their parameters during training, hyperparameters must be configured by the user. These include learning rates, batch sizes, regularization parameters, and architectural choices. Manual tuning is time-consuming and requires expert knowledge, making automated HPO essential for efficient model development.

In this guide, we'll explore HPO fundamentals and implement a simple random search algorithm to optimize a logistic regression model on the Fashion MNIST dataset.

## Prerequisites

First, let's set up our environment with the necessary imports:

```python
import numpy as np
import torch
from torch import nn
from scipy import stats
from d2l import torch as d2l
```

## 1. Understanding the HPO Problem

### 1.1 The Objective Function

The goal of HPO is to find hyperparameter configurations that minimize validation error. We define an objective function `f(x)` that maps from hyperparameter space `x ∈ X` to validation loss. Each evaluation requires training and validating a model, which can be computationally expensive.

For neural networks, training is stochastic (random initialization, mini-batch sampling), so our observations are noisy: `y ∼ f(x) + ε`, where `ε ∼ N(0, σ)`.

### 1.2 Configuration Space

The configuration space defines feasible hyperparameter values. For our logistic regression example, we'll optimize the learning rate with a log-uniform distribution between 1e-4 and 1:

```python
config_space = {"learning_rate": stats.loguniform(1e-4, 1)}
```

Log-uniform sampling is appropriate for learning rates since optimal values can span several orders of magnitude.

## 2. Implementing the Validation Error Computation

We need a way to compute validation error for our model. Let's extend the standard trainer:

```python
class HPOTrainer(d2l.Trainer):
    def validation_error(self):
        self.model.eval()
        accuracy = 0
        val_batch_idx = 0
        
        for batch in self.val_dataloader:
            with torch.no_grad():
                x, y = self.prepare_batch(batch)
                y_hat = self.model(x)
                accuracy += self.model.accuracy(y_hat, y)
            val_batch_idx += 1
            
        return 1 - accuracy / val_batch_idx
```

## 3. Defining the HPO Objective

Now we create our objective function that takes a configuration, trains a model, and returns validation error:

```python
def hpo_objective_softmax_classification(config, max_epochs=8):
    learning_rate = config["learning_rate"]
    trainer = d2l.HPOTrainer(max_epochs=max_epochs)
    data = d2l.FashionMNIST(batch_size=16)
    model = d2l.SoftmaxRegression(num_outputs=10, lr=learning_rate)
    trainer.fit(model=model, data=data)
    return d2l.numpy(trainer.validation_error())
```

## 4. Implementing Random Search

Random search independently samples from the configuration space until exhausting a predefined budget. Here's a sequential implementation:

```python
errors, values = [], []
num_iterations = 5

for i in range(num_iterations):
    # Sample learning rate from configuration space
    learning_rate = config_space["learning_rate"].rvs()
    print(f"Trial {i}: learning_rate = {learning_rate}")
    
    # Evaluate configuration
    y = hpo_objective_softmax_classification({"learning_rate": learning_rate})
    print(f"    validation_error = {y}")
    
    # Store results
    values.append(learning_rate)
    errors.append(y)
```

## 5. Identifying the Best Configuration

After running random search, we select the configuration with the lowest validation error:

```python
best_idx = np.argmin(errors)
print(f"Optimal learning rate = {values[best_idx]}")
print(f"Best validation error = {errors[best_idx]}")
```

## Key Insights

1. **Random Search Advantages**: Simple to implement, easily parallelizable, and doesn't require gradient computations.

2. **Limitations**: Doesn't adapt sampling based on previous observations and allocates equal resources to all configurations regardless of performance.

3. **Configuration Space Design**: Critical for success. Ranges that are too narrow may exclude optimal values, while ranges that are too broad increase search time.

## Practical Considerations

### Validation Set Protocol
In our example, we use the original Fashion MNIST test set for validation. In practice, you should:
1. Split the original training data into training and validation sets
2. Use the original test set only for final evaluation
3. Consider k-fold cross-validation for small datasets

### Beyond Random Search
While random search is a good baseline, more sophisticated methods include:
- Bayesian optimization (adapts sampling based on previous results)
- Early stopping strategies (terminates poorly performing configurations early)
- Multi-fidelity methods (uses cheaper approximations to guide search)

## Exercises for Further Learning

1. **Validation Protocol**: Why is using the test set for validation problematic? What should we do instead?

2. **Gradient-Based HPO**: Consider why gradient descent through the entire training process is challenging for HPO. What are the computational and numerical issues?

3. **Grid Search vs Random Search**: Research why random search can be more efficient than grid search when only a subset of hyperparameters significantly affects performance.

## Summary

Hyperparameter optimization is essential for achieving good model performance. Random search provides a simple yet effective baseline that outperforms grid search in many scenarios. By understanding the configuration space and objective function, you can implement basic HPO and build toward more sophisticated methods.

Remember: The goal is often to find good configurations quickly rather than the global optimum exactly, especially for computationally expensive models like deep neural networks.