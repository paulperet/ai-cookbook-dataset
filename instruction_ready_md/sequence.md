# Working with Sequences: A Practical Guide to Autoregressive Models

## Introduction

In previous tutorials, we've focused on models that process single feature vectors. Now, we'll shift our perspective to handle **sequences** - ordered lists of feature vectors indexed by time. This is crucial for many real-world applications like:
- Natural language processing (documents as word sequences)
- Time series analysis (stock prices, sensor readings)
- Medical data (patient trajectories over hospital stays)

Unlike independent samples, sequence elements are related - future values depend on past ones. This tutorial will guide you through building and training sequence models, starting with autoregressive approaches.

## Prerequisites

First, let's set up our environment. We'll use the D2L library which provides consistent interfaces across different deep learning frameworks.

```python
# Select your preferred framework
import sys
import os

# Framework selection (choose one)
FRAMEWORK = 'pytorch'  # Options: 'mxnet', 'pytorch', 'tensorflow', 'jax'

# Import the appropriate D2L module
if FRAMEWORK == 'mxnet':
    from d2l import mxnet as d2l
    from mxnet import autograd, np, npx, gluon, init
    from mxnet.gluon import nn
    npx.set_np()
elif FRAMEWORK == 'pytorch':
    from d2l import torch as d2l
    import torch
    from torch import nn
elif FRAMEWORK == 'tensorflow':
    from d2l import tensorflow as d2l
    import tensorflow as tf
elif FRAMEWORK == 'jax':
    from d2l import jax as d2l
    import jax
    from jax import numpy as jnp
    import numpy as np

import matplotlib.pyplot as plt
```

## Understanding Autoregressive Models

### The Core Challenge

Consider predicting stock prices. At each time step `t`, we observe price `x_t`. A trader wants to know:

```
P(x_t | x_{t-1}, ..., x_1)
```

The probability distribution of the next price given all previous prices. Estimating the entire distribution is complex, so we often focus on key statistics like expected value.

**The Problem**: The number of inputs varies with `t`! If `t=10`, we have 9 previous values; if `t=100`, we have 99. This makes it hard to use standard models that require fixed-length inputs.

### Practical Solutions

Two common strategies address this:

1. **Fixed Window Approach**: Only look back `τ` time steps, using `x_{t-1}, ..., x_{t-τ}`. This gives us fixed-length inputs.

2. **Latent State Approach**: Maintain a summary `h_t` of all past observations that gets updated over time:
   - Predict: `x̂_t = P(x_t | h_t)`
   - Update: `h_t = g(h_{t-1}, x_{t-1})`

## Markov Models: A Simplifying Assumption

When we can predict `x_t` using only the previous `τ` observations without losing predictive power, we say the sequence satisfies a **Markov condition**. The future is conditionally independent of the distant past, given recent history.

For first-order Markov models (`τ = 1`), the joint probability factorizes nicely:

```
P(x_1, ..., x_T) = P(x_1) ∏_{t=2}^T P(x_t | x_{t-1})
```

Even when this is only approximately true, Markov assumptions help overcome computational challenges.

## Hands-On: Building an Autoregressive Model

### Step 1: Create Synthetic Sequence Data

Let's start with a simple example using synthetic data. We'll generate a sine wave with added noise to simulate real-world time series data.

```python
class SequenceData(d2l.DataModule):
    def __init__(self, batch_size=16, T=1000, num_train=600, tau=4):
        """Initialize sequence dataset.
        
        Args:
            batch_size: Batch size for training
            T: Total sequence length
            num_train: Number of training examples
            tau: Lookback window size
        """
        self.save_hyperparameters()
        
        # Create time steps
        self.time = d2l.arange(1, T + 1, dtype=d2l.float32)
        
        # Generate synthetic data: sine wave + noise
        if FRAMEWORK in ['mxnet', 'pytorch']:
            self.x = d2l.sin(0.01 * self.time) + d2l.randn(T) * 0.2
        elif FRAMEWORK == 'tensorflow':
            self.x = d2l.sin(0.01 * self.time) + d2l.normal([T]) * 0.2
        elif FRAMEWORK == 'jax':
            key = d2l.get_key()
            self.x = d2l.sin(0.01 * self.time) + jax.random.normal(key, [T]) * 0.2
    
    def visualize(self):
        """Plot the generated sequence."""
        d2l.plot(self.time, self.x, 'time', 'x', 
                xlim=[1, 1000], figsize=(6, 3))
        plt.show()

# Create and visualize data
data = SequenceData()
print("Data shape:", data.x.shape)
data.visualize()
```

### Step 2: Prepare Training Examples

We'll use a fixed window approach. For each time step `t`, we create an example where:
- Label: `y = x_t`
- Features: `[x_{t-τ}, ..., x_{t-1}]`

```python
@d2l.add_to_class(SequenceData)
def get_dataloader(self, train=True):
    """Create data loader for training or validation."""
    # Create features using sliding window
    features = [self.x[i : self.T - self.tau + i] for i in range(self.tau)]
    self.features = d2l.stack(features, 1)
    
    # Create labels (next value in sequence)
    self.labels = d2l.reshape(self.x[self.tau:], (-1, 1))
    
    # Split into train/validation
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    
    return self.get_tensorloader([self.features, self.labels], train, i)

# Test the data loader
train_loader = data.get_dataloader(train=True)
val_loader = data.get_dataloader(train=False)

print(f"Training samples: {len(data.features[:data.num_train])}")
print(f"Validation samples: {len(data.features[data.num_train:])}")
```

### Step 3: Train a Linear Regression Model

We'll start with a simple linear regression model to predict the next value in the sequence.

```python
# Initialize model
model = d2l.LinearRegression(lr=0.01)
trainer = d2l.Trainer(max_epochs=5)

# Train the model
trainer.fit(model, data)

print("Training complete!")
```

### Step 4: Evaluate One-Step-Ahead Predictions

Let's see how well our model performs when predicting just one step into the future.

```python
def evaluate_one_step(model, data):
    """Evaluate one-step-ahead predictions."""
    if FRAMEWORK == 'jax':
        onestep_preds = model.apply({'params': trainer.state.params}, 
                                   data.features)
    else:
        onestep_preds = d2l.numpy(model(data.features))
    
    # Plot results
    d2l.plot(data.time[data.tau:], 
             [data.labels, onestep_preds], 
             'time', 'x',
             legend=['True values', '1-step predictions'], 
             figsize=(6, 3))
    plt.show()
    
    return onestep_preds

# Evaluate
onestep_preds = evaluate_one_step(model, data)
```

### Step 5: Multi-Step Ahead Predictions

Now let's try predicting further into the future. This is more challenging because we have to use our own predictions as inputs for future steps.

```python
def make_multistep_predictions(model, data, steps_ahead=100):
    """Make predictions multiple steps into the future."""
    if FRAMEWORK == 'mxnet' or FRAMEWORK == 'pytorch':
        multistep_preds = d2l.zeros(data.T)
        multistep_preds[:] = data.x
        
        for i in range(data.num_train + data.tau, data.T):
            # Use previous predictions as inputs
            features = d2l.reshape(multistep_preds[i-data.tau : i], (1, -1))
            multistep_preds[i] = model(features)
        
        multistep_preds = d2l.numpy(multistep_preds)
        
    elif FRAMEWORK == 'tensorflow':
        multistep_preds = tf.Variable(d2l.zeros(data.T))
        multistep_preds[:].assign(data.x)
        
        for i in range(data.num_train + data.tau, data.T):
            features = d2l.reshape(multistep_preds[i-data.tau : i], (1, -1))
            prediction = model(features)
            multistep_preds[i].assign(d2l.reshape(prediction, ()))
            
        multistep_preds = multistep_preds.numpy()
        
    elif FRAMEWORK == 'jax':
        multistep_preds = d2l.zeros(data.T)
        multistep_preds = multistep_preds.at[:].set(data.x)
        
        for i in range(data.num_train + data.tau, data.T):
            features = d2l.reshape(multistep_preds[i-data.tau : i], (1, -1))
            pred = model.apply({'params': trainer.state.params}, features)
            multistep_preds = multistep_preds.at[i].set(pred.item())
    
    return multistep_preds

# Generate multi-step predictions
multistep_preds = make_multistep_predictions(model, data)

# Compare one-step vs multi-step predictions
d2l.plot([data.time[data.tau:], 
          data.time[data.num_train + data.tau:]],
         [onestep_preds, 
          multistep_preds[data.num_train + data.tau:]],
         'time', 'x',
         legend=['1-step predictions', 'Multi-step predictions'],
         figsize=(6, 3))
plt.show()
```

### Step 6: Analyze Prediction Quality Over Different Horizons

Let's systematically examine how prediction quality degrades as we predict further into the future.

```python
def k_step_ahead_predictions(model, data, k_values=(1, 4, 16, 64)):
    """Generate k-step-ahead predictions for different k."""
    max_k = max(k_values)
    features = []
    
    # Initial features from actual data
    for i in range(data.tau):
        features.append(data.x[i : i + data.T - data.tau - max_k + 1])
    
    # Generate predictions for each step
    for i in range(max_k):
        if FRAMEWORK == 'jax':
            preds = model.apply({'params': trainer.state.params},
                               d2l.stack(features[i : i + data.tau], 1))
        else:
            preds = model(d2l.stack(features[i : i + data.tau], 1))
        
        features.append(d2l.reshape(preds, -1))
    
    # Extract predictions for each k
    predictions = {}
    for k in k_values:
        predictions[k] = features[data.tau + k - 1]
    
    return predictions

# Generate and plot predictions
k_values = (1, 4, 16, 64)
predictions = k_step_ahead_predictions(model, data, k_values)

# Plot results
d2l.plot(data.time[data.tau + max(k_values) - 1:],
         [d2l.numpy(predictions[k]) for k in k_values],
         'time', 'x',
         legend=[f'{k}-step predictions' for k in k_values],
         figsize=(6, 3))
plt.show()
```

## Key Insights and Practical Considerations

### Error Accumulation in Multi-Step Predictions

Notice how prediction quality degrades rapidly as we predict further ahead. This happens because:
1. Each prediction has some error `ε`
2. These errors compound when predictions become inputs for future steps
3. The model sees increasingly different inputs than it was trained on

This is analogous to weather forecasting: predictions for the next 24 hours are decent, but forecasts for next week are much less reliable.

### The Importance of Temporal Order

Always respect temporal order when training sequence models:
- **Never train on future data** to predict the past
- Use proper time-based splits for train/validation/test sets
- Be cautious about data leakage from future information

### Choosing the Right Model Architecture

For our simple example, linear regression worked reasonably well for one-step predictions but failed for multi-step forecasts. In practice, you might need:

1. **More sophisticated models**: RNNs, LSTMs, or Transformers that can maintain longer-term dependencies
2. **Larger context windows**: If `τ=4` isn't enough, try larger values
3. **Regularization**: Prevent overfitting to noise in the training data
4. **Ensemble methods**: Combine predictions from multiple models

## Exercises for Further Exploration

1. **Improve the model**:
   - Experiment with different window sizes (`τ`)
   - Try adding more features or using a neural network
   - Implement early stopping to prevent overfitting

2. **Real-world considerations**:
   - What happens if the data distribution changes over time (non-stationarity)?
   - How would you handle missing values in the sequence?
   - What metrics are most appropriate for sequence prediction tasks?

3. **Advanced topics**:
   - Implement a latent autoregressive model with hidden states
   - Experiment with different loss functions (MSE, MAE, etc.)
   - Try sequence-to-sequence models for more complex patterns

## Summary

In this tutorial, you've learned:
- How sequence data differs from independent samples
- The challenges of variable-length inputs in autoregressive models
- Practical strategies like fixed windows and Markov assumptions
- How to build, train, and evaluate sequence prediction models
- Why multi-step predictions are challenging and how errors accumulate

Remember: sequence modeling requires careful attention to temporal dependencies. Always validate your approach on held-out temporal data, and be realistic about how far into the future you can reliably predict.

## Next Steps

Ready to dive deeper? Consider exploring:
- Recurrent Neural Networks (RNNs) for longer-term dependencies
- Attention mechanisms for focusing on relevant parts of the sequence
- Transformer architectures for state-of-the-art sequence modeling
- Probabilistic forecasting to quantify prediction uncertainty

Happy modeling!