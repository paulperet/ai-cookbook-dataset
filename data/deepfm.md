# Implementing Deep Factorization Machines (DeepFM)

## Overview

Deep Factorization Machines (DeepFM) combine the strengths of Factorization Machines (FM) for low-order feature interactions and deep neural networks for high-order, non-linear feature combinations. This architecture automatically learns feature representations without extensive manual feature engineering, making it particularly effective for click-through rate (CTR) prediction tasks.

## Prerequisites

Before starting, ensure you have the necessary libraries installed. This tutorial uses MXNet.

```bash
pip install mxnet d2l
```

## Step 1: Import Required Libraries

Begin by importing the necessary modules from MXNet and the D2L library.

```python
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## Step 2: Define the DeepFM Model

The DeepFM model consists of three core components:
1. An **FM Component** to model pairwise (second-order) feature interactions.
2. A **Deep Component** (an MLP) to capture high-order and non-linear feature interactions.
3. A **Linear Component** for first-order (linear) feature weights.

These components share the same input embeddings, and their outputs are summed before applying a sigmoid activation for the final prediction.

```python
class DeepFM(nn.Block):
    def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        # Total number of unique features across all fields
        num_inputs = int(sum(field_dims))
        
        # Embedding layer for latent factor vectors (used by both FM and Deep components)
        self.embedding = nn.Embedding(num_inputs, num_factors)
        # Embedding layer for linear (first-order) terms
        self.fc = nn.Embedding(num_inputs, 1)
        # Dense layer for the linear component's bias
        self.linear_layer = nn.Dense(1, use_bias=True)
        
        # Calculate input dimension for the MLP: number of fields * embedding size
        self.embed_output_dim = len(field_dims) * num_factors
        input_dim = self.embed_output_dim
        
        # Build the MLP (Deep Component)
        self.mlp = nn.Sequential()
        for dim in mlp_dims:
            self.mlp.add(nn.Dense(dim, 'relu', True, in_units=input_dim))
            self.mlp.add(nn.Dropout(rate=drop_rate))
            input_dim = dim
        # Final output layer of the MLP
        self.mlp.add(nn.Dense(in_units=input_dim, units=1))

    def forward(self, x):
        # FM Component: Get embeddings for all features
        embed_x = self.embedding(x)
        
        # Calculate the second-order interaction term using the efficient FM formula
        square_of_sum = np.sum(embed_x, axis=1) ** 2
        sum_of_square = np.sum(embed_x ** 2, axis=1)
        fm_interaction = 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        
        # Linear Component: Sum of first-order feature weights
        linear_term = self.linear_layer(self.fc(x).sum(1))
        
        # Deep Component: Flatten embeddings and pass through MLP
        inputs = np.reshape(embed_x, (-1, self.embed_output_dim))
        deep_output = self.mlp(inputs)
        
        # Combine all components and apply sigmoid
        x = linear_term + fm_interaction + deep_output
        x = npx.sigmoid(x)
        return x
```

**Key Details:**
- `field_dims`: A list containing the number of unique values for each categorical feature field.
- `num_factors`: The dimensionality of the latent factor vectors (embedding size).
- `mlp_dims`: A list defining the number of neurons in each hidden layer of the MLP (e.g., `[30, 20, 10]`).
- The FM interaction term is computed efficiently using the identity: `sum_{i,j} <v_i, v_j> x_i x_j = 0.5 * ( (sum_i v_i x_i)^2 - sum_i (v_i x_i)^2 )`.

## Step 3: Load and Prepare the Dataset

We'll use the same CTR dataset as in the standard FM tutorial. The `CTRDataset` class handles loading and preprocessing.

```python
batch_size = 2048
data_dir = d2l.download_extract('ctr')

# Load training data
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
# Load test data, using the feature mappings from the training set
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)

field_dims = train_data.field_dims

# Create data loaders
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
```

## Step 4: Initialize the Model and Trainer

Instantiate the DeepFM model, initialize its weights, and set up the trainer and loss function.

```python
# Use all available GPUs, fallback to CPU if none
devices = d2l.try_all_gpus()

# Instantiate the model: 10 latent factors, MLP with layers [30, 20, 10]
net = DeepFM(field_dims, num_factors=10, mlp_dims=[30, 20, 10])
net.initialize(init.Xavier(), ctx=devices)

# Training configuration
lr, num_epochs, optimizer = 0.01, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
```

## Step 5: Train and Evaluate the Model

Use the `train_ch13` utility function to handle training across multiple devices (GPUs/CPUs) and evaluate performance on the test set.

```python
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

During training, you will see output similar to the following, showing the model's progress over epochs:

```
loss 0.510, train acc 0.767, test acc 0.780
...
loss 0.470, train acc 0.789, test acc 0.795
```

**Observation:** DeepFM typically converges faster and achieves a higher test accuracy compared to the standard FM model, demonstrating the benefit of adding a deep neural network component to capture complex feature interactions.

## Summary

In this tutorial, you implemented a Deep Factorization Machine (DeepFM) model for CTR prediction. You learned how to:

1. **Architect the Model:** Combine an FM component for low-order interactions with an MLP for high-order, non-linear interactions.
2. **Implement Efficient FM:** Use the optimized formula to compute second-order feature interactions.
3. **Train the Model:** Utilize MXNet's Gluon API to define, initialize, and train the model on a CTR dataset.

The key advantage of DeepFM is its ability to automatically learn both low and high-order feature combinations without manual feature engineering, often leading to superior performance over traditional FM.

## Exercises

To deepen your understanding, try the following modifications:

1. **Experiment with MLP Architecture:** Change the `mlp_dims` parameter (e.g., try `[50, 30]` or `[40, 20, 10, 5]`). Observe how the depth and width of the network affect training speed and final accuracy.
2. **Test on a Different Dataset:** Apply the model to the larger **Criteo** display advertising dataset. Compare its performance against the standard FM model. You may need to adjust hyperparameters like `num_factors`, `mlp_dims`, and learning rate.

---
*For further discussion, visit the [D2L forum topic](https://discuss.d2l.ai/t/407).*