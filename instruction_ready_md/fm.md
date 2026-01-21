# Factorization Machines: A Practical Guide

Factorization Machines (FM), introduced by Rendle (2010), are a powerful supervised algorithm for classification, regression, and ranking. They generalize linear regression and matrix factorization, capturing feature interactions efficiently—especially in high-dimensional, sparse data common in advertising and recommendation systems. This guide walks you through implementing and training a 2-way FM model for Click-Through Rate (CTR) prediction.

## Prerequisites

Ensure you have the necessary libraries installed. This tutorial uses MXNet.

```bash
pip install mxnet d2l
```

## 1. Understanding the 2-Way FM Model

The FM model for a feature vector \( x \in \mathbb{R}^d \) is:

\[
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
\]

- **Global bias and linear weights:** \( \mathbf{w}_0 \) and \( \mathbf{w}_i \) model linear effects.
- **Latent factor interactions:** \( \langle\mathbf{v}_i, \mathbf{v}_j\rangle \) captures pairwise feature interactions via embeddings \( \mathbf{V} \in \mathbb{R}^{d \times k} \).

A key innovation is the reformulation of the interaction term to compute in \( \mathcal{O}(kd) \) time instead of \( \mathcal{O}(kd^2) \), making FM scalable.

## 2. Implementing the Factorization Machine

We'll implement the FM model in MXNet. The model combines a linear block (for bias and weights) and an interaction block (for pairwise feature interactions).

```python
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()

class FM(nn.Block):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)

    def forward(self, x):
        # Efficient pairwise interaction computation
        square_of_sum = np.sum(self.embedding(x), axis=1) ** 2
        sum_of_square = np.sum(self.embedding(x) ** 2, axis=1)
        
        # Linear term + interaction term
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        x = npx.sigmoid(x)  # For binary classification (CTR)
        return x
```

**Key components:**
- `embedding`: Latent factor vectors \( \mathbf{v}_i \).
- `fc`: Linear weights \( \mathbf{w}_i \).
- `linear_layer`: Global bias \( \mathbf{w}_0 \).
- The forward pass computes the reformulated interaction term efficiently.

## 3. Loading the Advertising Dataset

We'll use a CTR dataset. The `d2l` library provides a convenient wrapper.

```python
from d2l import mxnet as d2l

batch_size = 2048
data_dir = d2l.download_extract('ctr')

# Load training and test data
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)

# Create data loaders
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
```

## 4. Training the Model

Initialize the model, set up the optimizer and loss function, then train.

```python
# Initialize model and move to available GPUs
devices = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=devices)

# Training configuration
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

# Train
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

**Training details:**
- **Embedding size:** 20 latent factors.
- **Optimizer:** Adam with learning rate 0.02.
- **Loss:** Sigmoid binary cross-entropy (for CTR classification).
- The `train_ch13` function handles training loops, metric logging, and GPU utilization.

## 5. Summary

- Factorization Machines efficiently model feature interactions, reducing the need for manual feature engineering.
- The reformulated interaction term enables linear-time computation, making FM suitable for large-scale sparse data.
- This implementation applies FM to CTR prediction, but the model is flexible for regression, classification, and ranking tasks.

## 6. Exercises

1. **Test on other datasets:** Experiment with Avazu, MovieLens, or Criteo datasets. Adjust `CTRDataset` loading accordingly.
2. **Embedding size ablation:** Vary `num_factors` (e.g., 10, 20, 50) and observe performance changes. Compare with matrix factorization trends.
3. **Extend to higher-order interactions:** Modify the model to include 3-way interactions (note: computational complexity increases).

---
*For further discussion, visit the [D2L forum](https://discuss.d2l.ai/t/406).*