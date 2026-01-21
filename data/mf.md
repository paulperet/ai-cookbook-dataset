# Matrix Factorization for Recommender Systems: A Practical Guide

This guide walks you through implementing a Matrix Factorization model for collaborative filtering, a foundational technique in recommender systems. We'll build a model to predict user ratings for movies using the MovieLens dataset.

## Prerequisites

First, ensure you have the necessary libraries installed. We'll be using MXNet and the D2L library.

```bash
pip install mxnet d2l
```

## Understanding Matrix Factorization

Matrix Factorization decomposes the user-item interaction matrix (e.g., ratings) into lower-dimensional latent factor matrices for users and items. The predicted rating from user *u* to item *i* is calculated as:

\[
\hat{\mathbf{R}}_{ui} = \mathbf{p}_u\mathbf{q}^\top_i + b_u + b_i
\]

Where:
- \(\mathbf{p}_u\) represents user *u*'s latent factors
- \(\mathbf{q}_i\) represents item *i*'s latent factors  
- \(b_u\) and \(b_i\) are user and item bias terms

The model is trained by minimizing the mean squared error between predicted and actual ratings with L2 regularization to prevent overfitting.

## Step 1: Import Required Libraries

```python
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()
```

## Step 2: Implement the Matrix Factorization Model

We'll create a neural network block that uses embedding layers to represent user and item latent factors, along with their bias terms.

```python
class MF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        super(MF, self).__init__(**kwargs)
        # User and item latent factor embeddings
        self.P = nn.Embedding(input_dim=num_users, output_dim=num_factors)
        self.Q = nn.Embedding(input_dim=num_items, output_dim=num_factors)
        # User and item bias embeddings
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        # Look up embeddings for the given user and item IDs
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        
        # Compute predicted rating: dot product + biases
        outputs = (P_u * Q_i).sum(axis=1) + np.squeeze(b_u) + np.squeeze(b_i)
        return outputs.flatten()
```

## Step 3: Create an Evaluation Metric

We'll use Root Mean Square Error (RMSE) to evaluate our model's performance. RMSE measures the difference between predicted and actual ratings.

```python
def evaluator(net, test_iter, devices):
    rmse = mx.metric.RMSE()
    rmse_list = []
    
    for idx, (users, items, ratings) in enumerate(test_iter):
        # Split data across available devices
        u = gluon.utils.split_and_load(users, devices, even_split=False)
        i = gluon.utils.split_and_load(items, devices, even_split=False)
        r_ui = gluon.utils.split_and_load(ratings, devices, even_split=False)
        
        # Generate predictions
        r_hat = [net(u, i) for u, i in zip(u, i)]
        
        # Update RMSE calculation
        rmse.update(labels=r_ui, preds=r_hat)
        rmse_list.append(rmse.get()[1])
    
    return float(np.mean(np.array(rmse_list)))
```

## Step 4: Implement the Training Loop

The training function handles the forward pass, backpropagation, and evaluation across multiple epochs.

```python
def train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices=d2l.try_all_gpus(), evaluator=None, **kwargs):
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        
        for i, values in enumerate(train_iter):
            timer.start()
            input_data = []
            values = values if isinstance(values, list) else [values]
            
            # Prepare data for multi-device training
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))
            
            train_feat = input_data[:-1] if len(values) > 1 else input_data
            train_label = input_data[-1]
            
            # Forward pass
            with autograd.record():
                preds = [net(*t) for t in zip(*train_feat)]
                ls = [loss(p, s) for p, s in zip(preds, train_label)]
            
            # Backward pass
            [l.backward() for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean() / len(devices)
            
            # Update parameters
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        
        # Evaluate on test set
        if len(kwargs) > 0:
            test_rmse = evaluator(net, test_iter, kwargs['inter_mat'], devices)
        else:
            test_rmse = evaluator(net, test_iter, devices)
        
        train_l = l / (i + 1)
        animator.add(epoch + 1, (train_l, test_rmse))
    
    print(f'train loss {metric[0] / metric[1]:.3f}, test RMSE {test_rmse:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}')
```

## Step 5: Train the Model

Now let's put everything together and train our matrix factorization model on the MovieLens 100K dataset.

```python
# Set up devices and load data
devices = d2l.try_all_gpus()
num_users, num_items, train_iter, test_iter = d2l.split_and_load_ml100k(
    test_ratio=0.1, batch_size=512)

# Initialize model with 30 latent factors
net = MF(30, num_users, num_items)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))

# Configure training parameters
lr, num_epochs, wd, optimizer = 0.002, 20, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})

# Train the model
train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                    devices, evaluator)
```

During training, you'll see the training loss and test RMSE decreasing over epochs. The final output will show metrics like:

```
train loss 0.123, test RMSE 0.945
12540.7 examples/sec on [gpu(0), gpu(1)]
```

## Step 6: Make Predictions

Once trained, you can use the model to predict ratings for specific user-item pairs. For example, let's predict the rating that user ID 20 would give to item ID 30:

```python
scores = net(np.array([20], dtype='int', ctx=devices[0]),
             np.array([30], dtype='int', ctx=devices[0]))
print(f"Predicted rating: {scores[0]:.3f}")
```

## Summary

In this guide, you've learned how to:
1. Implement a Matrix Factorization model using embedding layers in MXNet
2. Incorporate user and item bias terms to capture systematic rating tendencies
3. Train the model using L2 loss with weight decay regularization
4. Evaluate model performance using RMSE
5. Make predictions for specific user-item pairs

Matrix factorization remains a powerful baseline for recommender systems, effectively capturing latent patterns in user-item interactions.

## Exercises to Try

1. **Experiment with latent factor dimensions**: Try different values (e.g., 10, 50, 100) for the number of latent factors. How does this affect model performance and training time?

2. **Optimizer tuning**: Test different optimizers (SGD, RMSProp, AdamW) and learning rates. Can you achieve better RMSE with different configurations?

3. **Regularization strength**: Adjust the weight decay parameter to find the optimal balance between fitting the training data and generalizing to unseen data.

4. **Make more predictions**: Use the trained model to predict ratings for various user-movie combinations and compare them with actual ratings when available.

5. **Batch size effects**: Experiment with different batch sizes. How does this affect training stability and final performance?