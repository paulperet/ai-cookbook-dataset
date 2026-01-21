# Neural Collaborative Filtering for Personalized Ranking: A Step-by-Step Guide

This guide walks you through implementing the Neural Matrix Factorization (NeuMF) model for personalized ranking with implicit feedback. Implicit feedback—like clicks, purchases, or views—is abundant and indicative of user preferences. NeuMF combines the strengths of matrix factorization and deep learning to create a powerful recommendation model.

## Prerequisites

Ensure you have the necessary libraries installed. This guide uses MXNet.

```bash
pip install mxnet d2l
```

## 1. Import Libraries

We begin by importing the required modules.

```python
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import random

npx.set_np()
```

## 2. Define the NeuMF Model

The NeuMF model fuses two subnetworks:
1.  **Generalized Matrix Factorization (GMF):** A neural version of matrix factorization.
2.  **Multi-Layer Perceptron (MLP):** A deep network to capture complex user-item interactions.

The outputs of these networks are concatenated and passed through a final prediction layer.

```python
class NeuMF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, nums_hiddens, **kwargs):
        super(NeuMF, self).__init__(**kwargs)
        # Embeddings for GMF pathway
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        # Embeddings for MLP pathway
        self.U = nn.Embedding(num_users, num_factors)
        self.V = nn.Embedding(num_items, num_factors)
        
        # Build the MLP
        self.mlp = nn.Sequential()
        for num_hiddens in nums_hiddens:
            self.mlp.add(nn.Dense(num_hiddens, activation='relu', use_bias=True))
        
        # Final prediction layer
        self.prediction_layer = nn.Dense(1, activation='sigmoid', use_bias=False)

    def forward(self, user_id, item_id):
        # GMF pathway
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf  # Element-wise product
        
        # MLP pathway
        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp = self.mlp(np.concatenate([p_mlp, q_mlp], axis=1))
        
        # Concatenate and predict
        con_res = np.concatenate([gmf, mlp], axis=1)
        return self.prediction_layer(con_res)
```

## 3. Create a Dataset with Negative Sampling

For pairwise ranking loss, we need negative samples—items a user has *not* interacted with. The following custom dataset handles this.

```python
class PRDataset(gluon.data.Dataset):
    def __init__(self, users, items, candidates, num_items):
        self.users = users
        self.items = items
        self.cand = candidates
        self.all = set([i for i in range(num_items)])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        # Get all items not in the user's candidate set
        neg_items = list(self.all - set(self.cand[int(self.users[idx])]))
        # Randomly sample one negative item
        indices = random.randint(0, len(neg_items) - 1)
        return self.users[idx], self.items[idx], neg_items[indices]
```

## 4. Implement Evaluation Metrics

We evaluate the model using:
-   **Hit Rate @ k:** Whether the ground-truth item is in the top-k recommendations.
-   **Area Under the Curve (AUC):** Measures the model's ranking quality.

First, a helper function calculates these for a single user's ranked list.

```python
def hit_and_auc(rankedlist, test_matrix, k):
    hits_k = [(idx, val) for idx, val in enumerate(rankedlist[:k])
              if val in set(test_matrix)]
    hits_all = [(idx, val) for idx, val in enumerate(rankedlist)
                if val in set(test_matrix)]
    max = len(rankedlist) - 1
    auc = 1.0 * (max - hits_all[0][0]) / max if len(hits_all) > 0 else 0
    return len(hits_k), auc
```

Next, the main evaluation function computes the average Hit Rate and AUC across all users.

```python
def evaluate_ranking(net, test_input, seq, candidates, num_users, num_items, devices):
    ranked_list, ranked_items, hit_rate, auc = {}, {}, [], []
    all_items = set([i for i in range(num_users)])
    
    for u in range(num_users):
        # Get negative items for this user
        neg_items = list(all_items - set(candidates[int(u)]))
        user_ids, item_ids, x, scores = [], [], [], []
        
        [item_ids.append(i) for i in neg_items]
        [user_ids.append(u) for _ in neg_items]
        x.extend([np.array(user_ids)])
        
        if seq is not None:
            x.append(seq[user_ids, :])
        x.extend([np.array(item_ids)])
        
        # Create data loader and get scores for all items
        test_data_iter = gluon.data.DataLoader(
            gluon.data.ArrayDataset(*x), shuffle=False, last_batch="keep",
            batch_size=1024)
        
        for index, values in enumerate(test_data_iter):
            x = [gluon.utils.split_and_load(v, devices, even_split=False)
                 for v in values]
            scores.extend([list(net(*t).asnumpy()) for t in zip(*x)])
        
        scores = [item for sublist in scores for item in sublist]
        item_scores = list(zip(item_ids, scores))
        
        # Rank items by predicted score
        ranked_list[u] = sorted(item_scores, key=lambda t: t[1], reverse=True)
        ranked_items[u] = [r[0] for r in ranked_list[u]]
        
        # Calculate metrics for this user
        temp = hit_and_auc(ranked_items[u], test_input[u], 50)
        hit_rate.append(temp[0])
        auc.append(temp[1])
    
    return np.mean(np.array(hit_rate)), np.mean(np.array(auc))
```

## 5. Define the Training Loop

The training function uses pairwise ranking loss (BPR Loss) to ensure positive items are ranked higher than negative ones.

```python
def train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter,
                  num_users, num_items, num_epochs, devices, evaluator,
                  candidates, eval_step=1):
    timer, hit_rate, auc = d2l.Timer(), 0, 0
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['test hit rate', 'test AUC'])
    
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            input_data = []
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))
            
            with autograd.record():
                # Predict scores for positive and negative items
                p_pos = [net(*t) for t in zip(*input_data[:-1])]
                p_neg = [net(*t) for t in zip(*input_data[:-2], input_data[-1])]
                ls = [loss(p, n) for p, n in zip(p_pos, p_neg)]
            
            [l.backward(retain_graph=False) for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean()/len(devices)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        
        # Evaluate periodically
        with autograd.predict_mode():
            if (epoch + 1) % eval_step == 0:
                hit_rate, auc = evaluator(net, test_iter, test_seq_iter,
                                          candidates, num_users, num_items,
                                          devices)
                animator.add(epoch + 1, (hit_rate, auc))
    
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test hit rate {float(hit_rate):.3f}, test AUC {float(auc):.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
```

## 6. Prepare the Data

We use the MovieLens 100K dataset, treating ratings as implicit feedback (1 if rated, 0 otherwise). We split the data in "seq-aware" mode, holding out each user's latest interaction for testing.

```python
batch_size = 1024
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items, 'seq-aware')

users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")

# Create the training DataLoader with negative sampling
train_iter = gluon.data.DataLoader(
    PRDataset(users_train, items_train, candidates, num_items), batch_size,
    True, last_batch="rollover", num_workers=d2l.get_dataloader_workers())
```

## 7. Initialize and Train the Model

Create the NeuMF model with a three-layer MLP (each layer with 10 hidden units). We'll use the Adam optimizer and BPR Loss.

```python
# Set up devices (GPUs if available)
devices = d2l.try_all_gpus()

# Initialize the model
net = NeuMF(10, num_users, num_items, nums_hiddens=[10, 10, 10])
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))

# Configure training
lr, num_epochs, wd, optimizer = 0.01, 10, 1e-5, 'adam'
loss = d2l.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})

# Start training
train_ranking(net, train_iter, test_iter, loss, trainer, None, num_users,
              num_items, num_epochs, devices, evaluate_ranking, candidates)
```

## Summary

In this tutorial, you implemented the NeuMF model for personalized ranking with implicit feedback. Key takeaways:

*   NeuMF combines **Generalized Matrix Factorization** and a **Multi-Layer Perceptron** to capture both linear and non-linear user-item interactions.
*   **Pairwise ranking loss** (BPR) is used to train the model, ensuring positive items rank higher than negative samples.
*   The model is evaluated using **Hit Rate @ k** and **AUC**, standard metrics for ranking tasks.

## Exercises

To deepen your understanding, try the following modifications:

1.  **Latent Factor Size:** Experiment with different sizes for the user/item embeddings (e.g., 5, 20, 50). How does this affect model performance?
2.  **MLP Architecture:** Change the number of layers or neurons in the MLP (e.g., `[20, 10]` or `[10, 10, 10, 10]`). Observe the impact on training time and accuracy.
3.  **Optimization:** Test different optimizers (SGD, RMSProp), learning rates, or weight decay values.
4.  **Loss Function:** Replace the BPR loss with the hinge loss from the previous section and compare the results.

---
*For further discussion, visit the [D2L.ai forum](https://discuss.d2l.ai/t/403).*