# Building a Sequence-Aware Recommender with Caser

This guide walks you through implementing Caser (Convolutional Sequence Embedding Recommendation Model), a neural network architecture designed for sequential recommendation. Unlike traditional matrix factorization, Caser models users' short-term behavioral patterns by processing their interaction history as a sequence.

## Prerequisites

Ensure you have the necessary libraries installed. This implementation uses MXNet and the D2L library.

```bash
pip install mxnet d2l
```

## 1. Understanding the Caser Model

Caser treats a user's last `L` interactions as an `L x k` embedding matrix (where `k` is the embedding dimension). This matrix is processed by two parallel convolutional networks:

1.  **Vertical Convolutional Network:** Operates across the embedding dimension to capture point-level patterns (influence of individual items).
2.  **Horizontal Convolutional Network:** Operates across the sequence length to capture union-level patterns (influence of combinations of items).

The outputs are combined with a user's general (long-term) embedding to predict the next item.

## 2. Model Implementation

We'll start by defining the Caser model class. It includes embedding layers for users and items, the two convolutional networks, and a final prediction layer.

```python
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import random

npx.set_np()

class Caser(nn.Block):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16,
                 d_prime=4, drop_ratio=0.05, **kwargs):
        super(Caser, self).__init__(**kwargs)
        # Embedding layers
        self.P = nn.Embedding(num_users, num_factors)  # User general taste
        self.Q = nn.Embedding(num_items, num_factors)  # Item embeddings
        self.d_prime, self.d = d_prime, d

        # Vertical convolution layer
        self.conv_v = nn.Conv2D(d_prime, (L, 1), in_channels=1)

        # Horizontal convolution layers
        h = [i + 1 for i in range(L)]
        self.conv_h, self.max_pool = nn.Sequential(), nn.Sequential()
        for i in h:
            self.conv_h.add(nn.Conv2D(d, (i, num_factors), in_channels=1))
            self.max_pool.add(nn.MaxPool1D(L - i + 1))

        # Fully connected layer to combine convolutional outputs
        self.fc1_dim_v, self.fc1_dim_h = d_prime * num_factors, d * len(h)
        self.fc = nn.Dense(in_units=d_prime * num_factors + d * L,
                           activation='relu', units=num_factors)

        # Final prediction layers
        self.Q_prime = nn.Embedding(num_items, num_factors * 2)
        self.b = nn.Embedding(num_items, 1)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, user_id, seq, item_id):
        # seq: (batch_size, L) -> item_embs: (batch_size, 1, L, num_factors)
        item_embs = np.expand_dims(self.Q(seq), 1)
        user_emb = self.P(user_id)

        out_v, out_h = None, None
        out_hs = []

        # Vertical convolution
        if self.d_prime:
            out_v = self.conv_v(item_embs)
            out_v = out_v.reshape(out_v.shape[0], self.fc1_dim_v)

        # Horizontal convolution
        if self.d:
            for conv, maxp in zip(self.conv_h, self.max_pool):
                conv_out = np.squeeze(npx.relu(conv(item_embs)), axis=3)
                t = maxp(conv_out)
                pool_out = np.squeeze(t, axis=2)
                out_hs.append(pool_out)
            out_h = np.concatenate(out_hs, axis=1)

        # Combine features and add user general taste
        out = np.concatenate([out_v, out_h], axis=1)
        z = self.fc(self.dropout(out))  # Short-term intent representation
        x = np.concatenate([z, user_emb], axis=1)

        # Final prediction
        q_prime_i = np.squeeze(self.Q_prime(item_id))
        b = np.squeeze(self.b(item_id))
        res = (x * q_prime_i).sum(1) + b
        return res
```

## 3. Creating a Sequential Dataset

Standard recommendation datasets aren't formatted for sequence modeling. We need a custom Dataset class that creates training samples where each sample contains:
*   A user ID
*   A sequence of their last `L` interactions
*   The next item they interacted with (the target)
*   A negative sample item

```python
class SeqDataset(gluon.data.Dataset):
    def __init__(self, user_ids, item_ids, L, num_users, num_items, candidates):
        user_ids, item_ids = np.array(user_ids), np.array(item_ids)
        # Sort by user ID to group interactions per user
        sort_idx = np.array(sorted(range(len(user_ids)),
                                   key=lambda k: user_ids[k]))
        u_ids, i_ids = user_ids[sort_idx], item_ids[sort_idx]

        temp, u_ids, self.cand = {}, u_ids.asnumpy(), candidates
        self.all_items = set([i for i in range(num_items)])

        # Group item indices by user
        [temp.setdefault(u_ids[i], []).append(i) for i, _ in enumerate(u_ids)]
        temp = sorted(temp.items(), key=lambda x: x[0])
        u_ids = np.array([i[0] for i in temp])
        idx = np.array([i[1][0] for i in temp])

        # Calculate number of sequences
        self.ns = ns = int(sum([c - L if c >= L + 1 else 1 for c
                                in np.array([len(i[1]) for i in temp])]))

        # Initialize storage arrays
        self.seq_items = np.zeros((ns, L))
        self.seq_users = np.zeros(ns, dtype='int32')
        self.seq_tgt = np.zeros((ns, 1))
        self.test_seq = np.zeros((num_users, L))

        test_users, _uid = np.empty(num_users), None
        for i, (uid, i_seq) in enumerate(self._seq(u_ids, i_ids, idx, L + 1)):
            if uid != _uid:
                self.test_seq[uid][:] = i_seq[-L:]  # Store last L items for testing
                test_users[uid], _uid = uid, uid
            self.seq_tgt[i][:] = i_seq[-1:]         # Target item
            self.seq_items[i][:], self.seq_users[i] = i_seq[:L], uid

    def _win(self, tensor, window_size, step_size=1):
        """Generate sliding windows over a tensor."""
        if len(tensor) - window_size >= 0:
            for i in range(len(tensor), 0, -step_size):
                if i - window_size >= 0:
                    yield tensor[i - window_size:i]
                else:
                    break
        else:
            yield tensor

    def _seq(self, u_ids, i_ids, idx, max_len):
        """Generate sequences for each user."""
        for i in range(len(idx)):
            stop_idx = None if i >= len(idx) - 1 else int(idx[i + 1])
            for s in self._win(i_ids[int(idx[i]):stop_idx], max_len):
                yield (int(u_ids[i]), s)

    def __len__(self):
        return self.ns

    def __getitem__(self, idx):
        """Return a training sample with a negative item."""
        neg = list(self.all_items - set(self.cand[int(self.seq_users[idx])]))
        i = random.randint(0, len(neg) - 1)
        return (self.seq_users[idx], self.seq_items[idx], self.seq_tgt[idx], neg[i])
```

## 4. Loading and Preparing the Data

Now, let's load the MovieLens 100K dataset and prepare it for sequence-aware training.

```python
# Set hyperparameters
TARGET_NUM, L, batch_size = 1, 5, 4096

# Load and split data
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items, 'seq-aware')

# Load implicit feedback data
users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")

# Create sequential dataset
train_seq_data = SeqDataset(users_train, items_train, L, num_users, num_items, candidates)
train_iter = gluon.data.DataLoader(train_seq_data, batch_size, True,
                                   last_batch="rollover",
                                   num_workers=d2l.get_dataloader_workers())
test_seq_iter = train_seq_data.test_seq

# Inspect a sample
print(train_seq_data[0])
```

The output shows the structure of a training sample:
```
(array(0, dtype=int32), array([50., 52., 53., 54., 55.]), array([56.]), 123)
```
This represents:
- User ID: 0
- Last 5 items: [50, 52, 53, 54, 55]
- Target item: 56
- Negative item: 123

## 5. Training the Model

With our data prepared, we can now train the Caser model. We'll use the BPR (Bayesian Personalized Ranking) loss, which is effective for implicit feedback data.

```python
# Initialize model and training components
devices = d2l.try_all_gpus()
net = Caser(10, num_users, num_items, L)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))

# Training hyperparameters
lr, num_epochs, wd, optimizer = 0.04, 8, 1e-5, 'adam'
loss = d2l.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})

# Note: Full training may take significant time
# Uncomment to run training:
# d2l.train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter,
#                   num_users, num_items, num_epochs, devices,
#                   d2l.evaluate_ranking, candidates, eval_step=1)
```

## Summary

In this tutorial, you've implemented a sequence-aware recommender system using the Caser architecture:

1.  **Model Architecture:** Built a neural network with parallel vertical and horizontal convolutional layers to capture both point-level and union-level patterns in user interaction sequences.
2.  **Data Preparation:** Created a custom dataset class that transforms chronological interaction data into sequential training samples with negative sampling.
3.  **Training Pipeline:** Set up the complete training workflow using BPR loss for implicit feedback recommendation.

Key insights from Caser:
*   Modeling both short-term (sequential) and long-term (general) user interests improves recommendation accuracy
*   Convolutional networks can effectively capture patterns in interaction sequences
*   Sequence-aware models are particularly valuable for domains where user preferences evolve over time

## Exercises

To deepen your understanding, try these experiments:

1.  **Ablation Study:** Remove either the horizontal or vertical convolutional network. Which component contributes more to the model's performance?
2.  **Sequence Length:** Experiment with different values of `L` (the sequence length). Does considering longer historical interactions consistently improve accuracy?
3.  **Comparison:** Research session-based recommendation models. How do they differ from the sequence-aware approach implemented here?

For further discussion and community support, visit the [D2L discussion forum](https://discuss.d2l.ai/t/404).