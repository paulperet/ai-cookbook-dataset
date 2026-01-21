# Pretraining Word2Vec with Skip-Gram and Negative Sampling

This guide walks you through implementing and training a Word2Vec model using the Skip-Gram architecture with Negative Sampling. You'll learn to create the model, define the loss, train it on the PTB dataset, and then use the learned word embeddings to find semantically similar words.

## Prerequisites & Setup

First, ensure you have the necessary libraries installed. This tutorial provides code for both **MXNet** and **PyTorch**. Choose your preferred framework.

### Install Dependencies

If you haven't already, install the `d2l` library which contains the dataset loader and utilities.

```bash
pip install d2l
```

### Import Libraries

Depending on your chosen framework, import the required modules.

```python
# For MXNet
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```python
# For PyTorch
from d2l import torch as d2l
import math
import torch
from torch import nn
```

## Step 1: Load and Prepare the Dataset

We'll use the Penn Treebank (PTB) dataset, which is a standard corpus for training word embeddings. The `d2l.load_data_ptb` function handles batching and creates pairs of center words with their context and noise words for negative sampling.

Define your hyperparameters and load the data iterator and vocabulary.

```python
batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

*   **`batch_size`**: Number of examples per training batch.
*   **`max_window_size`**: Maximum context window size for sampling context words.
*   **`num_noise_words`**: Number of noise words to sample per center word for negative sampling.

## Step 2: Understand the Embedding Layer

The core of the Skip-Gram model is the embedding layer, which maps a token's index to a dense vector representation. The layer's weight matrix has dimensions `(vocabulary_size, embedding_dimension)`.

Let's create a small embedding layer to see how it works.

```python
# MXNet
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
print(embed.weight)
```

```python
# PyTorch
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, dtype={embed.weight.dtype})')
```

You can pass a tensor of token indices to the layer to get their vector representations.

```python
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
print(embed(x))
```

The output will have shape `(2, 3, 4)`—a batch of 2 sequences, each with 3 tokens, each represented by a 4-dimensional vector.

## Step 3: Implement the Skip-Gram Model Forward Pass

The forward pass of the Skip-Gram model takes:
1.  `center`: Indices of the center words (shape: `[batch_size, 1]`).
2.  `contexts_and_negatives`: Concatenated indices of context words and sampled noise words (shape: `[batch_size, max_len]`).

It uses two separate embedding layers: one for center words (`embed_v`) and one for context/negative words (`embed_u`). The model computes the dot product between each center word vector and all its associated context/negative word vectors via a batch matrix multiplication.

```python
# MXNet Implementation
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)  # Shape: (batch_size, 1, embed_size)
    u = embed_u(contexts_and_negatives)  # Shape: (batch_size, max_len, embed_size)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))  # Dot product
    return pred
```

```python
# PyTorch Implementation
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)  # Shape: (batch_size, 1, embed_size)
    u = embed_u(contexts_and_negatives)  # Shape: (batch_size, max_len, embed_size)
    pred = torch.bmm(v, u.permute(0, 2, 1))  # Batch matrix multiplication
    return pred
```

Let's verify the output shape with dummy data.

```python
# MXNet
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

```python
# PyTorch
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

The output shape should be `(2, 1, 4)`, confirming a prediction score for each center-context pair in the batch.

## Step 4: Define the Loss Function

We use binary cross-entropy loss for negative sampling. The target labels are 1 for real context words and 0 for noise (negative) words. A mask is used to account for variable sequence lengths due to padding.

```python
# MXNet: Use built-in SigmoidBCELoss
loss = gluon.loss.SigmoidBCELoss()
```

```python
# PyTorch: Define a custom loss with masking support
class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

To understand how the loss works with masking, consider this example:

```python
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])  # 1 for valid, 0 for padded
print(loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1))
```

The loss is normalized only over valid (non-padded) positions.

## Step 5: Initialize the Model

We create a model with two embedding layers: one for center words and one for context words. The embedding dimension is a key hyperparameter.

```python
embed_size = 100

# MXNet
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))

# PyTorch
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

## Step 6: Train the Model

The training loop iterates over the dataset, computes the loss using our `skip_gram` function, and updates the model parameters. It handles the masking logic to correctly normalize the loss.

```python
# MXNet Training Loop
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs])
    metric = d2l.Accumulator(2)  # Sum of losses, count of losses
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```python
# PyTorch Training Loop
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(module):
        if type(module) == nn.Embedding:
            nn.init.xavier_uniform_(module.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs])
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [data.to(device) for data in batch]
            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                 / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

Now, start the training process.

```python
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

You should see the loss decrease over epochs, and the final training speed in tokens per second.

## Step 7: Use the Learned Word Embeddings

After training, the embedding layers contain the learned word vectors. We can find words semantically similar to a query word by computing the cosine similarity between its vector and all other word vectors.

```python
# MXNet
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Skip the query word itself
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')
```

```python
# PyTorch
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) * torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')
```

Test it with a sample word.

```python
get_similar_tokens('chip', 3, net[0])
```

The output will list the top 3 most similar words to "chip" (e.g., 'intel', 'chips', 'processor') along with their cosine similarity scores.

## Summary

In this tutorial, you have:
1.  Loaded and prepared the PTB dataset for Word2Vec training.
2.  Implemented the Skip-Gram model forward pass using embedding layers.
3.  Defined a binary cross-entropy loss function compatible with negative sampling and padding masks.
4.  Initialized and trained the model using an efficient training loop.
5.  Applied the learned embeddings to find semantically similar words via cosine similarity.

This foundational knowledge enables you to train custom word embeddings for various NLP tasks.

## Exercises

1.  Experiment with different hyperparameters (e.g., `embed_size`, `lr`, `num_epochs`). How do they affect the quality of the similar words found?
2.  For very large corpora, consider dynamically sampling context and noise words for each center word in every epoch, rather than pre-sampling. This can improve model robustness and convergence. Can you modify the training loop to implement this?