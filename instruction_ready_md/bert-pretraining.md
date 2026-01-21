# Pretraining BERT: A Step-by-Step Guide

This guide walks you through the process of pretraining a BERT model on the WikiText-2 dataset. We'll cover loading the data, defining a smaller BERT model for demonstration, implementing the training loop with its dual loss functions, and finally using the pretrained model to generate contextual text representations.

## Prerequisites & Setup

First, ensure you have the necessary libraries installed and imported. This guide provides code for both MXNet and PyTorch.

```python
# For MXNet
# !pip install d2l # Uncomment if needed
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
npx.set_np()
```

```python
# For PyTorch
# !pip install d2l torch # Uncomment if needed
from d2l import torch as d2l
import torch
from torch import nn
```

## Step 1: Load and Prepare the Dataset

We start by loading the WikiText-2 dataset, which has been preprocessed into minibatches suitable for BERT's pretraining tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).

We set a batch size of 512 and a maximum sequence length of 64 tokens. (Note: The original BERT uses a length of 512, but we use a smaller value for faster demonstration).

```python
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
```

## Step 2: Define the BERT Model

The original BERT comes in Base (110M parameters) and Large (340M parameters) variants. For efficiency in this tutorial, we define a much smaller model.

Our model will have:
* 2 Transformer encoder blocks (layers)
* 128 hidden units
* 2 self-attention heads
* A feed-forward network with 256 hidden units

```python
# MXNet
net = d2l.BERTModel(len(vocab), num_hiddens=128, ffn_num_hiddens=256,
                    num_heads=2, num_blks=2, dropout=0.2)
devices = d2l.try_all_gpus()
net.initialize(init.Xavier(), ctx=devices)
loss = gluon.loss.SoftmaxCELoss()
```

```python
# PyTorch
net = d2l.BERTModel(len(vocab), num_hiddens=128,
                    ffn_num_hiddens=256, num_heads=2, num_blks=2, dropout=0.2)
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()
```

## Step 3: Implement the Training Loss Function

BERT is trained by minimizing the sum of two losses: the Masked Language Modeling (MLM) loss and the Next Sentence Prediction (NSP) loss. We define a helper function `_get_batch_loss_bert` to compute both.

**What it does:** For a given batch of data, it performs a forward pass through the network, calculates the individual MLM and NSP losses, and returns their sum.

```python
# MXNet version
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X_shards,
                         segments_X_shards, valid_lens_x_shards,
                         pred_positions_X_shards, mlm_weights_X_shards,
                         mlm_Y_shards, nsp_y_shards):
    mlm_ls, nsp_ls, ls = [], [], []
    for (tokens_X_shard, segments_X_shard, valid_lens_x_shard,
         pred_positions_X_shard, mlm_weights_X_shard, mlm_Y_shard,
         nsp_y_shard) in zip(
        tokens_X_shards, segments_X_shards, valid_lens_x_shards,
        pred_positions_X_shards, mlm_weights_X_shards, mlm_Y_shards,
        nsp_y_shards):
        # Forward pass
        _, mlm_Y_hat, nsp_Y_hat = net(
            tokens_X_shard, segments_X_shard, valid_lens_x_shard.reshape(-1),
            pred_positions_X_shard)
        # Compute masked language model loss
        mlm_l = loss(
            mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y_shard.reshape(-1),
            mlm_weights_X_shard.reshape((-1, 1)))
        mlm_l = mlm_l.sum() / (mlm_weights_X_shard.sum() + 1e-8)
        # Compute next sentence prediction loss
        nsp_l = loss(nsp_Y_hat, nsp_y_shard)
        nsp_l = nsp_l.mean()
        mlm_ls.append(mlm_l)
        nsp_ls.append(nsp_l)
        ls.append(mlm_l + nsp_l)
        npx.waitall()
    return mlm_ls, nsp_ls, ls
```

```python
# PyTorch version
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # Compute masked language model loss
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Compute next sentence prediction loss
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
```

## Step 4: Define the Training Loop

Now we define the main training function `train_bert`. Instead of training for a set number of epochs, this function trains for a specified number of iteration steps (`num_steps`), which is more practical for the lengthy BERT pretraining process.

The function handles:
* Optimizer setup (Adam)
* Loss accumulation and logging
* Progress visualization

```python
# MXNet training function
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': 0.01})
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for batch in train_iter:
            (tokens_X_shards, segments_X_shards, valid_lens_x_shards,
             pred_positions_X_shards, mlm_weights_X_shards,
             mlm_Y_shards, nsp_y_shards) = [gluon.utils.split_and_load(
                elem, devices, even_split=False) for elem in batch]
            timer.start()
            with autograd.record():
                mlm_ls, nsp_ls, ls = _get_batch_loss_bert(
                    net, loss, vocab_size, tokens_X_shards, segments_X_shards,
                    valid_lens_x_shards, pred_positions_X_shards,
                    mlm_weights_X_shards, mlm_Y_shards, nsp_y_shards)
            for l in ls:
                l.backward()
            trainer.step(1)
            mlm_l_mean = sum([float(l) for l in mlm_ls]) / len(mlm_ls)
            nsp_l_mean = sum([float(l) for l in nsp_ls]) / len(nsp_ls)
            metric.add(mlm_l_mean, nsp_l_mean, batch[0].shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

```python
# PyTorch training function
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net(*next(iter(train_iter))[:4])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

## Step 5: Pretrain the Model

Let's start the pretraining process for 50 steps to see the loss curves. In practice, BERT requires millions of steps for full pretraining.

```python
train_bert(train_iter, net, loss, len(vocab), devices, 50)
```

Expected output (values will vary):
```
MLM loss 7.692, NSP loss 0.721
XXXX.X sentence pairs/sec on [gpu(0), gpu(1)]
```

## Step 6: Generate Text Representations with the Pretrained BERT

Once pretrained, we can use BERT to generate contextual embeddings for text. The function `get_bert_encoding` returns the BERT representations for all tokens in the input.

```python
# MXNet
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = np.expand_dims(np.array(vocab[tokens], ctx=devices[0]),
                               axis=0)
    segments = np.expand_dims(np.array(segments, ctx=devices[0]), axis=0)
    valid_len = np.expand_dims(np.array(len(tokens), ctx=devices[0]), axis=0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

```python
# PyTorch
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

### Example 1: Single Sentence

Let's get the representation for the sentence "a crane is flying". The special `[CLS]` token's embedding (index 0) often serves as a pooled representation of the entire input.

```python
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# Tokens: '[CLS]', 'a', 'crane', 'is', 'flying', '[SEP]'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
print('Shape of encoded text:', encoded_text.shape)
print('Shape of [CLS] token embedding:', encoded_text_cls.shape)
print('First 3 elements of "crane" embedding:', encoded_text_crane[0][:3])
```

### Example 2: Sentence Pair

Now let's see how the representation of the word "crane" changes when its context changes, demonstrating BERT's context-sensitive nature.

```python
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# Tokens: '[CLS]', 'a', 'crane', 'driver', 'came', '[SEP]', 'he', 'just', 'left', '[SEP]'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
print('Shape of encoded pair:', encoded_pair.shape)
print('Shape of [CLS] token embedding:', encoded_pair_cls.shape)
print('First 3 elements of "crane" embedding in new context:', encoded_pair_crane[0][:3])
```

You should observe that the first three elements of the "crane" embedding differ between the two examples, confirming that BERT generates context-dependent representations.

## Summary

In this guide, you have:
1. Loaded and prepared the WikiText-2 dataset for BERT pretraining.
2. Defined a smaller BERT model architecture suitable for demonstration.
3. Implemented the combined Masked Language Modeling and Next Sentence Prediction loss function.
4. Pretrained the BERT model for a limited number of steps.
5. Used the pretrained model to generate contextual embeddings for text, observing how word representations change with context.

This pretrained model can now be fine-tuned for downstream NLP tasks like sentiment analysis, question answering, or named entity recognition.

## Exercises

1.  **Loss Discrepancy:** In the output, you likely noticed the MLM loss is much higher than the NSP loss. Why might this be? (Hint: Consider the complexity and nature of each task).
2.  **Model Scaling:** Try changing the maximum sequence length to 512 (the original BERT's length) and use the configurations for `BERT_LARGE` (24 layers, 1024 hidden units, 16 heads). Do you encounter any errors? What might be the cause?