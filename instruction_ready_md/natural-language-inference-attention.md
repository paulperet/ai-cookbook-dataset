# Natural Language Inference with Attention Mechanisms

## Overview

This guide implements a decomposable attention model for Natural Language Inference (NLI), as introduced by Parikh et al. (2016). This model achieves strong performance on the SNLI dataset without using recurrent or convolutional layers, relying instead on attention mechanisms and MLPs.

## Prerequisites

First, ensure you have the necessary libraries installed. We'll use either MXNet or PyTorch.

```bash
# Install required packages (if using pip)
# For MXNet: pip install mxnet d2l
# For PyTorch: pip install torch d2l
```

## 1. Model Architecture

The decomposable attention model consists of three sequential steps:
1. **Attending**: Soft alignment between tokens in premise and hypothesis sequences
2. **Comparing**: Comparison between aligned token representations
3. **Aggregating**: Classification based on aggregated comparisons

### 1.1 Setup and Imports

Let's start by importing the necessary modules.

```python
# MXNet version
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```python
# PyTorch version
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

### 1.2 MLP Helper Function

We'll define a multi-layer perceptron (MLP) function that will be used in multiple components of our model.

```python
# MXNet version
def mlp(num_hiddens, flatten):
    net = nn.Sequential()
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    return net
```

```python
# PyTorch version
def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)
```

### 1.3 Attending Step

The attending step computes soft alignments between tokens in the premise and hypothesis sequences. This is done efficiently using a decomposition trick that reduces complexity from quadratic to linear.

```python
# MXNet version
class Attend(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B):
        f_A = self.f(A)
        f_B = self.f(B)
        e = npx.batch_dot(f_A, f_B, transpose_b=True)
        beta = npx.batch_dot(npx.softmax(e), B)
        alpha = npx.batch_dot(npx.softmax(e.transpose(0, 2, 1)), A)
        return beta, alpha
```

```python
# PyTorch version
class Attend(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        f_A = self.f(A)
        f_B = self.f(B)
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha
```

### 1.4 Comparing Step

After obtaining aligned representations, we compare each token with its aligned counterpart from the other sequence.

```python
# MXNet version
class Compare(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(np.concatenate([A, beta], axis=2))
        V_B = self.g(np.concatenate([B, alpha], axis=2))
        return V_A, V_B
```

```python
# PyTorch version
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B
```

### 1.5 Aggregating Step

Finally, we aggregate the comparison vectors and make a prediction about the logical relationship.

```python
# MXNet version
class Aggregate(nn.Block):
    def __init__(self, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_hiddens=num_hiddens, flatten=True)
        self.h.add(nn.Dense(num_outputs))

    def forward(self, V_A, V_B):
        V_A = V_A.sum(axis=1)
        V_B = V_B.sum(axis=1)
        Y_hat = self.h(np.concatenate([V_A, V_B], axis=1))
        return Y_hat
```

```python
# PyTorch version
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat
```

### 1.6 Complete Model

Now let's assemble the complete decomposable attention model by combining all three components.

```python
# MXNet version
class DecomposableAttention(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_hiddens)
        self.compare = Compare(num_hiddens)
        self.aggregate = Aggregate(num_hiddens, 3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

```python
# PyTorch version
class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

## 2. Training the Model

### 2.1 Loading the Dataset

We'll use the SNLI dataset for training and evaluation.

```python
batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = d2l.load_data_snli(batch_size, num_steps)
```

### 2.2 Initializing the Model

We initialize the model with pretrained GloVe embeddings for better performance.

```python
# MXNet version
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
net.initialize(init.Xavier(), ctx=devices)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
```

```python
# PyTorch version
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds)
```

### 2.3 Training Loop

Now we train the model using the Adam optimizer and cross-entropy loss.

```python
# MXNet version
lr, num_epochs = 0.001, 4
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# Helper function for multi-input batching
def split_batch_multi_inputs(X, y, devices):
    X = list(zip(*[gluon.utils.split_and_load(
        feature, devices, even_split=False) for feature in X]))
    return (X, gluon.utils.split_and_load(y, devices, even_split=False))

d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               split_batch_multi_inputs)
```

```python
# PyTorch version
lr, num_epochs = 0.001, 4
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## 3. Making Predictions

After training, we can use the model to predict logical relationships between sentence pairs.

```python
# MXNet version
def predict_snli(net, vocab, premise, hypothesis):
    premise = np.array(vocab[premise], ctx=d2l.try_gpu())
    hypothesis = np.array(vocab[hypothesis], ctx=d2l.try_gpu())
    label = np.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), axis=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 else 'neutral'
```

```python
# PyTorch version
def predict_snli(net, vocab, premise, hypothesis):
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 else 'neutral'
```

Let's test our model with an example:

```python
predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'])
```

The output should indicate a "contradiction" relationship between these two sentences.

## 4. Summary

In this tutorial, we implemented a decomposable attention model for natural language inference that:

1. Uses attention mechanisms to create soft alignments between premise and hypothesis tokens
2. Compares aligned token representations using MLPs
3. Aggregates comparison results to predict logical relationships
4. Achieves linear computational complexity through a decomposition trick
5. Leverages pretrained word embeddings for better performance

## 5. Exercises

1. Experiment with different hyperparameter combinations (learning rate, hidden sizes, dropout rates) to improve test accuracy.
2. Consider the limitations of this model architecture. What types of linguistic phenomena might it struggle with?
3. How would you modify this approach for semantic similarity scoring (continuous values between 0 and 1) instead of three-way classification?

## Further Reading

- Parikh, A. P., Täckström, O., Das, D., & Uszkoreit, J. (2016). A decomposable attention model for natural language inference. *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*.
- SNLI dataset: https://nlp.stanford.edu/projects/snli/
- GloVe embeddings: https://nlp.stanford.edu/projects/glove/