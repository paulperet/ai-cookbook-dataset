# Sentiment Analysis with Convolutional Neural Networks (textCNN)

In this guide, you will implement a **textCNN** model for sentiment analysis. This approach treats text sequences as one-dimensional images, allowing you to use one-dimensional convolutional neural networks (CNNs) to capture local patterns like n-grams. You'll learn how to build, train, and evaluate this efficient architecture.

## Prerequisites

First, ensure you have the necessary libraries installed and import them. This tutorial provides code for both **MXNet** and **PyTorch**. Choose your preferred framework.

### MXNet Setup

```python
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

### PyTorch Setup

```python
from d2l import torch as d2l
import torch
from torch import nn
```

## 1. Load the Dataset

You'll use the IMDB movie review dataset for binary sentiment classification (positive/negative). The `d2l` library provides a convenient loader.

```python
batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

## 2. Understanding One-Dimensional Convolutions

A 1D convolution slides a kernel across a sequence, computing element-wise multiplications and summing them to produce an output sequence. This operation is fundamental for processing text.

### 2.1 Implement a Basic 1D Cross-Correlation

Let's start by implementing a simple 1D cross-correlation function.

```python
def corr1d(X, K):
    w = K.shape[0]
    Y = d2l.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
```

Test the function with a simple example:

```python
X, K = d2l.tensor([0, 1, 2, 3, 4, 5, 6]), d2l.tensor([1, 2])
corr1d(X, K)
```

**Output:**
```
[2., 5., 8., 11., 14., 17.]
```

### 2.2 Handle Multiple Input Channels

Text representations often have multiple channels (e.g., different embedding dimensions). For multi-channel input, you sum the cross-correlation results across all channels.

```python
def corr1d_multi_in(X, K):
    # Iterate through the channel dimension and sum the results
    return sum(corr1d(x, k) for x, k in zip(X, K))
```

Validate this with a 3-channel example:

```python
X = d2l.tensor([[0, 1, 2, 3, 4, 5, 6],
                [1, 2, 3, 4, 5, 6, 7],
                [2, 3, 4, 5, 6, 7, 8]])
K = d2l.tensor([[1, 2], [3, 4], [-1, -3]])
corr1d_multi_in(X, K)
```

**Output:**
```
[2., 8., 14., 20., 26., 32.]
```

**Key Insight:** A multi-input-channel 1D convolution is equivalent to a single-input-channel 2D convolution where the kernel height matches the input tensor height.

## 3. Max-Over-Time Pooling

To extract the most significant feature from each channel, you use **max-over-time pooling**. This operation takes the maximum value across the entire sequence for each channel, effectively capturing the strongest signal regardless of sequence length.

## 4. Build the textCNN Model

The textCNN model architecture is as follows:
1.  **Input:** A sequence of tokens, each represented by a `d`-dimensional vector.
2.  **Convolutional Layers:** Apply multiple 1D convolutional kernels with different widths to capture n-grams of various sizes.
3.  **Pooling:** Perform max-over-time pooling on each convolutional output channel.
4.  **Classification:** Concatenate the pooled features and pass them through a fully connected layer for sentiment prediction.

### 4.1 Define the Model Class

Here is the implementation for both frameworks.

#### MXNet Version

```python
class TextCNN(nn.Block):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # A second, non-trainable embedding layer
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        self.pool = nn.GlobalMaxPool1D() # Shared pooling layer
        # Create multiple 1D convolutional layers
        self.convs = nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # Concatenate both embedding outputs
        embeddings = np.concatenate((self.embedding(inputs), self.constant_embedding(inputs)), axis=2)
        # Rearrange dimensions for Conv1D: (batch, channels, length)
        embeddings = embeddings.transpose(0, 2, 1)
        # Apply each conv layer, pool, and concatenate results
        encoding = np.concatenate([
            np.squeeze(self.pool(conv(embeddings)), axis=-1)
            for conv in self.convs], axis=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

#### PyTorch Version

```python
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        self.pool = nn.AdaptiveAvgPool1d(1) # Adaptive pooling to get size 1
        self.relu = nn.ReLU()
        # Create multiple 1D convolutional layers
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(2 * embed_size, c, k))

    def forward(self, inputs):
        # Concatenate both embedding outputs
        embeddings = torch.cat((self.embedding(inputs), self.constant_embedding(inputs)), dim=2)
        # Rearrange dimensions for Conv1d: (batch, channels, length)
        embeddings = embeddings.permute(0, 2, 1)
        # Apply each conv layer, pool, and concatenate results
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```

### 4.2 Instantiate the Model

Create an instance with three convolutional layers having kernel widths of 3, 4, and 5, each with 100 output channels.

```python
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
```

#### MXNet Initialization

```python
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=devices)
```

#### PyTorch Initialization

```python
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)

def init_weights(module):
    if type(module) in (nn.Linear, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight)

net.apply(init_weights)
```

## 5. Load Pretrained Word Embeddings

Initialize the model with pretrained GloVe embeddings to provide a strong starting point. One embedding layer will be fine-tuned during training, while the other remains fixed.

```python
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
```

#### MXNet Weight Assignment

```python
net.embedding.weight.set_data(embeds)
net.constant_embedding.weight.set_data(embeds)
net.constant_embedding.collect_params().setattr('grad_req', 'null')
```

#### PyTorch Weight Assignment

```python
net.embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.data.copy_(embeds)
net.constant_embedding.weight.requires_grad = False
```

## 6. Train the Model

Now you can train the textCNN model. You'll use the Adam optimizer and cross-entropy loss.

```python
lr, num_epochs = 0.001, 5
```

#### MXNet Training Setup

```python
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

#### PyTorch Training Setup

```python
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## 7. Make Predictions

Finally, use your trained model to predict sentiment on new sentences.

```python
print(d2l.predict_sentiment(net, vocab, 'this movie is so great'))
print(d2l.predict_sentiment(net, vocab, 'this movie is so bad'))
```

**Expected Output:**
```
'positive'
'negative'
```

## Summary

In this tutorial, you successfully built a textCNN model for sentiment analysis. You learned that:

*   **1D CNNs** can effectively capture local text features like n-grams.
*   **Multi-channel 1D convolutions** are equivalent to specific 2D convolutions.
*   **Max-over-time pooling** extracts the most salient feature from each channel, accommodating variable-length sequences.
*   The **textCNN architecture** combines these layers to transform token embeddings into powerful sequence representations for classification.

## Next Steps & Exercises

To deepen your understanding, try the following:
1.  **Hyperparameter Tuning:** Compare the accuracy and efficiency of this textCNN model against the RNN model from the previous chapter.
2.  **Improve Accuracy:** Apply techniques like further regularization or different pooling strategies to boost performance.
3.  **Add Positional Encoding:** Experiment with adding positional information to the input embeddings. Does it improve results?

For further discussion, see the MXNet [forum](https://discuss.d2l.ai/t/393) or PyTorch [forum](https://discuss.d2l.ai/t/1425).