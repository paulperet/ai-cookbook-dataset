# Sentiment Analysis with RNNs and Pretrained Word Vectors

This guide walks you through building a sentiment analysis model using a Recurrent Neural Network (RNN) and pretrained GloVe word vectors. You'll learn how to represent text sequences with a bidirectional RNN and classify movie reviews from the IMDb dataset as positive or negative.

## Prerequisites

This tutorial uses either **MXNet** or **PyTorch**. Ensure you have the necessary libraries installed.

```bash
# For MXNet
pip install mxnet d2l

# For PyTorch
pip install torch d2l
```

## Step 1: Import Libraries and Load Data

First, import the required libraries and load the IMDb dataset. The `d2l` library provides a convenient function to load the data, which is already tokenized and split into training and testing sets.

```python
# For MXNet
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

# For PyTorch
from d2l import torch as d2l
import torch
from torch import nn

# Load the IMDb dataset
batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

## Step 2: Define the Bidirectional RNN Model

We'll create a `BiRNN` class that uses an embedding layer to convert tokens into pretrained GloVe vectors, processes the sequence with a bidirectional LSTM, and finally uses a dense layer for classification.

The key steps in the model are:
1. **Embedding Layer:** Converts token indices to dense vectors.
2. **Bidirectional LSTM:** Encodes the sequence from both directions.
3. **Concatenation:** The hidden states from the first and last time steps are concatenated to form a fixed-length representation of the entire sequence.
4. **Decoder (Fully Connected Layer):** Maps the concatenated representation to two output classes (positive/negative).

```python
# For MXNet
class BiRNN(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.T)
        outputs = self.encoder(embeddings)
        encoding = np.concatenate((outputs[0], outputs[-1]), axis=1)
        outs = self.decoder(encoding)
        return outs

# For PyTorch
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                               bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs
```

## Step 3: Instantiate the Model

Now, create an instance of the `BiRNN` model. We'll use an embedding size of 100 (to match the GloVe vectors), 100 hidden units, 2 layers, and attempt to use all available GPUs.

```python
embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)

# Initialize weights
# For MXNet
net.initialize(init.Xavier(), ctx=devices)

# For PyTorch
def init_weights(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.LSTM:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])
net.apply(init_weights)
```

## Step 4: Load Pretrained GloVe Word Vectors

To leverage pretrained knowledge and reduce overfitting, we'll load GloVe embeddings and set them as the weights of our embedding layer. These weights will be frozen (not updated during training).

```python
# Load the 100-dimensional GloVe embeddings
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')

# Retrieve vectors for tokens in our vocabulary
embeds = glove_embedding[vocab.idx_to_token]
print(f'Embedding shape: {embeds.shape}')

# Set the embedding layer weights and freeze them
# For MXNet
net.embedding.weight.set_data(embeds)
net.embedding.collect_params().setattr('grad_req', 'null')

# For PyTorch
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False
```

## Step 5: Train the Model

With the model and data ready, we can now train the bidirectional RNN for sentiment analysis. We'll use the Adam optimizer and cross-entropy loss.

```python
lr, num_epochs = 0.01, 5

# For MXNet
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()

# For PyTorch
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")

# Train the model
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## Step 6: Make Predictions

After training, let's create a helper function to predict the sentiment of new text sequences.

```python
# For MXNet
def predict_sentiment(net, vocab, sequence):
    sequence = np.array(vocab[sequence.split()], ctx=d2l.try_gpu())
    label = np.argmax(net(sequence.reshape(1, -1)), axis=1)
    return 'positive' if label == 1 else 'negative'

# For PyTorch
def predict_sentiment(net, vocab, sequence):
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
```

Now, test the model with a couple of example sentences.

```python
print(predict_sentiment(net, vocab, 'this movie is so great'))
# Output: positive

print(predict_sentiment(net, vocab, 'this movie is so bad'))
# Output: negative
```

## Summary

In this tutorial, you built a sentiment analysis model using a bidirectional RNN and pretrained GloVe word vectors. Key takeaways include:

* **Pretrained word vectors** provide rich representations for individual tokens, improving model performance, especially with limited training data.
* **Bidirectional RNNs** effectively capture context from both directions in a text sequence. By concatenating the hidden states from the initial and final time steps, we create a robust fixed-length representation of the entire sequence.
* This single text representation is then passed through a fully connected layer to predict sentiment categories.

## Exercises

To deepen your understanding, try the following modifications:

1. **Increase the number of epochs** or tune other hyperparameters (like learning rate, hidden size, or number of layers). Can you improve the training and testing accuracy?
2. **Use larger pretrained word vectors**, such as 300-dimensional GloVe embeddings. Does this increase classification accuracy?
3. **Experiment with different tokenization.** Install spaCy (`pip install spacy`) and the English model (`python -m spacy download en`). Replace the default tokenizer with spaCy's tokenizer. Note that token formats may differ (e.g., "new-york" in GloVe vs. "new york" in spaCy). Does this affect performance?

```python
# Example spaCy tokenizer setup
import spacy
spacy_en = spacy.load('en')
def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]
# Use this tokenizer when loading data or processing new sequences
```

For further discussion and community support, visit the [D2L discussion forum](https://discuss.d2l.ai/).