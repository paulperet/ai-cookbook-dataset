# Sentiment Analysis with the IMDb Dataset: A Practical Guide

This guide walks you through the process of loading and preprocessing the IMDb movie review dataset for sentiment analysis. Sentiment analysis classifies text, like reviews, into categories such as "positive" or "negative." We'll treat this as a text classification task, transforming variable-length reviews into fixed-length sequences for model training.

## Prerequisites

First, ensure you have the necessary libraries installed. This guide provides code for both **MXNet/Gluon** and **PyTorch** frameworks.

```bash
# Install the d2l library which contains utility functions used in this guide.
# The exact command may vary. Typically, you would use:
# pip install d2l
```

Now, import the required modules for your chosen framework.

```python
# For MXNet/Gluon
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```python
# For PyTorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

## Step 1: Download and Extract the Dataset

We'll use the Stanford Large Movie Review Dataset (IMDb). The following helper function downloads and extracts it to a local directory.

```python
# Register the dataset URL with the d2l library's data hub.
d2l.DATA_HUB['aclImdb'] = (
    d2l.DATA_URL + 'aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903'
)

# Download and extract the dataset. It will be placed in the '../data/aclImdb' directory.
data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

## Step 2: Read the Dataset into Memory

We need a function to read the raw text files from the `train` and `test` directories. Each review is paired with a label: `1` for positive and `0` for negative.

```python
def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels."""
    data, labels = [], []
    # Iterate through 'pos' and 'neg' subdirectories
    for label in ('pos', 'neg'):
        # Construct the path to the correct folder (train or test)
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        # Read each text file in the folder
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

# Read the training data
train_data = read_imdb(data_dir, is_train=True)
print(f'Number of training reviews: {len(train_data[0])}')

# Let's inspect the first three examples
for review, label in zip(train_data[0][:3], train_data[1][:3]):
    print(f'Label: {label}, Review (first 60 chars): {review[:60]}')
```

**Output:**
```
Number of training reviews: 25000
Label: 1, Review (first 60 chars): ...
Label: 0, Review (first 60 chars): ...
Label: 1, Review (first 60 chars): ...
```

## Step 3: Preprocess Text and Build Vocabulary

To prepare text for a model, we tokenize it (split into words) and create a vocabulary. We filter out infrequent words to keep the vocabulary manageable.

```python
# Tokenize all training reviews (split into word-level tokens)
train_tokens = d2l.tokenize(train_data[0], token='word')

# Build a vocabulary. Words appearing less than 5 times are ignored.
# The '<pad>' token is added for sequence padding.
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
print(f'Vocabulary size: {len(vocab)}')
```

### Analyze Review Lengths

It's useful to understand the distribution of review lengths before we standardize them.

```python
import matplotlib.pyplot as plt

# Plot a histogram of token counts per review
d2l.set_figsize()
plt.xlabel('# tokens per review')
plt.ylabel('count')
plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50))
plt.show()
```

The histogram will show that reviews vary significantly in length. To batch them efficiently, we need to standardize their length.

## Step 4: Standardize Sequence Length

We truncate long reviews and pad short ones to a fixed length of 500 tokens. This is a common preprocessing step for sequence models.

```python
num_steps = 500  # Our chosen fixed sequence length

# Convert each tokenized review into a tensor of length `num_steps`.
# Indices are looked up in the vocabulary, then truncated/padded.
train_features = d2l.tensor([
    d2l.truncate_pad(vocab[line], num_steps, vocab['<pad>']) for line in train_tokens
])
print(f'Shape of training features tensor: {train_features.shape}')
# Output: torch.Size([25000, 500]) or similar
```

## Step 5: Create Data Iterators

Data iterators efficiently yield mini-batches of data during training. We'll create one for the training set.

```python
batch_size = 64

# For MXNet
train_iter = d2l.load_array((train_features, train_data[1]), batch_size)

# For PyTorch (note the explicit tensor conversion for labels)
train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), batch_size)

# Inspect one batch
for X, y in train_iter:
    print(f'Batch feature shape (X): {X.shape}, Batch label shape (y): {y.shape}')
    break
print(f'Total number of batches: {len(train_iter)}')
```

## Step 6: Integrate Everything into a Utility Function

For reusability, we combine all steps into a single function that returns data iterators and the vocabulary.

```python
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    # 1. Download and extract data
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    # 2. Read raw data
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    # 3. Tokenize
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    # 4. Build vocabulary from training data only
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    # 5. Truncate and pad sequences
    # For MXNet, use `np.array`. For PyTorch, use `torch.tensor`.
    train_features = d2l.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = d2l.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    # 6. Create iterators
    # Framework-specific label tensor handling is required here.
    train_iter = d2l.load_array((train_features, d2l.tensor(train_data[1])), batch_size)
    test_iter = d2l.load_array((test_features, d2l.tensor(test_data[1])), batch_size, is_train=False)
    return train_iter, test_iter, vocab

# Example usage
train_iter, test_iter, vocab = load_data_imdb(batch_size=64)
```

## Summary

In this guide, you have:
1. **Downloaded the IMDb sentiment analysis dataset.**
2. **Read and inspected the raw text and labels.**
3. **Preprocessed the text by tokenizing and building a vocabulary.**
4. **Standardized review lengths by truncating and padding.**
5. **Created efficient data iterators for mini-batch training.**
6. **Encapsulated the entire pipeline in a reusable function.**

The dataset is now ready to be fed into a neural network model for sentiment classification. Key hyperparameters you can experiment with include the sequence length (`num_steps`), batch size, and vocabulary minimum frequency (`min_freq`).

## Exercises

1.  **Hyperparameter Tuning:** Which hyperparameters in this pipeline (e.g., `num_steps`, `min_freq`, `batch_size`) most significantly impact training speed and model performance? How would you modify them?
2.  **Extend to New Data:** Try adapting the `load_data_imdb` function to load and preprocess the [Amazon Reviews dataset](https://snap.stanford.edu/data/web-Amazon.html). What additional steps might be necessary?