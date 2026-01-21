# Natural Language Inference and the Dataset

In this guide, you will learn about Natural Language Inference (NLI) and how to work with the Stanford Natural Language Inference (SNLI) dataset. NLI is a fundamental task for understanding relationships between pairs of sentences, which is crucial for applications like information retrieval and question answering.

## What is Natural Language Inference?

Natural Language Inference determines whether a **hypothesis** can logically be inferred from a **premise**. The relationship between a premise-hypothesis pair falls into one of three categories:

*   **Entailment**: The hypothesis can be inferred from the premise.
*   **Contradiction**: The negation of the hypothesis can be inferred from the premise.
*   **Neutral**: Neither entailment nor contradiction holds.

For example:
*   **Premise**: Two women are hugging each other.
    **Hypothesis**: Two women are showing affection.
    **Label**: Entailment
*   **Premise**: A man is running the coding example from Dive into Deep Learning.
    **Hypothesis**: The man is sleeping.
    **Label**: Contradiction
*   **Premise**: The musicians are performing for us.
    **Hypothesis**: The musicians are famous.
    **Label**: Neutral

## Setting Up the Environment

Before we begin, ensure you have the necessary libraries installed. This tutorial provides code for both **MXNet/Gluon** and **PyTorch**. Choose the framework you are using.

```python
# For MXNet/Gluon
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re
npx.set_np()

# For PyTorch
from d2l import torch as d2l
import torch
from torch import nn
import os
import re
```

## Step 1: Downloading the SNLI Dataset

The Stanford Natural Language Inference (SNLI) Corpus contains over 500,000 labeled English sentence pairs. We'll start by downloading and extracting it.

First, we register the dataset URL with the `d2l` library's data hub and define the download location.

```python
# This cell is common to both frameworks
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

## Step 2: Reading and Parsing the Dataset

The raw dataset file contains more information than we need. We will write a function to extract just the premises, hypotheses, and their corresponding labels, while cleaning the text.

```python
def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove parentheses and extra whitespace
        s = re.sub('\(', '', s)
        s = re.sub('\)', '', s)
        s = re.sub('\s{2,}', ' ', s)
        return s.strip()

    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt' if is_train else 'snli_1.0_test.txt')

    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]

    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]

    return premises, hypotheses, labels
```

Let's test this function by loading the training data and inspecting the first three examples.

```python
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

**Expected Output:**
```
premise: A person on a horse jumps over a broken down airplane.
hypothesis: A person is training his horse for a competition.
label: 2
premise: A person on a horse jumps over a broken down airplane.
hypothesis: A person is at a diner, ordering an omelette.
label: 1
premise: A person on a horse jumps over a broken down airplane.
hypothesis: A person is outdoors, on a horse.
label: 0
```

Now, let's check the label distribution to ensure the dataset is balanced.

```python
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

**Expected Output (approximate):**
```
[183416, 183187, 182764]
[3368, 3237, 3219]
```

## Step 3: Creating a Custom Dataset Class

To efficiently load and preprocess the data for training, we need to create a custom Dataset class. This class will handle tokenization, vocabulary building, and padding/truncation of sequences to a fixed length (`num_steps`).

### For MXNet/Gluon:

```python
class SNLIDataset(gluon.data.Dataset):
    """A customized dataset to load the SNLI dataset for MXNet."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])

        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab

        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

### For PyTorch:

```python
class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset for PyTorch."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])

        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab

        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

## Step 4: Creating Data Loaders

Finally, we write a function to tie everything together. It downloads the data, creates the `SNLIDataset` objects for training and testing, and returns the corresponding DataLoader objects along with the vocabulary. **Crucially, the test set uses the vocabulary built from the training set** to ensure no unseen tokens appear during evaluation.

### For MXNet/Gluon:

```python
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary for MXNet."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

### For PyTorch:

```python
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary for PyTorch."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

## Step 5: Putting It All to Work

Let's load the data with a batch size of 128 and a sequence length of 50 tokens. We'll then inspect the vocabulary size and the shape of a batch.

```python
train_iter, test_iter, vocab = load_data_snli(128, 50)
print(f'Vocabulary size: {len(vocab)}')
```

**Expected Output (approximate):**
```
read 549367 examples
read 9824 examples
Vocabulary size: 18678
```

Now, let's examine the first batch from the training iterator. Notice that unlike single-sequence tasks, we have two inputs: `X[0]` for premises and `X[1]` for hypotheses.

```python
for X, Y in train_iter:
    print(f'Premises batch shape: {X[0].shape}')
    print(f'Hypotheses batch shape: {X[1].shape}')
    print(f'Labels batch shape: {Y.shape}')
    break
```

**Expected Output:**
```
Premises batch shape: torch.Size([128, 50])  # or (128, 50) for MXNet
Hypotheses batch shape: torch.Size([128, 50])
Labels batch shape: torch.Size([128])
```

## Summary

In this tutorial, you learned:
*   The concept of **Natural Language Inference (NLI)** and its three relationship types: entailment, contradiction, and neutral.
*   How to download and preprocess the **Stanford Natural Language Inference (SNLI)** dataset.
*   How to build a custom Dataset class to handle tokenization, vocabulary creation, and sequence padding for sentence pairs.
*   How to create DataLoader objects for efficient batch processing during model training and evaluation.

You are now ready to use this prepared data to train models for the NLI task.

## Exercises

1.  Machine translation has long been evaluated based on superficial $n$-gram matching between an output translation and a ground-truth translation. Can you design a measure for evaluating machine translation results by using natural language inference?
2.  How can we change hyperparameters (like `min_freq` in the `Vocab` constructor) to reduce the vocabulary size?