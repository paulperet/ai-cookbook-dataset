# Preparing Data for Word2Vec: A Step-by-Step Guide

This guide walks you through preparing a text dataset for training a Word2Vec model using the skip-gram architecture with negative sampling. We'll use the Penn Tree Bank (PTB) corpus and implement key preprocessing steps like subsampling and minibatch creation.

## Prerequisites and Setup

First, ensure you have the necessary libraries installed. This guide provides code for both PyTorch and MXNet frameworks.

```bash
# Install the d2l library which contains utilities for this tutorial
pip install d2l
```

Now, import the required modules.

```python
import collections
import math
import os
import random

# For PyTorch
import torch
from d2l import torch as d2l

# For MXNet (uncomment if using MXNet)
# from mxnet import gluon, np
# from d2l import mxnet as d2l
```

## Step 1: Reading and Tokenizing the Dataset

We'll use the Penn Tree Bank (PTB) dataset, which consists of sentences from Wall Street Journal articles.

```python
# Helper function to download and load the PTB dataset
def read_ptb():
    """Load the PTB dataset into a list of tokenized sentences."""
    # Download and extract the dataset
    data_dir = d2l.download_extract('ptb')
    # Read the training set
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    # Split the text into sentences and then into word tokens
    return [line.split() for line in raw_text.split('\n')]

# Load the dataset
sentences = read_ptb()
print(f'Number of sentences: {len(sentences)}')
```

## Step 2: Building the Vocabulary

Next, we create a vocabulary from the corpus. Words appearing less than 10 times are replaced with a special `<unk>` (unknown) token.

```python
# Build vocabulary, filtering out rare words
vocab = d2l.Vocab(sentences, min_freq=10)
print(f'Vocabulary size: {len(vocab)}')
```

## Step 3: Subsampling High-Frequency Words

Common words like "the" or "a" appear too frequently and provide less meaningful signal for training. We subsample them probabilistically to speed up training and improve model quality.

The probability of keeping a word \( w_i \) is given by:
\[ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right) \]
where \( f(w_i) \) is the word's relative frequency and \( t \) is a hyperparameter (set to \(10^{-4}\)).

```python
def subsample(sentences, vocab):
    """Subsample high-frequency words from the tokenized sentences."""
    # Remove unknown tokens from the sentences
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    # Count token frequencies
    counter = collections.Counter([
        token for line in sentences for token in line])
    num_tokens = sum(counter.values())

    # Define the keep probability for each token
    def keep(token):
        return (random.uniform(0, 1) <
                math.sqrt(1e-4 / counter[token] * num_tokens))

    # Apply subsampling
    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

# Apply subsampling
subsampled, counter = subsample(sentences, vocab)
```

Let's verify the effect of subsampling by comparing counts of a frequent and an infrequent word.

```python
def compare_counts(token):
    before = sum([line.count(token) for line in sentences])
    after = sum([line.count(token) for line in subsampled])
    return f'Count of "{token}": before={before}, after={after}'

print(compare_counts('the'))   # High-frequency word
print(compare_counts('join'))  # Low-frequency word
```

## Step 4: Converting Tokens to Indices

Now we convert the subsampled sentences into sequences of vocabulary indices for efficient processing.

```python
corpus = [vocab[line] for line in subsampled]
print('First three lines of the corpus (indices):')
print(corpus[:3])
```

## Step 5: Extracting Center and Context Words

For the skip-gram model, we need to create (center word, context word) pairs. We define a function that, for each center word, samples a random context window size and collects the surrounding words as context.

```python
def get_centers_and_contexts(corpus, max_window_size):
    """Extract center words and their context words from the corpus."""
    centers, contexts = [], []
    for line in corpus:
        # Need at least 2 words to form a pair
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):
            # Randomly sample a context window size
            window_size = random.randint(1, max_window_size)
            # Determine the start and end indices for the context window
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Remove the center word index
            indices.remove(i)
            # Append the context words
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

Let's test this function on a small dataset.

```python
# Create a tiny dataset for demonstration
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('Dataset:', tiny_dataset)
centers, contexts = get_centers_and_contexts(tiny_dataset, 2)
for center, context in zip(centers, contexts):
    print(f'Center {center} has contexts {context}')
```

Now, apply it to the full PTB corpus with a maximum window size of 5.

```python
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
num_pairs = sum([len(contexts) for contexts in all_contexts])
print(f'Number of center-context pairs: {num_pairs}')
```

## Step 6: Generating Negative Samples

For negative sampling, we need to draw noise words that are *not* in the context of a given center word. We'll sample according to a distribution where each word's probability is proportional to its frequency raised to the power of 0.75.

First, we implement a helper class for efficient random sampling.

```python
class RandomGenerator:
    """Draw random samples according to given sampling weights."""
    def __init__(self, sampling_weights):
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Cache k random samples for efficiency
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

Now, generate negative samples for each context.

```python
def get_negatives(all_contexts, vocab, counter, K):
    """Generate K noise words for each context."""
    # Calculate sampling weights (index 0 is the <unk> token)
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Ensure the noise word is not actually a context word
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

# Generate 5 negative samples per context word
all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## Step 7: Batching Examples for Training

We need to package our data into minibatches for efficient training. Since context windows vary in size, we pad sequences to a uniform length and create mask variables to ignore padding during loss calculation.

```python
def batchify(data):
    """
    Convert a list of examples into a minibatch.
    Each example is a tuple: (center, context_words, negative_words)
    """
    # Find the maximum combined length of context and negative words
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers.append(center)
        # Concatenate context and negative words, then pad with zeros
        contexts_negatives.append(
            context + negative + [0] * (max_len - cur_len))
        # Create mask: 1 for real words, 0 for padding
        masks.append([1] * cur_len + [0] * (max_len - cur_len))
        # Create labels: 1 for context words, 0 for negative words
        labels.append([1] * len(context) + [0] * (max_len - len(context)))
    
    # Convert to tensors
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)),
            d2l.tensor(contexts_negatives),
            d2l.tensor(masks),
            d2l.tensor(labels))
```

Test the batching function with a simple example.

```python
# Create two example data points
x_1 = (1, [2, 2], [3, 3, 3, 3])  # Center 1, two context words, four negative words
x_2 = (1, [2, 2, 2], [3, 3])     # Center 1, three context words, two negative words

batch = batchify((x_1, x_2))
names = ['centers', 'contexts_negatives', 'masks', 'labels']

for name, data in zip(names, batch):
    print(f'{name} = {data}')
```

## Step 8: Creating the Data Loader

Finally, we combine all steps into a single function that returns a data iterator and the vocabulary.

### For PyTorch Users:

```python
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Load the PTB dataset and return a DataLoader and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    # Define a custom Dataset class
    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    
    # Create the DataLoader
    data_iter = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True,
        collate_fn=batchify, num_workers=num_workers)
    
    return data_iter, vocab
```

### For MXNet Users:

```python
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Load the PTB dataset and return a DataLoader and vocabulary."""
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    
    dataset = gluon.data.ArrayDataset(all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(
        dataset, batch_size, shuffle=True, batchify_fn=batchify,
        num_workers=d2l.get_dataloader_workers())
    
    return data_iter, vocab
```

Now, let's create a data loader and inspect the first batch.

```python
data_iter, vocab = load_data_ptb(batch_size=512, max_window_size=5, num_noise_words=5)

# Examine the first batch
for batch in data_iter:
    for name, data in zip(['centers', 'contexts_negatives', 'masks', 'labels'], batch):
        print(f'{name} shape: {data.shape}')
    break
```

## Summary

In this guide, you've learned how to prepare text data for Word2Vec training:

1. **Subsampling high-frequency words** to speed up training and improve embedding quality.
2. **Extracting center-context pairs** with variable window sizes for the skip-gram model.
3. **Generating negative samples** using a frequency-based distribution.
4. **Batching variable-length sequences** with proper masking and labeling for efficient training.

The prepared data is now ready for training a Word2Vec model using negative sampling.

## Exercises

1. Experiment with disabling subsampling. How does it affect the running time and memory usage?
2. Adjust the cache size `k` in the `RandomGenerator` class. What value provides the best balance between memory and speed?
3. Identify other hyperparameters (like `max_window_size` or `num_noise_words`) that might impact data loading performance.

For further discussion, visit the [D2L AI forum](https://discuss.d2l.ai/).