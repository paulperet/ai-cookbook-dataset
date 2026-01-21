# Text Sequence Preprocessing: From Raw Text to Numerical Indices

In this guide, you will learn the fundamental preprocessing pipeline for converting raw text into numerical sequences suitable for machine learning models. We'll work with H. G. Wells' *The Time Machine* to demonstrate each step: loading text, tokenization, vocabulary building, and sequence conversion.

## Prerequisites

First, ensure you have the necessary libraries installed. This guide supports multiple frameworks (MXNet, PyTorch, TensorFlow, JAX). The `d2l` library provides common utilities.

```bash
# Install the d2l library for deep learning utilities
pip install d2l
```

Now, import the required modules for your chosen framework.

```python
import collections
import re
import random

# Framework-specific imports
# Choose one of the following blocks based on your preference:

# For MXNet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

# For PyTorch
# from d2l import torch as d2l
# import torch

# For TensorFlow
# from d2l import tensorflow as d2l
# import tensorflow as tf

# For JAX
# from d2l import jax as d2l
# import jax
# from jax import numpy as jnp
```

## Step 1: Load and Inspect the Raw Text

We begin by downloading and reading the raw text of *The Time Machine*. The following class handles data loading.

```python
class TimeMachine(d2l.DataModule):
    """The Time Machine dataset."""
    def _download(self):
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root,
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

# Instantiate the dataset and load raw text
data = TimeMachine()
raw_text = data._download()
print(raw_text[:60])  # Display the first 60 characters
```

**Output:**
```
The Time Machine, by H. G. Wells [1898]
```

## Step 2: Preprocess the Text

Real-world text often contains punctuation and mixed capitalization. For simplicity, we'll remove non-alphabetic characters and convert everything to lowercase.

```python
@d2l.add_to_class(TimeMachine)
def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

text = data._preprocess(raw_text)
print(text[:60])
```

**Output:**
```
the time machine by h g wells
```

## Step 3: Tokenize the Text

Tokens are the atomic units of text. You can tokenize by characters, words, or subwords. Here, we'll split the text into a list of individual characters.

```python
@d2l.add_to_class(TimeMachine)
def _tokenize(self, text):
    return list(text)

tokens = data._tokenize(text)
print(','.join(tokens[:30]))
```

**Output:**
```
t,h,e, ,t,i,m,e, ,m,a,c,h,i,n,e, ,b,y, ,h, ,g, ,w,e,l,l,s
```

## Step 4: Build a Vocabulary

To convert tokens into numerical inputs, we need a vocabulary that maps each unique token to an index. The `Vocab` class handles this, including special tokens for unknown values.

```python
class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']
```

Now, construct a vocabulary from our character tokens and convert a sample to indices.

```python
vocab = Vocab(tokens)
indices = vocab[tokens[:10]]
print('indices:', indices)
print('words:', vocab.to_tokens(indices))
```

**Output:**
```
indices: [20, 7, 4, 1, 20, 8, 12, 4, 1, 12]
words: ['t', 'h', 'e', ' ', 't', 'i', 'm', 'e', ' ', 'm']
```

## Step 5: Package the Pipeline

Let's consolidate the preprocessing steps into a single `build` method that returns the tokenized corpus and vocabulary.

```python
@d2l.add_to_class(TimeMachine)
def build(self, raw_text, vocab=None):
    tokens = self._tokenize(self._preprocess(raw_text))
    if vocab is None:
        vocab = Vocab(tokens)
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab

corpus, vocab = data.build(raw_text)
print(len(corpus), len(vocab))
```

**Output:**
```
170580 28
```

Our corpus contains 170,580 character tokens, and the vocabulary size is 28 (26 letters plus space and the unknown token).

## Step 6: Analyze Word Frequency Statistics

While we tokenized by character, it's instructive to examine word-level statistics. Let's build a word vocabulary and inspect the most frequent terms.

```python
words = text.split()
vocab = Vocab(words)
print(vocab.token_freqs[:10])
```

**Output:**
```
[('the', 2261), ('i', 1267), ('and', 1245), ('of', 1155), ('a', 816), ('to', 695), ('was', 552), ('in', 541), ('that', 443), ('my', 440)]
```

The most frequent words are often stop words (e.g., "the", "and", "of"), which are common but not very descriptive. Word frequency typically follows a power-law distribution (Zipf's law). Let's visualize the frequency distribution.

```python
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

The log-log plot shows a roughly linear relationship, confirming Zipf's law: the frequency of a word is inversely proportional to its rank.

## Step 7: Extend to N-grams

We can also analyze sequences of words (bigrams, trigrams). Let's examine their frequency distributions.

```python
# Bigrams (pairs of consecutive words)
bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
bigram_vocab = Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])

# Trigrams (triples of consecutive words)
trigram_tokens = ['--'.join(triple) for triple in zip(
    words[:-2], words[1:-1], words[2:])]
trigram_vocab = Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])
```

**Output:**
```
[('of--the', 309), ('in--the', 169), ('i--had', 130), ('i--was', 112), ('and--the', 109), ('the--time', 102), ('it--was', 99), ('to--the', 85), ('as--i', 78), ('of--a', 73)]
[('the--time--traveller', 59), ('the--time--machine', 30), ('the--medical--man', 24), ('it--seemed--to', 16), ('it--was--a', 15), ('here--and--there', 15), ('seemed--to--me', 14), ('i--did--not', 14), ('i--saw--that', 13), ('i--began--to', 13)]
```

Now, compare the frequency distributions of unigrams, bigrams, and trigrams.

```python
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

The plot shows that all n-gram distributions follow Zipf's law, with steeper decay for longer n-grams. This highlights the rich structure in language and the sparsity of longer sequences, motivating the use of deep learning models for language tasks.

## Summary

In this tutorial, you've implemented a complete text preprocessing pipeline:

1. **Loaded raw text** from a source.
2. **Preprocessed** by removing punctuation and lowercasing.
3. **Tokenized** the text into characters (or words).
4. **Built a vocabulary** to map tokens to numerical indices.
5. **Converted text** into a sequence of indices for model input.
6. **Analyzed frequency statistics**, observing Zipf's law for unigrams and n-grams.

This pipeline is foundational for any NLP task, from language modeling to text classification. The same principles apply whether you're working with characters, words, or subwords, and regardless of the dataset size.

## Exercises

1. Experiment with the `min_freq` parameter in the `Vocab` constructor. How does increasing `min_freq` affect vocabulary size and which tokens are retained?
2. Estimate the exponent of the Zipfian distribution for unigrams, bigrams, and trigrams in this corpus by fitting a line to the log-log frequency plot.
3. Apply this preprocessing pipeline to another text corpus (e.g., a different book or dataset). Compare vocabulary sizes and Zipfian exponents with those from *The Time Machine*.