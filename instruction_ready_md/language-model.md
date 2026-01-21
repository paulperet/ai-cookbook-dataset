# A Guide to Language Models

This guide introduces the core concepts of language models, from the basic probability theory to practical data handling for training neural models. You will learn how language models estimate the likelihood of text sequences and how to prepare data to train them.

## Prerequisites

This tutorial uses a custom utility class `d2l.TimeMachine`. Ensure you have the necessary deep learning framework installed.

```python
# Installation and imports will depend on your chosen framework (PyTorch, TensorFlow, JAX, or MXNet).
# The following is a generic import statement.
from d2l import torch as d2l  # Change 'torch' to 'tensorflow', 'jax', or 'mxnet' as needed.
import torch  # Or tf, jax.numpy, mxnet.np
```

## 1. Understanding Language Models

A language model's primary goal is to estimate the joint probability of a sequence of tokens (e.g., words or characters). For a sequence of length *T* with tokens *x₁, x₂, ..., xₜ*, this is:

$$P(x_1, x_2, \ldots, x_T)$$

Using the chain rule of probability, this decomposes into a product of conditional probabilities:

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1})$$

For example:
`P("deep", "learning", "is", "fun") = P("deep") * P("learning" | "deep") * P("is" | "deep", "learning") * P("fun" | "deep", "learning", "is")`

### 1.1 N-gram Models and Their Limitations

Directly modeling long dependencies is complex. A common simplification is the **Markov assumption**, which states that the probability of the next token depends only on the last *n-1* tokens. This leads to **n-gram models**:
*   **Unigram:** `P(x₁, x₂, x₃, x₄) ≈ P(x₁)P(x₂)P(x₃)P(x₄)`
*   **Bigram:** `P(x₁, x₂, x₃, x₄) ≈ P(x₁)P(x₂|x₁)P(x₃|x₂)P(x₄|x₃)`
*   **Trigram:** `P(x₁, x₂, x₃, x₄) ≈ P(x₁)P(x₂|x₁)P(x₃|x₁,x₂)P(x₄|x₂,x₃)`

Parameters for these models are estimated from word frequencies in a large corpus. However, n-gram models face significant issues:
1.  **Sparsity:** Many plausible word combinations (especially for n>2) may never appear in the training data, leading to zero probability estimates.
2.  **Storage:** They require storing counts for all observed n-grams, which becomes massive.
3.  **Lack of Generalization:** They fail to capture semantic similarity (e.g., "cat" and "feline").

Techniques like **Laplace Smoothing** (adding a small constant to all counts) can alleviate the zero-probability problem but don't solve the core limitations. This is why modern approaches use **neural networks**, which can generalize better and handle long-range dependencies.

## 2. Evaluating Language Models: Perplexity

We need a metric to evaluate and compare language models. **Perplexity** is the standard measure. It is derived from the average cross-entropy loss per token over a sequence:

$$\text{Perplexity} = \exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right)$$

**How to interpret perplexity:**
*   **Lower is better.** It represents the "average branching factor" or the number of equally likely choices the model believes it has for the next token.
*   **Best Case:** A perfect predictor has a perplexity of 1.
*   **Worst Case:** A model that always assigns zero probability to the correct token has infinite perplexity.
*   **Uniform Baseline:** A model that predicts a uniform distribution over a vocabulary of size `V` has a perplexity of `V`. Any useful model must perform better than this.

## 3. Preparing Data for Neural Language Models

To train neural language models effectively, we need to structure our text data into input-target pairs for minibatch training.

### 3.1 The Core Idea: Next-Token Prediction

We frame language modeling as a **next-token prediction** task. Given an input sequence of `n` tokens, the target is the same sequence shifted by one token.

**Example:** For an input sequence `["It", "is", "raining"]`, the target sequence would be `["is", "raining", "outside"]`.

### 3.2 Implementing Sequence Partitioning

We implement this using a data utility class. The key steps are:
1.  **Tokenize the corpus** and convert it to a sequence of indices.
2.  **Partition** the long sequence into consecutive, overlapping subsequences of length `num_steps + 1` (the extra token is for the target).
3.  **Split** each subsequence into an input (`X`) and a target (`Y`), where `Y` is `X` shifted by one position.

Let's examine the code that accomplishes this within a dataset class.

```python
@d2l.add_to_class(d2l.TimeMachine)
def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
    super(d2l.TimeMachine, self).__init__()
    self.save_hyperparameters()
    # corpus: list of token indices, vocab: token-to-index mapping
    corpus, self.vocab = self.build(self._download())
    
    # Create all consecutive (overlapping) subsequences of length num_steps+1
    array = d2l.tensor([corpus[i:i+num_steps+1]
                        for i in range(len(corpus)-num_steps)])
    # Input X is all but the last token, target Y is all but the first token
    self.X, self.Y = array[:,:-1], array[:,1:]
```

### 3.3 Creating the Data Loader

Next, we need a method to create data loaders that yield random minibatches for training and validation.

```python
@d2l.add_to_class(d2l.TimeMachine)
def get_dataloader(self, train):
    # Define indices for training or validation split
    idx = slice(0, self.num_train) if train else slice(
        self.num_train, self.num_train + self.num_val)
    # Return a loader for the (X, Y) tensors within the specified indices
    return self.get_tensorloader([self.X, self.Y], train, idx)
```

### 3.4 Inspecting the Data

Let's instantiate the dataset and inspect a single minibatch to see the input-target structure.

```python
# Create a dataset loader with batch_size=2 and sequence length (num_steps)=10
data = d2l.TimeMachine(batch_size=2, num_steps=10)

# Get the training data loader and fetch the first batch
for X, Y in data.train_dataloader():
    print('Input (X) shape:', X.shape)
    print('Target (Y) shape:', Y.shape)
    print('\nFirst input sequence (indices):', X[0])
    print('Corresponding target sequence :', Y[0])
    break
```

**Expected Output:**
```
Input (X) shape: torch.Size([2, 10])
Target (Y) shape: torch.Size([2, 10])

First input sequence (indices): tensor([ 8,  2, 11,  ... , 14])
Corresponding target sequence : tensor([ 2, 11,  5,  ... ,  4])
```
You can see that `Y` is indeed `X` shifted one position to the left. The model's task is to predict the token at position `t+1` given all tokens up to position `t`.

## Summary

In this guide, you learned:
*   The fundamental probability objective of a language model.
*   The limitations of traditional n-gram models and the motivation for neural approaches.
*   How **perplexity** is used to evaluate language model quality.
*   The practical steps to structure text data into input-target pairs for training neural language models via next-token prediction.

This data preparation pipeline is the foundation for training the neural language models (like RNNs and Transformers) you will encounter in subsequent tutorials.