# Approximate Training for Word Embeddings

## Introduction

In word embedding models like skip-gram and CBOW, the standard softmax operation requires calculating probabilities over the entire vocabulary. When vocabularies contain hundreds of thousands or millions of words, this becomes computationally expensive. This guide introduces two approximate training methods that dramatically reduce computational cost: **negative sampling** and **hierarchical softmax**.

## Prerequisites

This tutorial assumes familiarity with:
- Word2Vec models (skip-gram and CBOW)
- Basic probability concepts
- Vector operations and sigmoid functions

## Negative Sampling

Negative sampling modifies the training objective to avoid computing probabilities over the entire vocabulary. Instead, it focuses on distinguishing actual context words from randomly sampled "noise" words.

### Mathematical Foundation

Given a center word $w_c$ and a context word $w_o$, we model the probability that $w_o$ appears in $w_c$'s context window as:

$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c)$$

where $\sigma$ is the sigmoid function:

$$\sigma(x) = \frac{1}{1+\exp(-x)}$$

### The Negative Sampling Objective

For a text sequence of length $T$ with context window size $m$, we want to maximize:

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)})$$

where the conditional probability is approximated as:

$$ P(w^{(t+j)} \mid w^{(t)}) = P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k)$$

Here, $K$ noise words $w_k$ are sampled from a predefined distribution $P(w)$.

### Loss Function

The logarithmic loss becomes:

$$
\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$

**Key Insight:** The computational cost now depends linearly on $K$ (typically 5-20) rather than the vocabulary size, making training much faster.

## Hierarchical Softmax

Hierarchical softmax uses a binary tree structure where each leaf node represents a word in the vocabulary. Instead of computing probabilities over all words, we compute probabilities along paths in the tree.

### Tree Structure

Consider a binary tree where:
- Each leaf node represents a vocabulary word
- $L(w)$ is the number of nodes on the path from root to word $w$
- $n(w,j)$ is the $j$-th node on this path
- $\mathbf{u}_{n(w, j)}$ is the vector associated with node $n(w,j)$

### Probability Calculation

The conditional probability is approximated as:

$$P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \textrm{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right)$$

where $[\![x]\!] = 1$ if $x$ is true, otherwise $[\![x]\!] = -1$.

### Example Calculation

For a word $w_3$ reached by going left, right, then left from the root:

$$P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c)$$

**Key Insight:** Since $L(w)-1 = \mathcal{O}(\log_2|\mathcal{V}|)$, the computational cost grows logarithmically with vocabulary size rather than linearly.

### Normalization Property

Because $\sigma(x)+\sigma(-x) = 1$, hierarchical softmax maintains the normalization property:

$$\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1$$

## Implementation Considerations

### Choosing Between Methods

1. **Negative Sampling**:
   - Simpler to implement
   - Works well with large vocabularies
   - Requires careful sampling distribution $P(w)$ (typically $P(w) \propto \text{freq}(w)^{3/4}$)

2. **Hierarchical Softmax**:
   - More complex to implement (requires building and maintaining a tree)
   - Better for frequent words
   - No sampling noise

### Practical Tips

- For negative sampling, typical values are $K = 5$ for small datasets and $K = 15$ for large datasets
- For hierarchical softmax, use a Huffman tree to minimize average path length for frequent words
- Both methods can be applied to CBOW by modifying the gradient calculations appropriately

## Summary

- **Negative sampling** reduces computational cost by training a binary classifier to distinguish real context words from noise words. Cost is linear in $K$ (number of noise words).
- **Hierarchical softmax** reduces computational cost by organizing the vocabulary in a binary tree and computing probabilities along paths. Cost is logarithmic in vocabulary size.

## Exercises

1. **Noise Word Sampling**: How can we sample noise words in negative sampling? Typically, we use a unigram distribution raised to the 3/4 power: $P(w) \propto \text{freq}(w)^{3/4}$.

2. **Verification**: Verify that $\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1$ holds for hierarchical softmax. This follows from the property $\sigma(x) + \sigma(-x) = 1$ and the tree structure.

3. **CBOW Adaptation**: To train the continuous bag-of-words model:
   - For negative sampling: Average the context word vectors, then apply the same negative sampling procedure
   - For hierarchical softmax: Use the averaged context vector in place of $\mathbf{v}_c$ in the probability calculations

## Further Reading

For implementation details and community discussions, visit the [D2L discussion forum](https://discuss.d2l.ai/t/382).