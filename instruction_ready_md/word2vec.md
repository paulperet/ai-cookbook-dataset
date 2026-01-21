# Word Embedding with word2vec: A Practical Guide

## Introduction

Natural language is a complex system where words serve as the fundamental units of meaning. To process language computationally, we need effective ways to represent words numerically. This guide explores **word embedding**—the technique of mapping words to real-valued vectors—and specifically focuses on the word2vec models that have become foundational in natural language processing.

## Why One-Hot Vectors Fall Short

In previous neural network implementations, you might have used one-hot vectors to represent words. For a vocabulary of size N, each word gets a vector of length N with a single 1 at its index position and 0s elsewhere.

While simple to construct, one-hot vectors have a critical limitation: they cannot express semantic similarity between words. The cosine similarity between any two different one-hot vectors is always 0, making them useless for capturing relationships like synonyms or related concepts.

## The word2vec Solution

The word2vec tool addresses this limitation by learning dense vector representations where semantically similar words have similar vectors. Word2vec contains two self-supervised models:

1. **Skip-gram**: Predicts context words given a center word
2. **Continuous Bag of Words (CBOW)**: Predicts a center word given its context words

Both models learn by predicting words from their surrounding context in text corpora, without requiring labeled data.

## The Skip-Gram Model

### Conceptual Foundation

The skip-gram model operates on a simple principle: given a center word, predict the words that appear around it within a defined context window.

Consider the sequence: "the", "man", "loves", "his", "son"
- Center word: "loves"
- Context window size: 2
- Context words: "the", "man", "his", "son"

The model learns to maximize: P("the"|"loves") · P("man"|"loves") · P("his"|"loves") · P("son"|"loves")

### Mathematical Formulation

Each word has two vector representations:
- **vᵢ**: Center word vector
- **uᵢ**: Context word vector

The conditional probability of context word wₒ given center word w_c is:

```python
P(wₒ | w_c) = exp(uₒᵀ v_c) / Σᵢ exp(uᵢᵀ v_c)
```

For a text sequence of length T, the likelihood function becomes:

```python
Πₜ Π_{j=-m to m, j≠0} P(w⁽ᵗ⁺ʲ⁾ | w⁽ᵗ⁾)
```

### Training Process

During training, we maximize this likelihood (or minimize the negative log-likelihood):

```python
Loss = -Σₜ Σ_{j=-m to m, j≠0} log P(w⁽ᵗ⁺ʲ⁾ | w⁽ᵗ⁾)
```

The gradient with respect to the center word vector v_c is:

```python
∂log P(wₒ | w_c)/∂v_c = uₒ - Σⱼ P(wⱼ | w_c) uⱼ
```

This gradient calculation requires computing probabilities for all words in the vocabulary, which can be computationally expensive for large vocabularies.

### Practical Implementation Considerations

After training, you typically use the center word vectors (vᵢ) as your word representations. These vectors capture semantic relationships—similar words will have vectors pointing in similar directions in the vector space.

## The Continuous Bag of Words (CBOW) Model

### Conceptual Foundation

CBOW flips the skip-gram approach: instead of predicting context from a center word, it predicts a center word from its surrounding context.

Using our same example sequence:
- Context words: "the", "man", "his", "son"
- Center word: "loves"

The model learns: P("loves" | "the", "man", "his", "son")

### Mathematical Formulation

In CBOW, the roles of the vectors are reversed:
- **vᵢ**: Context word vector
- **uᵢ**: Center word vector

The conditional probability averages the context vectors:

```python
P(w_c | w_{o₁}, ..., w_{o₂ₘ}) = exp(u_cᵀ v̄_o) / Σᵢ exp(uᵢᵀ v̄_o)
```

Where v̄_o = (v_{o₁} + ... + v_{o₂ₘ}) / (2m) is the average context vector.

### Training Process

The training objective minimizes:

```python
Loss = -Σₜ log P(w⁽ᵗ⁾ | w⁽ᵗ⁻ᵐ⁾, ..., w⁽ᵗ⁻¹⁾, w⁽ᵗ⁺¹⁾, ..., w⁽ᵗ⁺ᵐ⁾)
```

The gradient with respect to a context word vector is:

```python
∂log P(w_c | W_o)/∂v_{oᵢ} = (1/2m)(u_c - Σⱼ P(wⱼ | W_o) uⱼ)
```

### Practical Implementation Considerations

Unlike skip-gram, CBOW typically uses the context word vectors (vᵢ) as the final word representations. The averaging of context vectors makes CBOW more efficient but potentially less sensitive to rare words.

## Key Differences and When to Use Each

| Aspect | Skip-Gram | CBOW |
|--------|-----------|------|
| Prediction direction | Center → Context | Context → Center |
| Training speed | Slower | Faster |
| Performance on rare words | Better | Worse |
| Typical use case | Large datasets, rare words | Smaller datasets, frequent words |

## Implementation Tips

1. **Vocabulary Size**: Both models require computing softmax over the entire vocabulary, which is O(V) complexity. For large vocabularies, consider using negative sampling or hierarchical softmax.

2. **Vector Dimensions**: Typical dimensions range from 50-300. Higher dimensions capture more nuance but require more data and computation.

3. **Context Window**: The window size m is a hyperparameter. Smaller windows (2-5) capture syntactic relationships, while larger windows capture more topical associations.

4. **Training Data**: More data generally leads to better embeddings. Pre-trained word2vec embeddings are available for many languages.

## Exercises for Practice

1. Analyze the computational complexity of gradient calculations in both models. What optimization techniques could help with large vocabularies?

2. How would you handle multi-word expressions like "new york"? Consider modifying the training data or creating composite representations.

3. For skip-gram embeddings, what relationship exists between the dot product of two word vectors and their cosine similarity? Why might semantically similar words have high cosine similarity in the learned space?

## Summary

Word2vec revolutionized NLP by providing efficient, semantically meaningful word representations. The skip-gram and CBOW models offer complementary approaches to learning these embeddings from unlabeled text. Skip-gram excels at capturing nuanced relationships and handling rare words, while CBOW offers faster training and good performance on frequent words. Understanding both models gives you the foundation to work with modern word embeddings and their applications in downstream NLP tasks.