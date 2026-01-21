# Word Embedding with Global Vectors (GloVe)

This guide explains the GloVe (Global Vectors) model for word embedding, which leverages global corpus statistics for efficient and effective training.

## Prerequisites

Before diving into the model, ensure you have a basic understanding of:
*   Word embeddings and their purpose.
*   The skip-gram model architecture.
*   Concepts like conditional probability and cross-entropy loss.

## 1. From Skip-Gram to Global Statistics

The skip-gram model predicts context words given a center word. Let's reinterpret it using global corpus statistics.

### 1.1 Defining the Model and Global Counts

In the skip-gram model, the conditional probability of context word \( w_j \) given center word \( w_i \) is:

\[
q_{ij} = \frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \exp(\mathbf{u}_k^\top \mathbf{v}_i)},
\]

where \( \mathbf{v}_i \) and \( \mathbf{u}_i \) are the center and context word vectors for word \( w_i \), and \( \mathcal{V} \) is the vocabulary.

Now, consider the entire corpus. For a center word \( w_i \), all its context words form a **multiset** \( \mathcal{C}_i \). Let \( x_{ij} \) be the **multiplicity** of word \( w_j \) in this multiset. This value \( x_{ij} \) is the **global co-occurrence count** of word \( w_j \) (context) with word \( w_i \) (center) across the entire corpus.

### 1.2 The Global Loss Function

Using these global counts, the skip-gram model's loss function can be equivalently expressed as:

\[
-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.
\tag{1}
\]

Let \( x_i = |\mathcal{C}_i| \) be the total number of context words for \( w_i \). We can define the empirical conditional probability from the corpus as \( p_{ij} = x_{ij} / x_i \). This allows us to rewrite the loss as:

\[
-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.
\tag{2}
\]

This formulation shows the loss is the cross-entropy between the empirical distribution \( p_{ij} \) and the model's predicted distribution \( q_{ij} \), weighted by the frequency \( x_i \) of the center word.

### 1.3 Limitations of Cross-Entropy for This Task

While common, cross-entropy has drawbacks here:
1.  **Computational Cost:** Calculating the denominator of \( q_{ij} \) requires a sum over the entire vocabulary, which is expensive for large vocabularies.
2.  **Weighting Issues:** It can assign excessive importance to rare events from large corpora.

## 2. The GloVe Model

The GloVe model addresses these issues by modifying the skip-gram objective to use a weighted least squares loss on the co-occurrence statistics.

### 2.1 Model Changes

GloVe makes three key changes:

1.  **Work with Logarithms:** Instead of modeling probabilities \( p_{ij} \) and \( q_{ij} \), GloVe works with the non-probabilistic quantities \( p'_{ij}=x_{ij} \) and \( q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i) \). The core term becomes \( (\log\,p'_{ij} - \log\,q'_{ij})^2 = (\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij})^2 \).
2.  **Add Bias Terms:** Two scalar bias parameters are added for each word: a center word bias \( b_i \) and a context word bias \( c_j \).
3.  **Introduce a Weight Function:** A weighting function \( h(x_{ij}) \) is applied to each loss term. This function assigns lower weight to very frequent co-occurrences. A common choice is:
    \[
    h(x) = \begin{cases}
    (x/c)^\alpha & \text{if } x < c \\
    1 & \text{otherwise}
    \end{cases}
    \]
    with typical values \( \alpha = 0.75 \) and \( c = 100 \).

### 2.2 The GloVe Loss Function

Combining these elements, the GloVe model is trained by minimizing the following loss function:

\[
\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.
\tag{3}
\]

**Key Efficiency Note:** Since \( h(0) = 0 \), terms where \( x_{ij} = 0 \) (words never co-occur) contribute nothing to the loss. Therefore, training can proceed efficiently by sampling only **non-zero** \( x_{ij} \) co-occurrence counts, which are precomputed from the corpus. This focus on global statistics gives GloVe its name.

### 2.3 Symmetry in GloVe

Note that co-occurrence is symmetric: \( x_{ij} = x_{ji} \). Consequently, GloVe fits the symmetric \( \log \, x_{ij} \), unlike the asymmetric \( p_{ij} \) in word2vec. This makes the center word vector \( \mathbf{v}_i \) and context word vector \( \mathbf{u}_i \) for the same word **mathematically equivalent** in the model. In practice, they often differ due to random initialization, so the final word vector is typically taken as the sum \( \mathbf{v}_i + \mathbf{u}_i \).

## 3. Interpreting GloVe via Co-occurrence Ratios

We can derive the GloVe objective from a different, intuitive perspective based on word relationships.

### 3.1 The Ratio of Probabilities

Consider the conditional probabilities \( p_{ij} = P(w_j | w_i) \). The **ratio** \( p_{ij} / p_{ik} \) for a given center word \( w_i \) and two context words \( w_j, w_k \) reveals their semantic relationship relative to \( w_i \).

For example, given center words "ice" and "steam":
*   For \( w_k = \text{solid} \) (related to ice, not steam), \( p_{\text{ice, solid}} / p_{\text{steam, solid}} \) is large (>1).
*   For \( w_k = \text{gas} \) (related to steam, not ice), the ratio is small (<1).
*   For \( w_k = \text{water} \) (related to both), the ratio is close to 1.
*   For \( w_k = \text{fashion} \) (related to neither), the ratio is also close to 1.

### 3.2 Designing a Function to Fit Ratios

We want to design a function \( f \) of the word vectors that approximates this ratio:

\[
f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.
\tag{4}
\]

A sensible choice, given the ratio is a scalar and the function must satisfy \( f(x)f(-x)=1 \), is the exponential function: \( f(x) = \exp(x) \). This leads us to model:

\[
\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{\exp(\mathbf{u}_k^\top \mathbf{v}_i)} \approx \frac{p_{ij}}{p_{ik}}.
\]

### 3.3 Recovering the GloVe Objective

From the above, we set \( \exp(\mathbf{u}_j^\top \mathbf{v}_i) \approx \alpha p_{ij} \) for some constant \( \alpha \). Substituting \( p_{ij} = x_{ij}/x_i \) and taking logarithms yields:

\[
\mathbf{u}_j^\top \mathbf{v}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i.
\]

The terms \( \log\,\alpha \) and \( \log\,x_i \) can be absorbed into the bias terms \( b_i \) and \( c_j \), giving us the core relationship GloVe aims to fit:

\[
\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.
\tag{5}
\]

Minimizing the weighted squared error of this equation directly gives us the GloVe loss function in Equation (3).

## Summary

*   GloVe is a word embedding model that efficiently uses **precomputed global word-word co-occurrence statistics** from a corpus.
*   It addresses computational and weighting issues of the skip-gram model by using a **weighted least squares loss** (Equation 3) instead of cross-entropy.
*   The model is **symmetric** by design, making the center and context vectors for a word theoretically equivalent.
*   The objective can be derived from the intuitive goal of having word vector dot products capture the **ratios of co-occurrence probabilities** between words.

## Exercises

1.  How could the distance between words \( w_i \) and \( w_j \) in a context window be used to refine the calculation of the co-occurrence probability \( p_{ij} \)? (Hint: Refer to Section 4.2 of the GloVe paper).
2.  In the GloVe model, for any given word, are its center word bias \( b_i \) and context word bias \( c_i \) mathematically equivalent? Justify your answer.