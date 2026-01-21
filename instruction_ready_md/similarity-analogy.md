# Working with Pretrained Word Vectors: Similarity and Analogy

This guide demonstrates how to use pretrained word vectors (embeddings) to perform semantic tasks. We'll load popular GloVe embeddings and use them to find words with similar meanings and solve word analogies.

## Prerequisites & Setup

First, ensure you have the necessary libraries installed. This guide uses the `d2l` library for utilities and either MXNet or PyTorch for tensor operations.

```bash
# Install d2l if you haven't already
pip install d2l
```

Now, import the required modules. Choose the framework block that matches your environment.

```python
# For MXNet users
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```python
# For PyTorch users
from d2l import torch as d2l
import torch
from torch import nn
import os
```

## Step 1: Download Pretrained Embeddings

We'll use pretrained GloVe embeddings of various dimensions. The following code registers the datasets with the `d2l` download utility.

```python
# Register GloVe and fastText datasets for download
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                 '0b8703943ccdb6eb788e6f091b8946e82231bc4d')
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                  'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                   'b5116e234e9eb9076672cfeabf5469f3eec904fa')
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                            'c1816da3821ae9f43899be655002f6c723e91b88')
```

## Step 2: Create an Embedding Loader Class

To load and interact with the embedding files, we define a `TokenEmbedding` class. This class handles reading the vector file, building vocabulary mappings, and retrieving vectors for tokens.

```python
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx) for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

## Step 3: Load the 50-Dimensional GloVe Embeddings

Let's load the GloVe embeddings trained on 6 billion tokens with 50 dimensions. The `TokenEmbedding` instance will automatically download the file on first use.

```python
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

Check the size of the loaded vocabulary. It should contain 400,000 words plus a special `<unk>` token for unknowns.

```python
len(glove_6b50d)
```

**Output:**
```
400001
```

You can look up the index for a word and retrieve a word by its index.

```python
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

**Output:**
```
(3367, 'beautiful')
```

## Step 4: Find Semantically Similar Words

To find words with similar meanings, we calculate the cosine similarity between a query word's vector and all other vectors in the vocabulary. We'll implement a k-nearest neighbors (k-NN) function.

### 4.1 Implement the k-NN Function

```python
# MXNet version
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```python
# PyTorch version
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

### 4.2 Create a Helper Function to Get Similar Tokens

This function uses the k-NN search to print the *k* most similar words for a given query.

```python
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word itself
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

### 4.3 Find Words Similar to "chip"

Let's find the three most similar words to "chip".

```python
get_similar_tokens('chip', 3, glove_6b50d)
```

**Output:**
```
cosine sim=0.856: chips
cosine sim=0.749: intel
cosine sim=0.749: electronics
```

### 4.4 Test with Other Words

Try it with "baby" and "beautiful".

```python
get_similar_tokens('baby', 3, glove_6b50d)
```

**Output:**
```
cosine sim=0.839: babies
cosine sim=0.800: boy
cosine sim=0.792: girl
```

```python
get_similar_tokens('beautiful', 3, glove_6b50d)
```

**Output:**
```
cosine sim=0.921: lovely
cosine sim=0.893: gorgeous
cosine sim=0.830: wonderful
```

## Step 5: Solve Word Analogies

Word analogies test relational understanding. For an analogy *a* : *b* :: *c* : *d*, we find *d* by calculating the vector: **vec(*c*) + vec(*b*) - vec(*a*)**. The word whose vector is closest to this result is the answer.

### 5.1 Implement the Analogy Function

```python
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]  # vec(b) - vec(a) + vec(c)
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]
```

### 5.2 Test a "Male-Female" Analogy

Solve: *man* : *woman* :: *son* : ?

```python
get_analogy('man', 'woman', 'son', glove_6b50d)
```

**Output:**
```
'daughter'
```

### 5.3 Test a "Capital-Country" Analogy

Solve: *beijing* : *china* :: *tokyo* : ?

```python
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

**Output:**
```
'japan'
```

### 5.4 Test a "Adjective-Superlative" Analogy

Solve: *bad* : *worst* :: *big* : ?

```python
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

**Output:**
```
'biggest'
```

### 5.5 Test a "Present-Past Tense" Analogy

Solve: *do* : *did* :: *go* : ?

```python
get_analogy('do', 'did', 'go', glove_6b50d)
```

**Output:**
```
'went'
```

## Summary

In this tutorial, you learned to:
1.  Download and load pretrained GloVe word vectors.
2.  Find semantically similar words using cosine similarity and k-NN search.
3.  Solve word analogies by performing vector arithmetic.

Pretrained word vectors capture rich semantic and syntactic relationships from large text corpora, making them powerful tools for downstream NLP tasks.

## Exercises

1.  **Test fastText embeddings:** Load the English fastText embeddings (`TokenEmbedding('wiki.en')`) and repeat the similarity and analogy tasks. Compare the results with GloVe.
2.  **Optimize for large vocabularies:** When the vocabulary contains millions of words, the linear scan used in `knn` becomes slow. Research and propose methods (e.g., approximate nearest neighbor libraries like FAISS or Annoy) to speed up similarity searches.

## Further Reading

*   **GloVe Project Page:** [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
*   **fastText Project Page:** [https://fasttext.cc/](https://fasttext.cc/)
*   **Discussion Forum:** [https://discuss.d2l.ai/](https://discuss.d2l.ai/) (Check the chapter-specific threads for this topic).