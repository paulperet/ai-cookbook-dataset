# FastText and Byte Pair Encoding: A Practical Guide to Subword Embeddings

## Overview

This guide explores two fundamental techniques for handling word morphology in natural language processing: FastText's subword embeddings and Byte Pair Encoding (BPE). You'll learn how these methods address limitations in traditional word embeddings and implement BPE from scratch.

## Prerequisites

This tutorial requires basic Python knowledge and an understanding of word embeddings. We'll use only Python's standard library.

```python
import collections
```

## 1. Understanding the FastText Model

### The Problem with Traditional Word Embeddings

In models like word2vec, each word form ("helps", "helped", "helping") gets its own vector representation without sharing parameters. This approach:
- Fails to capture morphological relationships
- Performs poorly on rare or unseen words
- Wastes parameters on similar word forms

### FastText's Solution: Subword Embeddings

FastText introduces **subword embeddings** where words are represented as the sum of their character n-gram vectors. Here's how it works:

1. **Add boundary markers**: Wrap words with `<` and `>` to distinguish prefixes/suffixes
2. **Extract n-grams**: Generate all character sequences of specified lengths (typically 3-6)
3. **Sum subword vectors**: A word's vector = sum of all its subword vectors

**Example**: For "where" with n=3:
- Subwords: `<wh`, `whe`, `her`, `ere`, `re>`, `<where>`
- Vector: `v_where = z_<wh + z_whe + z_her + z_ere + z_re> + z_<where>`

### Advantages and Trade-offs

**Benefits**:
- Better representations for rare and out-of-vocabulary words
- Shared parameters across morphologically similar words
- Can handle misspellings and morphological variations

**Costs**:
- Larger vocabulary size
- Higher computational complexity (summing multiple vectors)
- More model parameters

## 2. Implementing Byte Pair Encoding (BPE)

Byte Pair Encoding provides a data-driven way to discover optimal subwords of variable lengths. Let's implement it step by step.

### Step 1: Initialize the Symbol Vocabulary

We start with basic characters and special symbols:

```python
symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

The `_` symbol marks word endings, while `[UNK]` handles unknown symbols.

### Step 2: Prepare Token Frequencies

We need word frequencies from our training data. Let's use a simple example dataset:

```python
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}

for token, freq in raw_token_freqs.items():
    # Insert spaces between characters for initial segmentation
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]

print("Initial token frequencies:")
print(token_freqs)
```

Output:
```
{'f a s t _': 4, 'f a s t e r _': 3, 't a l l _': 5, 't a l l e r _': 4}
```

### Step 3: Find the Most Frequent Symbol Pair

We need a function to identify which consecutive symbols appear together most frequently:

```python
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # Count frequency of each symbol pair
            pairs[symbols[i], symbols[i + 1]] += freq
    
    # Return the pair with highest frequency
    return max(pairs, key=pairs.get)
```

### Step 4: Merge Symbol Pairs

When we find a frequent pair, we merge them into a new symbol:

```python
def merge_symbols(max_freq_pair, token_freqs, symbols):
    # Add new merged symbol to vocabulary
    new_symbol = ''.join(max_freq_pair)
    symbols.append(new_symbol)
    
    # Update tokens with merged symbols
    new_token_freqs = {}
    old_pair_str = ' '.join(max_freq_pair)
    
    for token, freq in token_freqs.items():
        # Replace separated pair with merged symbol
        new_token = token.replace(old_pair_str, new_symbol)
        new_token_freqs[new_token] = freq
    
    return new_token_freqs
```

### Step 5: Run the BPE Algorithm

Now let's perform multiple merging iterations:

```python
num_merges = 10
print("Performing Byte Pair Encoding merges:")

for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'Merge #{i + 1}: {max_freq_pair} -> {"".join(max_freq_pair)}')
```

Output:
```
Merge #1: ('t', 'a') -> ta
Merge #2: ('ta', 'l') -> tal
Merge #3: ('tal', 'l') -> tall
Merge #4: ('f', 'a') -> fa
Merge #5: ('fa', 's') -> fas
Merge #6: ('fas', 't') -> fast
Merge #7: ('e', 'r') -> er
Merge #8: ('er', '_') -> er_
Merge #9: ('tall', '_') -> tall_
Merge #10: ('fast', '_') -> fast_
```

### Step 6: Examine Results

Let's see what symbols we've learned:

```python
print("\nFinal symbol vocabulary:")
print(symbols)
print(f"\nTotal symbols: {len(symbols)}")
```

And check how our original words are now segmented:

```python
print("\nSegmented words:")
print(list(token_freqs.keys()))
```

Output:
```
['fast_', 'fast er_', 'tall_', 'tall er_']
```

Notice that "faster_" became "fast er_" and "taller_" became "tall er_", showing that BPE learned meaningful subwords.

## 3. Applying BPE to New Words

We can use our learned symbols to segment unseen words:

```python
def segment_BPE(tokens, symbols):
    outputs = []
    
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        
        # Greedy longest-match segmentation
        while start < len(token) and start < end:
            if token[start:end] in symbols:
                cur_output.append(token[start:end])
                start = end
                end = len(token)
            else:
                end -= 1
        
        # Handle any remaining characters
        if start < len(token):
            cur_output.append('[UNK]')
        
        outputs.append(' '.join(cur_output))
    
    return outputs

# Test on new words
tokens = ['tallest_', 'fatter_']
segmented = segment_BPE(tokens, symbols)

print("Segmenting new words:")
for original, segmented in zip(tokens, segmented):
    print(f"{original} -> {segmented}")
```

Output:
```
tallest_ -> tall e s t_
fatter_ -> f a t t er_
```

## Key Takeaways

1. **FastText** uses subword embeddings to capture morphological information, improving representations for rare and unseen words.

2. **Byte Pair Encoding** is a data-driven compression algorithm that discovers optimal subwords by iteratively merging frequent symbol pairs.

3. **Practical Benefits**:
   - BPE creates a fixed-size vocabulary of variable-length subwords
   - Learned subwords transfer to new datasets
   - Handles morphology better than fixed n-grams

4. **Implementation Notes**:
   - BPE is greedy and frequency-based
   - Start with character-level vocabulary
   - Merge most frequent pairs iteratively
   - Use longest-match segmentation for new words

## Exercises for Further Learning

1. **Vocabulary Size Challenge**: With ~300 million possible 6-grams in English, how would you manage vocabulary size? (Hint: Consider pruning infrequent n-grams or using hashing tricks)

2. **CBOW Adaptation**: How would you design a subword embedding model based on the continuous bag-of-words architecture instead of skip-gram?

3. **Merge Operations**: If you start with `n` symbols and want a vocabulary of size `m`, how many merge operations do you need?

4. **Phrase Extraction**: How could you extend BPE to discover multi-word phrases instead of just subwords?

## Next Steps

This implementation demonstrates the core concepts. For production use, consider:
- Using established libraries like Hugging Face's `tokenizers`
- Experimenting with different vocabulary sizes
- Applying these techniques to your specific domain
- Exploring BPE variants used in models like GPT and BERT

By understanding these fundamental techniques, you're better equipped to work with modern NLP models that rely heavily on subword representations.