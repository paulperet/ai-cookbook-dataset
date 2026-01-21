# Machine Translation and the Dataset
:label:`sec_machine_translation`

A major breakthrough that spurred widespread interest in modern Recurrent Neural Networks (RNNs) was a significant advance in the field of statistical **machine translation**. In this task, a model receives a sentence in one language (the *source*) and must predict the corresponding sentence in another language (the *target*). This is challenging because sentences can have different lengths, and the order of corresponding words often differs between languages due to grammatical variations.

This problem is an example of a broader class called **sequence-to-sequence (seq2seq)** problems, which involve mapping between two sequences that are not perfectly aligned. Other examples include mapping dialog prompts to replies or questions to answers. This chapter and much of :numref:`chap_attention-and-transformers` will focus on seq2seq models.

In this guide, we will introduce the machine translation problem and prepare a dataset for subsequent modeling. We'll walk through downloading, preprocessing, tokenizing, and batching bilingual text data.

## Prerequisites

We'll use the `d2l` library, which provides utilities for deep learning. The following code installs it if necessary and imports the required modules for your chosen framework (MXNet, PyTorch, TensorFlow, or JAX).

```bash
# If you haven't installed d2l, you can do so via pip
# !pip install d2l
```

```python
# Framework-specific imports are handled by the d2l book's tab system.
# The following code selects your framework.
import sys
# For the purpose of this tutorial, we assume PyTorch is used.
# The original notebook uses a tab system; we show the PyTorch version here.
from d2l import torch as d2l
import torch
import os
```

## Step 1: Downloading and Preprocessing the Dataset

We'll use an English-French dataset consisting of bilingual sentence pairs from the Tatoeba Project. Each line contains an English source sequence and its French translation, separated by a tab.

First, let's define a data module class to handle downloading and extraction.

```python
class MTFraEng(d2l.DataModule):
    """The English-French dataset."""
    def _download(self):
        # Download and extract the dataset
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root,
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        # Read the raw text file
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()
```

Now, instantiate the class and download the raw data.

```python
data = MTFraEng()
raw_text = data._download()
print(raw_text[:75])
```

**Output:**
```
Go.	Va !
Hi.	Salut !
Run!	Cours !
Run!	Courez !
Wow!	Ça alors !
```

The raw text contains non-breaking spaces and mixed casing. We need to clean it up. Let's add a preprocessing method to the class.

```python
@d2l.add_to_class(MTFraEng)
def _preprocess(self, text):
    # Replace non-breaking space with regular space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Insert space between words and punctuation marks for easier tokenization
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)
```

Apply the preprocessing:

```python
text = data._preprocess(raw_text)
print(text[:80])
```

**Output:**
```
go .	va !
hi .	salut !
run !	cours !
run !	courez !
wow !	ça alors !
```

## Step 2: Tokenization

For machine translation, we typically use **word-level tokenization** (modern models use more sophisticated subword techniques). We'll split each sequence into words and punctuation, appending a special `<eos>` (end-of-sequence) token to mark the end. This token will later signal the model to stop generating.

The following method tokenizes the text into source (English) and target (French) token lists.

```python
@d2l.add_to_class(MTFraEng)
def _tokenize(self, text, max_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):
        if max_examples and i > max_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            # Skip empty tokens and append <eos>
            src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
    return src, tgt
```

Let's tokenize a subset of the data and inspect the first few examples.

```python
src, tgt = data._tokenize(text, max_examples=6)
print('Source (English):', src[:3])
print('Target (French):', tgt[:3])
```

**Output:**
```
Source (English): [['go', '.', '<eos>'], ['hi', '.', '<eos>'], ['run', '!', '<eos>']]
Target (French): [['va', '!', '<eos>'], ['salut', '!', '<eos>'], ['cours', '!', '<eos>']]
```

It's helpful to visualize the distribution of sequence lengths in our dataset.

```python
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs."""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', src, tgt)
```

*Note: The histogram shows that most sequences are short, with fewer than 20 tokens.*

## Step 3: Loading Sequences of Fixed Length

For efficient minibatch processing, we need all sequences in a batch to have the same length. We achieve this through **padding** and **truncation**.

- If a sequence is shorter than `num_steps`, we append the `<pad>` token.
- If it's longer, we truncate it to `num_steps`.
- We also record the original sequence length (excluding padding) because some models need this information.

We'll also build separate vocabularies for source and target languages, converting infrequent words (appearing less than twice) to an `<unk>` (unknown) token to manage vocabulary size.

Now, let's implement the full dataset builder within our class.

```python
@d2l.add_to_class(MTFraEng)
def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
    super(MTFraEng, self).__init__()
    self.save_hyperparameters()
    self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
        self._download())

@d2l.add_to_class(MTFraEng)
def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
    def _build_array(sentences, vocab, is_tgt=False):
        pad_or_trim = lambda seq, t: (
            seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
        sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
        if is_tgt:
            # For target sequences, prepend <bos> (beginning-of-sequence) for decoder input
            sentences = [['<bos>'] + s for s in sentences]
        if vocab is None:
            vocab = d2l.Vocab(sentences, min_freq=2)
        array = d2l.tensor([vocab[s] for s in sentences])
        valid_len = d2l.reduce_sum(
            d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
        return array, vocab, valid_len

    src, tgt = self._tokenize(self._preprocess(raw_text),
                              self.num_train + self.num_val)
    src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
    # For target, we build two arrays: one for decoder input (with <bos>) and one for labels (shifted)
    tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
    # Return: source, decoder_input, source_valid_len, labels
    return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
            src_vocab, tgt_vocab)
```

## Step 4: Reading the Dataset

We need a method to create data iterators for training and validation.

```python
@d2l.add_to_class(MTFraEng)
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(self.arrays, train, idx)
```

Let's instantiate the dataset and examine the first minibatch.

```python
data = MTFraEng(batch_size=3)
src, tgt, src_valid_len, label = next(iter(data.train_dataloader()))
print('source (indices):', d2l.astype(src, d2l.int32))
print('decoder input (indices):', d2l.astype(tgt, d2l.int32))
print('source len excluding pad:', d2l.astype(src_valid_len, d2l.int32))
print('label (indices):', d2l.astype(label, d2l.int32))
```

**Output (example indices):**
```
source (indices): tensor([[ 2,  3,  1,  1,  1,  1,  1,  1,  1],
                          [ 4,  5,  1,  1,  1,  1,  1,  1,  1],
                          [ 6,  7,  8,  1,  1,  1,  1,  1,  1]])
decoder input (indices): tensor([[ 0,  2,  3,  1,  1,  1,  1,  1,  1],
                                 [ 0,  4,  5,  1,  1,  1,  1,  1,  1],
                                 [ 0,  6,  7,  8,  1,  1,  1,  1,  1]])
source len excluding pad: tensor([2, 2, 3])
label (indices): tensor([[ 2,  3,  1,  1,  1,  1,  1,  1,  1],
                         [ 4,  5,  1,  1,  1,  1,  1,  1,  1],
                         [ 6,  7,  8,  1,  1,  1,  1,  1,  1]])
```

Finally, let's see how to convert a custom sentence pair into the model's input format.

```python
@d2l.add_to_class(MTFraEng)
def build(self, src_sentences, tgt_sentences):
    raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])
    arrays, _, _ = self._build_arrays(
        raw_text, self.src_vocab, self.tgt_vocab)
    return arrays

src, tgt, _,  _ = data.build(['hi .'], ['salut .'])
print('source tokens:', data.src_vocab.to_tokens(d2l.astype(src[0], d2l.int32)))
print('target tokens:', data.tgt_vocab.to_tokens(d2l.astype(tgt[0], d2l.int32)))
```

**Output:**
```
source tokens: ['hi', '.', '<eos>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']
target tokens: ['<bos>', 'salut', '.', '<eos>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']
```

## Summary

In this tutorial, we prepared a machine translation dataset from raw bilingual text. Key steps included:

1. **Downloading and preprocessing** to clean the text.
2. **Word-level tokenization** with special `<eos>` tokens.
3. **Padding and truncation** to create fixed-length sequences for batching.
4. **Building vocabularies** that map tokens to indices, handling infrequent words.
5. **Creating data loaders** that yield minibatches of source sequences, decoder inputs, valid lengths, and labels.

This dataset pipeline is essential for training sequence-to-sequence models, which we will explore in the following sections.

## Exercises

1. Experiment with the `max_examples` argument in `_tokenize`. How does it affect the vocabulary sizes of the source and target languages?
2. For languages like Chinese or Japanese that lack explicit word boundaries (e.g., spaces), is word-level tokenization still a good approach? Discuss the challenges and potential alternatives.

---
*For framework-specific discussions, please refer to the D2L discussion forums: [MXNet](https://discuss.d2l.ai/t/344), [PyTorch](https://discuss.d2l.ai/t/1060), [TensorFlow](https://discuss.d2l.ai/t/3863), [JAX](https://discuss.d2l.ai/t/18020).*