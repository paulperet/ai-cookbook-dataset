# Preparing a Dataset for BERT Pretraining

This guide walks you through creating a dataset suitable for pretraining a BERT model. We will focus on generating data for BERT's two key pretraining tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). While the original BERT was trained on massive corpora, we'll use the smaller, more manageable WikiText-2 dataset for demonstration, which retains useful features like original punctuation and case.

## Prerequisites & Setup

First, ensure you have the necessary libraries installed and imported. This tutorial provides code for both PyTorch and MXNet frameworks.

```bash
# Install the d2l library which contains utilities used in this guide
# pip install d2l
```

Choose your framework and import the required modules:

```python
# For MXNet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import random
npx.set_np()
```

```python
# For PyTorch
from d2l import torch as d2l
import os
import random
import torch
```

## Step 1: Download and Read the WikiText-2 Dataset

We start by downloading the WikiText-2 dataset and writing a function to load and preprocess it. The function reads the training file, converts text to lowercase, splits paragraphs into sentences using the period as a delimiter, and shuffles the paragraphs.

```python
# Download the dataset
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

def _read_wiki(data_dir):
    """Read the WikiText-2 dataset."""
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # Convert to lowercase and split into sentences on the period.
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs
```

## Step 2: Create Helper Functions for Pretraining Tasks

BERT pretraining requires data formatted for two tasks. Let's build the helper functions to generate examples for each.

### 2.1 Generating Next Sentence Prediction (NSP) Data

The NSP task is a binary classification to predict if two sentences are consecutive. The following function creates a training example, randomly making the second sentence a non-consecutive one 50% of the time.

```python
def _get_next_sentence(sentence, next_sentence, paragraphs):
    """Generate a single example for the next sentence prediction task."""
    if random.random() < 0.5:
        is_next = True
    else:
        # Randomly select a sentence from the corpus as a negative example
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```

Next, we process a full paragraph to generate multiple NSP examples, ensuring the combined token length respects the maximum sequence length (accounting for `[CLS]` and `[SEP]` tokens).

```python
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    """Generate NSP data from a single paragraph."""
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # Check length constraint: Account for [CLS] and two [SEP] tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        # Get token IDs and segment IDs (0 for first sentence, 1 for second)
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
```

### 2.2 Generating Masked Language Modeling (MLM) Data

For the MLM task, we mask 15% of tokens in an input sequence. For each selected token, we replace it with:
* The `[MASK]` token 80% of the time.
* The original token 10% of the time.
* A random vocabulary token 10% of the time.

The function below performs this replacement.

```python
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    """Replace tokens for the masked language modeling task."""
    # Create a copy of the input tokens
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Shuffle candidate positions and select up to `num_mlm_preds`
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% probability: replace with [MASK]
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% probability: keep original
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% probability: replace with random word
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        # Record the position and the original token (the label)
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels
```

Now, we apply this function to a token sequence. We first identify candidate positions (excluding special tokens like `[CLS]` and `[SEP]`), then call the replacement function.

```python
def _get_mlm_data_from_tokens(tokens, vocab):
    """Generate MLM data from a tokenized sequence."""
    candidate_pred_positions = []
    # Collect indices of tokens that are not special tokens
    for i, token in enumerate(tokens):
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # Calculate 15% of tokens to predict
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    # Get masked input and prediction labels
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    # Sort predictions by position for consistent ordering
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    # Convert tokens and labels to their vocabulary indices
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```

## Step 3: Assemble and Pad the Final Dataset

We need to combine the NSP and MLM data and pad sequences to a uniform length for batching. The padding function below handles this for both MXNet and PyTorch.

```python
# For MXNet
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        # Pad token IDs and segment IDs
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype='int32'))
        all_segments.append(np.array(segments + [0] * (
            max_len - len(segments)), dtype='int32'))
        # Record the actual length (excluding padding)
        valid_lens.append(np.array(len(token_ids), dtype='float32'))
        # Pad prediction positions and labels for MLM
        all_pred_positions.append(np.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype='int32'))
        # Create weights: 1.0 for real predictions, 0.0 for padding
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype='float32'))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```python
# For PyTorch
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        # Pad token IDs and segment IDs
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        # Pad prediction positions and labels for MLM
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Create weights for the loss function
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

## Step 4: Create the Custom Dataset Class

Now, we integrate everything into a PyTorch/MXNet Dataset class. This class tokenizes the raw text, builds a vocabulary, generates NSP and MLM examples, and pads them.

```python
# For MXNet
class _WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Tokenize each paragraph into sentences and then into word tokens
        paragraphs = [d2l.tokenize(paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        # Build vocabulary, filtering tokens appearing less than 5 times
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Generate NSP examples
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Augment each example with MLM data
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad all examples to the same length
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```python
# For PyTorch
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Tokenization and vocabulary building
        paragraphs = [d2l.tokenize(paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Generate NSP examples
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Augment with MLM data
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad examples
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

## Step 5: Load the Dataset and Inspect a Batch

Finally, we create a convenience function to download the dataset and create a DataLoader, then inspect the shape of a single batch.

```python
def load_data_wiki(batch_size, max_len):
    """Download WikiText-2 and create a DataLoader."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    # For MXNet, use: gluon.data.DataLoader(...)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab

# Load data with a batch size of 512 and max sequence length of 64
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

# Inspect the first batch
for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print('Token IDs shape:', tokens_X.shape)
    print('Segment IDs shape:', segments_X.shape)
    print('Valid lengths shape:', valid_lens_x.shape)
    print('Prediction positions shape:', pred_positions_X.shape)
    print('MLM weights shape:', mlm_weights_X.shape)
    print('MLM label shape:', mlm_Y.shape)
    print('NSP label shape:', nsp_y.shape)
    break
```

You should see output similar to:
```
Token IDs shape: torch.Size([512, 64])
Segment IDs shape: torch.Size([512, 64])
Valid lengths shape: torch.Size([512])
Prediction positions shape: torch.Size([512, 10])
MLM weights shape: torch.Size([512, 10])
MLM label shape: torch.Size([512, 10])
NSP label shape: torch.Size([512])
```

Notice that `pred_positions_X` has a second dimension of 10, which corresponds to 15% of the max length (64), the number of masked tokens predicted per sequence.

Let's also check the size of the vocabulary we built:

```python
print(f'Vocabulary size: {len(vocab)}')
```

## Summary

In this tutorial, you have learned how to construct a dataset for BERT pretraining from raw text. Key steps included:
1. Downloading and preprocessing the WikiText-2 corpus.
2. Implementing helper functions to generate examples for the Next Sentence Prediction (NSP) task.
3. Implementing helper functions to generate examples for the Masked Language Modeling (MLM) task, including the 80%/10%/10% masking strategy.
4. Combining and padding the data into batches suitable for model training.
5. Creating a custom Dataset class and DataLoader to stream the data efficiently.

The resulting dataset provides token IDs, segment IDs, valid lengths, masked token positions, and labels required to train a BERT model on both its core pretraining objectives.

## Exercises

1. **Advanced Sentence Splitting:** We used the period as the only sentence delimiter. Try integrating a more robust tokenizer, like spaCy or NLTK. For NLTK, you would:
   ```bash
   pip install nltk
   ```
   ```python
   import nltk
   nltk.d