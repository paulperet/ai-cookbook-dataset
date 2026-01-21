# Fine-Tuning BERT for Natural Language Inference

## Overview

This guide walks you through fine-tuning a pretrained BERT model for the Natural Language Inference (NLI) task on the SNLI dataset. You will learn how to:
1. Load a pretrained BERT model.
2. Prepare the SNLI dataset in a format suitable for BERT.
3. Add a simple classification head on top of BERT for the NLI task.
4. Fine-tune the model and evaluate its performance.

## Prerequisites

Ensure you have the necessary libraries installed. This tutorial provides code for both PyTorch and MXNet frameworks.

```bash
# Install the d2l library which contains utilities for this tutorial
pip install d2l
```

Import the required modules:

```python
# For PyTorch
from d2l import torch as d2l
import json
import multiprocessing
import torch
from torch import nn
import os

# For MXNet
# from d2l import mxnet as d2l
# import json
# import multiprocessing
# from mxnet import gluon, np, npx
# from mxnet.gluon import nn
# import os
# npx.set_np()
```

## Step 1: Load a Pretrained BERT Model

We'll use a small version of BERT ("bert.small") for demonstration. The `load_pretrained_model` function downloads the model and its vocabulary, then initializes a BERT model with the pretrained weights.

```python
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_blks, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Load vocabulary
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
    
    # Initialize BERT model architecture
    bert = d2l.BERTModel(
        len(vocab), num_hiddens, ffn_num_hiddens=ffn_num_hiddens,
        num_heads=num_heads, num_blks=num_blks, dropout=dropout, max_len=max_len
    )
    
    # Load pretrained weights
    # For PyTorch
    bert.load_state_dict(torch.load(os.path.join(data_dir, 'pretrained.params')))
    # For MXNet
    # bert.load_parameters(os.path.join(data_dir, 'pretrained.params'), ctx=devices)
    
    return bert, vocab
```

Now, load the small BERT model onto your available GPUs (or CPU).

```python
devices = d2l.try_all_gpus()
bert, vocab = load_pretrained_model(
    'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_blks=2, dropout=0.1, max_len=512, devices=devices
)
```

## Step 2: Prepare the SNLI Dataset for BERT

The SNLI dataset consists of premise-hypothesis pairs labeled with entailment, contradiction, or neutral. We need to format these pairs into BERT's input structure: a single sequence with special tokens `[CLS]`, premise, `[SEP]`, hypothesis, `[SEP]`, and segment IDs to distinguish the two sentences.

We'll create a custom dataset class, `SNLIBERTDataset`, that tokenizes the text, truncates sequences to a maximum length, and generates token IDs, segment IDs, and valid lengths.

```python
class SNLIBERTDataset(torch.utils.data.Dataset):  # Use gluon.data.Dataset for MXNet
    def __init__(self, dataset, max_len, vocab=None):
        # Tokenize all premises and hypotheses
        all_premise_hypothesis_tokens = [
            [p_tokens, h_tokens] for p_tokens, h_tokens in zip(
                *[d2l.tokenize([s.lower() for s in sentences]) for sentences in dataset[:2]]
            )
        ]
        
        self.labels = torch.tensor(dataset[2])  # For MXNet: np.array(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        
        # Preprocess all pairs in parallel
        (self.all_token_ids, self.all_segments, self.valid_lens) = self._preprocess(
            all_premise_hypothesis_tokens
        )
        print(f'read {len(self.all_token_ids)} examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # Use 4 worker processes
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        
        # For PyTorch
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))
        # For MXNet
        # return (np.array(all_token_ids, dtype='int32'),
        #         np.array(all_segments, dtype='int32'),
        #         np.array(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        
        # Pad sequences to max_len
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Reserve 3 tokens for [CLS], [SEP], and [SEP]
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

Now, download the SNLI dataset and create DataLoader instances for training and testing.

```python
# Adjust batch_size and max_len based on your GPU memory
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')

# Load and preprocess datasets
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)

# Create data loaders
train_iter = torch.utils.data.DataLoader(
    train_set, batch_size, shuffle=True, num_workers=num_workers
)
test_iter = torch.utils.data.DataLoader(
    test_set, batch_size, num_workers=num_workers
)
```

## Step 3: Define the BERT Classifier for NLI

For the NLI task, we add a simple MLP on top of BERT. This MLP takes the `[CLS]` token's embedding (which aggregates information from the entire input sequence) and outputs scores for the three classes: entailment, contradiction, and neutral.

```python
class BERTClassifier(nn.Module):  # Use nn.Block for MXNet
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder      # Pretrained BERT encoder
        self.hidden = bert.hidden        # BERT's hidden layer
        self.output = nn.LazyLinear(3)   # For MXNet: nn.Dense(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        # Use the [CLS] token representation (first token) for classification
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

Instantiate the classifier. During fine-tuning, only the parameters of the output layer are trained from scratch, while the BERT encoder and its hidden layer are fine-tuned.

```python
net = BERTClassifier(bert)
# For MXNet, initialize the output layer:
# net.output.initialize(ctx=devices)
```

## Step 4: Fine-Tune BERT

Now, we'll train the model using the SNLI training set. We use the Adam optimizer with a small learning rate and cross-entropy loss.

```python
lr, num_epochs = 1e-4, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)  # For MXNet: gluon.Trainer(...)
loss = nn.CrossEntropyLoss(reduction='none')         # For MXNet: gluon.loss.SoftmaxCrossEntropyLoss()

# Perform a forward pass to initialize lazy layers (PyTorch specific)
net(next(iter(train_iter))[0])

# Train the model
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

**Note:** The `train_ch13` function handles training and evaluation across multiple GPUs. Due to computational constraints, the achieved accuracy might be modest. You can improve it by training for more epochs, tuning hyperparameters, or using a larger BERT model.

## Summary

In this tutorial, you learned how to:
- Load a pretrained BERT model.
- Adapt the SNLI dataset into BERT's input format.
- Build a classifier on top of BERT for the NLI task.
- Fine-tune BERT for a downstream application.

Fine-tuning leverages BERT's powerful pretrained representations, requiring only a small task-specific head to achieve strong performance on tasks like natural language inference.

## Exercises

1. **Scale Up:** Try fine-tuning the larger "bert.base" model by changing the model parameters in `load_pretrained_model` to: `num_hiddens=768`, `ffn_num_hiddens=3072`, `num_heads=12`, `num_blks=12`. Increase the number of training epochs and tune other hyperparameters. Can you achieve a testing accuracy above 0.86?

2. **Truncation Strategy:** The current truncation method removes tokens from the longer sequence until the total length fits. Consider an alternative: truncate based on the ratio of the lengths of the premise and hypothesis (e.g., keep more tokens from the longer sequence). Compare the two methods—what are the advantages and disadvantages of each?