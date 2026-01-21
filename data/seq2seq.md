# Sequence-to-Sequence Learning for Machine Translation

In this guide, you will learn how to build and train a sequence-to-sequence (seq2seq) model for machine translation. This architecture uses two RNNs: an **encoder** that processes the input sequence and a **decoder** that generates the output sequence. We'll implement this model step-by-step, train it on an English-French dataset, and evaluate its predictions using the BLEU score.

## Prerequisites

First, ensure you have the necessary libraries installed. The code supports multiple deep learning frameworks (MXNet, PyTorch, TensorFlow, JAX). Import the required modules based on your chosen framework.

```python
import collections
import math

# Framework-specific imports
# For MXNet
# from mxnet import np, npx, init, gluon, autograd
# from mxnet.gluon import nn, rnn
# npx.set_np()

# For PyTorch
# import torch
# from torch import nn
# from torch.nn import functional as F

# For TensorFlow
# import tensorflow as tf

# For JAX
# from flax import linen as nn
# from functools import partial
# import jax
# from jax import numpy as jnp
# import optax

# Import the D2L library for utilities
from d2l import mxnet as d2l  # Change to torch, tensorflow, or jax as needed
```

## 1. Understanding the Encoder-Decoder Architecture

The core idea is to map a variable-length input sequence (e.g., an English sentence) to a variable-length output sequence (e.g., a French translation). The encoder RNN compresses the input into a fixed-size context vector. The decoder RNN then uses this context to generate the output token by token.

**Key Design Points:**
- The encoder processes the entire input sequence.
- The decoder is initialized with the encoder's final hidden state.
- During training, we use **teacher forcing**: the decoder receives the ground truth previous token as input at each step.
- Special tokens `<bos>` (beginning of sequence) and `<eos>` (end of sequence) mark the start and end of sequences.

## 2. Implementing the Encoder

The encoder transforms an input sequence into a context variable. We'll implement it as a multilayer GRU RNN with an embedding layer.

```python
class Seq2SeqEncoder(d2l.Encoder):
    """The RNN encoder for sequence-to-sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(num_hiddens, num_layers, dropout)
        # Framework-specific initialization may be required here
        
    def forward(self, X, *args):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(d2l.transpose(X))
        # embs shape: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs)
        # outputs shape: (num_steps, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state
```

**Let's test the encoder with a dummy input to verify its output shapes.**

```python
vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 9
encoder = Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
X = d2l.zeros((batch_size, num_steps))

# Forward pass (framework-specific)
if framework in ['mxnet', 'pytorch', 'tensorflow']:
    enc_outputs, enc_state = encoder(X)
elif framework == 'jax':
    (enc_outputs, enc_state), _ = encoder.init_with_output(d2l.get_key(), X)

print(f"Encoder outputs shape: {enc_outputs.shape}")
print(f"Encoder state shape: {enc_state.shape}")
```

You should see:
- `enc_outputs` shape: `(num_steps=9, batch_size=4, num_hiddens=16)`
- `enc_state` shape: `(num_layers=2, batch_size=4, num_hiddens=16)`

## 3. Implementing the Decoder

The decoder predicts each subsequent token in the target sequence. At each step, it takes the previous token, the current hidden state, and the context from the encoder.

```python
class Seq2SeqDecoder(d2l.Decoder):
    """The RNN decoder for sequence to sequence learning."""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout)
        self.dense = nn.Dense(vocab_size)  # Output layer
        
    def init_state(self, enc_all_outputs, *args):
        return enc_all_outputs  # Returns encoder outputs and hidden state

    def forward(self, X, state):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(d2l.transpose(X))
        # embs shape: (num_steps, batch_size, embed_size)
        enc_output, hidden_state = state
        # Use the final encoder hidden state as context
        context = enc_output[-1]  # Shape: (batch_size, num_hiddens)
        # Broadcast context to match sequence length
        context = context.repeat(embs.shape[0], 1, 1)
        # Concatenate embeddings and context
        embs_and_context = d2l.concat((embs, context), -1)
        # RNN forward pass
        outputs, hidden_state = self.rnn(embs_and_context, hidden_state)
        # Project to vocabulary size
        outputs = d2l.swapaxes(self.dense(outputs), 0, 1)
        # outputs shape: (batch_size, num_steps, vocab_size)
        return outputs, [enc_output, hidden_state]
```

**Test the decoder with the same dummy input.**

```python
decoder = Seq2SeqDecoder(vocab_size, embed_size, num_hiddens, num_layers)
state = decoder.init_state(enc_outputs)  # Use encoder outputs as initial state

if framework in ['mxnet', 'pytorch', 'tensorflow']:
    dec_outputs, state = decoder(X, state)
elif framework == 'jax':
    (dec_outputs, state), _ = decoder.init_with_output(d2l.get_key(), X, state)

print(f"Decoder outputs shape: {dec_outputs.shape}")
```

The decoder outputs should have shape `(batch_size=4, num_steps=9, vocab_size=10)`.

## 4. Combining Encoder and Decoder

Now, let's combine them into a complete seq2seq model.

```python
class Seq2Seq(d2l.EncoderDecoder):
    """The RNN encoder--decoder for sequence to sequence learning."""
    def __init__(self, encoder, decoder, tgt_pad, lr):
        super().__init__(encoder, decoder)
        self.save_hyperparameters()
        
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        
    def configure_optimizers(self):
        # Use Adam optimizer
        if framework == 'mxnet':
            return gluon.Trainer(self.parameters(), 'adam', {'learning_rate': self.lr})
        elif framework == 'pytorch':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        elif framework == 'tensorflow':
            return tf.keras.optimizers.Adam(learning_rate=self.lr)
        elif framework == 'jax':
            return optax.adam(learning_rate=self.lr)
```

## 5. Loss Function with Masking

We need to ignore padding tokens when calculating the loss. We'll mask them out.

```python
@d2l.add_to_class(Seq2Seq)
def loss(self, Y_hat, Y):
    l = super(Seq2Seq, self).loss(Y_hat, Y, averaged=False)
    mask = d2l.astype(d2l.reshape(Y, -1) != self.tgt_pad, d2l.float32)
    return d2l.reduce_sum(l * mask) / d2l.reduce_sum(mask)
```

## 6. Training the Model

Let's train the model on the English-French machine translation dataset.

```python
# Load the dataset
data = d2l.MTFraEng(batch_size=128)

# Define model hyperparameters
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2

# Instantiate encoder and decoder
encoder = Seq2SeqEncoder(len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

# Create the seq2seq model
model = Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'], lr=0.005)

# Train the model
trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
```

## 7. Making Predictions

During prediction, we feed the previously predicted token back into the decoder at each step.

```python
@d2l.add_to_class(d2l.EncoderDecoder)
def predict_step(self, batch, device, num_steps, save_attention_weights=False):
    src, tgt, src_valid_len, _ = batch
    enc_all_outputs = self.encoder(src, src_valid_len)
    dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
    outputs = [d2l.expand_dims(tgt[:, 0], 1)]
    for _ in range(num_steps):
        Y, dec_state = self.decoder(outputs[-1], dec_state)
        outputs.append(d2l.argmax(Y, 2))
    return d2l.concat(outputs[1:], 1)
```

## 8. Evaluating with BLEU Score

We'll use the BLEU metric to evaluate translation quality by comparing n-gram overlap between predicted and target sequences.

```python
def bleu(pred_seq, label_seq, k):
    """Compute the BLEU score."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```

## 9. Translating Example Sentences

Let's test the trained model on a few English sentences and compute their BLEU scores.

```python
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']

# Generate predictions
preds, _ = model.predict_step(data.build(engs, fras), d2l.try_gpu(), data.num_steps)

# Print translations and BLEU scores
for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)
    print(f'{en} => {" ".join(translation)}, bleu: {bleu(" ".join(translation), fr, k=2):.3f}')
```

## Summary

In this guide, you built a complete RNN-based sequence-to-sequence model for machine translation. Key steps included:

1. **Encoder:** Processes the input sequence into a context vector.
2. **Decoder:** Generates the output sequence token by token using teacher forcing during training.
3. **Training:** Combined encoder and decoder with a masked loss function to ignore padding tokens.
4. **Prediction:** Used the decoder autoregressively to translate new sentences.
5. **Evaluation:** Applied the BLEU score to measure translation quality.

This foundational architecture can be extended with attention mechanisms (covered later) to improve performance on longer sequences.

## Exercises

1. Experiment with hyperparameters (e.g., hidden size, layers, dropout) to improve translation accuracy.
2. Train the model without masking the loss. How does performance change?
3. Modify the architecture so the encoder and decoder can have different numbers of layers or hidden units. How would you initialize the decoder's hidden state?
4. Replace teacher forcing with using the model's own predictions during training. What impact does this have?
5. Implement the model using LSTM instead of GRU.
6. Explore other output layer designs for the decoder (e.g., tied embeddings).