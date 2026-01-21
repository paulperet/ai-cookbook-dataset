# Implementing the Bahdanau Attention Mechanism

## Overview

In this tutorial, you will implement the Bahdanau attention mechanism for sequence-to-sequence learning. This mechanism allows the decoder to dynamically focus on different parts of the input sequence when generating each output token, addressing the limitations of fixed-context models when handling long sequences.

## Prerequisites

First, ensure you have the necessary libraries installed. The code supports multiple deep learning frameworks.

```bash
# Install the required deep learning framework of your choice
# For example, for PyTorch:
# pip install torch
```

Now, import the required modules. The code is designed to work with MXNet, PyTorch, TensorFlow, or JAX.

```python
# Framework-specific imports are handled by the d2l book
from d2l import torch as d2l  # Example for PyTorch
import torch
from torch import nn
```

## Step 1: Understanding the Attention Mechanism

The Bahdanau attention mechanism enhances the standard encoder-decoder architecture by allowing the decoder to access all encoder hidden states, not just the final one. At each decoding step, the mechanism computes a context vector as a weighted sum of the encoder hidden states, where the weights are determined by an alignment model.

The context vector \( \mathbf{c}_{t'} \) at decoding step \( t' \) is computed as:

\[
\mathbf{c}_{t'} = \sum_{t=1}^{T} \alpha(\mathbf{s}_{t' - 1}, \mathbf{h}_{t}) \mathbf{h}_{t}
\]

Here, \( \mathbf{s}_{t' - 1} \) is the previous decoder hidden state (the query), and \( \mathbf{h}_{t} \) are the encoder hidden states (keys and values). The alignment weights \( \alpha \) are computed using an additive attention scoring function.

## Step 2: Defining the Attention Decoder Base Class

We start by defining a base class for attention-based decoders. This class provides an interface that includes a property for accessing attention weights.

```python
class AttentionDecoder(d2l.Decoder):
    """The base attention-based decoder interface."""
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError
```

## Step 3: Implementing the Seq2Seq Attention Decoder

Next, we implement the actual decoder that incorporates the attention mechanism. The decoder uses an RNN (GRU) and an additive attention module.

### For PyTorch:

```python
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.attention = d2l.AdditiveAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.LazyLinear(vocab_size)
        self.apply(d2l.init_seq2seq)

    def init_state(self, enc_outputs, enc_valid_lens):
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights
```

**Explanation:**
- The decoder's state is initialized with encoder outputs, the final encoder hidden state, and valid lengths.
- At each decoding step, the previous decoder hidden state serves as the query for the attention mechanism.
- The attention context is concatenated with the current input embedding and fed into the RNN.
- The RNN output is collected and passed through a dense layer to produce vocabulary-sized logits.

## Step 4: Testing the Decoder

Let's verify the implementation by creating a small batch and checking the output shapes.

```python
vocab_size, embed_size, num_hiddens, num_layers = 10, 8, 16, 2
batch_size, num_steps = 4, 7

encoder = d2l.Seq2SeqEncoder(vocab_size, embed_size, num_hiddens, num_layers)
decoder = Seq2SeqAttentionDecoder(vocab_size, embed_size, num_hiddens,
                                  num_layers)

X = d2l.zeros((batch_size, num_steps), dtype=torch.long)
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)

print(f'Output shape: {output.shape}')  # Should be (batch_size, num_steps, vocab_size)
print(f'Encoder outputs shape: {state[0].shape}')  # Should be (batch_size, num_steps, num_hiddens)
print(f'Decoder hidden state shape: {state[1][0].shape}')  # Should be (batch_size, num_hiddens)
```

## Step 5: Training the Model

Now, we'll train the full encoder-decoder model with attention on a machine translation dataset.

```python
data = d2l.MTFraEng(batch_size=128)
embed_size, num_hiddens, num_layers, dropout = 256, 256, 2, 0.2

encoder = d2l.Seq2SeqEncoder(
    len(data.src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(data.tgt_vocab), embed_size, num_hiddens, num_layers, dropout)

model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                    lr=0.005)
trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)
trainer.fit(model, data)
```

**Note:** The training process uses the same setup as the standard sequence-to-sequence model, but with the attention-enhanced decoder.

## Step 6: Translating Sentences and Computing BLEU Score

After training, let's use the model to translate a few English sentences to French and evaluate the translations.

```python
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']

preds, _ = model.predict_step(
    data.build(engs, fras), d2l.try_gpu(), data.num_steps)

for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)
    print(f'{en} => {translation}, bleu: {d2l.bleu(" ".join(translation), fr, k=2):.3f}')
```

## Step 7: Visualizing Attention Weights

To understand how the attention mechanism works, we can visualize the attention weights for a specific sentence.

```python
_, dec_attention_weights = model.predict_step(
    data.build([engs[-1]], [fras[-1]]), d2l.try_gpu(), data.num_steps, True)

attention_weights = d2l.concat(
    [step[0][0][0] for step in dec_attention_weights], 0)
attention_weights = d2l.reshape(attention_weights, (1, 1, -1, data.num_steps))

# Visualize the heatmap (excluding the end-of-sequence token)
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key positions', ylabel='Query positions')
```

The heatmap shows how each decoder query (output position) attends to different encoder key positions (input words). You should see non-uniform weights, indicating selective focus.

## Summary

In this tutorial, you implemented the Bahdanau attention mechanism for sequence-to-sequence models. Key takeaways:

- The attention mechanism allows the decoder to dynamically focus on relevant parts of the input sequence.
- The context vector is computed as a weighted sum of encoder hidden states, with weights determined by an alignment model.
- This approach significantly improves the model's ability to handle long sequences compared to fixed-context models.

## Exercises

1. **Experiment with LSTM:** Replace the GRU layers in the decoder with LSTM layers. Observe any changes in performance or training dynamics.

2. **Change the Attention Scoring Function:** Modify the attention mechanism to use the scaled dot-product scoring function instead of additive attention. Compare the training efficiency and translation quality.

---
*This tutorial is based on the D2L.ai chapter on the Bahdanau attention mechanism. For framework-specific discussions, refer to the D2L.ai forums.*