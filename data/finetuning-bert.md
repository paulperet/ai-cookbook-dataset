# Fine-Tuning BERT for Sequence-Level and Token-Level Applications

## Introduction

In previous chapters, we explored various neural network architectures for natural language processing (NLP), including RNNs, CNNs, attention mechanisms, and MLPs. While these models are valuable, designing a specialized architecture for every NLP task is impractical. BERT (Bidirectional Encoder Representations from Transformers), introduced in earlier sections, offers a powerful solution: a single pretrained model that can be adapted with minimal changes to a wide range of NLP problems.

This guide walks through the practical process of fine-tuning BERT for both sequence-level and token-level applications. We'll cover the architectural adjustments needed and how BERT's input representations are transformed for different tasks.

## Prerequisites

Before starting, ensure you have the necessary libraries installed. This tutorial assumes you have a working environment with PyTorch or TensorFlow and the Hugging Face `transformers` library.

```bash
pip install torch transformers
```

## 1. Single Text Classification

Single text classification involves taking one text sequence as input and predicting a categorical label. Common examples include sentiment analysis (positive/negative) and grammatical acceptability judgment.

### How BERT Handles Single Text

BERT uses special tokens to structure its input:
- `[CLS]`: A classification token prepended to every input sequence. Its final hidden state is used as the aggregate sequence representation for classification tasks.
- `[SEP]`: A separator token that marks the end of a single text or separates two text segments.

For single text classification, the entire input sequence is fed into BERT, and the representation corresponding to the `[CLS]` token is extracted. This representation captures information about the whole sequence.

### Architecture Adjustment

To adapt BERT for classification, we add a small multilayer perceptron (MLP) on top of the BERT encoder:

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertForSequenceClassification(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation for classification
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Pass through classifier
        logits = self.classifier(cls_embedding)
        return logits
```

### Fine-Tuning Process

During fine-tuning:
1. The parameters of the pretrained BERT model are updated (fine-tuned)
2. The parameters of the newly added classification layers are learned from scratch
3. The model is trained end-to-end on your specific classification dataset

## 2. Text Pair Classification or Regression

Text pair applications involve two text sequences as input. Examples include:
- **Natural language inference**: Determining if a hypothesis entails, contradicts, or is neutral to a premise
- **Semantic textual similarity**: Predicting a similarity score between two sentences (regression task)

### Input Representation for Text Pairs

For text pairs, BERT uses the following structure:
```
[CLS] Sentence A [SEP] Sentence B [SEP]
```

The `[CLS]` token representation still serves as the aggregate representation for the entire input pair.

### Architecture Adjustment

The architecture is similar to single text classification, but the input formatting differs:

```python
class BertForTextPairClassification(nn.Module):
    def __init__(self, bert_model_name, num_labels, regression=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.regression = regression
        
        if regression:
            # For regression tasks (e.g., similarity scoring)
            self.regressor = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1)  # Single continuous output
            )
        else:
            # For classification tasks
            self.classifier = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_labels)
            )
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        if self.regression:
            return self.regressor(cls_embedding).squeeze(-1)
        else:
            return self.classifier(cls_embedding)
```

### Loss Functions

- For classification: Cross-entropy loss
- For regression: Mean squared error loss

## 3. Text Tagging (Token-Level Classification)

Text tagging assigns a label to each token in the input sequence. Common applications include:
- Part-of-speech tagging
- Named entity recognition
- Chunking

### Architecture Adjustment

Unlike sequence classification where we use only the `[CLS]` token, for token-level tasks we process the representation of every token:

```python
class BertForTokenClassification(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get hidden states for all tokens
        sequence_output = outputs.last_hidden_state
        
        # Apply classifier to each token
        logits = self.classifier(sequence_output)
        return logits
```

### Handling Special Tokens

When computing the loss for token classification, we typically mask out the special tokens (`[CLS]`, `[SEP]`, `[PAD]`) since they don't correspond to actual words in the original text.

## 4. Question Answering

Question answering involves predicting a text span within a passage that answers a given question. The Stanford Question Answering Dataset (SQuAD) is a popular benchmark for this task.

### Input Representation

The question and passage are combined as:
```
[CLS] Question [SEP] Passage [SEP]
```

### Architecture Adjustment

For question answering, we need to predict both the start and end positions of the answer span within the passage:

```python
class BertForQuestionAnswering(nn.Module):
    def __init__(self, bert_model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Separate classifiers for start and end positions
        self.qa_outputs = nn.Linear(hidden_size, 2)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Predict start and end logits for each token
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
```

### Training Objective

The training objective maximizes the log-likelihood of the correct start and end positions:

```python
def compute_qa_loss(start_logits, end_logits, start_positions, end_positions):
    # Ignore positions with -100 (usually padding or non-passage tokens)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    
    return (start_loss + end_loss) / 2
```

### Inference Strategy

During inference, we find the span with the highest combined score:

```python
def predict_span(start_logits, end_logits, max_answer_length=30):
    # Convert logits to probabilities
    start_probs = torch.softmax(start_logits, dim=-1)
    end_probs = torch.softmax(end_logits, dim=-1)
    
    # Find the best span
    best_score = -float('inf')
    best_span = (0, 0)
    
    # Consider all possible start and end positions
    for i in range(len(start_probs)):
        for j in range(i, min(i + max_answer_length, len(end_probs))):
            score = start_probs[i] * end_probs[j]
            if score > best_score:
                best_score = score
                best_span = (i, j)
    
    return best_span
```

## 5. Fine-Tuning Best Practices

### Learning Rate Scheduling

BERT fine-tuning typically uses a smaller learning rate than training from scratch:

```python
from transformers import AdamW, get_linear_schedule_with_warmup

def setup_optimizer(model, train_dataloader, epochs, learning_rate=2e-5):
    # Prepare optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    
    # Setup learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler
```

### Gradient Accumulation

For large models or limited GPU memory, use gradient accumulation:

```python
accumulation_steps = 4
optimizer.zero_grad()

for step, batch in enumerate(train_dataloader):
    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss
    
    # Normalize loss for gradient accumulation
    loss = loss / accumulation_steps
    loss.backward()
    
    # Update weights only after accumulating gradients
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

## Summary

Fine-tuning BERT for downstream NLP tasks involves:

1. **Understanding the task type**: Determine if it's sequence-level (classification/regression) or token-level (tagging/QA)
2. **Adjusting the architecture**: Add task-specific layers on top of BERT
3. **Formatting inputs correctly**: Use `[CLS]` and `[SEP]` tokens appropriately
4. **Setting up optimization**: Use appropriate learning rates and schedules
5. **Training end-to-end**: Fine-tune all BERT parameters while learning new layers from scratch

The key advantage of BERT is its versatility—the same pretrained model can be adapted with minimal architectural changes to solve diverse NLP problems, often achieving state-of-the-art results with relatively little task-specific data.

## Next Steps

1. Experiment with different BERT variants (RoBERTa, DistilBERT, ALBERT)
2. Try multi-task learning by fine-tuning on related tasks simultaneously
3. Explore domain-adaptive pretraining for specialized applications
4. Implement ensemble methods combining multiple fine-tuned models

Remember that while this guide provides the foundational patterns, each specific application may require additional task-specific adjustments and optimizations.