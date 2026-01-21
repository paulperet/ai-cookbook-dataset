# Annotate Text Data Using Active Learning with Cleanlab

In this guide, you will learn how to use **active learning** to efficiently improve a fine-tuned Hugging Face Transformer for text classification while minimizing the number of human annotations required. When labeling resources are limited, active learning helps prioritize which data points annotators should label to maximize model performance.

## What is Active Learning?

Active Learning is an iterative process that selects the most informative data points for annotation to improve a supervised model under a fixed labeling budget. The [ActiveLab](https://arxiv.org/abs/2301.11856) algorithm is particularly effective when dealing with noisy human annotations, as it intelligently decides whether to collect an additional label for a previously annotated example (if its current label seems suspect) or for a new, unlabeled example. After gathering new annotations for a batch of data, the model is retrained and evaluated.

ActiveLab consistently outperforms random selection, often reducing error rates by approximately 50% across various labeling budgets.

## Prerequisites

First, install the required libraries.

```bash
pip install datasets==2.20.0 transformers==4.25.1 scikit-learn==1.1.2 matplotlib==3.5.3 cleanlab
```

Now, import the necessary modules.

```python
import pandas as pd
pd.set_option('max_colwidth', None)
import numpy as np
import random
import transformers
import datasets
import matplotlib.pyplot as plt

from cleanlab.multiannotator import get_majority_vote_label, get_active_learning_scores, get_label_quality_multiannotator
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, Dataset, DatasetDict, ClassLabel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from scipy.special import softmax
from datetime import datetime
```

## Step 1: Load and Organize the Dataset

We'll use the **Stanford Politeness Corpus**, a binary text classification task where each phrase is labeled as polite (1) or impolite (0). The dataset includes:

- **`X_labeled_full`**: An initial training set of 100 text examples, each with 2 annotations.
- **`X_unlabeled`**: A large pool of 1900 unlabeled text examples available for annotation.
- **`test`**: A held-out test set with high-confidence labels derived from a consensus of 5 annotators per example.
- **`extra_annotations`**: A pool of additional annotations to simulate the collection of new labels.

Download and load the data.

```python
labeled_data_file = {"labeled": "X_labeled_full.csv"}
unlabeled_data_file = {"unlabeled": "X_labeled_full.csv"}
test_data_file = {"test": "test.csv"}

X_labeled_full = load_dataset("Cleanlab/stanford-politeness", split="labeled", data_files=labeled_data_file)
X_unlabeled = load_dataset("Cleanlab/stanford-politeness", split="unlabeled", data_files=unlabeled_data_file)
test = load_dataset("Cleanlab/stanford-politeness", split="test", data_files=test_data_file)

# Download the extra annotations file
!wget -nc -O 'extra_annotations.npy' 'https://huggingface.co/datasets/Cleanlab/stanford-politeness/resolve/main/extra_annotations.npy?download=true'

extra_annotations = np.load("extra_annotations.npy", allow_pickle=True).item()
```

Convert the datasets to pandas DataFrames for easier manipulation.

```python
X_labeled_full = X_labeled_full.to_pandas()
X_labeled_full.set_index('id', inplace=True)
X_unlabeled = X_unlabeled.to_pandas()
X_unlabeled.set_index('id', inplace=True)
test = test.to_pandas()
```

## Step 2: Explore the Data

Let's examine the structure of each dataset.

### Multi-Annotated Training Data (`X_labeled_full`)

Each row represents a text example, and columns `a6`, `a12`, etc., contain annotations from different annotators (NaN indicates a missing annotation).

```python
X_labeled_full.head()
```

### Unlabeled Data (`X_unlabeled`)

This DataFrame contains only the text column.

```python
X_unlabeled.head()
```

### Extra Annotations Pool (`extra_annotations`)

This dictionary maps example IDs to dictionaries of additional annotations from various annotators.

```python
# View a random sample
{k: extra_annotations[k] for k in random.sample(list(extra_annotations.keys()), 5)}
```

### Test Set Examples

View a few examples from the test set to understand the classification task.

```python
num_to_label = {0: 'Impolite', 1: 'Polite'}
for i in range(2):
    print(f"{num_to_label[i]} examples:")
    subset = test[test.label == i][['text']].sample(n=3, random_state=2)
    print(subset)
```

## Next Steps

Now that the data is loaded and inspected, you are ready to proceed with the active learning workflow. The subsequent steps will involve:

1. **Training an initial classifier** on the small labeled dataset.
2. **Using ActiveLab** to score unlabeled examples and select which ones to annotate next.
3. **Simulating annotation collection** from the `extra_annotations` pool.
4. **Retraining the model** with the expanded dataset and evaluating its performance on the test set.

This iterative process will demonstrate how active learning can significantly improve model accuracy while minimizing annotation effort.