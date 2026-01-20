# Annotate text data using Active Learning with Cleanlab

Authored by: [Aravind Putrevu](https://huggingface.co/aravindputrevu)

In this notebook, I highlight the use of [active learning](https://arxiv.org/abs/2301.11856) to improve a fine-tuned Hugging Face Transformer for text classification, while keeping the total number of collected labels from human annotators low. When resource constraints prevent you from acquiring labels for the entirety of your data, active learning aims to save both time and money by selecting which examples data annotators should spend their effort labeling.

## What is Active Learning?

Active Learning helps prioritize what data to label in order to maximize the performance of a supervised machine learning model trained on the labeled data. This process usually happens iteratively — at each round, active learning tells us which examples we should collect additional annotations for to improve our current model the most under a limited labeling budget. [ActiveLab](https://arxiv.org/abs/2301.11856) is an active learning algorithm that is particularly useful when the labels coming from human annotators are noisy and when we should collect one more annotation for a previously annotated example (whose label seems suspect) vs. for a not-yet-annotated example.  After collecting these new annotations for a batch of data to increase our training dataset, we re-train our model and evaluate its test accuracy.

Active learning with ActiveLab is much better than random selection when it comes to collecting additional annotations for Transformer models. It consistently produces much better models with approximately 50% less error rate, regardless of the total labeling budget.

The rest of this notebook walks through the open-source code you can use to achieve these results.

## Setting up the environment


```python
!pip install datasets==2.20.0 transformers==4.25.1 scikit-learn==1.1.2 matplotlib==3.5.3 cleanlab
```


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

## Collecting and Organizing Data

Here we download the data that we need for this notebook.


```python
labeled_data_file = {"labeled": "X_labeled_full.csv"}
unlabeled_data_file = {"unlabeled": "X_labeled_full.csv"}
test_data_file = {"test": "test.csv"}

X_labeled_full = load_dataset("Cleanlab/stanford-politeness", split="labeled", data_files=labeled_data_file)
X_unlabeled = load_dataset("Cleanlab/stanford-politeness", split="unlabeled", data_files=unlabeled_data_file)
test = load_dataset("Cleanlab/stanford-politeness", split="test", data_files=test_data_file)

!wget -nc -O 'extra_annotations.npy' 'https://huggingface.co/datasets/Cleanlab/stanford-politeness/resolve/main/extra_annotations.npy?download=true'

extra_annotations = np.load("extra_annotations.npy",allow_pickle=True).item()
```


```python
X_labeled_full = X_labeled_full.to_pandas()
X_labeled_full.set_index('id', inplace=True)
X_unlabeled = X_unlabeled.to_pandas()
X_unlabeled.set_index('id', inplace=True)
test = test.to_pandas()
```

## Classifying the Politeness of Text

We are using [Stanford Politeness Corpus](https://convokit.cornell.edu/documentation/wiki_politeness.html) as the Dataset.

It is structured as a binary text classification task, to classify whether each phrase is polite or impolite. Human annotators are given a selected text phrase and they provide an (imperfect) annotation regarding its politeness: **0** for impolite and **1** for polite.

Training a Transformer classifier on the annotated data, we measure model accuracy over a set of held-out test examples, where I feel confident about their ground truth labels because they are derived from a consensus amongst 5 annotators who labeled each of these examples.

As for the training data, we have:

- `X_labeled_full`: our initial training set with just a small set of 100 text examples labeled with 2 annotations per example.
- `X_unlabeled`: large set of 1900 unlabeled text examples we can consider having annotators label.
- `extra_annotations`: pool of additional annotations we pull from when an annotation is requested for an example

## Visualize Data


```python
# Multi-annotated Data
X_labeled_full.head()
```





  <div id="df-3a720cd1-6b98-47c0-be0c-0a8c328eff10" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>a6</th>
      <th>a12</th>
      <th>a16</th>
      <th>a19</th>
      <th>a20</th>
      <th>a22</th>
      <th>a39</th>
      <th>a42</th>
      <th>a52</th>
      <th>...</th>
      <th>a157</th>
      <th>a158</th>
      <th>a178</th>
      <th>a180</th>
      <th>a185</th>
      <th>a193</th>
      <th>a196</th>
      <th>a197</th>
      <th>a215</th>
      <th>a216</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>450d326d</th>
      <td>&lt;url&gt;. Congrats, or should I say good luck?</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6a22f4ec</th>
      <td>Can I get some time to finish what I am doing without everything being deleted??</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>823f1104</th>
      <td>Ok. Thank you for clarifying. Could you be more specific as to what you are specifying as "the claim" so that I may find relevant information to refute?</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7677905a</th>
      <td>One wonders, of course, who "Elliott of Macedon" would have been. Probably something analogous to Brian of Nazareth but in a Macedonian phalanx?</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>a1ce799b</th>
      <td>So, let me make sure I understand this. You think that, if we remove an image as it does not meet the NFCC, you would then be able to upload the same image, only this time, it would meet the NFCC?</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>





```python
# Unlabeled Data
X_unlabeled.head()
```





  <div id="df-50691c9d-9a9c-40bd-acde-bf5c4bfad34b" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>486aff36</th>
      <td>The review has been up there for something like six weeks, I notice. Think you'll be able to take care of those last couple of things?</td>
    </tr>
    <tr>
      <th>201d7655</th>
      <td>How many other states follow the same pattern?  And do we really need it to?</td>
    </tr>
    <tr>
      <th>c9125774</th>
      <td>You added the name Ken Taylor to the &lt;url&gt; page but there is no such person listed on the DOD website as having received that award. Who were you refering to?</td>
    </tr>
    <tr>
      <th>593ac8fb</th>
      <td>I found &lt;url&gt; whilst looking for something else. Any use to you?</td>
    </tr>
    <tr>
      <th>d1fdcdba</th>
      <td>If it were me I'd want to try and find out more about how/why this happened first before I continued to use that software. Have you asked at the talk page I mentioned above?</td>
    </tr>
  </tbody>
</table>
</div>





```python
# extra_annotations contains the annotations that we will use when an additional annotation is requested.
extra_annotations

# Random sample of extra_annotations to see format.
{k:extra_annotations[k] for k in random.sample(extra_annotations.keys(), 5)}
```

    <ipython-input-6-d9c8ad254414>:5: DeprecationWarning: Sampling from a set deprecated
    since Python 3.9 and will be removed in a subsequent version.
      {k:extra_annotations[k] for k in random.sample(extra_annotations.keys(), 5)}





    {'4235a537': {'a6': 0.0, 'a12': 0.0, 'a98': 0.0, 'a99': 0.0, 'a119': 0.0},
     '3d961d64': {'a68': 0.0, 'a70': 0.0, 'a79': 0.0, 'a99': 0.0, 'a199': 1.0},
     '4a5e75dc': {'a60': 1.0, 'a102': 1.0, 'a130': 1.0, 'a148': 1.0, 'a174': 1.0},
     '369a8b74': {'a65': 1.0, 'a68': 1.0, 'a71': 1.0, 'a157': 0.0, 'a161': 0.0},
     '356a4a74': {'a61': 0.0, 'a70': 1.0, 'a139': 0.0, 'a145': 1.0, 'a198': 1.0}}



# View Some Examples From Test Set


```python
num_to_label = {0:'Impolite', 1:"Polite"}
for i in range(2):
    print(f"{num_to_label[i]} examples:")
    subset=test[test.label==i][['text']].sample(n=3, random_state=2)
    print(subset)
```

    Impolite examples:




  <div id="df-41902311-0bdf-4503-8ea1-f7c599550be4" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>120</th>
      <td>And wasting our time as well. I can only repeat: why don't you do constructive work by adding contents about your beloved Makedonia?</td>
    </tr>
    <tr>
      <th>150</th>
      <td>Rather than tell me how wrong I was to close certain afd's maybe your time would be better spent dealing with the current afd backlog &lt;url&gt;. If my decisions were so wrong why haven't you re-opened them?</td>
    </tr>
    <tr>
      <th>326</th>
      <td>This was supposed to have been moved to &lt;url&gt; per the CFD. Why wasn't it moved?</td>
    </tr>
  </tbody>
</table>
</div>



    Polite examples:




  <div id="df-32540cf4-ec14-46c7-a81a-83647ef26abf" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</