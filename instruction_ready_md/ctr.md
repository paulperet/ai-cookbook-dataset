# Building a Feature-Rich Recommender System for Click-Through Rate Prediction

## Introduction

Interaction data, such as clicks and views, is fundamental for understanding user preferences. However, this data is often sparse and noisy. To build more robust and accurate recommendation models, we can incorporate side information—features about items, user profiles, and the context of the interaction. This approach is especially valuable for predicting Click-Through Rate (CTR), a critical metric in online advertising.

CTR measures the percentage of users who click on a specific link or advertisement. It's calculated as:

$$ \textrm{CTR} = \frac{\#\textrm{Clicks}} {\#\textrm{Impressions}} \times 100 \% .$$

Accurate CTR prediction is essential not only for targeted advertising but also for general recommender systems, email campaigns, and search engines. In this guide, you will learn how to build a data pipeline for a CTR prediction task using an anonymized advertising dataset.

## Prerequisites

This tutorial uses MXNet and the D2L library. Ensure you have the necessary packages installed.

```bash
pip install mxnet d2l
```

## Step 1: Import Required Libraries

We'll start by importing the necessary modules for data handling and processing.

```python
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## Step 2: Download the Dataset

The dataset we'll use is an anonymized online advertising dataset. It contains 34 categorical feature fields and a binary target variable indicating a click (1) or no click (0). The real semantics of the features are hidden for privacy.

First, we register the dataset URL with the D2L data hub and download it to a local directory.

```python
# Register and download the dataset
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

The dataset contains a training set (15,000 samples) and a test set (3,000 samples).

## Step 3: Create a Custom Dataset Class

To efficiently load and preprocess the data, we'll create a custom `CTRDataset` class that inherits from `gluon.data.Dataset`. This class will handle:
*   Reading the CSV file.
*   Mapping categorical features to integer indices.
*   Filtering out rare feature values (below a minimum threshold).
*   Preparing the data in a format suitable for model training.

```python
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)

        # Read and process the data file
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                # Create label: [0, 1] for click, [1, 0] for no-click
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                # Store feature values and count their occurrences
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1

        # If no mapper/defaults provided, create them by filtering rare features
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}

        # Calculate field dimensions (unique values per feature + 1 for default)
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        # Precompute offsets for embedding lookup in a potential model
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy()[:-1]))

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        # Convert categorical strings to integer indices, using default index for unseen/rare values
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        # Add offsets to create a global index for each feature value
        return feat + self.offsets, self.data[idx]['y']
```

**Key operations of the `__getitem__` method:**
1.  It retrieves a data instance.
2.  Each categorical feature value is looked up in the `feat_mapper` dictionary. If the value was seen frequently during initialization, it gets a unique integer ID. If it was rare or unseen, it's assigned a default ID.
3.  The feature indices are then shifted by a precomputed `offsets` array. This is a common technique for factorization machines or embedding-based models, where each feature field gets a contiguous block of indices in a combined embedding table.

## Step 4: Load and Inspect the Training Data

Now, let's instantiate the dataset for the training split and examine the first sample.

```python
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
sample_feat, sample_label = train_data[0]
print('Processed Feature Indices:', sample_feat)
print('Label:', sample_label)
```

**Expected Output:**
```
Processed Feature Indices: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33]
Label: [0.0]
```

The output shows the 34 processed feature indices for the first sample. The label `[0.0]` indicates this advertisement was not clicked. The `CTRDataset` is now ready to be used with a `DataLoader` for batch training.

## Summary

In this guide, you have:
1.  Understood the importance of side features and CTR prediction in recommender systems.
2.  Downloaded an anonymized advertising dataset.
3.  Built a custom `CTRDataset` class that efficiently loads, preprocesses, and encodes categorical features for CTR prediction, framing it as a binary classification problem.

This data pipeline is a foundational step. The processed data, where each categorical feature is mapped to a unique integer index, is the standard input format for many advanced CTR prediction models like Factorization Machines, DeepFM, and DCN.

## Exercises

1.  **Adapt to Other Datasets:** The `CTRDataset` class can be adapted for other popular CTR benchmarks. Try loading the [Criteo Display Advertising Challenge Dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/). Note that the Criteo dataset contains both categorical and real-valued numerical features. How would you modify the `__init__` and `__getitem__` methods to handle continuous features?

2.  **Explore the Avazu Dataset:** Attempt to load the [Avazu Click-Through Rate Prediction Dataset](https://www.kaggle.com/c/avazu-ctr-prediction) using the same class. What adjustments are needed for the different data format or feature counts?

---
*For further discussion on this topic, visit the [D2L forum](https://discuss.d2l.ai/t/405).*