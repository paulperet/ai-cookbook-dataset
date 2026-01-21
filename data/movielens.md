# Working with the MovieLens 100K Dataset for Recommendation Systems

This guide walks you through loading, exploring, and preparing the classic MovieLens 100K dataset for building recommendation models.

## Prerequisites

First, ensure you have the necessary packages installed. We'll use `d2l` (Dive into Deep Learning utilities), `mxnet`, `pandas`, and `os`.

```python
# If you haven't installed d2l, you can do so via pip:
# !pip install d2l

from d2l import mxnet as d2l
from mxnet import gluon, np
import os
import pandas as pd
```

## Step 1: Download and Load the Dataset

We'll start by defining a helper function to download the MovieLens 100K dataset and load it into a pandas DataFrame.

```python
# Register the dataset URL with d2l's data hub
d2l.DATA_HUB['ml-100k'] = (
    'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

def read_data_ml100k():
    """
    Downloads and extracts the MovieLens 100K dataset.
    Returns:
        data (pd.DataFrame): The loaded rating data.
        num_users (int): Number of unique users.
        num_items (int): Number of unique items.
    """
    # Download and extract the dataset
    data_dir = d2l.download_extract('ml-100k')
    # Define column names for the main ratings file (u.data)
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    # Load the data. The file is tab-separated.
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t',
                       names=names, engine='python')
    # Calculate the number of unique users and items
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items
```

Now, let's load the data and inspect its basic statistics.

```python
# Load the dataset
data, num_users, num_items = read_data_ml100k()

# Calculate the sparsity of the user-item interaction matrix
sparsity = 1 - len(data) / (num_users * num_items)

print(f'Number of users: {num_users}')
print(f'Number of items: {num_items}')
print(f'Matrix sparsity: {sparsity:.3%}')
print('\nFirst five records:')
print(data.head())
```

**Output:**
```
Number of users: 943
Number of items: 1682
Matrix sparsity: 93.695%

First five records:
   user_id  item_id  rating  timestamp
0      196      242       3  881250949
1      186      302       3  891717742
2       22      377       1  878887116
3      244       51       2  880606923
4      166      346       1  886397596
```

The output shows we have 943 users and 1682 movies. The interaction matrix is over 93% sparse, meaning most user-movie pairs are unrated—a common challenge in recommendation systems.

## Step 2: Explore the Rating Distribution

Understanding the distribution of ratings helps gauge user behavior. Let's visualize it.

```python
# Plot a histogram of the ratings
d2l.plt.hist(data['rating'], bins=5, ec='black')
d2l.plt.xlabel('Rating')
d2l.plt.ylabel('Count')
d2l.plt.title('Distribution of Ratings in MovieLens 100K')
d2l.plt.show()
```

This will display a histogram showing ratings are roughly normally distributed, centered around 3-4 stars.

## Step 3: Split the Dataset

To evaluate our models, we need to split the data into training and test sets. We'll define a function supporting two modes: `random` and `seq-aware`.

*   **Random Split:** Shuffles interactions randomly. (Default: 90% train, 10% test).
*   **Seq-aware Split:** For each user, their most recent interaction (by timestamp) is held out for testing. This mode is crucial for sequence-aware or next-item prediction tasks.

```python
def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """
    Splits the dataset into training and test sets.
    Args:
        data (pd.DataFrame): The loaded rating data.
        num_users (int): Number of users.
        num_items (int): Number of items.
        split_mode (str): 'random' or 'seq-aware'.
        test_ratio (float): Proportion of data to use for testing.
    Returns:
        train_data, test_data (pd.DataFrame): The split datasets.
    """
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        # Organize data per user
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            # Keep the most recent item for the test set
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        # Create training list sorted by timestamp for each user
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        # Format test and train data
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else: # random split
        # Create a random mask for the training set
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data
```

## Step 4: Load Data for Model Training

After splitting, we need to convert the DataFrames into formats suitable for model input (lists and interaction matrices). This function handles both explicit (actual ratings 1-5) and implicit (binary, 1 for interaction) feedback.

```python
def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    """
    Converts DataFrame into lists and an interaction matrix.
    Args:
        data (pd.DataFrame): Input data.
        num_users (int): Number of users.
        num_items (int): Number of items.
        feedback (str): 'explicit' or 'implicit'.
    Returns:
        users, items, scores (list): Lists of indices and ratings.
        inter (np.ndarray or dict): Interaction matrix or dictionary.
    """
    users, items, scores = [], [], []
    # Initialize the interaction structure
    if feedback == 'explicit':
        inter = np.zeros((num_items, num_users))
    else:
        inter = {} # For implicit feedback, use a dictionary

    for line in data.itertuples():
        # Convert to zero-based indices
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        # Populate the interaction structure
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter
```

## Step 5: Create a Complete Data Pipeline

Finally, we combine all steps into a single pipeline function that returns data loaders ready for training.

```python
def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    """
    Complete pipeline: Download, split, and load the MovieLens 100K data.
    Args:
        split_mode (str): 'random' or 'seq-aware'.
        feedback (str): 'explicit' or 'implicit'.
        test_ratio (float): Test set ratio.
        batch_size (int): Batch size for DataLoader.
    Returns:
        num_users, num_items (int): Dataset dimensions.
        train_iter, test_iter (DataLoader): Data loaders for training and testing.
    """
    # 1. Load raw data
    data, num_users, num_items = read_data_ml100k()
    # 2. Split the data
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    # 3. Convert to lists/matrices
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    # 4. Create MXNet Datasets
    train_set = gluon.data.ArrayDataset(
        np.array(train_u), np.array(train_i), np.array(train_r))
    test_set = gluon.data.ArrayDataset(
        np.array(test_u), np.array(test_i), np.array(test_r))
    # 5. Create DataLoaders
    # Training loader shuffles data and rolls over the last incomplete batch.
    train_iter = gluon.data.DataLoader(
        train_set, shuffle=True, last_batch='rollover',
        batch_size=batch_size)
    test_iter = gluon.data.DataLoader(test_set, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter
```

You can now use this function to get data loaders for your model. For example, to get data for an explicit feedback model with a sequence-aware split:

```python
num_users, num_items, train_iter, test_iter = split_and_load_ml100k(
    split_mode='seq-aware', feedback='explicit', batch_size=128)
```

## Summary

In this guide, you learned how to:
1.  Download and load the MovieLens 100K dataset.
2.  Analyze its key statistics, including high sparsity.
3.  Split the data using both random and sequence-aware methods.
4.  Process the data into formats suitable for training recommendation models with explicit or implicit feedback.
5.  Create a complete, reusable data pipeline that outputs DataLoader objects.

The MovieLens dataset is a foundational resource for recommendation system research and development. The functions defined here will be used as building blocks in subsequent tutorials for building collaborative filtering and sequence-aware models.

## Next Steps & Exercises

*   **Explore Other Datasets:** Investigate similar public datasets like Amazon Reviews, Netflix Prize, or Yelp for recommendation tasks.
*   **Visit MovieLens:** Go to the [MovieLens website](https://movielens.org/) to understand the context of the data and its real-world application.
*   **Experiment:** Try changing the `split_mode` and `feedback` parameters in the final pipeline to see how the prepared data changes.