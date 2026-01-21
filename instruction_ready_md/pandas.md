# Data Preprocessing with Pandas

In this guide, you'll learn how to load and preprocess real-world data using the pandas library. We'll cover reading CSV files, handling missing values, and converting data into a tensor format suitable for machine learning.

## Prerequisites

First, ensure you have the necessary libraries installed. We'll use `pandas` for data manipulation and your chosen deep learning framework for tensor operations.

```bash
pip install pandas
```

Depending on your chosen framework, install one of:
- `mxnet`
- `pytorch`
- `tensorflow`
- `jax`

Now, let's import the required modules.

```python
import os
import pandas as pd
```

## Step 1: Create a Sample Dataset

We'll start by creating a small CSV file to simulate a housing dataset. This file contains three columns: `NumRooms` (number of rooms), `RoofType` (type of roof), and `Price` (house price). Some entries are intentionally missing.

```python
# Create the data directory if it doesn't exist
os.makedirs(os.path.join('..', 'data'), exist_ok=True)

# Define the file path
data_file = os.path.join('..', 'data', 'house_tiny.csv')

# Write the CSV data
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
```

## Step 2: Load the Dataset with Pandas

Use pandas' `read_csv` function to load the data into a DataFrame. This function automatically parses the CSV structure and converts missing values (`NA`) to `NaN` (Not a Number).

```python
data = pd.read_csv(data_file)
print(data)
```

**Output:**
```
   NumRooms RoofType   Price
0       NaN      NaN  127500
1       2.0      NaN  106000
2       4.0    Slate  178100
3       NaN      NaN  140000
```

## Step 3: Separate Inputs and Targets

In supervised learning, we separate the features (inputs) from the target variable we want to predict. Here, `Price` is our target, while `NumRooms` and `RoofType` are inputs.

```python
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
print("Inputs:")
print(inputs)
print("\nTargets:")
print(targets)
```

## Step 4: Handle Missing Values

Missing data is common in real-world datasets. We'll handle it using two techniques: one for categorical data and another for numerical data.

### 4.1 Impute Categorical Variables

For the categorical column `RoofType`, we can treat `NaN` as its own category using `pd.get_dummies`. This creates binary columns for each category, including `NaN`.

```python
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

**Output:**
```
   NumRooms  RoofType_Slate  RoofType_nan
0       NaN               0             1
1       2.0               0             1
2       4.0               1             0
3       NaN               0             1
```

### 4.2 Impute Numerical Variables

For the numerical column `NumRooms`, we'll replace `NaN` values with the column's mean.

```python
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

**Output:**
```
   NumRooms  RoofType_Slate  RoofType_nan
0       3.0               0             1
1       2.0               0             1
2       4.0               1             0
3       3.0               0             1
```

## Step 5: Convert to Tensor Format

Now that all values are numerical, we can convert the pandas DataFrames into tensors for use in deep learning frameworks.

### PyTorch

```python
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
print("Input tensor X:", X)
print("Target tensor y:", y)
```

### TensorFlow

```python
import tensorflow as tf

X = tf.constant(inputs.to_numpy(dtype=float))
y = tf.constant(targets.to_numpy(dtype=float))
print("Input tensor X:", X)
print("Target tensor y:", y)
```

### MXNet

```python
from mxnet import np

X = np.array(inputs.to_numpy(dtype=float))
y = np.array(targets.to_numpy(dtype=float))
print("Input tensor X:", X)
print("Target tensor y:", y)
```

### JAX

```python
from jax import numpy as jnp

X = jnp.array(inputs.to_numpy(dtype=float))
y = jnp.array(targets.to_numpy(dtype=float))
print("Input tensor X:", X)
print("Target tensor y:", y)
```

## Summary

You've successfully loaded a CSV dataset, separated inputs from targets, handled missing values through imputation, and converted the data into tensor format. These are foundational steps in any data preprocessing pipeline.

## Next Steps

- Explore larger datasets from sources like the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets).
- Practice selecting columns by name using pandas' `loc` and `iloc` methods.
- Investigate advanced data types (text, images) and libraries like `Pillow` for image processing.

Remember, data quality is critical—always visualize and inspect your data for outliers or errors before training models.