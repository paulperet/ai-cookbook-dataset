# Predicting House Prices on Kaggle: A Practical Guide
:label:`sec_kaggle_house`

This guide walks you through a complete machine learning workflow using a real-world dataset from Kaggle. You'll learn how to preprocess mixed data types, design a model, perform hyperparameter tuning via cross-validation, and submit predictions to a competition.

## Prerequisites

Ensure you have the necessary libraries installed. The code supports multiple deep learning frameworks.

```bash
# Install core data science libraries
pip install pandas matplotlib
# Install your preferred deep learning framework
# pip install torch torchvision
# pip install tensorflow
# pip install mxnet
# pip install jax jaxlib
```

First, import the required modules.

```python
%matplotlib inline
import pandas as pd

# Framework-specific imports
# For PyTorch:
import torch
from torch import nn
from d2l import torch as d2l

# For TensorFlow:
# import tensorflow as tf
# from d2l import tensorflow as d2l

# For MXNet:
# from mxnet import gluon, autograd, init, np, npx
# from mxnet.gluon import nn
# npx.set_np()
# from d2l import mxnet as d2l

# For JAX:
# import jax
# from jax import numpy as jnp
# import numpy as np
# from d2l import jax as d2l
```

## Step 1: Understanding the Competition

The [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition on Kaggle provides a dataset of house sales in Ames, Iowa. Your goal is to predict the final sale price of each home based on 79 explanatory features.

**Key details:**
- **Training data:** 1460 examples with features and the target `SalePrice`
- **Test data:** 1459 examples with features only (for which you must predict prices)
- **Evaluation metric:** Root Mean Squared Logarithmic Error (RMSLE)

## Step 2: Loading the Data

We'll create a custom data module to download and load the Kaggle dataset. The following class handles downloading from a predefined URL and caching the data.

```python
class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            # Download and load training data
            self.raw_train = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_train.csv',
                self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'
            ))
            # Download and load test/validation data
            self.raw_val = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_test.csv',
                self.root,
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'
            ))
```

Let's instantiate the data module and examine its dimensions.

```python
data = KaggleHouse(batch_size=64)
print(f"Training shape: {data.raw_train.shape}")
print(f"Validation/Test shape: {data.raw_val.shape}")
```

**Output:**
```
Training shape: (1460, 81)
Validation/Test shape: (1459, 80)
```

The training data has 81 columns (80 features + 1 target), while the test data has only the 80 features.

## Step 3: Exploring the Raw Data

Before preprocessing, let's inspect the first few rows to understand the data structure.

```python
print(data.raw_train.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])
```

You'll notice the first column is an `Id` (identifier) which should be removed before modeling, as it doesn't contain predictive information. The last column is our target variable `SalePrice`.

## Step 4: Preprocessing the Data

Real-world data requires careful preprocessing. Our dataset contains:
- Numerical features with missing values
- Categorical features
- Features on different scales

We'll implement a comprehensive preprocessing method.

### 4.1 Standardizing Numerical Features

First, we standardize numerical features to zero mean and unit variance. This helps optimization and ensures no feature dominates due to scale.

```python
@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    label = 'SalePrice'
    # Combine train and test features for consistent preprocessing
    features = pd.concat(
        (self.raw_train.drop(columns=['Id', label]),
         self.raw_val.drop(columns=['Id']))
    )
    
    # Standardize numerical columns
    numeric_features = features.dtypes[features.dtypes != 'object'].index
    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std())
    )
    # Replace missing numerical values with 0 (after standardization, mean is 0)
    features[numeric_features] = features[numeric_features].fillna(0)
    
    # One-hot encode categorical features
    features = pd.get_dummies(features, dummy_na=True)
    
    # Split back into training and validation sets
    self.train = features[:self.raw_train.shape[0]].copy()
    self.train[label] = self.raw_train[label]
    self.val = features[self.raw_train.shape[0]:].copy()
```

Apply the preprocessing:

```python
data.preprocess()
print(f"Processed training shape: {data.train.shape}")
```

**Output:**
```
Processed training shape: (1460, 331)
```

One-hot encoding increased our feature count from 79 to 331 (excluding ID and label).

### 4.2 Preparing Data Loaders

We need to create data loaders for training. Since house prices vary widely, we'll predict the logarithm of prices to focus on relative errors.

```python
@d2l.add_to_class(KaggleHouse)
def get_dataloader(self, train):
    label = 'SalePrice'
    data_split = self.train if train else self.val
    if label not in data_split:
        return
    
    # Helper function to convert DataFrame to tensor
    get_tensor = lambda x: d2l.tensor(x.values.astype(float), dtype=d2l.float32)
    
    # Features (X) and log-transformed labels (Y)
    tensors = (
        get_tensor(data_split.drop(columns=[label])),  # X
        d2l.reshape(d2l.log(get_tensor(data_split[label])), (-1, 1))  # Y
    )
    return self.get_tensorloader(tensors, train)
```

## Step 5: Defining the Evaluation Metric

For house prices, relative error matters more than absolute error. A \$100,000 mistake on a \$125,000 house is catastrophic, while the same error on a \$4 million house might be acceptable.

The competition uses Root Mean Squared Logarithmic Error (RMSLE):

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log(y_i + 1) - \log(\hat{y}_i + 1)\right)^2}$$

By predicting log prices, we naturally optimize for this metric.

## Step 6: Model Selection with K-Fold Cross-Validation

We'll use K-fold cross-validation to select hyperparameters and avoid overfitting. First, implement a function to split data into K folds.

```python
def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k
    for j in range(k):
        idx = range(j * fold_size, (j+1) * fold_size)
        # Create fold with j-th segment as validation
        rets.append(KaggleHouse(
            data.batch_size,
            data.train.drop(index=idx),
            data.train.loc[idx]
        ))
    return rets
```

Now implement the cross-validation training loop:

```python
def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        # Initialize a fresh model for each fold
        model = d2l.LinearRegression(lr)
        model.board.yscale = 'log'
        if i != 0:
            model.board.display = False  # Only show plots for first fold
        
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    
    print(f'Average validation log MSE = {sum(val_loss)/len(val_loss)}')
    return models
```

## Step 7: Training and Validation

Let's train a simple linear regression model as our baseline. We'll use 5-fold cross-validation.

```python
trainer = d2l.Trainer(max_epochs=10)
models = k_fold(trainer, data, k=5, lr=0.01)
```

The average validation error gives us a reliable estimate of how our model performs on unseen data. If training error is much lower than validation error, we're overfitting and might need regularization or a simpler model.

## Step 8: Making Predictions for Submission

After selecting our model through cross-validation, we generate predictions on the test set by averaging predictions from all K models.

```python
# Generate predictions from each model
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = [model(d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
             for model in models]

# For JAX, the syntax differs slightly
if tab.selected('jax'):
    preds = [model.apply({'params': trainer.state.params},
             d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
             for model in models]

# Average predictions and convert from log scale back to prices
ensemble_preds = d2l.reduce_mean(d2l.exp(d2l.concat(preds, 1)), 1)

# Create submission file
submission = pd.DataFrame({
    'Id': data.raw_val.Id,
    'SalePrice': d2l.numpy(ensemble_preds)
})
submission.to_csv('submission.csv', index=False)
```

## Step 9: Submitting to Kaggle

1. Log in to your Kaggle account and navigate to the [House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
2. Click "Submit Predictions" or "Late Submission"
3. Upload your `submission.csv` file
4. Click "Make Submission" to see your score on the leaderboard

## Summary

In this tutorial, you've implemented a complete pipeline for a Kaggle competition:

1. **Data loading and inspection:** Understanding the structure of real-world data
2. **Preprocessing:** Handling missing values, standardizing numerical features, and encoding categorical variables
3. **Model design:** Starting with a simple linear regression baseline
4. **Validation:** Using K-fold cross-validation for reliable performance estimation
5. **Prediction and submission:** Creating competition-ready output files

## Next Steps and Exercises

1. **Improve your score:** Submit your predictions. How does your linear model compare to the leaderboard?
2. **Handle missing data thoughtfully:** Consider when mean imputation might be inappropriate. What if values aren't missing at random?
3. **Tune hyperparameters:** Experiment with learning rates, batch sizes, and training epochs using cross-validation.
4. **Try more complex models:** Implement neural networks with layers, dropout, and weight decay.
5. **Ablation study:** What happens if you skip feature standardization? Test this by modifying the preprocessing step.

By working through these exercises, you'll develop deeper intuition for what matters in practical machine learning projects.