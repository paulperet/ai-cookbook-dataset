# Synthetic Regression Data: A Practical Guide
:label:`sec_synthetic-regression-data`

Machine learning extracts patterns from data. While synthetic data might seem artificial—since we define its underlying patterns ourselves—it's an invaluable tool for learning and validation. By generating data where the *true* parameters are known, we can verify that our models and algorithms work correctly and can recover those parameters.

This guide walks you through creating, loading, and inspecting synthetic data for linear regression, demonstrating core principles applicable to real-world datasets.

## Prerequisites

First, ensure you have the necessary imports. This tutorial supports multiple frameworks (MXNet, PyTorch, TensorFlow, JAX). The `d2l` book library provides cross-framework utilities.

```python
%matplotlib inline
# Framework-specific imports are handled in the code blocks below.
```

## 1. Generating the Dataset

We'll generate a simple 2D dataset for clarity. The goal is to create 1000 examples from a linear model with added noise.

The data follows this formula:
$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \boldsymbol{\epsilon}.$$

Here:
*   $\mathbf{X}$ is our 1000×2 design matrix (features).
*   $\mathbf{w} = [2, -3.4]^\top$ and $b = 4.2$ are the *true* parameters we aim to recover.
*   $\boldsymbol{\epsilon}$ is Gaussian noise with mean 0 and standard deviation 0.01.

We implement this in a `SyntheticRegressionData` class, inheriting from `d2l.DataModule` for consistent data handling.

```python
class SyntheticRegressionData(d2l.DataModule):  #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val

        # Generate features from a standard normal distribution
        if tab.selected('pytorch') or tab.selected('mxnet'):
            self.X = d2l.randn(n, len(w))
            noise = d2l.randn(n, 1) * noise
        if tab.selected('tensorflow'):
            self.X = tf.random.normal((n, w.shape[0]))
            noise = tf.random.normal((n, 1)) * noise
        if tab.selected('jax'):
            key = jax.random.PRNGKey(0)
            key1, key2 = jax.random.split(key)
            self.X = jax.random.normal(key1, (n, w.shape[0]))
            noise = jax.random.normal(key2, (n, 1)) * noise

        # Generate labels: y = Xw + b + noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```

Now, instantiate the dataset with our chosen parameters.

```python
data = SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
```

### Inspecting the Data

Let's examine the first example to understand the structure. Each feature is a 2D vector, and each label is a scalar.

```python
print('features:', data.X[0], '\nlabel:', data.y[0])
```

**Output:**
```
features: [ 0.707  1.572]
label: [3.254]
```

## 2. Building a Custom Data Loader

Training models requires iterating over the dataset in minibatches. We'll implement a basic `get_dataloader` method. This method yields batches of `(features, labels)`, shuffling the data during training for randomness but keeping validation order fixed for reproducibility.

```python
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        random.shuffle(indices)  # Shuffle for training
    else:
        indices = list(range(self.num_train, self.num_train + self.num_val))

    for i in range(0, len(indices), self.batch_size):
        if tab.selected('mxnet', 'pytorch', 'jax'):
            batch_indices = d2l.tensor(indices[i: i + self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
        if tab.selected('tensorflow'):
            j = tf.constant(indices[i: i + self.batch_size])
            yield tf.gather(self.X, j), tf.gather(self.y, j)
```

### Testing the Data Loader

Fetch the first training batch to verify shapes. The batch size determines the first dimension of both features and labels.

```python
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

**Output:**
```
X shape: (32, 2)
y shape: (32, 1)
```

> **Note on Python Design:** The `iter(data.train_dataloader())` call works because we added the `get_dataloader` method to the class *after* creating the `data` object. This showcases Python's dynamic, object-oriented flexibility.

## 3. Using Framework-Optimized Data Loaders

While our custom loader is educational, real applications use highly optimized, built-in data loaders. These handle large datasets efficiently, streaming from disk and managing memory smartly.

We first add a general `get_tensorloader` method to the base `DataModule` class, which each framework implements.

```python
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    if tab.selected('mxnet'):
        dataset = gluon.data.ArrayDataset(*tensors)
        return gluon.data.DataLoader(dataset, self.batch_size, shuffle=train)
    if tab.selected('pytorch'):
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)
    if tab.selected('jax'):
        # JAX uses TensorFlow's data loading utilities
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(tensors).shuffle(
                buffer_size=shuffle_buffer).batch(self.batch_size))
    if tab.selected('tensorflow'):
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)
```

Now, update our `SyntheticRegressionData` class to use this efficient loader.

```python
@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```

### Benefits of the Built-in Loader

The new loader behaves identically but is more robust. It also supports useful features like querying the number of batches.

```python
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
print('Number of batches:', len(data.train_dataloader()))
```

**Output:**
```
X shape: (32, 2)
y shape: (32, 1)
Number of batches: 32
```

## Summary

In this guide, you learned to:
1.  **Generate synthetic data** for a known linear regression problem.
2.  **Implement a basic data loader** to iterate over data in minibatches, with shuffling for training.
3.  **Leverage framework-optimized data loaders** for efficiency and additional functionality.

Data loaders abstract data loading and preprocessing, allowing the same algorithm to work with diverse data sources. The simple 2D linear model provides a perfect testbed for verifying regression algorithms before applying them to complex, real-world data.

## Exercises

Test your understanding with these challenges:

1.  **Handling Remainders:** What happens if the dataset size isn't divisible by the batch size? Check your framework's API (`DataLoader` in PyTorch, `batch` in TF, etc.) to see how it handles the final, potentially smaller, batch.
2.  **Large-Scale Data:**
    *   How would you manage data that doesn't fit in memory?
    *   Design an algorithm to shuffle data stored on disk efficiently. (Hint: Look into *pseudorandom permutation generators*).
3.  **On-the-Fly Generation:** Modify the data generator to produce new data dynamically each time the iterator is called, rather than pre-generating all data.
4.  **Deterministic Generation:** Ensure your random data generator produces the *exact same* sequence of data every time it runs, which is crucial for reproducibility in experiments.

---
*Discussions: [MXNet](https://discuss.d2l.ai/t/6662) | [PyTorch](https://discuss.d2l.ai/t/6663) | [TensorFlow](https://discuss.d2l.ai/t/6664) | [JAX](https://discuss.d2l.ai/t/17975)*