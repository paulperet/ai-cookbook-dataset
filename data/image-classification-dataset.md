# Working with the Fashion-MNIST Dataset

## Introduction

This guide introduces the Fashion-MNIST dataset, a modern benchmark for image classification. While the classic MNIST dataset of handwritten digits was foundational, its simplicity makes it less suitable for evaluating today's powerful models. Fashion-MNIST offers a more challenging and relevant alternative, consisting of 70,000 grayscale images across 10 clothing categories.

You will learn how to load, inspect, and prepare this dataset for training machine learning models using popular deep learning frameworks.

## Prerequisites

Ensure you have the necessary libraries installed. The code is framework-agnostic, supporting MXNet, PyTorch, TensorFlow, and JAX. The `d2l` library provides common utilities.

```bash
# Installation commands would be framework-specific.
# For example, for PyTorch:
# pip install torch torchvision
```

## 1. Loading the Dataset

All major frameworks provide utilities to download and load Fashion-MNIST. We'll encapsulate this in a `DataModule` class for consistency.

First, import the necessary modules. The `%%tab` magic from the source is replaced by framework-specific code blocks.

**For MXNet:**
```python
import time
from d2l import mxnet as d2l
from mxnet import gluon, npx
from mxnet.gluon.data.vision import transforms
npx.set_np()
d2l.use_svg_display()
```

**For PyTorch:**
```python
import time
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms
d2l.use_svg_display()
```

**For TensorFlow:**
```python
import time
from d2l import tensorflow as d2l
import tensorflow as tf
d2l.use_svg_display()
```

**For JAX:**
```python
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import time
import tensorflow as tf
import tensorflow_datasets as tfds
d2l.use_svg_display()
```

### 1.1 Define the DataModule Class

Now, define the `FashionMNIST` class which handles downloading and structuring the data. The implementation varies slightly by framework.

**For MXNet:**
```python
class FashionMNIST(d2l.DataModule):
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = gluon.data.vision.FashionMNIST(
            train=True).transform_first(trans)
        self.val = gluon.data.vision.FashionMNIST(
            train=False).transform_first(trans)
```

**For PyTorch:**
```python
class FashionMNIST(d2l.DataModule):
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
```

**For TensorFlow and JAX:**
```python
class FashionMNIST(d2l.DataModule):
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()
```

### 1.2 Instantiate and Inspect the Dataset

Create an instance of the dataset. We'll resize the images to 32x32 pixels for demonstration.

```python
data = FashionMNIST(resize=(32, 32))
```

Check the size of the training and validation sets. The dataset contains 60,000 training and 10,000 validation images.

**For MXNet/PyTorch:**
```python
len(data.train), len(data.val)
```
**Output:**
```
(60000, 10000)
```

**For TensorFlow/JAX:**
```python
len(data.train[0]), len(data.val[0])
```
**Output:**
```
(60000, 10000)
```

Examine the shape of a single image. After resizing, each image is a 32x32 grayscale tensor.

```python
data.train[0][0].shape
```
**Output (for a batch size of 1, resized to 32x32):**
```
torch.Size([1, 32, 32])  # Example PyTorch output
```

## 2. Understanding the Labels

Fashion-MNIST has 10 categories. Let's create a helper method to convert numeric labels (0-9) to human-readable names.

```python
@d2l.add_to_class(FashionMNIST)
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

## 3. Creating a Data Loader

Efficient training requires batching and shuffling data. We'll implement a `get_dataloader` method that returns a data iterator.

**For MXNet:**
```python
@d2l.add_to_class(FashionMNIST)
def get_dataloader(self, train):
    data = self.train if train else self.val
    return gluon.data.DataLoader(data, self.batch_size, shuffle=train,
                                 num_workers=self.num_workers)
```

**For PyTorch:**
```python
@d2l.add_to_class(FashionMNIST)
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
```

**For TensorFlow and JAX:**
The process involves normalizing pixel values (scaling to [0,1]), casting labels to integers, and applying resizing.

```python
@d2l.add_to_class(FashionMNIST)
def get_dataloader(self, train):
    data = self.train if train else self.val
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
    shuffle_buf = len(data[0]) if train else 1
    if tab.selected('tensorflow'):  # Framework selection logic
        return tf.data.Dataset.from_tensor_slices(process(*data)).batch(
            self.batch_size).map(resize_fn).shuffle(shuffle_buf)
    if tab.selected('jax'):
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(process(*data)).batch(
                self.batch_size).map(resize_fn).shuffle(shuffle_buf))
```

### 3.1 Test the Data Loader

Let's fetch one batch to verify its shape and data types.

```python
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
```
**Example Output:**
```
torch.Size([64, 1, 32, 32]) torch.float32 torch.Size([64]) torch.int64
```

### 3.2 Measure Data Loading Speed

It's useful to ensure data loading isn't a bottleneck. Let's time a full pass through the training data.

```python
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
```
**Example Output:**
```
2.34 sec
```

## 4. Visualizing the Data

Visual inspection helps catch data issues early. We'll add a `visualize` method to display images and their labels.

**Note:** The `show_images` utility function is assumed to be provided by the `d2l` library. Its implementation is omitted here as it's a visualization helper.

```python
@d2l.add_to_class(FashionMNIST)
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    # Framework-specific squeezing for channel dimension
    if tab.selected('mxnet', 'pytorch'):
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
    if tab.selected('tensorflow'):
        d2l.show_images(tf.squeeze(X), nrows, ncols, titles=labels)
    if tab.selected('jax'):
        d2l.show_images(jnp.squeeze(X), nrows, ncols, titles=labels)
```

Now, visualize a batch from the validation set.

```python
batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```
This will display a grid of images with their corresponding labels (e.g., "t-shirt", "sneaker").

## Summary

You have successfully set up the Fashion-MNIST dataset for image classification tasks. Key steps included:
1. **Loading the dataset** using framework-specific utilities.
2. **Inspecting dataset dimensions** (60,000 training, 10,000 validation images, 10 classes).
3. **Creating an efficient data loader** that batches and shuffles data.
4. **Visualizing samples** to verify data integrity.

Fashion-MNIST serves as a practical benchmark for evaluating models, from simple linear classifiers to complex convolutional networks. The data is represented as tensors of shape `(batch_size, channels, height, width)`. For grayscale images, `channels=1`.

Efficient data loading is critical for performance. The implemented data iterators leverage each framework's optimized pipelines to prevent I/O from slowing down training.

## Exercises

1. **Batch Size Impact:** Experiment with reducing `batch_size` to 1. How does it affect reading performance and memory usage?
2. **Performance Profiling:** Is the current data loader fast enough? Use a system profiler (e.g., `cProfile` in Python) to identify bottlenecks. Consider optimizing by:
   - Increasing the number of worker processes.
   - Using pinned memory (for GPU training).
   - Pre-fetching batches.
3. **Explore Other Datasets:** Check your framework's documentation for other built-in datasets (e.g., CIFAR-10, ImageNet subsets). How do their structures compare to Fashion-MNIST?

---
*This tutorial is based on the Fashion-MNIST dataset by Xiao et al., 2017. For further discussion, refer to the [D2L.ai forums](https://discuss.d2l.ai).*