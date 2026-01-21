# Implementing a Naive Bayes Classifier for Digit Recognition

In this tutorial, we'll build a Naive Bayes classifier from scratch to recognize handwritten digits from the MNIST dataset. This probabilistic model, while simple, demonstrates fundamental concepts in machine learning and serves as an excellent introduction to classification tasks.

## Prerequisites

First, ensure you have the necessary libraries installed. We'll use a deep learning framework helper library (`d2l`) for visualization and data handling. Choose your preferred framework (MXNet, PyTorch, or TensorFlow).

```bash
# Install required packages (if needed)
# pip install d2l
```

Now, let's import the necessary modules.

```python
# For MXNet
from d2l import mxnet as d2l
import math
from mxnet import gluon, np, npx
npx.set_np()

# For PyTorch
from d2l import torch as d2l
import math
import torch
import torchvision

# For TensorFlow
from d2l import tensorflow as d2l
import math
import tensorflow as tf

# Set up visualization
d2l.use_svg_display()
```

## Step 1: Load and Prepare the MNIST Dataset

The MNIST dataset contains 60,000 training and 10,000 test images of handwritten digits (0-9). Each image is 28×28 pixels. We'll preprocess the images by converting pixel values to binary (0 or 1) to simplify our probability calculations.

### MXNet Implementation

```python
def transform(data, label):
    # Convert to float and threshold at 128 to create binary features
    return np.floor(data.astype('float32') / 128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
```

### PyTorch Implementation

```python
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    lambda x: torch.floor(x * 255 / 128).squeeze(dim=0)
])

mnist_train = torchvision.datasets.MNIST(
    root='./temp', train=True, transform=data_transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./temp', train=False, transform=data_transform, download=True)
```

### TensorFlow Implementation

```python
# Load MNIST data
((train_images, train_labels), (test_images, test_labels)) = tf.keras.datasets.mnist.load_data()

# Convert to binary features (0 or 1)
train_images = tf.floor(tf.constant(train_images / 128, dtype=tf.float32))
test_images = tf.floor(tf.constant(test_images / 128, dtype=tf.float32))

train_labels = tf.constant(train_labels, dtype=tf.int32)
test_labels = tf.constant(test_labels, dtype=tf.int32)
```

### Verify Data Loading

Let's examine a sample image to understand our data structure.

```python
# Access a single training example
# MXNet/PyTorch
image, label = mnist_train[2]
# TensorFlow
image, label = train_images[2], train_labels[2]

print(f"Image shape: {image.shape}, Label: {label}")
print(f"Data type: {image.dtype}")
```

The output should show a shape of `(28, 28)` for the image and a scalar label.

## Step 2: Understanding the Probabilistic Model

Our goal is to classify images by finding the most probable digit given the pixel values. Formally, we want to compute:

$$\hat{y} = \mathrm{argmax}_y \> p(y \mid \mathbf{x})$$

where $\mathbf{x}$ represents our 784 pixels (28×28 flattened) and $y$ is a digit from 0-9.

Using Bayes' theorem, we can rewrite this as:

$$\hat{y} = \mathrm{argmax}_y \> p(\mathbf{x} \mid y) p(y)$$

The denominator $p(\mathbf{x})$ is constant for all $y$, so we can ignore it for classification.

## Step 3: The Naive Bayes Assumption

Computing $p(\mathbf{x} \mid y)$ directly would require estimating probabilities for $2^{784}$ possible pixel combinations - an impossible task. The Naive Bayes classifier makes a key simplifying assumption: **pixels are conditionally independent given the digit class**.

This allows us to factor the probability:

$$p(\mathbf{x} \mid y) = \prod_{i=1}^{784} p(x_i \mid y)$$

Now we only need to estimate $p(x_i \mid y)$ for each pixel $i$ and digit $y$, which is much more manageable.

## Step 4: Training the Model

Training involves estimating two sets of probabilities:
1. $P_y[y]$: Prior probability of each digit
2. $P_{xy}[i, y]$: Probability that pixel $i$ is "on" (value 1) for digit $y$

### Estimate Prior Probabilities $P_y$

We simply count how often each digit appears in the training set.

```python
# MXNet
X, Y = mnist_train[:]  # All training examples
n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()

# PyTorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], dim=0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])
n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()

# TensorFlow
X = train_images
Y = train_labels
n_y = tf.Variable(tf.zeros(10))
for y in range(10):
    n_y[y].assign(tf.reduce_sum(tf.cast(Y == y, tf.float32)))
P_y = n_y / tf.reduce_sum(n_y)

print("Prior probabilities P_y:", P_y)
```

### Estimate Pixel Probabilities $P_{xy}$

For each digit class, we count how often each pixel is "on" (value 1). We apply **Laplace smoothing** by adding 1 to each count to avoid zero probabilities.

```python
# MXNet
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y.asnumpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

# PyTorch
n_x = torch.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = torch.tensor(X.numpy()[Y.numpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

# TensorFlow
n_x = tf.Variable(tf.zeros((10, 28, 28)))
for y in range(10):
    n_x[y].assign(tf.cast(tf.reduce_sum(
        X.numpy()[Y.numpy() == y], axis=0), tf.float32))
P_xy = (n_x + 1) / tf.reshape((n_y + 2), (10, 1, 1))

# Visualize the learned pixel probabilities
d2l.show_images(P_xy, 2, 5)
```

The visualization shows the "average" appearance of each digit based on our probability estimates.

## Step 5: Making Predictions

To classify a new image $\mathbf{t} = (t_1, t_2, \ldots, t_{784})$ with $t_i \in \{0,1\}$, we compute:

$$\hat{y} = \mathrm{argmax}_y \> P_y[y] \prod_{i=1}^{784} P_{xy}[i, y]^{t_i} (1 - P_{xy}[i, y])^{1-t_i}$$

### Initial Implementation (Numerically Unstable)

Let's first implement this formula directly:

```python
# MXNet
def bayes_pred(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(axis=1)  # p(x|y)
    return np.array(p_xy) * P_y

# PyTorch
def bayes_pred(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(dim=1)  # p(x|y)
    return p_xy * P_y

# TensorFlow
def bayes_pred(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = tf.math.reduce_prod(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy * P_y

# Test on a sample image
image, label = mnist_test[0]  # or test_images[0] for TensorFlow
probabilities = bayes_pred(image)
print("Raw probabilities:", probabilities)
```

You'll notice the probabilities are all zero or extremely small! This is due to **numerical underflow** - multiplying 784 small probabilities results in a number too small for computers to represent accurately.

### Stable Implementation Using Logarithms

To fix this, we work with log probabilities instead:

$$\hat{y} = \mathrm{argmax}_y \> \log P_y[y] + \sum_{i=1}^{784} \left[t_i \log P_{xy}[i, y] + (1-t_i) \log (1 - P_{xy}[i, y])\right]$$

```python
# Precompute log probabilities
# MXNet
log_P_xy = np.log(P_xy)
log_P_xy_neg = np.log(1 - P_xy)
log_P_y = np.log(P_y)

# PyTorch
log_P_xy = torch.log(P_xy)
log_P_xy_neg = torch.log(1 - P_xy)
log_P_y = torch.log(P_y)

# TensorFlow
log_P_xy = tf.math.log(P_xy)
log_P_xy_neg = tf.math.log(1 - P_xy)
log_P_y = tf.math.log(P_y)

def bayes_pred_stable(x):
    # MXNet
    x = np.expand_dims(x, axis=0)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)
    return p_xy + log_P_y
    
    # PyTorch
    x = x.unsqueeze(0)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)
    return p_xy + log_P_y
    
    # TensorFlow
    x = tf.expand_dims(x, axis=0)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = tf.math.reduce_sum(tf.reshape(p_xy, (10, -1)), axis=1)
    return p_xy + log_P_y

# Test the stable version
log_probs = bayes_pred_stable(image)
print("Log probabilities:", log_probs)
predicted_digit = log_probs.argmax()  # or tf.argmax for TensorFlow
print(f"Predicted: {predicted_digit}, Actual: {label}")
print(f"Correct prediction: {predicted_digit == label}")
```

## Step 6: Batch Prediction and Evaluation

Now let's create a function to predict multiple images and evaluate our classifier's accuracy.

```python
def predict(X):
    # MXNet
    return [bayes_pred_stable(x).argmax(axis=0).astype(np.int32) for x in X]
    
    # PyTorch
    return [bayes_pred_stable(x).argmax(dim=0).type(torch.int32).item() for x in X]
    
    # TensorFlow
    return [tf.argmax(bayes_pred_stable(x), axis=0, output_type=tf.int32).numpy() for x in X]

# Test on a small batch
# MXNet/PyTorch
X_test, y_test = mnist_test[:18]
# TensorFlow
X_test = tf.stack([test_images[i] for i in range(18)], axis=0)
y_test = tf.constant([test_labels[i].numpy() for i in range(18)])

preds = predict(X_test)
d2l.show_images(X_test, 2, 9, titles=[str(d) for d in preds])
```

### Compute Overall Accuracy

Finally, let's evaluate our classifier on the entire test set.

```python
# MXNet
X_test, y_test = mnist_test[:]
preds = np.array(predict(X_test), dtype=np.int32)
accuracy = float((preds == y_test).sum()) / len(y_test)

# PyTorch
X_test = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))], dim=0)
y_test = torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))])
preds = torch.tensor(predict(X_test), dtype=torch.int32)
accuracy = float((preds == y_test).sum()) / len(y_test)

# TensorFlow
X_test = test_images
y_test = test_labels
preds = tf.constant(predict(X_test), dtype=tf.int32)
accuracy = tf.reduce_sum(tf.cast(preds == y_test, tf.float32)).numpy() / len(y_test)

print(f"Test accuracy: {accuracy:.4f}")
```

You should achieve an accuracy around 84%. While modern deep learning models achieve over 99% accuracy on MNIST, our Naive Bayes classifier performs reasonably well given its simplicity and the fact that it ignores pixel correlations.

## Summary

In this tutorial, we implemented a Naive Bayes classifier for digit recognition:

1. **Loaded and preprocessed** the MNIST dataset, converting images to binary features
2. **Understood the probabilistic framework** for classification using Bayes' theorem
3. **Applied the Naive Bayes assumption** of conditional independence between features
4. **Trained the model** by estimating prior probabilities and pixel probabilities with Laplace smoothing
5. **Implemented prediction** using log probabilities to avoid numerical underflow
6. **Evaluated the classifier** achieving approximately 84% accuracy on the test set

The Naive Bayes classifier, while based on a strong independence assumption that doesn't hold for real images, demonstrates how probabilistic reasoning can be applied to classification tasks and serves as a foundation for more sophisticated models.

## Exercises

1. Consider the XOR dataset with inputs `[[0,0], [0,1], [1,0], [1,1]]` and labels `[0,1,1,0]`. What probabilities would a Naive Bayes classifier learn from this data? Would it classify points correctly? Which assumptions are violated?

2. What would happen if we didn't use Laplace smoothing and encountered a pixel value at test time that never appeared during training for a particular class?

3. The Naive Bayes classifier is a simple Bayesian network. Research how allowing dependencies between input variables (as in more complex Bayesian networks) could help solve problems like XOR classification.