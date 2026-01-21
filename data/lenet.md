# Implementing LeNet: A Classic Convolutional Neural Network

This guide walks you through implementing LeNet-5, one of the earliest and most influential convolutional neural networks (CNNs). You'll build the model, inspect its architecture, and train it on the Fashion-MNIST dataset.

## Prerequisites

First, install and import the necessary libraries. The code supports multiple deep learning frameworks (MXNet, PyTorch, TensorFlow, JAX). Choose your preferred one.

```bash
# Installation commands would typically go here.
# Since we're using the d2l library, ensure it's installed.
```

```python
# Framework-specific imports
# For PyTorch
import torch
from torch import nn
from d2l import torch as d2l

# For MXNet
# from mxnet import np, npx, gluon, init, autograd
# from mxnet.gluon import nn
# npx.set_np()
# from d2l import mxnet as d2l

# For TensorFlow
# import tensorflow as tf
# from d2l import tensorflow as d2l

# For JAX
# from d2l import jax as d2l
# from flax import linen as nn
# import jax
# from jax import numpy as jnp
```

## 1. Understanding the LeNet Architecture

LeNet-5, introduced by Yann LeCun in 1998, was designed for handwritten digit recognition. Its success helped popularize CNNs. The architecture consists of two main parts:
1.  **A Convolutional Encoder:** Two convolutional layers, each followed by a sigmoid activation and average pooling.
2.  **A Dense Block:** Three fully connected (linear) layers.

The network processes a 28x28 single-channel (grayscale) image and outputs a probability distribution over 10 classes.

## 2. Implementing the LeNet Model

Now, you will define the LeNet class. The implementation uses a `Sequential` container to chain the layers. We initialize the weights using the Xavier initialization method for stable training.

```python
# PyTorch Implementation
def init_cnn(module):
    """Initialize weights for CNNs using Xavier initialization."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

class LeNet(d2l.Classifier):
    """The LeNet-5 model."""
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(num_classes))
```

**Key Components Explained:**
*   `nn.LazyConv2d`: Defines a 2D convolutional layer. The "lazy" variant infers the input channels from the first forward pass.
*   `nn.Sigmoid`: The activation function used in the original LeNet (modern networks typically use ReLU).
*   `nn.AvgPool2d`: Performs 2x2 average pooling with a stride of 2, reducing spatial dimensions.
*   `nn.Flatten`: Reshapes the 4D tensor (batch, channels, height, width) into a 2D tensor (batch, features) for the dense layers.
*   The final dense layer has `num_classes` outputs (10 for Fashion-MNIST).

## 3. Inspecting the Model Architecture

Let's verify the model's layer-by-layer transformations by passing a dummy input and printing the output shapes. This confirms the architecture matches the theoretical design.

```python
@d2l.add_to_class(d2l.Classifier)
def layer_summary(self, X_shape):
    """Print the output shape after each layer for a given input shape."""
    X = d2l.randn(*X_shape)
    for layer in self.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

# Instantiate the model and inspect it
model = LeNet()
model.layer_summary((1, 1, 28, 28))
```

**Expected Output:**
```
LazyConv2d output shape:	 torch.Size([1, 6, 28, 28])
Sigmoid output shape:	 torch.Size([1, 6, 28, 28])
AvgPool2d output shape:	 torch.Size([1, 6, 14, 14])
LazyConv2d output shape:	 torch.Size([1, 16, 10, 10])
Sigmoid output shape:	 torch.Size([1, 16, 10, 10])
AvgPool2d output shape:	 torch.Size([1, 16, 5, 5])
Flatten output shape:	 torch.Size([1, 400])
LazyLinear output shape:	 torch.Size([1, 120])
Sigmoid output shape:	 torch.Size([1, 120])
LazyLinear output shape:	 torch.Size([1, 84])
Sigmoid output shape:	 torch.Size([1, 84])
LazyLinear output shape:	 torch.Size([1, 10])
```

**What's happening?**
1.  The first conv layer (5x5 kernel, padding=2) preserves the 28x28 spatial size but increases channels to 6.
2.  The first pooling layer halves the spatial dimensions to 14x14.
3.  The second conv layer (5x5 kernel, no padding) reduces the spatial size to 10x10 and increases channels to 16.
4.  The second pooling layer reduces spatial dimensions to 5x5.
5.  The `Flatten` layer reshapes the (1, 16, 5, 5) tensor into a 1D vector of length 400 (16 * 5 * 5).
6.  The three dense layers progressively reduce dimensions to 120, 84, and finally 10 (the number of classes).

## 4. Training LeNet on Fashion-MNIST

Now, let's train the model. We'll use the Fashion-MNIST dataset, cross-entropy loss, and stochastic gradient descent. The `d2l.Trainer` class handles the training loop, device placement, and logging.

```python
# Initialize the trainer and data loader
trainer = d2l.Trainer(max_epochs=10, num_gpus=1) # Use GPU if available
data = d2l.FashionMNIST(batch_size=128)

# Create the model and initialize its weights (PyTorch specific)
model = LeNet(lr=0.1)
model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)

# Start training
trainer.fit(model, data)
```

**Training Details:**
*   **Dataset:** Fashion-MNIST (60,000 training, 10,000 test images of 10 clothing categories).
*   **Batch Size:** 128.
*   **Optimizer:** SGD with a learning rate of 0.1.
*   **Epochs:** 10.
*   The `apply_init` function initializes the convolutional and linear layers using the `init_cnn` function defined earlier.

After training, you should observe the training and validation accuracy increasing over epochs, demonstrating that LeNet can effectively learn to classify fashion items.

## 5. Summary and Next Steps

You have successfully implemented and trained the LeNet-5 CNN. While modern architectures use ReLU and max-pooling, LeNet's core design—alternating convolutional and pooling layers followed by dense layers—remains foundational.

**Key Takeaways:**
*   CNNs preserve spatial structure in image data, unlike flattening used in MLPs.
*   Convolutional layers extract hierarchical features (edges, textures, patterns).
*   Pooling layers provide spatial invariance and reduce computational cost.
*   The transition from convolutional to dense layers requires flattening the feature maps.

## Exercises for Further Exploration

1.  **Modernize LeNet:** Replace average pooling with max-pooling and sigmoid activations with ReLU. How does this affect performance and training speed?
2.  **Architecture Search:** Experiment with the network architecture:
    *   Change the convolution kernel size (e.g., 3x3 or 7x7).
    *   Adjust the number of output channels in each conv layer.
    *   Add or remove convolutional/dense layers.
    *   Tune hyperparameters like learning rate, weight initialization, and number of epochs.
3.  **Dataset Change:** Train your modified network on the original MNIST digit dataset. Compare results.
4.  **Activation Visualization:** Write code to extract and visualize the feature maps (activations) from the first two convolutional layers for different input images (e.g., a sweater vs. a coat). What patterns do the filters detect?
5.  **Robustness Test:** Feed the network significantly different images (e.g., of cats, cars, or random noise). Observe how the activations and final predictions change. This helps understand the model's limitations and generalization.

By completing these exercises, you'll deepen your understanding of CNN design choices and their impact on model performance.