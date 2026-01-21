# Generative Adversarial Networks (GANs) Tutorial

## Introduction

In this tutorial, you will implement a Generative Adversarial Network (GAN) from scratch. While GANs are famously used for generating photorealistic images, we'll start with a simpler task: learning to generate data from a 2D Gaussian distribution. This foundational example will help you understand the core adversarial training dynamics that make GANs work.

## Prerequisites

First, let's set up our environment by importing the necessary libraries. We'll use a deep learning framework (choose one: MXNet, PyTorch, or TensorFlow) and some utilities from the D2L library.

```python
# For MXNet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```python
# For PyTorch
from d2l import torch as d2l
import torch
from torch import nn
```

```python
# For TensorFlow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Step 1: Generate the "Real" Data

We'll create our training dataset by sampling from a 2D Gaussian distribution and applying a linear transformation. This gives us a simple but non-trivial distribution for our GAN to learn.

```python
# Generate 1000 samples from a standard normal distribution
X = d2l.normal(0.0, 1, (1000, 2))

# Define a transformation matrix and bias
A = d2l.tensor([[1, 2], [-0.1, 0.5]])
b = d2l.tensor([1, 2])

# Apply linear transformation: data = X * A + b
data = d2l.matmul(X, A) + b
```

Let's examine the first 100 points to understand our data's structure:

```python
d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(data[:100, 0]), d2l.numpy(data[:100, 1]))
print(f'The covariance matrix is\n{d2l.matmul(A.T, A)}')
```

Finally, we'll create a data iterator to serve batches during training:

```python
batch_size = 8
data_iter = d2l.load_array((data,), batch_size)
```

## Step 2: Build the Generator Network

The generator's job is to transform random noise into data that resembles our real distribution. For this simple example, we'll use a single linear layer.

```python
# MXNet
net_G = nn.Sequential()
net_G.add(nn.Dense(2))

# PyTorch
net_G = nn.Sequential(nn.Linear(2, 2))

# TensorFlow
net_G = tf.keras.layers.Dense(2)
```

## Step 3: Build the Discriminator Network

The discriminator acts as a binary classifier, distinguishing between real data and generated (fake) data. We'll use a slightly more complex multi-layer perceptron.

```python
# MXNet
net_D = nn.Sequential()
net_D.add(nn.Dense(5, activation='tanh'),
          nn.Dense(3, activation='tanh'),
          nn.Dense(1))

# PyTorch
net_D = nn.Sequential(
    nn.Linear(2, 5), nn.Tanh(),
    nn.Linear(5, 3), nn.Tanh(),
    nn.Linear(3, 1))

# TensorFlow
net_D = tf.keras.models.Sequential([
    tf.keras.layers.Dense(5, activation="tanh", input_shape=(2,)),
    tf.keras.layers.Dense(3, activation="tanh"),
    tf.keras.layers.Dense(1)
])
```

## Step 4: Define the Training Functions

The training process involves alternating updates to the discriminator and generator. Let's define helper functions for each update step.

### Update the Discriminator

The discriminator is trained to maximize its ability to distinguish real from fake data.

```python
# MXNet
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    batch_size = X.shape[0]
    ones = np.ones((batch_size,), ctx=X.ctx)
    zeros = np.zeros((batch_size,), ctx=X.ctx)
    with autograd.record():
        real_Y = net_D(X)
        fake_X = net_G(Z)
        fake_Y = net_D(fake_X.detach())
        loss_D = (loss(real_Y, ones) + loss(fake_Y, zeros)) / 2
    loss_D.backward()
    trainer_D.step(batch_size)
    return float(loss_D.sum())
```

```python
# PyTorch
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape)) +
              loss(fake_Y, zeros.reshape(fake_Y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D
```

```python
# TensorFlow
def update_D(X, Z, net_D, net_G, loss, optimizer_D):
    batch_size = X.shape[0]
    ones = tf.ones((batch_size,))
    zeros = tf.zeros((batch_size,))
    fake_X = net_G(Z)
    with tf.GradientTape() as tape:
        real_Y = net_D(X)
        fake_Y = net_D(fake_X)
        loss_D = (loss(ones, tf.squeeze(real_Y)) + loss(
            zeros, tf.squeeze(fake_Y))) * batch_size / 2
    grads_D = tape.gradient(loss_D, net_D.trainable_variables)
    optimizer_D.apply_gradients(zip(grads_D, net_D.trainable_variables))
    return loss_D
```

### Update the Generator

The generator is trained to fool the discriminator by producing increasingly realistic data.

```python
# MXNet
def update_G(Z, net_D, net_G, loss, trainer_G):
    batch_size = Z.shape[0]
    ones = np.ones((batch_size,), ctx=Z.ctx)
    with autograd.record():
        fake_X = net_G(Z)
        fake_Y = net_D(fake_X)
        loss_G = loss(fake_Y, ones)
    loss_G.backward()
    trainer_G.step(batch_size)
    return float(loss_G.sum())
```

```python
# PyTorch
def update_G(Z, net_D, net_G, loss, trainer_G):
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()
    fake_X = net_G(Z)
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G
```

```python
# TensorFlow
def update_G(Z, net_D, net_G, loss, optimizer_G):
    batch_size = Z.shape[0]
    ones = tf.ones((batch_size,))
    with tf.GradientTape() as tape:
        fake_X = net_G(Z)
        fake_Y = net_D(fake_X)
        loss_G = loss(ones, tf.squeeze(fake_Y)) * batch_size
    grads_G = tape.gradient(loss_G, net_G.trainable_variables)
    optimizer_G.apply_gradients(zip(grads_G, net_G.trainable_variables))
    return loss_G
```

## Step 5: Train the GAN

Now we'll write the main training loop that alternates between updating the discriminator and generator.

```python
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    # Initialize loss function and optimizers
    loss = gluon.loss.SigmoidBCELoss()  # MXNet
    # loss = nn.BCEWithLogitsLoss(reduction='sum')  # PyTorch
    # loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)  # TensorFlow
    
    # Initialize network weights
    net_D.initialize(init=init.Normal(0.02), force_reinit=True)  # MXNet
    net_G.initialize(init=init.Normal(0.02), force_reinit=True)
    # PyTorch/TensorFlow initialization code would go here
    
    # Create optimizers
    trainer_D = gluon.Trainer(net_D.collect_params(), 'adam', {'learning_rate': lr_D})
    trainer_G = gluon.Trainer(net_G.collect_params(), 'adam', {'learning_rate': lr_G})
    
    # Set up visualization
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        
        for X in data_iter:
            batch_size = X.shape[0]
            Z = np.random.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        
        # Visualize generated examples
        Z = np.random.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).asnumpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        
        # Record losses
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

## Step 6: Run the Training

Let's set our hyperparameters and start training:

```python
lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 20
train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G,
      latent_dim, d2l.numpy(data[:100]))
```

During training, you'll see the discriminator and generator losses evolve. The visualization will show how the generated samples (orange) gradually match the real data distribution (blue).

## Summary

In this tutorial, you've implemented a complete Generative Adversarial Network:

1. **Generator Network**: Transforms random noise into synthetic data
2. **Discriminator Network**: Distinguishes between real and generated data
3. **Adversarial Training**: The generator tries to fool the discriminator, while the discriminator tries to correctly identify real vs. fake data

The key insight is that both networks improve through this competitive process: the generator produces more realistic data, while the discriminator becomes better at detection.

## Next Steps

Try experimenting with:
- Different network architectures for the generator and discriminator
- Various hyperparameter settings (learning rates, latent dimension)
- More complex datasets
- The equilibrium question: Can the generator become so good that the discriminator can't distinguish real from fake data?

This foundational understanding will help you tackle more advanced GAN applications like image generation, style transfer, and data augmentation.