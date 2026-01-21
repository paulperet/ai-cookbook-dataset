# Visualizing Models, Data, and Training with TensorBoard

In this tutorial, you will learn how to use TensorBoard to visualize and track your PyTorch model's training process, architecture, and performance. We'll use the Fashion-MNIST dataset to build a simple convolutional neural network (CNN) and demonstrate TensorBoard's key features.

## Prerequisites

Ensure you have the necessary libraries installed. You can install them via pip if needed.

```bash
pip install torch torchvision matplotlib tensorboard
```

Now, let's import the required modules.

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```

## Step 1: Load and Prepare the Data

We'll use the Fashion-MNIST dataset, applying transformations to normalize the pixel values.

```python
# Define transformations: convert to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load training and test datasets
trainset = torchvision.datasets.FashionMNIST(
    './data',
    download=True,
    train=True,
    transform=transform
)
testset = torchvision.datasets.FashionMNIST(
    './data',
    download=True,
    train=False,
    transform=transform
)

# Create data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# Class labels for Fashion-MNIST
classes = (
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
)
```

## Step 2: Define a Helper Function for Displaying Images

This function will help us visualize images in a Matplotlib-compatible format.

```python
def matplotlib_imshow(img, one_channel=False):
    """
    Display an image, handling single-channel (grayscale) inputs.
    """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
```

## Step 3: Define the Neural Network Model

We'll define a simple CNN adapted for the 28x28 grayscale images of Fashion-MNIST.

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
net = Net()
```

## Step 4: Define Loss Function and Optimizer

We'll use Cross-Entropy Loss and Stochastic Gradient Descent (SGD) with momentum.

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

## Step 5: Set Up TensorBoard

Import TensorBoard's `SummaryWriter` to log data for visualization.

```python
from torch.utils.tensorboard import SummaryWriter

# Create a SummaryWriter instance
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
```

This creates a directory `runs/fashion_mnist_experiment_1` where TensorBoard logs will be stored.

## Step 6: Write Images to TensorBoard

Let's log a grid of sample training images to TensorBoard.

```python
# Get a batch of random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Create a grid of images
img_grid = torchvision.utils.make_grid(images)

# Display the grid (optional, for local verification)
matplotlib_imshow(img_grid, one_channel=True)

# Log the image grid to TensorBoard
writer.add_image('four_fashion_mnist_images', img_grid)
```

To view the logged images, run TensorBoard from your terminal:

```bash
tensorboard --logdir=runs
```

Then navigate to `http://localhost:6006` in your browser.

## Step 7: Visualize the Model Graph

TensorBoard can visualize your model's architecture. Let's log the computational graph.

```python
writer.add_graph(net, images)
writer.close()
```

Refresh TensorBoard and click on the "Graphs" tab to explore the model structure.

## Step 8: Visualize High-Dimensional Data with Embeddings

We can project high-dimensional image data into a lower-dimensional space for visualization.

First, define a helper function to randomly select data points.

```python
def select_n_random(data, labels, n=100):
    """
    Select n random datapoints and their corresponding labels from a dataset.
    """
    assert len(data) == len(labels)
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]
```

Now, log embeddings of a subset of training images.

```python
# Select random images and labels
images, labels = select_n_random(trainset.data, trainset.targets)

# Get class names for the selected labels
class_labels = [classes[lab] for lab in labels]

# Flatten images to 1D vectors (28*28 = 784 dimensions)
features = images.view(-1, 28 * 28)

# Log embeddings to TensorBoard
writer.add_embedding(
    features,
    metadata=class_labels,
    label_img=images.unsqueeze(1)
)
writer.close()
```

In TensorBoard, go to the "Projector" tab to interact with the 3D projection of the data.

## Step 9: Track Model Training

Instead of printing loss values, we'll log them to TensorBoard. We'll also log model predictions on sample batches.

First, define helper functions for generating predictions and plotting.

```python
def images_to_probs(net, images):
    """
    Generate predictions and corresponding probabilities for a batch of images.
    """
    output = net(images)
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

def plot_classes_preds(net, images, labels):
    """
    Create a matplotlib figure showing model predictions vs. actual labels.
    """
    preds, probs = images_to_probs(net, images)
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title(
            "{0}, {1:.1f}%\n(label: {2})".format(
                classes[preds[idx]],
                probs[idx] * 100.0,
                classes[labels[idx]]
            ),
            color=("green" if preds[idx] == labels[idx].item() else "red")
        )
    return fig
```

Now, train the model and log loss and prediction figures every 1000 batches.

```python
running_loss = 0.0
for epoch in range(1):  # Loop over the dataset once for demonstration
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Log every 1000 mini-batches
        if i % 1000 == 999:
            # Log scalar: training loss
            writer.add_scalar(
                'training loss',
                running_loss / 1000,
                epoch * len(trainloader) + i
            )

            # Log figure: predictions vs. actuals
            writer.add_figure(
                'predictions vs. actuals',
                plot_classes_preds(net, inputs, labels),
                global_step=epoch * len(trainloader) + i
            )
            running_loss = 0.0

print('Finished Training')
```

In TensorBoard, check the "Scalars" tab for the loss curve and the "Images" tab for prediction visualizations.

## Step 10: Evaluate Model with Precision-Recall Curves

After training, we can assess per-class performance using precision-recall curves.

First, gather predictions and labels from the test set.

```python
class_probs = []
class_label = []

with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]
        class_probs.append(class_probs_batch)
        class_label.append(labels)

# Concatenate all batches
test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_label = torch.cat(class_label)
```

Define a helper function to log precision-recall curves for each class.

```python
def add_pr_curve_tensorboard(class_index, test_probs, test_label, global_step=0):
    """
    Log a precision-recall curve for a specific class.
    """
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs[:, class_index]
    writer.add_pr_curve(
        classes[class_index],
        tensorboard_truth,
        tensorboard_probs,
        global_step=global_step
    )
    writer.close()
```

Log PR curves for all classes.

```python
for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_label)
```

In TensorBoard, navigate to the "PR Curves" tab to examine the precision-recall performance for each class.

## Conclusion

You've now learned how to integrate TensorBoard with PyTorch to:
- Visualize training data and model architecture.
- Track training loss and model predictions interactively.
- Evaluate model performance with precision-recall curves.

TensorBoard provides an interactive and comprehensive suite of tools to monitor and debug your deep learning experiments, going beyond static print statements or Jupyter Notebook visualizations.