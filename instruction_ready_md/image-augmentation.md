# Image Augmentation Guide

This guide walks you through implementing common image augmentation techniques to improve the generalization of your deep learning models. You'll learn how to apply transformations like flipping, cropping, and color adjustments, then train a model using these augmented images.

## Prerequisites

First, install the required libraries and import the necessary modules.

```bash
pip install torch torchvision matplotlib
```

```python
import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
```

## 1. Understanding Image Augmentation

Image augmentation creates variations of your training images through random transformations. This serves two main purposes:
- **Expands your dataset** by generating similar but distinct examples
- **Improves model generalization** by reducing reliance on specific attributes like object position or color

We'll use a sample cat image to demonstrate each technique.

```python
# Load and display the sample image
img = plt.imread('../img/cat1.jpg')
plt.imshow(img)
plt.show()
```

## 2. Creating a Visualization Helper

To compare different augmentation effects, we'll create a helper function that applies a transformation multiple times and displays the results.

```python
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    """Apply augmentation multiple times and display results."""
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(scale*num_cols, scale*num_rows))
    for i, ax in enumerate(axes.flat):
        ax.imshow(Y[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()
```

## 3. Basic Spatial Transformations

### 3.1 Horizontal Flipping
Flipping images horizontally preserves object categories while creating variation. This is one of the most common augmentations.

```python
horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
apply(img, horizontal_flip)
```

### 3.2 Vertical Flipping
Vertical flipping is less common but can be useful for certain datasets.

```python
vertical_flip = torchvision.transforms.RandomVerticalFlip(p=0.5)
apply(img, vertical_flip)
```

### 3.3 Random Cropping
Random cropping helps models become less sensitive to object position by showing objects in different locations and scales.

```python
random_crop = torchvision.transforms.RandomResizedCrop(
    size=(200, 200),
    scale=(0.1, 1.0),      # Crop 10% to 100% of original area
    ratio=(0.5, 2.0)       # Aspect ratio between 0.5 and 2.0
)
apply(img, random_crop)
```

## 4. Color Transformations

### 4.1 Adjusting Brightness
Random brightness changes help models become less sensitive to lighting conditions.

```python
brightness_aug = torchvision.transforms.ColorJitter(
    brightness=0.5,  # Random brightness between 50% and 150%
    contrast=0,
    saturation=0,
    hue=0
)
apply(img, brightness_aug)
```

### 4.2 Adjusting Hue
Changing hue can help models focus on shapes rather than colors.

```python
hue_aug = torchvision.transforms.ColorJitter(
    brightness=0,
    contrast=0,
    saturation=0,
    hue=0.5  # Maximum hue shift
)
apply(img, hue_aug)
```

### 4.3 Combined Color Adjustments
You can combine multiple color transformations for more robust augmentation.

```python
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5,
    contrast=0.5,
    saturation=0.5,
    hue=0.5
)
apply(img, color_aug)
```

## 5. Combining Multiple Augmentations

In practice, you'll want to combine several augmentation techniques. Use `Compose` to chain transformations together.

```python
combined_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    color_aug,
    random_crop
])
apply(img, combined_augs)
```

## 6. Training with Image Augmentation

Now let's apply these techniques to train a model on the CIFAR-10 dataset, which has more color and size variation than Fashion-MNIST.

### 6.1 Setting Up Data Loaders

First, define the augmentations for training and testing. Note that we only apply random augmentations during training.

```python
# Training augmentations (include random operations)
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()  # Convert to tensor and normalize to [0,1]
])

# Testing augmentations (no random operations)
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
```

Next, create data loading functions:

```python
def load_cifar10(is_train, augs, batch_size):
    """Load CIFAR-10 dataset with specified augmentations."""
    dataset = torchvision.datasets.CIFAR10(
        root="../data",
        train=is_train,
        transform=augs,
        download=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=2  # Adjust based on your system
    )
    return dataloader
```

### 6.2 Training Function

We'll use a ResNet-18 model and train it with our augmented data.

```python
def train_with_data_aug(train_augs, test_augs, net, lr=0.001, num_epochs=10):
    """Train model with image augmentation."""
    batch_size = 256
    devices = ['cuda:0'] if torch.cuda.is_available() else ['cpu']
    
    # Load data
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    
    # Initialize model
    net = net.to(devices[0])
    
    # Loss and optimizer
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        net.train()
        train_loss, train_acc = 0.0, 0.0
        
        for X, y in train_iter:
            X, y = X.to(devices[0]), y.to(devices[0])
            
            trainer.zero_grad()
            pred = net(X)
            l = loss(pred, y)
            l.backward()
            trainer.step()
            
            train_loss += l.item()
            train_acc += (pred.argmax(1) == y).float().mean().item()
        
        # Calculate averages
        avg_loss = train_loss / len(train_iter)
        avg_acc = train_acc / len(train_iter)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.3f}, Accuracy = {avg_acc:.3f}")
    
    return net
```

### 6.3 Running the Training

```python
# Initialize model (simplified ResNet-18)
net = d2l.resnet18(10, 3)  # 10 classes, 3 color channels
net.apply(d2l.init_cnn)    # Initialize weights

# Train with augmentation
train_with_data_aug(train_augs, test_augs, net)
```

## 7. Key Takeaways

1. **Image augmentation creates training diversity** by applying random transformations to your images
2. **Use different strategies**:
   - Spatial: flipping, cropping, rotation
   - Color: brightness, contrast, saturation, hue adjustments
3. **Apply augmentations only during training** - use deterministic transforms for testing
4. **Combine multiple augmentations** using `Compose` for maximum effect
5. **Monitor results** to ensure augmentations improve rather than hurt performance

## 8. Exercises

1. **Compare with and without augmentation**: Train the model using `train_with_data_aug(test_augs, test_augs)` (no augmentation) and compare the test accuracy with the augmented version. Does this support the argument that augmentation reduces overfitting?

2. **Experiment with combinations**: Try different combinations of the augmentation methods covered. Which combination gives the best test accuracy on CIFAR-10?

3. **Explore additional augmentations**: Check the PyTorch documentation for other augmentation methods like rotation, perspective transforms, or auto-augment policies. How do these affect your results?

## Next Steps

- Experiment with the augmentation parameters to find optimal values for your specific dataset
- Try implementing custom augmentation functions for domain-specific transformations
- Consider using automated augmentation search techniques like AutoAugment or RandAugment for more sophisticated strategies

Remember: The goal of augmentation is to create realistic variations that your model might encounter in production, not to distort images beyond recognition. Always visualize your augmented images to ensure they remain meaningful.