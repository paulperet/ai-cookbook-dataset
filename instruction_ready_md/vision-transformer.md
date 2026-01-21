# Vision Transformers (ViT) Implementation Guide

## Overview
This guide walks through implementing Vision Transformers (ViTs), which adapt the Transformer architecture for computer vision tasks. Unlike traditional CNNs, ViTs process images by splitting them into patches and treating them as sequences, similar to how Transformers handle text tokens.

## Prerequisites

First, install the required libraries:

```bash
pip install torch torchvision  # For PyTorch
# or
pip install flax jaxlib  # For JAX
```

## 1. Patch Embedding

Vision Transformers begin by splitting images into fixed-size patches and projecting them into embeddings.

### Implementation

```python
import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)
```

**How it works:**
- The `PatchEmbedding` class uses a convolutional layer with kernel and stride sizes equal to the patch size
- This effectively splits the image into patches and linearly projects them in one operation
- The output shape is `(batch_size, num_patches, num_hiddens)`

### Testing the Patch Embedding

```python
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = torch.zeros(batch_size, 3, img_size, img_size)
output = patch_emb(X)
print(f"Output shape: {output.shape}")
# Output shape: torch.Size([4, 36, 512])
```

## 2. Vision Transformer MLP

The MLP in ViT differs from the original Transformer's position-wise FFN:

```python
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()  # Smoother alternative to ReLU
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))
```

**Key differences:**
- Uses GELU activation instead of ReLU for smoother gradients
- Applies dropout after each linear layer for regularization

## 3. Vision Transformer Encoder Block

The encoder block uses pre-normalization (normalization before attention/MLP), which improves training stability:

```python
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = MultiHeadAttention(num_hiddens, num_heads,
                                            dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        # Pre-norm: normalize before attention
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        # Pre-norm: normalize before MLP
        return X + self.mlp(self.ln2(X))
```

**Note:** The `MultiHeadAttention` implementation should be imported from your Transformer utilities.

## 4. Complete Vision Transformer

Now we assemble the complete ViT architecture:

```python
class ViT(nn.Module):
    """Vision Transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        
        # Special classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        
        # Learnable positional embeddings
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        
        self.dropout = nn.Dropout(emb_dropout)
        
        # Stack of Transformer blocks
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, num_classes)
        )

    def forward(self, X):
        # 1. Create patch embeddings
        X = self.patch_embedding(X)
        
        # 2. Add classification token to each sequence
        cls_tokens = self.cls_token.expand(X.shape[0], -1, -1)
        X = torch.cat((cls_tokens, X), dim=1)
        
        # 3. Add positional embeddings and apply dropout
        X = self.dropout(X + self.pos_embedding)
        
        # 4. Pass through Transformer blocks
        for blk in self.blks:
            X = blk(X)
        
        # 5. Use only the classification token for final prediction
        return self.head(X[:, 0])
```

## 5. Training the Vision Transformer

Let's train the ViT on Fashion-MNIST:

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

# Hyperparameters
img_size, patch_size = 96, 16
num_hiddens, mlp_num_hiddens = 512, 2048
num_heads, num_blks = 8, 2
emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
batch_size = 128
num_epochs = 10

# Prepare data
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, and optimizer
model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, lr, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1}, Batch: {batch_idx}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    print(f'Epoch {epoch+1} Summary: Loss: {epoch_loss:.4f}, '
          f'Accuracy: {epoch_acc:.2f}%')
```

## Key Insights and Discussion

1. **Performance Characteristics**: On smaller datasets like Fashion-MNIST, ViTs may not outperform well-tuned ResNets. This is because Transformers lack the inductive biases (translation invariance, locality) that make CNNs efficient for vision tasks.

2. **Scalability Advantage**: The true strength of ViTs emerges with larger datasets. When trained on massive datasets (300M+ images), ViTs demonstrate superior scalability and often outperform CNNs.

3. **Computational Considerations**: Self-attention has quadratic complexity with respect to sequence length, making ViTs computationally expensive for high-resolution images. Techniques like Swin Transformers address this by introducing hierarchical attention patterns.

4. **Architecture Variants**: Recent improvements include:
   - **DeiT**: Data-efficient training strategies
   - **Swin Transformers**: Linear complexity with respect to image size through shifted window attention

## Exercises for Further Exploration

1. **Image Size Impact**: Experiment with different `img_size` values (e.g., 64, 128, 256) and observe how training time and accuracy change.

2. **Alternative Pooling Strategies**: Instead of using only the classification token, try averaging all patch representations:
   ```python
   # Replace in forward() method:
   # return self.head(X[:, 0])  # Original
   return self.head(X.mean(dim=1))  # Average pooling
   ```

3. **Hyperparameter Tuning**: Experiment with:
   - Different numbers of Transformer blocks (2, 4, 8, 12)
   - Varying attention heads (4, 8, 16)
   - Adjusting dropout rates for regularization
   - Learning rate schedules

## Summary

Vision Transformers represent a paradigm shift in computer vision, demonstrating that the Transformer architecture can effectively process visual data when properly adapted. While they may not always outperform CNNs on small datasets, their superior scalability makes them increasingly important as dataset sizes grow. The implementation presented here provides a foundation for understanding and experimenting with this powerful architecture.