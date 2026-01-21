# Learning PyTorch: A Step-by-Step Tutorial

## Introduction
This tutorial introduces the fundamental concepts of PyTorch through practical, self-contained examples. You'll learn how PyTorch combines the power of n-dimensional Tensors (similar to NumPy) with automatic differentiation for building and training neural networks.

We'll use a concrete problem throughout: fitting a third-order polynomial to approximate the sine function. Our network will have four parameters and will be trained using gradient descent to minimize the Euclidean distance between predictions and true values.

## Prerequisites
Before starting, ensure you have the necessary packages installed:

```bash
pip install torch numpy
```

## Part 1: Understanding Tensors

### Step 1: Warm-up with NumPy
Before diving into PyTorch, let's implement our polynomial fitting using NumPy to understand the core concepts. NumPy provides n-dimensional arrays but lacks GPU acceleration and automatic differentiation.

```python
import numpy as np
import math

# Create random input and output data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# Randomly initialize weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')
```

### Step 2: Introducing PyTorch Tensors
PyTorch Tensors are similar to NumPy arrays but can leverage GPUs for acceleration. Let's recreate our example using PyTorch Tensors:

```python
import torch
import math

# Create Tensors to hold input and output
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), dtype=torch.float)
b = torch.randn((), dtype=torch.float)
c = torch.randn((), dtype=torch.float)
d = torch.randn((), dtype=torch.float)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    # Update weights
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
```

## Part 2: Automatic Differentiation with Autograd

### Step 3: Using Tensors with Autograd
Manually computing gradients becomes impractical for complex networks. PyTorch's autograd automates this process by tracking operations on Tensors with `requires_grad=True`.

```python
import torch
import math

# Create Tensors with gradient tracking
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Randomly initialize weights with gradient tracking
a = torch.randn((), dtype=torch.float, requires_grad=True)
b = torch.randn((), dtype=torch.float, requires_grad=True)
c = torch.randn((), dtype=torch.float, requires_grad=True)
d = torch.randn((), dtype=torch.float, requires_grad=True)

learning_rate = 1e-6
for t in range(2000):
    # Forward pass
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # Compute loss
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute gradients
    loss.backward()

    # Manually update weights (no gradient tracking during update)
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        # Zero gradients for next iteration
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')
```

### Step 4: Defining Custom Autograd Functions
You can create custom autograd operations by subclassing `torch.autograd.Function`. Here's an example using Legendre polynomials:

```python
import torch
import math

class LegendrePolynomial3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)

# Prepare data
x = torch.linspace(-math.pi, math.pi, 2000, dtype=torch.float)
y = torch.sin(x)

# Initialize weights
a = torch.full((), 0.0, dtype=torch.float, requires_grad=True)
b = torch.full((), -1.0, dtype=torch.float, requires_grad=True)
c = torch.full((), 0.0, dtype=torch.float, requires_grad=True)
d = torch.full((), 0.3, dtype=torch.float, requires_grad=True)

learning_rate = 5e-6
for t in range(2000):
    P3 = LegendrePolynomial3.apply
    
    # Forward pass
    y_pred = a + b * P3(c + d * x)
    
    # Compute loss
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())
    
    # Backward pass
    loss.backward()
    
    # Update weights
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad
        
        # Zero gradients
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f'Result: y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)')
```

## Part 3: Building Neural Networks with `nn` Module

### Step 5: Using the `nn` Module
The `nn` package provides higher-level abstractions for building neural networks. Let's recreate our polynomial model using `nn`:

```python
import torch
import math

# Create Tensors
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Prepare input tensor (x, x^2, x^3)
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# Define model
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for t in range(2000):
    # Forward pass
    y_pred = model(xx)
    
    # Compute loss
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    
    # Zero gradients
    model.zero_grad()
    
    # Backward pass
    loss.backward()
    
    # Update weights
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# Extract final layer weights
linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
```

### Step 6: Using Optimizers
PyTorch's `optim` package provides various optimization algorithms. Let's use RMSprop:

```python
import torch
import math

# Create Tensors
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Prepare input tensor
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

# Define model
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

# Define loss function and optimizer
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

for t in range(2000):
    # Forward pass
    y_pred = model(xx)
    
    # Compute loss
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    
    # Zero gradients, backward pass, optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Extract final layer weights
linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')
```

### Step 7: Creating Custom Modules
For more complex models, you can define custom modules by subclassing `nn.Module`:

```python
import torch
import math

class Polynomial3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

# Create data
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Create model
model = Polynomial3()

# Define loss and optimizer
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

for t in range(2000):
    # Forward pass
    y_pred = model(x)
    
    # Compute loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')
```

### Step 8: Dynamic Graphs and Weight Sharing
PyTorch supports dynamic computational graphs. Here's an example with weight sharing:

```python
import random
import torch
import math

class DynamicNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        for _ in range(random.randint(3, 5)):
            y = y + self.e * x ** 4
        return y

    def string(self):
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ?'

# Create data
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Create model
model = DynamicNet()

# Define loss and optimizer
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)

for t in range(30000):
    # Forward pass
    y_pred = model(x)
    
    # Compute loss
    loss = criterion(y_pred, y)
    if t % 2000 == 1999:
        print(t, loss.item())
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'Result: {model.string()}')
```

## Conclusion
You've now learned the fundamental concepts of PyTorch through practical examples. You started with basic Tensors, progressed to automatic differentiation with autograd, and finally built neural networks using the `nn` module and optimizers. These concepts form the foundation for more advanced deep learning applications with PyTorch.

Remember that PyTorch's dynamic computational graph and intuitive API make it an excellent choice for both research and production applications. As you continue learning, explore the official PyTorch documentation for more advanced features and best practices.