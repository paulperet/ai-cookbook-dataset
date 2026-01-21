# Data Manipulation with Tensors

This guide introduces the fundamental data structure in deep learning: the tensor. Tensors are multi-dimensional arrays that allow you to store and manipulate numerical data efficiently. If you're familiar with NumPy, you'll find tensors very similar, but with key enhancements like automatic differentiation and GPU acceleration.

## Prerequisites

First, import the necessary libraries for your chosen framework.

```python
# MXNet
from mxnet import np, npx
npx.set_np()

# PyTorch
import torch

# TensorFlow
import tensorflow as tf

# JAX
import jax
from jax import numpy as jnp
```

## 1. Creating Tensors

Tensors can be created in several ways: from sequences, with specific values, or randomly initialized.

### 1.1 Creating a Range Vector

Use the framework's `arange` (or `range` in TensorFlow) function to create a vector of consecutive numbers.

```python
# MXNet
x = np.arange(12)

# PyTorch
x = torch.arange(12, dtype=torch.float32)

# TensorFlow
x = tf.range(12, dtype=tf.float32)

# JAX
x = jnp.arange(12)
```

### 1.2 Inspecting Tensor Properties

Check the total number of elements (size) and the shape of your tensor.

```python
# Size/Number of elements
# MXNet & JAX
x.size
# PyTorch
x.numel()
# TensorFlow
tf.size(x)

# Shape
x.shape
```

### 1.3 Reshaping Tensors

You can change a tensor's shape without changing its data using the `reshape` method. The `-1` argument automatically infers the correct dimension.

```python
# Reshape the 12-element vector into a 3x4 matrix
# MXNet, PyTorch, JAX
X = x.reshape(3, 4)
# TensorFlow
X = tf.reshape(x, (3, 4))

# Using -1 to infer a dimension
X_alt = x.reshape(-1, 4)  # Results in a 3x4 matrix
```

### 1.4 Creating Tensors with Specific Values

Create tensors filled with zeros, ones, random values, or from a Python list.

```python
# Zeros tensor of shape (2, 3, 4)
# MXNet
np.zeros((2, 3, 4))
# PyTorch
torch.zeros((2, 3, 4))
# TensorFlow
tf.zeros((2, 3, 4))
# JAX
jnp.zeros((2, 3, 4))

# Ones tensor
# MXNet
np.ones((2, 3, 4))
# PyTorch
torch.ones((2, 3, 4))
# TensorFlow
tf.ones((2, 3, 4))
# JAX
jnp.ones((2, 3, 4))

# Random tensor from a normal distribution
# MXNet
np.random.normal(0, 1, size=(3, 4))
# PyTorch
torch.randn(3, 4)
# TensorFlow
tf.random.normal(shape=[3, 4])
# JAX (requires a random key)
jax.random.normal(jax.random.PRNGKey(0), (3, 4))

# Tensor from a nested list
# MXNet
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# PyTorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# TensorFlow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# JAX
jnp.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## 2. Indexing and Slicing

Access and modify tensor elements using indexing and slicing, similar to Python lists.

### 2.1 Reading Elements

```python
# Access the last row
X[-1]

# Slice rows 1 and 2 (index 1 to 3, exclusive)
X[1:3]
```

### 2.2 Writing Elements

**Note:** TensorFlow tensors are immutable. Use `tf.Variable` for mutable state. JAX arrays are also immutable; updates create new arrays.

```python
# MXNet & PyTorch: Direct assignment
X[1, 2] = 17

# TensorFlow: Use a Variable
X_var = tf.Variable(X)
X_var[1, 2].assign(9)

# JAX: Create a new array with the update
X_new = X.at[1, 2].set(17)
```

### 2.3 Assigning to Multiple Elements

Assign the same value to a slice of the tensor.

```python
# Set the first two rows to 12
# MXNet & PyTorch
X[:2, :] = 12

# TensorFlow
X_var[:2, :].assign(tf.ones(X_var[:2,:].shape, dtype=tf.float32) * 12)

# JAX
X_new_2 = X_new.at[:2, :].set(12)
```

## 3. Basic Operations

Tensors support elementwise operations, which apply a function to each element individually.

### 3.1 Unary Operations

```python
# Exponential function applied elementwise
# MXNet
np.exp(x)
# PyTorch
torch.exp(x)
# TensorFlow
tf.exp(x)
# JAX
jnp.exp(x)
```

### 3.2 Binary Operations

Perform elementwise addition, subtraction, multiplication, division, and exponentiation.

```python
# Define two vectors
# MXNet
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
# PyTorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
# TensorFlow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
# JAX
x = jnp.array([1.0, 2, 4, 8])
y = jnp.array([2, 2, 2, 2])

# Elementwise operations
x + y, x - y, x * y, x / y, x ** y
```

### 3.3 Concatenation

Join tensors along a specified axis.

```python
# Create two matrices
X = np.arange(12).reshape(3, 4)  # Adjust for your framework
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# Concatenate along rows (axis 0)
# MXNet
np.concatenate([X, Y], axis=0)
# PyTorch
torch.cat((X, Y), dim=0)
# TensorFlow
tf.concat([X, Y], axis=0)
# JAX
jnp.concatenate((X, Y), axis=0)

# Concatenate along columns (axis 1)
# MXNet
np.concatenate([X, Y], axis=1)
# PyTorch
torch.cat((X, Y), dim=1)
# TensorFlow
tf.concat([X, Y], axis=1)
# JAX
jnp.concatenate((X, Y), axis=1)
```

### 3.4 Logical Operations and Reductions

Create a binary tensor via logical statements and compute sums.

```python
# Elementwise comparison
X == Y

# Sum all elements
# MXNet, PyTorch, JAX
X.sum()
# TensorFlow
tf.reduce_sum(X)
```

## 4. Broadcasting

Broadcasting allows elementwise operations on tensors with different shapes by automatically expanding dimensions of size 1.

```python
# Create two matrices with incompatible shapes
# MXNet
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
# PyTorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
# TensorFlow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
# JAX
a = jnp.arange(3).reshape((3, 1))
b = jnp.arange(2).reshape((1, 2))

# Broadcasting adds them as if a were replicated along columns and b along rows
a + b
```

## 5. Memory Management

Performing operations often allocates new memory. For efficiency, especially with large model parameters, you can use in-place operations to update tensors without allocating new memory.

**Note:** TensorFlow uses a graph-based approach for optimization, and JAX arrays are immutable.

```python
# Demonstrating new memory allocation (all frameworks)
Y = Y + X  # This creates a new tensor

# In-place operations (MXNet & PyTorch)
Z = np.zeros_like(Y)  # or torch.zeros_like(Y)
Z[:] = X + Y  # Result is placed in existing Z memory

# Using += for in-place (MXNet & PyTorch)
X += Y  # Updates X without new allocation

# TensorFlow uses Variables and graph compilation for efficiency
@tf.function
def computation(X, Y):
    # TensorFlow prunes unused values and reuses memory internally
    A = X + Y
    B = A + Y
    C = B + Y
    return C + Y
```

## 6. Converting Between Formats

Convert tensors to NumPy arrays and Python scalars for interoperability.

### 6.1 Converting to/from NumPy

```python
# Tensor to NumPy array
# MXNet
A = X.asnumpy()
# PyTorch (shares memory with the tensor)
A = X.numpy()
# TensorFlow
A = X.numpy()
# JAX
A = jax.device_get(X)

# NumPy array to tensor
# MXNet
B = np.array(A)
# PyTorch (shares memory with the array)
B = torch.from_numpy(A)
# TensorFlow
B = tf.constant(A)
# JAX
B = jax.device_put(A)
```

### 6.2 Converting to Python Scalars

Extract a single value from a size-1 tensor.

```python
# Create a size-1 tensor
# MXNet
a = np.array([3.5])
# PyTorch
a = torch.tensor([3.5])
# TensorFlow (convert to NumPy first)
a = tf.constant([3.5]).numpy()
# JAX
a = jnp.array([3.5])

# Extract the scalar value
a.item(), float(a), int(a)
```

## Summary

You've learned the essentials of tensor manipulation:
- Creating tensors with various initializations
- Inspecting and reshaping tensor dimensions
- Accessing and modifying elements via indexing and slicing
- Performing elementwise mathematical and logical operations
- Using broadcasting to operate on tensors with different shapes
- Managing memory with in-place operations
- Converting between tensor formats and Python objects

These operations form the foundation for building and training neural networks, where tensors are used to store data, parameters, and gradients.

## Exercises

1. Experiment with different comparison operators. Change `X == Y` to `X < Y` or `X > Y` and observe the resulting tensor.
2. Test the broadcasting mechanism with 3-dimensional tensors. Create tensors with shapes like `(2, 3, 1)` and `(1, 3, 4)` and perform elementwise addition. Is the result what you expected?