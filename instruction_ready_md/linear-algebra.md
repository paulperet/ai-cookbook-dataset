# Linear Algebra Fundamentals for Machine Learning

This guide introduces the essential linear algebra concepts you'll need for machine learning and deep learning. We'll start with scalars and build up to matrix operations, all while using practical code examples.

## Prerequisites

First, let's import the necessary libraries. Choose your preferred framework:

```python
# For MXNet
# !pip install mxnet
from mxnet import np, npx
npx.set_np()

# For PyTorch
# !pip install torch
import torch

# For TensorFlow
# !pip install tensorflow
import tensorflow as tf

# For JAX
# !pip install jax jaxlib
from jax import numpy as jnp
```

## 1. Scalars: The Basic Building Blocks

Scalars are single numbers - the simplest mathematical objects. In code, we represent them as tensors containing just one element.

Let's create two scalars and perform basic arithmetic:

```python
# Create scalar tensors
if framework == 'mxnet':
    x = np.array(3.0)
    y = np.array(2.0)
elif framework == 'pytorch':
    x = torch.tensor(3.0)
    y = torch.tensor(2.0)
elif framework == 'tensorflow':
    x = tf.constant(3.0)
    y = tf.constant(2.0)
elif framework == 'jax':
    x = jnp.array(3.0)
    y = jnp.array(2.0)

# Perform operations
print(f"Addition: {x + y}")
print(f"Multiplication: {x * y}")
print(f"Division: {x / y}")
print(f"Exponentiation: {x ** y}")
```

## 2. Vectors: Ordered Collections of Scalars

Vectors are fixed-length arrays of scalars. In machine learning, vectors often represent features or data points.

### Creating and Accessing Vectors

```python
# Create a vector [0, 1, 2]
if framework == 'mxnet':
    x = np.arange(3)
elif framework == 'pytorch':
    x = torch.arange(3)
elif framework == 'tensorflow':
    x = tf.range(3)
elif framework == 'jax':
    x = jnp.arange(3)

print(f"Vector: {x}")
print(f"Third element: {x[2]}")
print(f"Vector length: {len(x)}")
print(f"Vector shape: {x.shape}")
```

**Key Insight:** The `shape` attribute tells us the tensor's dimensions. For vectors, this is a single-element tuple representing the number of components.

## 3. Matrices: Two-Dimensional Arrays

Matrices are 2nd-order tensors with rows and columns. They're essential for representing datasets where rows are samples and columns are features.

### Creating and Manipulating Matrices

```python
# Create a 3x2 matrix
if framework == 'mxnet':
    A = np.arange(6).reshape(3, 2)
elif framework == 'pytorch':
    A = torch.arange(6).reshape(3, 2)
elif framework == 'tensorflow':
    A = tf.reshape(tf.range(6), (3, 2))
elif framework == 'jax':
    A = jnp.arange(6).reshape(3, 2)

print(f"Matrix A:\n{A}")
print(f"Shape: {A.shape}")
```

### Matrix Transposition

The transpose flips rows and columns. For matrix A with elements aᵢⱼ, the transpose Aᵀ has elements aⱼᵢ.

```python
# Get the transpose
if framework == 'tensorflow':
    print(f"Transpose:\n{tf.transpose(A)}")
else:
    print(f"Transpose:\n{A.T}")
```

### Symmetric Matrices

A symmetric matrix equals its own transpose: A = Aᵀ.

```python
# Create a symmetric matrix
if framework == 'mxnet':
    S = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
elif framework == 'pytorch':
    S = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
elif framework == 'tensorflow':
    S = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
elif framework == 'jax':
    S = jnp.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])

# Check symmetry
if framework == 'tensorflow':
    print(f"Is symmetric? {tf.reduce_all(S == tf.transpose(S)).numpy()}")
else:
    print(f"Is symmetric? {(S == S.T).all()}")
```

## 4. Tensors: General n-dimensional Arrays

Tensors generalize vectors and matrices to arbitrary numbers of dimensions. Images, for example, are 3rd-order tensors (height × width × channels).

```python
# Create a 3rd-order tensor (2×3×4)
if framework == 'mxnet':
    T = np.arange(24).reshape(2, 3, 4)
elif framework == 'pytorch':
    T = torch.arange(24).reshape(2, 3, 4)
elif framework == 'tensorflow':
    T = tf.reshape(tf.range(24), (2, 3, 4))
elif framework == 'jax':
    T = jnp.arange(24).reshape(2, 3, 4)

print(f"3rd-order tensor shape: {T.shape}")
```

## 5. Basic Tensor Arithmetic

### Elementwise Operations

Operations between same-shaped tensors occur element-by-element:

```python
# Create two matrices
if framework == 'mxnet':
    A = np.arange(6).reshape(2, 3)
    B = A.copy()  # Independent copy
elif framework == 'pytorch':
    A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    B = A.clone()  # Independent copy
elif framework == 'tensorflow':
    A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
    B = A  # TensorFlow tensors are immutable
elif framework == 'jax':
    A = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    B = A  # JAX arrays are immutable

print(f"A + B:\n{A + B}")
```

### Hadamard (Elementwise) Product

The Hadamard product multiplies corresponding elements:

```python
print(f"Hadamard product A ⊙ B:\n{A * B}")
```

### Scalar-Tensor Operations

Scalars broadcast to match tensor shapes:

```python
a = 2
if framework == 'mxnet':
    X = np.arange(24).reshape(2, 3, 4)
elif framework == 'pytorch':
    X = torch.arange(24).reshape(2, 3, 4)
elif framework == 'tensorflow':
    X = tf.reshape(tf.range(24), (2, 3, 4))
elif framework == 'jax':
    X = jnp.arange(24).reshape(2, 3, 4)

print(f"Scalar addition shape: {(a + X).shape}")
print(f"Scalar multiplication shape: {(a * X).shape}")
```

## 6. Reduction Operations

Reduction operations collapse tensors along specified axes.

### Summation

```python
# Vector sum
if framework == 'mxnet':
    x = np.arange(3, dtype=np.float32)
    vector_sum = x.sum()
elif framework == 'pytorch':
    x = torch.arange(3, dtype=torch.float32)
    vector_sum = x.sum()
elif framework == 'tensorflow':
    x = tf.range(3, dtype=tf.float32)
    vector_sum = tf.reduce_sum(x)
elif framework == 'jax':
    x = jnp.arange(3, dtype=jnp.float32)
    vector_sum = x.sum()

print(f"Vector sum: {vector_sum}")

# Matrix sum along specific axes
print(f"Sum along rows (axis=0): {A.sum(axis=0) if framework != 'tensorflow' else tf.reduce_sum(A, axis=0)}")
print(f"Sum along columns (axis=1): {A.sum(axis=1) if framework != 'tensorflow' else tf.reduce_sum(A, axis=1)}")
```

### Mean (Average)

```python
if framework == 'mxnet':
    print(f"Matrix mean: {A.mean()}")
    print(f"Mean along rows: {A.mean(axis=0)}")
elif framework == 'pytorch':
    print(f"Matrix mean: {A.mean()}")
    print(f"Mean along rows: {A.mean(axis=0)}")
elif framework == 'tensorflow':
    print(f"Matrix mean: {tf.reduce_mean(A).numpy()}")
    print(f"Mean along rows: {tf.reduce_mean(A, axis=0)}")
elif framework == 'jax':
    print(f"Matrix mean: {A.mean()}")
    print(f"Mean along rows: {A.mean(axis=0)}")
```

### Non-Reduction Sum

Keep dimensions while summing for broadcasting:

```python
if framework == 'tensorflow':
    sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
else:
    sum_A = A.sum(axis=1, keepdims=True)

print(f"Sum with keepdims:\n{sum_A}")
print(f"Shape: {sum_A.shape}")

# Broadcasting division
print(f"A normalized by row sums:\n{A / sum_A}")
```

## 7. Dot Products

The dot product combines two vectors to produce a scalar, summing elementwise products.

```python
# Create vectors
if framework == 'mxnet':
    x = np.arange(3, dtype=np.float32)
    y = np.ones(3)
    dot_product = np.dot(x, y)
elif framework == 'pytorch':
    x = torch.arange(3, dtype=torch.float32)
    y = torch.ones(3, dtype=torch.float32)
    dot_product = torch.dot(x, y)
elif framework == 'tensorflow':
    x = tf.range(3, dtype=tf.float32)
    y = tf.ones(3, dtype=tf.float32)
    dot_product = tf.tensordot(x, y, axes=1)
elif framework == 'jax':
    x = jnp.arange(3, dtype=jnp.float32)
    y = jnp.ones(3, dtype=jnp.float32)
    dot_product = jnp.dot(x, y)

print(f"Dot product: {dot_product}")

# Equivalent: elementwise multiplication then sum
if framework == 'mxnet':
    alt_method = np.sum(x * y)
elif framework == 'pytorch':
    alt_method = torch.sum(x * y)
elif framework == 'tensorflow':
    alt_method = tf.reduce_sum(x * y)
elif framework == 'jax':
    alt_method = jnp.sum(x * y)

print(f"Alternative calculation: {alt_method}")
```

## 8. Matrix-Vector Products

Matrix-vector multiplication transforms vectors from one space to another, computing dot products between matrix rows and the vector.

```python
# Matrix-vector multiplication
if framework == 'mxnet':
    result = np.dot(A, x)
elif framework == 'pytorch':
    result = torch.mv(A, x)  # or A @ x
elif framework == 'tensorflow':
    result = tf.linalg.matvec(A, x)
elif framework == 'jax':
    result = jnp.matmul(A, x)

print(f"A shape: {A.shape}, x shape: {x.shape}")
print(f"Ax = {result}")
```

## 9. Matrix-Matrix Multiplication

Matrix multiplication combines two matrices to produce a third, computing dot products between rows of the first and columns of the second.

```python
# Create another matrix
if framework == 'mxnet':
    B = np.ones((3, 4))
    product = np.dot(A, B)
elif framework == 'pytorch':
    B = torch.ones(3, 4)
    product = torch.mm(A, B)  # or A @ B
elif framework == 'tensorflow':
    B = tf.ones((3, 4), tf.float32)
    product = tf.matmul(A, B)
elif framework == 'jax':
    B = jnp.ones((3, 4))
    product = jnp.matmul(A, B)

print(f"A shape: {A.shape}, B shape: {B.shape}")
print(f"AB shape: {product.shape}")
print(f"Product:\n{product}")
```

## 10. Norms: Measuring Vector and Matrix Sizes

Norms quantify the magnitude of vectors and matrices.

### Vector Norms

```python
# Create a vector
if framework == 'mxnet':
    u = np.array([3, -4])
    l2_norm = np.linalg.norm(u)
    l1_norm = np.abs(u).sum()
elif framework == 'pytorch':
    u = torch.tensor([3.0, -4.0])
    l2_norm = torch.norm(u)
    l1_norm = torch.abs(u).sum()
elif framework == 'tensorflow':
    u = tf.constant([3.0, -4.0])
    l2_norm = tf.norm(u)
    l1_norm = tf.reduce_sum(tf.abs(u))
elif framework == 'jax':
    u = jnp.array([3.0, -4.0])
    l2_norm = jnp.linalg.norm(u)
    l1_norm = jnp.linalg.norm(u, ord=1)

print(f"ℓ₂ norm (Euclidean): {l2_norm}")
print(f"ℓ₁ norm (Manhattan): {l1_norm}")
```

### Matrix Norms

```python
# Frobenius norm (like ℓ₂ norm for matrices)
if framework == 'mxnet':
    F_norm = np.linalg.norm(np.ones((4, 9)))
elif framework == 'pytorch':
    F_norm = torch.norm(torch.ones((4, 9)))
elif framework == 'tensorflow':
    F_norm = tf.norm(tf.ones((4, 9)))
elif framework == 'jax':
    F_norm = jnp.linalg.norm(jnp.ones((4, 9)))

print(f"Frobenius norm of 4×9 ones matrix: {F_norm}")
```

## Key Takeaways

1. **Scalars, vectors, matrices, and tensors** are the fundamental objects with 0, 1, 2, and n axes respectively.
2. **Elementwise operations** (like Hadamard product) work on same-shaped tensors.
3. **Reduction operations** (sum, mean) collapse tensors along specified axes.
4. **Dot products and matrix multiplications** are not elementwise and produce outputs with potentially different shapes.
5. **Norms** measure magnitudes, with ℓ₁ and ℓ₂ being common vector norms and Frobenius for matrices.

## Practice Exercises

Test your understanding with these exercises:

1. Prove that (Aᵀ)ᵀ = A for any matrix A.
2. Show that (A + B)ᵀ = Aᵀ + Bᵀ.
3. Is A + Aᵀ always symmetric? Why?
4. For a tensor X of shape (2, 3, 4), what is len(X)?
5. What does A / A.sum(axis=1) compute?
6. Calculate distances using different norms - which corresponds to Manhattan travel?
7. Experiment with norms on higher-order tensors.

These concepts form the foundation for understanding more advanced machine learning algorithms and neural network operations.