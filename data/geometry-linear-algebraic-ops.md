# A Guide to the Geometry and Operations of Linear Algebra

This guide explores the geometric interpretations of linear algebra concepts and introduces key operations essential for deep learning. We'll move beyond the basics to understand vectors as geometric objects, examine dot products and angles, and work with hyperplanes, linear transformations, and tensor operations.

## Prerequisites

We'll use standard linear algebra libraries. The following imports are required.

```python
# For MXNet
from mxnet import np, npx
npx.set_np()

# For PyTorch
import torch

# For TensorFlow
import tensorflow as tf
```

## 1. The Geometry of Vectors

A vector is fundamentally a list of numbers. We can interpret it in two primary geometric ways: as a **point** in space or as a **direction**.

### 1.1 Vectors as Points

Consider a vector as coordinates defining a point's location relative to an origin. For example, the vector `v = [1, 7, 0, 1]` represents a point in a 4-dimensional space. In 2D or 3D, we can easily visualize this.

```python
# A simple vector
v = [1, 7, 0, 1]
```

This perspective allows us to abstract complex problems (like image classification) into tasks involving clusters of points in space.

### 1.2 Vectors as Directions

A vector can also represent a direction and magnitude. The vector `[3, 2]` means "move 3 units right and 2 units up." All arrows pointing in the same direction with the same length represent the same vector.

This view makes operations like vector addition intuitive: you follow one direction, then the other.

```python
# Vector addition conceptually
u = [1, 2]
v = [3, 1]
# The sum w = u + v = [4, 3] is the point reached by following u then v.
```

## 2. Dot Products and Angles

The dot product between two vectors `u` and `v` is defined as:

\[
\mathbf{u} \cdot \mathbf{v} = \sum_i u_i \cdot v_i
\]

It has a crucial geometric interpretation: it's related to the cosine of the angle between the vectors.

\[
\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos(\theta)
\]

We can rearrange this to compute the angle \(\theta\):

\[
\theta = \arccos\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}\right)
\]

Let's implement a function to compute the angle between two vectors.

```python
def angle(v, w):
    """Compute the angle between vectors v and w."""
    # For MXNet
    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))
    
    # For PyTorch
    # return torch.acos(v.dot(w) / (torch.norm(v) * torch.norm(w)))
    
    # For TensorFlow
    # return tf.acos(tf.tensordot(v, w, axes=1) / (tf.norm(v) * tf.norm(w)))

# Example calculation
v = [0, 1, 2]
w = [2, 3, 4]
print(f"Angle: {angle(v, w)}")
```

**Key Insight:** Two vectors are **orthogonal** if their dot product is zero (\(\theta = 90^\circ\)). This property is useful for understanding geometric relationships in high-dimensional data.

### 2.1 Cosine Similarity

In machine learning, we often use **cosine similarity** to measure vector closeness:

\[
\cos(\theta) = \frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}
\]

It ranges from 1 (same direction) to -1 (opposite directions), with 0 indicating orthogonality. For high-dimensional random vectors with mean 0, the cosine similarity is typically near 0.

## 3. Working with Hyperplanes

A hyperplane is a high-dimensional generalization of a line (2D) or plane (3D). In a \(d\)-dimensional space, a hyperplane has \(d-1\) dimensions and divides the space into two half-spaces.

Consider a weight vector \(\mathbf{w} = [2, 1]^\top\). The set of points \(\mathbf{v}\) satisfying \(\mathbf{w}\cdot\mathbf{v} = 1\) forms a line perpendicular to \(\mathbf{w}\). The inequality \(\mathbf{w}\cdot\mathbf{v} > 1\) defines one half-space, while \(\mathbf{w}\cdot\mathbf{v} < 1\) defines the other.

In higher dimensions, \(\mathbf{w}\cdot\mathbf{v} = 1\) defines a plane. These hyperplanes are fundamental to linear classification models, where they act as **decision boundaries** separating different classes.

### 3.1 Practical Example: Classifying Fashion-MNIST

Let's build a simple classifier for t-shirts vs trousers from the Fashion-MNIST dataset using a hyperplane defined by the difference between class means.

First, load the data and compute average images.

```python
# For MXNet
import gluon
from gluon.data.vision import FashionMNIST

# Load dataset
train = FashionMNIST(train=True)
test = FashionMNIST(train=False)

# Separate t-shirts (class 0) and trousers (class 1)
X_train_0 = np.stack([x[0] for x in train if x[1] == 0]).astype(float)
X_train_1 = np.stack([x[0] for x in train if x[1] == 1]).astype(float)
X_test = np.stack([x[0] for x in test if x[1] == 0 or x[1] == 1]).astype(float)
y_test = np.stack([x[1] for x in test if x[1] == 0 or x[1] == 1]).astype(float)

# Compute class averages
ave_0 = np.mean(X_train_0, axis=0)  # Average t-shirt
ave_1 = np.mean(X_train_1, axis=0)  # Average trousers
```

Now, define our weight vector as the difference between averages and set a classification threshold.

```python
# Define weight vector and threshold
w = (ave_1 - ave_0).T
threshold = -1500000  # Eyeballed from data

# Make predictions
predictions = X_test.reshape(2000, -1).dot(w.flatten()) > threshold

# Calculate accuracy
accuracy = np.mean(predictions.astype(y_test.dtype) == y_test, dtype=np.float64)
print(f"Test accuracy: {accuracy:.3f}")
```

This simple linear classifier demonstrates how hyperplanes can separate classes. In practice, we would learn the threshold from data rather than setting it manually.

## 4. Geometry of Linear Transformations

Matrices represent linear transformations. When we multiply a vector by a matrix \(\mathbf{A}\), we transform it in ways that can include scaling, rotating, and skewing—but the transformation must be uniform across the entire space.

Consider a 2×2 matrix:

\[
\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
\]

When applied to a vector \([x, y]^\top\), we get:

\[
\mathbf{A}\mathbf{v} = x\begin{bmatrix}a \\ c\end{bmatrix} + y\begin{bmatrix}b \\ d\end{bmatrix}
\]

This shows that the transformation of any vector can be understood by how it transforms the basis vectors \([1,0]^\top\) and \([0,1]^\top\).

### 4.1 Visualizing Transformations

Let's examine a specific transformation:

\[
\mathbf{A} = \begin{bmatrix} 1 & 2 \\ -1 & 3 \end{bmatrix}
\]

This matrix skews and rotates the 2D grid while preserving its fundamental structure. Some matrices, however, can cause more severe distortions.

```python
# Example transformation matrix
A = np.array([[1, 2], [-1, 3]])
v = np.array([2, -1])
transformed_v = A.dot(v)
print(f"Original: {v}, Transformed: {transformed_v}")
```

## 5. Linear Dependence and Rank

Vectors are **linearly dependent** if one can be written as a combination of others. For example, if \(\mathbf{b}_1 = -2 \cdot \mathbf{b}_2\), then these vectors are linearly dependent.

The **rank** of a matrix is the maximum number of linearly independent columns. A full-rank \(n \times n\) matrix has rank \(n\), meaning it doesn't compress the space into a lower dimension.

```python
# Check matrix rank
B = np.array([[2, 4], [-1, -2]])
rank_B = np.linalg.matrix_rank(B)
print(f"Rank of B: {rank_B}")  # Output: 1 (columns are dependent)
```

## 6. Matrix Invertibility and Determinants

A matrix is **invertible** if there exists a matrix \(\mathbf{A}^{-1}\) such that \(\mathbf{A}^{-1}\mathbf{A} = \mathbf{I}\), where \(\mathbf{I}\) is the identity matrix.

The **determinant** measures how much a matrix scales areas (in 2D) or volumes (in higher dimensions). A nonzero determinant indicates an invertible matrix.

```python
# Compute determinant
A = np.array([[1, -1], [2, 3]])
det_A = np.linalg.det(A)
print(f"Determinant of A: {det_A}")

# Check invertibility
is_invertible = not np.isclose(det_A, 0)
print(f"Is A invertible? {is_invertible}")
```

**Important Note:** While matrix inversion is useful theoretically, in practice we often solve linear equations directly using more numerically stable methods than computing the inverse explicitly.

## 7. Tensor Operations and Einstein Notation

Tensors generalize matrices to higher dimensions. **Tensor contractions** are the tensor equivalent of matrix multiplication, and **Einstein notation** provides a concise way to express them.

### 7.1 Einstein Summation Examples

Many linear algebra operations can be expressed compactly using Einstein notation:

- Dot product: `"i,i->"`
- Matrix-vector multiplication: `"ij,j->i"`
- Matrix multiplication: `"ij,jk->ik"`
- Trace: `"ii->"`

```python
# Define tensors
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 2])
B = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

# Matrix-vector multiplication using einsum
result_einsum = np.einsum("ij, j -> i", A, v)
result_dot = A.dot(v)
print(f"Einsum result: {result_einsum}")
print(f"Dot result: {result_dot}")
print(f"Results match: {np.allclose(result_einsum, result_dot)}")

# More complex tensor contraction
complex_result = np.einsum("ijk, il, j -> kl", B, A, v)
print(f"Complex contraction shape: {complex_result.shape}")
```

Einstein notation is both human-readable and computationally efficient, making it invaluable for expressing complex tensor operations in deep learning.

## Summary

In this guide, we've explored:

1. **Vector Geometry**: Vectors as points or directions in space
2. **Dot Products & Angles**: Measuring similarity and orthogonality between vectors
3. **Hyperplanes**: Decision boundaries for classification tasks
4. **Linear Transformations**: How matrices uniformly transform space
5. **Linear Dependence & Rank**: Understanding when matrices compress space
6. **Invertibility & Determinants**: Measuring how matrices scale volumes
7. **Tensor Operations**: Using Einstein notation for concise expressions

These geometric perspectives provide intuition for the linear algebra operations that form the foundation of deep learning models.

## Exercises

1. Compute the angle between \(\vec v_1 = [1, 0, -1, 2]^\top\) and \(\vec v_2 = [3, 1, 0, 1]^\top\).
2. Verify whether \(\begin{bmatrix}1 & 2\\0&1\end{bmatrix}\) and \(\begin{bmatrix}1 & -2\\0&1\end{bmatrix}\) are inverses.
3. If a shape has area \(100\textrm{m}^2\), what is its area after transformation by \(\begin{bmatrix}2 & 3\\1 & 2\end{bmatrix}\)?
4. Determine which sets of vectors are linearly independent.
5. If \(A = \begin{bmatrix}c\\d\end{bmatrix}\cdot\begin{bmatrix}a & b\end{bmatrix}\), is its determinant always zero?
6. What condition on matrix \(A\) ensures \(Ae_1\) and \(Ae_2\) remain orthogonal?
7. Express \(\textrm{tr}(\mathbf{A}^4)\) in Einstein notation.