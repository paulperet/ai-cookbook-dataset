# A Practical Guide to Eigendecompositions

Eigendecompositions are a cornerstone of linear algebra with profound implications in machine learning and numerical computing. This guide will walk you through the core concepts, demonstrate practical computations, and show a key application in neural network initialization.

## Prerequisites

Ensure you have the necessary libraries installed. We'll use NumPy for demonstrations, but equivalent PyTorch and TensorFlow code is provided.

```bash
pip install numpy matplotlib
```

For the full interactive experience, you can use Jupyter Notebook:

```bash
pip install notebook
```

## 1. Understanding Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors reveal how a matrix transforms space. Consider matrix A:

```python
import numpy as np

A = np.array([[2, 0],
              [0, -1]])
```

When applied to vector `v = [x, y]`, we get `Av = [2x, -y]`. This stretches x by 2 and flips y.

However, some vectors don't change direction:
- `[1, 0]` → `[2, 0]` (stretched by 2)
- `[0, 1]` → `[0, -1]` (stretched by -1)

These special vectors are **eigenvectors**, and their scaling factors are **eigenvalues**.

Formally, for matrix A, vector v, and scalar λ:
```
A v = λ v
```
Here, v is an eigenvector and λ is its corresponding eigenvalue.

## 2. Finding Eigenvalues and Eigenvectors

### 2.1 The Characteristic Equation

From `A v = λ v`, we rearrange to:
```
(A - λI) v = 0
```
where I is the identity matrix.

For non-zero v to exist, `(A - λI)` must be singular (non-invertible), meaning:
```
det(A - λI) = 0
```
This is the **characteristic equation**.

### 2.2 Example Calculation

Let's find eigenvalues and eigenvectors for:

```python
A = np.array([[2, 1],
              [2, 3]])
```

First, compute eigenvalues by solving `det(A - λI) = 0`:

```python
import sympy as sp

λ = sp.symbols('λ')
A_sym = sp.Matrix([[2, 1], [2, 3]])
char_poly = (A_sym - λ * sp.eye(2)).det()
eigenvalues = sp.solve(char_poly, λ)
print(f"Eigenvalues: {eigenvalues}")
```

This gives eigenvalues λ = 4 and λ = 1.

Now find eigenvectors by solving `(A - λI)v = 0` for each λ:

For λ = 1:
```python
# Solve (A - I)v = 0
v1 = np.linalg.solve(A - 1*np.eye(2), [0, 0])
print(f"Eigenvector for λ=1: {v1}")
```

For λ = 4:
```python
# Solve (A - 4I)v = 0
v2 = np.linalg.solve(A - 4*np.eye(2), [0, 0])
print(f"Eigenvector for λ=4: {v2}")
```

### 2.3 Using Built-in Functions

NumPy provides efficient computation:

```python
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")
```

Note: NumPy normalizes eigenvectors to unit length, while our manual calculation produced vectors of arbitrary length. Both are valid—eigenvectors are defined up to a scalar multiple.

## 3. Matrix Decomposition via Eigendecomposition

If we collect eigenvectors as columns in matrix W and eigenvalues on diagonal of Σ:

```python
W = np.column_stack([v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)])
Σ = np.diag(eigenvalues)

print(f"W (eigenvectors):\n{W}")
print(f"Σ (eigenvalues):\n{Σ}")
```

The eigendecomposition states:
```
A = W Σ W⁻¹
```

Let's verify:

```python
A_reconstructed = W @ Σ @ np.linalg.inv(W)
print(f"Original A:\n{A}")
print(f"Reconstructed A:\n{A_reconstructed}")
print(f"Reconstruction error: {np.max(np.abs(A - A_reconstructed))}")
```

## 4. Operations Simplified by Eigendecomposition

### 4.1 Matrix Powers

Computing Aⁿ directly is expensive for large n. With eigendecomposition:
```
Aⁿ = W Σⁿ W⁻¹
```
Only Σ needs exponentiation (diagonal matrix powers are trivial).

```python
n = 10
A_power_direct = np.linalg.matrix_power(A, n)
A_power_eig = W @ np.linalg.matrix_power(Σ, n) @ np.linalg.inv(W)

print(f"Direct A¹⁰:\n{A_power_direct}")
print(f"Via eigendecomposition:\n{A_power_eig}")
```

### 4.2 Matrix Inverse

If A is invertible (all eigenvalues ≠ 0):
```
A⁻¹ = W Σ⁻¹ W⁻¹
```
Simply invert the eigenvalues on Σ's diagonal.

```python
A_inv_direct = np.linalg.inv(A)
A_inv_eig = W @ np.linalg.inv(Σ) @ np.linalg.inv(W)

print(f"Direct inverse:\n{A_inv_direct}")
print(f"Via eigendecomposition:\n{A_inv_eig}")
```

### 4.3 Determinant

The determinant equals the product of eigenvalues:
```
det(A) = λ₁ × λ₂ × ... × λₙ
```

```python
det_direct = np.linalg.det(A)
det_eig = np.prod(eigenvalues)

print(f"Direct determinant: {det_direct}")
print(f"Product of eigenvalues: {det_eig}")
```

## 5. Special Case: Symmetric Matrices

For symmetric matrices (A = Aᵀ), eigenvectors are orthogonal:
```
A = W Σ Wᵀ  (where W⁻¹ = Wᵀ)
```

```python
# Create a symmetric matrix
A_sym = np.array([[2, 1],
                  [1, 2]])

eigvals_sym, eigvecs_sym = np.linalg.eig(A_sym)
print(f"Symmetric matrix eigenvectors:\n{eigvecs_sym}")
print(f"Check orthogonality: {eigvecs_sym.T @ eigvecs_sym}")
```

## 6. Gershgorin Circle Theorem: Eigenvalue Approximation

For large matrices, computing eigenvalues exactly can be expensive. The Gershgorin Circle Theorem provides bounds:

For matrix A with entries aᵢⱼ, define radius rᵢ = Σ_{j≠i} |aᵢⱼ|. Each eigenvalue lies in at least one disc centered at aᵢᵢ with radius rᵢ.

### 6.1 Example Application

```python
A_large = np.array([[1.0, 0.1, 0.1, 0.1],
                    [0.1, 3.0, 0.2, 0.3],
                    [0.1, 0.2, 5.0, 0.5],
                    [0.1, 0.3, 0.5, 9.0]])

# Compute Gershgorin discs
n = A_large.shape[0]
for i in range(n):
    center = A_large[i, i]
    radius = np.sum(np.abs(A_large[i, :])) - np.abs(center)
    print(f"Disc {i}: center={center:.1f}, radius={radius:.1f}, "
          f"range=[{center-radius:.1f}, {center+radius:.1f}]")

# Compare with actual eigenvalues
actual_eigenvalues = np.linalg.eigvals(A_large)
print(f"\nActual eigenvalues: {np.sort(actual_eigenvalues)}")
```

Each eigenvalue should fall within at least one of these ranges.

## 7. Practical Application: Neural Network Initialization

Eigendecomposition helps understand why careful weight initialization is crucial in deep learning.

### 7.1 The Problem

Consider a deep network with repeated matrix multiplication:
```
v_out = Aᴺ v_in
```
If A stretches vectors, small input changes explode. If A shrinks vectors, information vanishes. We need balanced growth.

### 7.2 Experiment with Random Matrix

```python
import matplotlib.pyplot as plt

np.random.seed(42)
k = 5
A_random = np.random.randn(k, k)

# Track norm growth
v = np.random.randn(k, 1)
norms = [np.linalg.norm(v)]

for i in range(1, 100):
    v = A_random @ v
    norms.append(np.linalg.norm(v))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(norms)
plt.xlabel('Iteration')
plt.ylabel('Vector norm')
plt.title('Uncontrolled growth')

plt.subplot(1, 2, 2)
ratios = [norms[i]/norms[i-1] for i in range(1, len(norms))]
plt.plot(ratios)
plt.xlabel('Iteration')
plt.ylabel('Growth ratio')
plt.title('Stabilizing growth factor')
plt.tight_layout()
plt.show()
```

### 7.3 Connection to Principal Eigenvalue

The growth rate converges to the largest eigenvalue's magnitude:

```python
eigenvalues = np.linalg.eigvals(A_random)
largest_eig = np.max(np.abs(eigenvalues))
print(f"Largest eigenvalue magnitude: {largest_eig:.4f}")
print(f"Final growth ratio: {ratios[-1]:.4f}")
```

### 7.4 Proper Normalization

To prevent explosion, normalize by the principal eigenvalue:

```python
A_normalized = A_random / largest_eig

# Repeat experiment
v = np.random.randn(k, 1)
norms_normalized = [np.linalg.norm(v)]

for i in range(1, 100):
    v = A_normalized @ v
    norms_normalized.append(np.linalg.norm(v))

plt.plot(norms_normalized)
plt.xlabel('Iteration')
plt.ylabel('Vector norm')
plt.title('Controlled growth after normalization')
plt.show()
```

Now the norm stabilizes, which is desirable for neural network training.

## 8. Key Insights

1. **Eigenvectors** are transformation-invariant directions
2. **Eigenvalues** quantify stretching along eigenvectors
3. **Eigendecomposition** simplifies matrix operations
4. **Gershgorin's Theorem** provides eigenvalue bounds without full computation
5. **Principal eigenvalue** determines long-term behavior of iterated matrix multiplication

## 9. Exercises

Test your understanding:

1. Find eigenvalues and eigenvectors for:
   ```python
   B = np.array([[2, 1],
                 [1, 2]])
   ```

2. What happens with defective matrices? Try:
   ```python
   C = np.array([[2, 1],
                 [0, 2]])
   ```
   This matrix doesn't have enough eigenvectors—it's "defective."

3. Use Gershgorin's Theorem: Without computing, could the smallest eigenvalue of this matrix be less than 0.5?
   ```python
   D = np.array([[3.0, 0.1, 0.3, 1.0],
                 [0.1, 1.0, 0.1, 0.2],
                 [0.3, 0.1, 5.0, 0.0],
                 [1.0, 0.2, 0.0, 1.8]])
   ```

## 10. Further Reading

- Golub & Van Loan, "Matrix Computations" for algorithmic details
- The "circular law" for random matrix eigenvalue distributions
- Research on neural network initialization (Pennington et al., 2017)

Eigendecomposition is more than a mathematical curiosity—it's a practical tool for understanding and controlling linear transformations in machine learning systems.