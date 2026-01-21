# A Practical Guide to Forward and Backward Propagation

This guide walks you through the fundamental mechanics of training neural networks: the forward pass, which computes predictions, and the backward pass (backpropagation), which calculates gradients for learning. We'll build intuition by deriving the math for a simple one-hidden-layer network with L2 regularization.

## Prerequisites

This tutorial is conceptual and does not require specific library installations. Familiarity with basic linear algebra, calculus (the chain rule), and neural network concepts is assumed.

## 1. Defining Our Network and Forward Propagation

Forward propagation is the process of calculating predictions by passing input data through the network's layers in sequence. Let's define our simple network mathematically.

Consider an input vector `x ∈ ℝ^d`. Our network has:
*   A hidden layer with weight matrix `W^(1) ∈ ℝ^(h×d)` (no bias for simplicity).
*   An elementwise activation function `ϕ` (e.g., ReLU, sigmoid).
*   An output layer with weight matrix `W^(2) ∈ ℝ^(q×h)`.

The forward pass proceeds in these steps:

**Step 1: Compute the hidden layer's pre-activation.**
```math
z = W^(1) x
```
Here, `z ∈ ℝ^h` is an intermediate variable.

**Step 2: Apply the activation function.**
```math
h = ϕ(z)
```
The vector `h ∈ ℝ^h` is the hidden layer's activation output, another intermediate variable.

**Step 3: Compute the network's raw output.**
```math
o = W^(2) h
```
The vector `o ∈ ℝ^q` is the model's prediction before the final loss function.

**Step 4: Calculate the loss and regularization.**
Let `l(o, y)` be our loss function (e.g., Mean Squared Error, Cross-Entropy) for a true label `y`. The loss for this single example is:
```math
L = l(o, y)
```
We also add L2 regularization (weight decay) to prevent overfitting. The regularization term `s` is:
```math
s = (λ / 2) ( ||W^(1)||_F^2 + ||W^(2)||_F^2 )
```
where `λ` is a hyperparameter and `||·||_F` denotes the Frobenius norm (the L2 norm of the flattened matrix).

**Step 5: Compute the final objective.**
The total objective `J` (the value we aim to minimize during training) is the sum of the loss and the regularization penalty:
```math
J = L + s
```

## 2. Visualizing the Process with a Computational Graph

A computational graph helps visualize the flow of data and dependencies between operations. For our network, the graph flows from the input `x` (bottom-left) to the objective `J` (top-right).

*   **Variables** (like `x`, `h`, `W^(1)`, `J`) are often represented as squares.
*   **Operations** (like matrix multiplication `@`, activation `ϕ`, loss `l`) are represented as circles.
*   Arrows show the direction of data flow during the **forward pass**.

While we can't render the image here, the graph structure for our network would be:
`x → (multiply with W^(1)) → z → (apply ϕ) → h → (multiply with W^(2)) → o → (compute loss l) → L → (add to s) → J`
The regularization term `s` also branches directly from the parameters `W^(1)` and `W^(2)`.

## 3. Calculating Gradients via Backpropagation

Backpropagation is the algorithm for calculating the gradient of the objective `J` with respect to every model parameter (`W^(1)` and `W^(2)`). It works by applying the **chain rule** from calculus, moving backwards through the computational graph.

We need the gradients `∂J/∂W^(1)` and `∂J/∂W^(2)` to update the weights. Let's derive them step-by-step, starting from the output and moving towards the input.

**Step 1: Gradient at the objective.**
The gradients of `J` with respect to its immediate components are simple:
```math
∂J/∂L = 1  and  ∂J/∂s = 1
```

**Step 2: Gradient at the output `o`.**
We use the chain rule: `∂J/∂o = (∂J/∂L) * (∂L/∂o) = ∂L/∂o`.
The specific form of `∂L/∂o ∈ ℝ^q` depends on our chosen loss function `l`.

**Step 3: Gradients for the regularization term.**
The gradients of the regularization term `s` with respect to the parameters are straightforward:
```math
∂s/∂W^(1) = λ W^(1)  and  ∂s/∂W^(2) = λ W^(2)
```

**Step 4: Gradient for the output layer weights `W^(2)`.**
The gradient must account for both the path through the loss `L` and the regularization `s`. Applying the chain rule:
```math
∂J/∂W^(2) = (∂J/∂o) (∂o/∂W^(2)) + (∂J/∂s) (∂s/∂W^(2)) = (∂J/∂o) h^⊤ + λ W^(2)
```
Here, `(∂o/∂W^(2)) = h^⊤` comes from the derivative of `o = W^(2) h` with respect to `W^(2)`. The result `∂J/∂W^(2)` is a matrix in `ℝ^(q×h)`.

**Step 5: Backpropagate to the hidden layer output `h`.**
We now move backward to find how `J` changes with `h`:
```math
∂J/∂h = (∂J/∂o) (∂o/∂h) = (W^(2))^⊤ (∂J/∂o)
```
The result `∂J/∂h ∈ ℝ^h`.

**Step 6: Gradient through the activation function.**
We pass the gradient back through the elementwise activation function `ϕ`. This requires the derivative `ϕ'` evaluated at the forward pass value `z`:
```math
∂J/∂z = (∂J/∂h) ⊙ ϕ'(z)
```
The symbol `⊙` denotes elementwise multiplication (Hadamard product). The result `∂J/∂z ∈ ℝ^h`.

**Step 7: Gradient for the hidden layer weights `W^(1)`.**
Finally, we reach our first set of parameters. Again, we account for both the data and regularization paths:
```math
∂J/∂W^(1) = (∂J/∂z) (∂z/∂W^(1)) + (∂J/∂s) (∂s/∂W^(1)) = (∂J/∂z) x^⊤ + λ W^(1)
```
Here, `(∂z/∂W^(1)) = x^⊤` comes from the derivative of `z = W^(1) x`. The result `∂J/∂W^(1)` is a matrix in `ℝ^(h×d)`.

## 4. The Training Loop: Interleaving Forward and Backward Passes

Training a neural network is a cycle that alternates these two processes:

1.  **Forward Pass:** Using the *current* parameters (`W^(1)`, `W^(2)`), compute all intermediate values (`z`, `h`, `o`, `L`, `s`, `J`) by moving through the graph from input to output.
2.  **Backward Pass:** Using the *stored* intermediate values from the forward pass (`h`, `z`, etc.), compute all gradients (`∂J/∂W^(2)`, `∂J/∂W^(1)`) by moving backward through the graph.
3.  **Parameter Update:** Use the calculated gradients (e.g., with Stochastic Gradient Descent: `W ← W - η * ∂J/∂W`) to adjust the parameters and reduce the objective `J`.

**Key Insight:** The forward and backward passes are deeply interdependent. The backward pass reuses intermediate values (`h`, `z`) computed during the forward pass to avoid recalculating them. This efficiency is critical but has a memory cost—these values must be kept in memory until the backward pass is complete. This is why training a model requires significantly more memory than making a prediction (inference), and why training very deep networks or using large batch sizes can lead to out-of-memory errors.

## Summary

*   **Forward Propagation** sequentially computes and stores intermediate variables from the input layer to the output layer and the final objective.
*   **Backpropagation** sequentially computes the gradients of the objective with respect to all parameters and intermediate variables by applying the chain rule in reverse order.
*   During **training**, these two passes alternate. The forward pass determines the current network behavior and loss, while the backward pass determines how to adjust the parameters to improve. The need to store intermediate values for the backward pass is a primary contributor to the high memory consumption of neural network training.

## Exercises

Test your understanding with these challenges:

1.  If the input `X` to a scalar function `f` is an `n × m` matrix, what are the dimensions of the gradient `∂f/∂X`?
2.  Add a bias term `b ∈ ℝ^h` to the hidden layer of our model (note: biases are typically not regularized).
    *   Sketch the new computational graph.
    *   Derive the updated forward and backpropagation equations.
3.  Analyze the memory footprint. Estimate the memory required for training vs. prediction (inference) for our model, considering the storage of parameters, activations, and gradients.
4.  Imagine you need to compute second derivatives (Hessians). How would the computational graph change? Would this calculation be faster or slower than first-order backpropagation?
5.  Consider a computational graph too large for a single GPU.
    *   Is it possible to partition the graph across multiple GPUs?
    *   What are the trade-offs compared to simply using a smaller batch size?