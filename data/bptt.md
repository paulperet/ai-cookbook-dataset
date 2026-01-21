# A Guide to Backpropagation Through Time (BPTT) for RNNs

## Introduction

In previous experiments with Recurrent Neural Networks (RNNs), you likely encountered the need for gradient clipping to prevent training instability caused by massive gradients. These exploding gradients often originate from backpropagating across long sequences. Before diving into modern RNN architectures, let's examine the mechanics of backpropagation in sequence models mathematically. This discussion will clarify the concepts of *vanishing* and *exploding* gradients.

Backpropagation in RNNs is called **Backpropagation Through Time (BPTT)**. This technique involves unrolling the RNN's computational graph across time steps, transforming it into a deep feedforward network where the same parameters repeat at each step. We can then apply the chain rule to backpropagate gradients, summing contributions for each parameter across all its occurrences.

However, long sequences (common in text with over a thousand tokens) pose computational and optimization challenges. An input from the first step undergoes hundreds of matrix multiplications before producing an output and computing its gradient. Let's analyze the problems and practical solutions.

## 1. Analyzing Gradients in RNNs

We begin with a simplified RNN model to build intuition, ignoring specific hidden state update details. We denote:
- $h_t$: hidden state at step $t$
- $x_t$: input at step $t$
- $o_t$: output at step $t$

The transformations are:
$$
\begin{aligned}
h_t &= f(x_t, h_{t-1}, w_\textrm{h}),\\
o_t &= g(h_t, w_\textrm{o}),
\end{aligned}
$$
where $f$ and $g$ are the hidden and output layer transformations, $w_\textrm{h}$ and $w_\textrm{o}$ are their weights.

The forward pass iterates through $(x_t, h_t, o_t)$ triples. The loss over $T$ steps is:
$$
L = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).
$$

For backpropagation, computing $\partial L / \partial w_\textrm{h}$ is tricky due to recurrence:
$$
\frac{\partial L}{\partial w_\textrm{h}} = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_\textrm{o})}{\partial h_t} \frac{\partial h_t}{\partial w_\textrm{h}}.
$$

The term $\partial h_t / \partial w_\textrm{h}$ is complex because $h_t$ depends on $h_{t-1}$, which also depends on $w_\textrm{h}$. Applying the chain rule recursively gives:
$$
\frac{\partial h_t}{\partial w_\textrm{h}} = \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}} + \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.
$$

This recursion leads to a general form:
$$
\frac{\partial h_t}{\partial w_\textrm{h}}=\frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_\textrm{h})}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_\textrm{h})}{\partial w_\textrm{h}}.
$$

The product term involves many Jacobian matrices. For large $t$, this computation is problematic. Let's explore strategies to handle it.

### 1.1 Strategy 1: Full Computation
Computing the full sum exactly is slow and numerically unstable. Small changes can amplify dramatically (like the butterfly effect), harming generalization. This method is rarely used.

### 1.2 Strategy 2: Truncated Time Steps
We approximate the gradient by truncating the sum after $\tau$ steps, ignoring contributions beyond $\partial h_{t-\tau} / \partial w_\textrm{h}$. This **Truncated Backpropagation Through Time** works well in practice, biasing the model toward short-term dependencies and improving stability.

### 1.3 Strategy 3: Randomized Truncation
We replace $\partial h_t / \partial w_\textrm{h}$ with a random variable $z_t$ that is correct in expectation but truncates randomly. A random variable $\xi_t$ controls truncation:
- $P(\xi_t = 0) = 1-\pi_t$
- $P(\xi_t = \pi_t^{-1}) = \pi_t$
- $E[\xi_t] = 1$

Then:
$$
z_t = \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.
$$

When $\xi_t = 0$, recurrence stops. This yields a weighted sum over varying sequence lengths. However, in practice, randomized truncation often doesn't outperform regular truncation due to increased variance and the sufficiency of short-term dependencies.

## 2. Detailed BPTT for a Simple RNN

Let's derive gradients for a concrete RNN without biases and with identity activation ($\phi(x)=x$). For time step $t$:
- Input: $\mathbf{x}_t \in \mathbb{R}^d$
- Target: $y_t$
- Hidden state: $\mathbf{h}_t \in \mathbb{R}^h$
- Output: $\mathbf{o}_t \in \mathbb{R}^q$

The computations are:
$$
\begin{aligned}
\mathbf{h}_t &= \mathbf{W}_\textrm{hx} \mathbf{x}_t + \mathbf{W}_\textrm{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_\textrm{qh} \mathbf{h}_{t},
\end{aligned}
$$
with weight matrices $\mathbf{W}_\textrm{hx} \in \mathbb{R}^{h \times d}$, $\mathbf{W}_\textrm{hh} \in \mathbb{R}^{h \times h}$, $\mathbf{W}_\textrm{qh} \in \mathbb{R}^{q \times h}$.

The loss over $T$ steps is:
$$
L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).
$$

We'll compute gradients for each parameter.

### 2.1 Gradient with Respect to Output $\mathbf{o}_t$
This is straightforward:
$$
\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.
$$

### 2.2 Gradient for Output Weight $\mathbf{W}_\textrm{qh}$
Since $L$ depends on $\mathbf{W}_\textrm{qh}$ via all $\mathbf{o}_t$:
$$
\frac{\partial L}{\partial \mathbf{W}_\textrm{qh}} = \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top.
$$

### 2.3 Gradient for Final Hidden State $\mathbf{h}_T$
At the last step, $L$ depends on $\mathbf{h}_T$ only through $\mathbf{o}_T$:
$$
\frac{\partial L}{\partial \mathbf{h}_T} = \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.
$$

### 2.4 Gradient for Intermediate Hidden States $\mathbf{h}_t$ ($t < T$)
For $t < T$, $L$ depends on $\mathbf{h}_t$ via $\mathbf{h}_{t+1}$ and $\mathbf{o}_t$. Recurrently:
$$
\frac{\partial L}{\partial \mathbf{h}_t} = \mathbf{W}_\textrm{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.
$$

Expanding this recurrence reveals the core issue:
$$
\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_\textrm{hh}^\top\right)}^{T-i} \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.
$$

This involves powers of $\mathbf{W}_\textrm{hh}^\top$. If its eigenvalues are $<1$, they vanish; if $>1$, they explode—leading to vanishing/exploding gradients. Truncation or gradient detachment mitigates this.

### 2.5 Gradients for Hidden Weights $\mathbf{W}_\textrm{hx}$ and $\mathbf{W}_\textrm{hh}$
Using the chain rule:
$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_\textrm{hx}} &= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_\textrm{hh}} &= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top.
\end{aligned}
$$

Here, $\partial L/\partial \mathbf{h}_t$ is computed recursively as above. During BPTT, we cache these values to avoid redundant computation.

## 3. Summary

- **BPTT** applies backpropagation to RNNs by unrolling them across time.
- **Truncation** (regular or randomized) is essential for computational feasibility and numerical stability.
- **High matrix powers** cause vanishing/exploding gradients, which truncation alleviates.
- **Caching intermediate gradients** during BPTT improves efficiency.

## 4. Exercises

1. **Eigenvalue Analysis**:
   - Show that for a symmetric matrix $\mathbf{M}$ with eigenvalues $\lambda_i$, $\mathbf{M}^k$ has eigenvalues $\lambda_i^k$.
   - Prove that for a random vector $\mathbf{x}$, $\mathbf{M}^k \mathbf{x}$ aligns with the dominant eigenvector $\mathbf{v}_1$ as $k$ increases.
   - Relate this to gradient behavior in RNNs: gradients can explode or vanish based on $\mathbf{W}_\textrm{hh}$'s eigenvalues.

2. **Alternative Methods for Gradient Explosion**:
   - Besides gradient clipping, consider **weight initialization** schemes (e.g., orthogonal initialization) to control eigenvalue spread.
   - **Gradient norm regularization** or **layer normalization** can also stabilize training.

---
*For further discussion, visit the [community forum](https://discuss.d2l.ai/t/334).*