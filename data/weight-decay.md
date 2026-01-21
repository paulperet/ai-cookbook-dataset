# Weight Decay: A Regularization Technique for Linear Models

## Overview

In this guide, you'll learn about **weight decay**, a fundamental regularization technique used to prevent overfitting in machine learning models. We'll explore the theoretical foundations, implement it from scratch, and then use built-in framework features for a more concise implementation.

## Prerequisites

First, let's set up our environment by importing the necessary libraries. The code supports multiple deep learning frameworks (MXNet, PyTorch, TensorFlow, JAX) - choose the one you're working with.

```python
%matplotlib inline
from d2l import torch as d2l  # Change to mxnet, tensorflow, or jax as needed
import torch  # Change to your framework
from torch import nn
```

## 1. Understanding the Problem of Overfitting

When training machine learning models, we often face the challenge of **overfitting** - where a model performs well on training data but poorly on unseen validation data. While collecting more training data can help, it's not always feasible due to cost or time constraints.

In polynomial regression, we control model capacity by limiting polynomial degree. However, with high-dimensional data, the number of possible features (monomials) grows rapidly. For `k` variables, the number of monomials of degree `d` is `(k-1+d) choose (k-1)`. This exponential growth necessitates more sophisticated regularization techniques.

## 2. Introducing Weight Decay

Weight decay (also known as **L₂ regularization**) operates by restricting the values that model parameters can take. The core idea: among all possible functions, the simplest is `f(x) = 0`, and we can measure complexity by how far parameters deviate from zero.

For linear regression with loss function:
```
L(w, b) = (1/n) * Σ (1/2) * (wᵀx⁽ⁱ⁾ + b - y⁽ⁱ⁾)²
```

We add a penalty term to restrict the weight vector's size:
```
L(w, b) + (λ/2) * ||w||²
```

Where:
- `λ` is the regularization constant (hyperparameter)
- `||w||²` is the squared L₂ norm of the weight vector

**Why squared norm?** For computational convenience - the derivative is simpler, and squaring emphasizes larger weights more heavily.

## 3. Generating Synthetic Data

Let's create a high-dimensional dataset to demonstrate overfitting and how weight decay helps:

```python
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        
        # Generate input features
        self.X = d2l.randn(n, num_inputs)
        
        # Generate noise
        noise = d2l.randn(n, 1) * 0.01
        
        # True parameters
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        
        # Generate labels: y = 0.05 + Σ 0.01*x_i + noise
        self.y = d2l.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

We'll use 200 input dimensions but only 20 training examples to make overfitting pronounced.

## 4. Implementing Weight Decay from Scratch

### 4.1 Defining the L₂ Penalty Function

```python
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

### 4.2 Creating the Regularized Model

We extend a basic linear regression model to include weight decay:

```python
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        # Original loss + regularization term
        return (super().loss(y_hat, y) + self.lambd * l2_penalty(self.w))
```

### 4.3 Training Without Regularization

Let's first see what happens without weight decay (`λ = 0`):

```python
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):    
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    print('L2 norm of w:', float(l2_penalty(model.w)))

# Train without regularization
train_scratch(0)
```

You'll observe that the training error decreases rapidly, but validation error remains high - classic overfitting.

### 4.4 Training With Weight Decay

Now let's apply weight decay with `λ = 3`:

```python
train_scratch(3)
```

Notice that training error increases slightly, but validation error decreases significantly. The model generalizes better because the weight decay penalty prevents weights from growing too large.

## 5. Concise Implementation Using Framework Features

Most deep learning frameworks have built-in support for weight decay, making implementation cleaner and more efficient.

### 5.1 PyTorch Implementation

```python
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        # Apply weight decay only to weights, not biases
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
```

### 5.2 TensorFlow Implementation

```python
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        # Apply L2 regularization to the kernel (weights)
        self.net = tf.keras.layers.Dense(
            1, kernel_regularizer=tf.keras.regularizers.l2(wd),
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01)
        )
        
    def loss(self, y_hat, y):
        # Add regularization losses to the main loss
        return super().loss(y_hat, y) + self.net.losses
```

### 5.3 Training the Concise Model

```python
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)
print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))
```

The results should be similar to our from-scratch implementation, but the code is cleaner and runs faster.

## 6. Key Insights

1. **Weight decay vs. L₁ regularization**: While we used L₂ regularization (weight decay), L₁ regularization (lasso) is also common. L₂ tends to distribute weights evenly, while L₁ drives some weights to zero for feature selection.

2. **Bias terms**: Typically, we don't regularize bias terms since they don't contribute to feature interactions in the same way weights do.

3. **The update rule**: With weight decay, the weight update becomes:
   ```
   w ← (1 - ηλ)w - (η/|B|) * Σ x⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b - y⁽ⁱ⁾)
   ```
   The `(1 - ηλ)` term causes weights to "decay" toward zero at each step.

4. **Choosing λ**: This is a hyperparameter you should tune using validation data. Too small and you under-regularize; too large and you underfit.

## 7. Practical Recommendations

1. **Start simple**: Begin without regularization to establish a baseline.
2. **Use validation**: Always tune regularization parameters on a validation set, not your test set.
3. **Framework features**: Leverage built-in regularization in your deep learning framework - it's optimized and less error-prone.
4. **Layer-specific regularization**: In deep networks, you might want different regularization for different layers.

## Exercises

1. Experiment with different λ values and plot training/validation accuracy. What pattern emerges?
2. Use a validation set to find the optimal λ. How sensitive is performance to small changes in λ?
3. How would the update equations change if we used L₁ regularization instead of L₂?
4. Consider other strategies for dealing with overfitting besides weight decay.

Weight decay is a fundamental tool in your machine learning toolkit. By controlling model complexity through parameter regularization, you can build models that generalize better to unseen data while maintaining good performance on your training set.