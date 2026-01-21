# Implementing Double Backward for Custom Autograd Functions

## Overview
This guide demonstrates how to create custom PyTorch autograd functions that support double backward computation (second-order gradients). You'll learn the nuances of gradient tracking, proper tensor saving techniques, and common pitfalls to avoid when implementing higher-order gradient support.

## Prerequisites

```python
import torch
import torchviz
```

## 1. Understanding Autograd Behavior

Custom autograd functions interact with PyTorch's gradient tracking in specific ways:

- **During forward pass**: Operations inside `forward()` are not recorded in the computation graph
- **During backward pass**: When `create_graph=True`, the backward computation itself is recorded
- **Tensor saving**: Use `ctx.save_for_backward()` for tensors needed in backward computation

## 2. Basic Example: Saving Inputs

Let's start with a simple squaring function that properly supports double backward.

```python
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save input for backward pass
        ctx.save_for_backward(x)
        return x**2

    @staticmethod
    def backward(ctx, grad_out):
        # Retrieve saved tensor
        x, = ctx.saved_tensors
        # Compute gradient: d(x²)/dx = 2x
        return grad_out * 2 * x
```

### Testing the Implementation

```python
# Create test tensor with gradient tracking
x = torch.rand(3, 3, requires_grad=True, dtype=torch.double)

# Verify first-order gradients
torch.autograd.gradcheck(Square.apply, x)

# Verify second-order gradients
torch.autograd.gradgradcheck(Square.apply, x)
```

**Why this works**: The input `x` has `grad_fn` attached, allowing autograd to track operations performed on it during backward.

## 3. Saving Outputs Instead of Inputs

Sometimes you need to save outputs rather than inputs. Here's an exponential function example:

```python
class Exp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Compute and save output
        result = torch.exp(x)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_out):
        # Retrieve saved output
        result, = ctx.saved_tensors
        # Gradient: d(exp(x))/dx = exp(x)
        return result * grad_out
```

### Testing the Exponential Function

```python
x = torch.tensor(1., requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(Exp.apply, x)
torch.autograd.gradgradcheck(Exp.apply, x)
```

## 4. Handling Intermediate Results (Correct Approach)

When you need intermediate results for backward computation, you must include them as outputs to ensure proper gradient tracking.

```python
class Sinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Compute intermediate results
        expx = torch.exp(x)
        expnegx = torch.exp(-x)
        
        # Save intermediates for backward
        ctx.save_for_backward(expx, expnegx)
        
        # Return both main output AND intermediates
        return (expx - expnegx) / 2, expx, expnegx

    @staticmethod
    def backward(ctx, grad_out, _grad_out_exp, _grad_out_negexp):
        # Retrieve saved tensors
        expx, expnegx = ctx.saved_tensors
        
        # Compute gradient for sinh(x)
        grad_input = grad_out * (expx + expnegx) / 2
        
        # Accumulate gradients from intermediate outputs
        grad_input += _grad_out_exp * expx
        grad_input -= _grad_out_negexp * expnegx
        
        return grad_input

def sinh(x):
    # Wrapper that returns only the main output
    return Sinh.apply(x)[0]
```

### Testing the Sinh Function

```python
x = torch.rand(3, 3, requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(sinh, x)
torch.autograd.gradgradcheck(sinh, x)
```

**Key Insight**: By returning intermediate results as outputs, you ensure they become part of the computation graph, enabling proper gradient flow during double backward.

## 5. Common Mistake: Incorrect Intermediate Handling

Here's what happens when you don't properly handle intermediate results:

```python
class SinhBad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        expx = torch.exp(x)
        expnegx = torch.exp(-x)
        
        # WRONG: Storing intermediates directly on ctx
        ctx.expx = expx
        ctx.expnegx = expnegx
        
        return (expx - expnegx) / 2

    @staticmethod
    def backward(ctx, grad_out):
        expx = ctx.expx
        expnegx = ctx.expnegx
        
        # This computes first-order gradient correctly
        grad_input = grad_out * (expx + expnegx) / 2
        return grad_input
```

**The Problem**: The intermediate results `expx` and `expnegx` are detached from the computation graph, preventing second-order gradient computation.

## 6. Manual Backward Tracking for External Operations

When your backward computation uses non-PyTorch operations, you need to manually create the backward graph:

```python
# External operations (could be NumPy, SciPy, or C++ extensions)
def cube_forward(x):
    return x**3

def cube_backward(grad_out, x):
    return grad_out * 3 * x**2

def cube_backward_backward(grad_out, sav_grad_out, x):
    return grad_out * sav_grad_out * 6 * x

def cube_backward_backward_grad_out(grad_out, x):
    return grad_out * 3 * x**2

# Main custom function
class Cube(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return cube_forward(x)

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        # Use another custom function for backward
        return CubeBackward.apply(grad_out, x)

# Custom function for backward computation
class CubeBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grad_out, x):
        ctx.save_for_backward(x, grad_out)
        return cube_backward(grad_out, x)

    @staticmethod
    def backward(ctx, grad_out):
        x, sav_grad_out = ctx.saved_tensors
        # Compute gradients for both inputs
        dx = cube_backward_backward(grad_out, sav_grad_out, x)
        dgrad_out = cube_backward_backward_grad_out(grad_out, x)
        return dgrad_out, dx
```

### Testing the Cube Function

```python
x = torch.tensor(2., requires_grad=True, dtype=torch.double)
torch.autograd.gradcheck(Cube.apply, x)
torch.autograd.gradgradcheck(Cube.apply, x)
```

## 7. Visualization with Torchviz

To understand the computation graphs, you can visualize them:

```python
# Example visualization for the Square function
x = torch.tensor(1., requires_grad=True)
out = Square.apply(x)
grad_x, = torch.autograd.grad(out, x, create_graph=True)

# Create visualization
torchviz.make_dot((grad_x, x, out), 
                  {"grad_x": grad_x, "x": x, "out": out})
```

## Key Takeaways

1. **Input/Output Saving**: Use `ctx.save_for_backward()` for tensors needed in backward
2. **Intermediate Results**: Include them as function outputs to maintain gradient tracking
3. **External Operations**: Create nested custom functions when backward uses non-PyTorch code
4. **Testing**: Always use `gradcheck()` and `gradgradcheck()` to verify gradient correctness

## Best Practices

- Always test with `torch.autograd.gradcheck` and `torch.autograd.gradgradcheck`
- Use double precision (`dtype=torch.double`) for gradient checking to reduce numerical errors
- Visualize computation graphs when debugging complex gradient flows
- Remember that operations in `forward()` are not tracked by autograd

By following these patterns, you can create custom autograd functions that fully support higher-order gradient computation in PyTorch.