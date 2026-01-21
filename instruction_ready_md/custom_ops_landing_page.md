# PyTorch Custom Operators Guide

## Introduction

PyTorch provides an extensive library of built-in operators (like `torch.add` or `torch.sum`) for tensor manipulation. However, there are scenarios where you need to integrate custom operations—such as specialized C++/CUDA kernels or opaque Python functions—and have them work seamlessly with PyTorch subsystems like `torch.compile`, autograd, and `torch.vmap`. This guide explains when and how to create custom operators using PyTorch's registration APIs.

## Prerequisites

Before you begin, ensure you have PyTorch installed. If you plan to integrate C++ or CUDA code, you'll also need a compatible compiler and CUDA toolkit (if using NVIDIA GPUs).

```bash
pip install torch
```

## When to Create a Custom Operator

**Do not create a custom operator if your operation can be expressed as a composition of existing PyTorch operators.** Instead, write it as a regular Python function. Use custom operator registration only when you need to interface with external code that PyTorch doesn't natively understand, such as:

- Custom C/C++ or CUDA kernels.
- Python bindings to external C++/CUDA libraries.
- Opaque Python callables that should be treated as atomic operations by `torch.compile` or `torch.export`.

## Why Register a Custom Operator?

You might be tempted to pass a tensor's data pointer directly to a pybind11-wrapped kernel. While this works, it **does not compose** with PyTorch's core subsystems. Registering your operation via the official APIs ensures compatibility with:

- **Autograd:** For automatic differentiation.
- **torch.compile:** For graph capture and optimization.
- **torch.vmap:** For automatic vectorization.
- **torch.export:** For capturing the operation in an exportable graph.

## Choosing the Right Approach

PyTorch offers two primary ways to author custom operators. Choose based on your use case:

### 1. Authoring from Python

**Use this approach if:**
- You have a Python function you want PyTorch to treat as an opaque callable.
- You have Python bindings to C++/CUDA kernels and need them to work with `torch.compile` or `torch.export`.
- You are working in a Python environment (not a C++-only setting like AOTInductor).

**Next Steps:** Proceed to the [Python Custom Operators Tutorial](python-custom-ops-tutorial).

### 2. Authoring from C++ (with CUDA or SYCL)

**Use this approach if:**
- You have custom C++ and/or CUDA code.
- You plan to use the operator with **AOTInductor** for Python-less inference.
- You are targeting Intel GPUs and need to integrate custom SYCL code.

**Next Steps:** Proceed to the [C++ Custom Operators Tutorial](cpp-custom-ops-tutorial). For Intel GPU (SYCL) integration, see the [C++ Custom Operators Tutorial for SYCL](cpp-custom-ops-tutorial-sycl).

## Reference: The Custom Operators Manual

For detailed information not covered in the tutorials, consult the [Custom Operators Manual](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU). This manual serves as a reference guide—we recommend completing one of the tutorials above first, then using the manual for specific questions.

## Summary

Creating a custom operator is the right choice when you need to integrate non-PyTorch code and maintain compatibility with PyTorch's advanced features. Start by choosing the appropriate tutorial (Python or C++) based on your integration needs.