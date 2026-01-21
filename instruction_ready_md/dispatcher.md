# Extending PyTorch with Custom Operators Using the Dispatcher

## Introduction

This guide walks you through implementing custom PyTorch operators using the dispatcher system. The dispatcher is PyTorch's internal component responsible for routing function calls to the appropriate implementation based on various factors like device type, autograd requirements, and other cross-cutting concerns.

> **Note:** This tutorial covers the dispatcher-based approach for PyTorch versions prior to 2.4. For newer versions, please refer to the official Custom Operators documentation.

## Prerequisites

Before starting, ensure you have:
- Basic familiarity with C++ and PyTorch's C++ API
- Understanding of how to register custom operators
- Knowledge of custom autograd functions

## 1. Setting Up the Operator Schema

First, define the schema for your operator. This specifies the type signature that all implementations must follow.

```cpp
// op.cpp
#include <torch/library.h>

TORCH_LIBRARY(myops, m) {
  m.def("myadd(Tensor self, Tensor other) -> Tensor");
}
```

This creates a namespace `myops` with an operator `myadd` that takes two tensors and returns a tensor.

## 2. Implementing Backend Kernels

### 2.1 CPU Implementation

Create a simple CPU implementation of the addition operator:

```cpp
// op.cpp
torch::Tensor myadd_cpu(const torch::Tensor& self, const torch::Tensor& other) {
  return self + other;
}
```

Register this implementation specifically for CPU tensors:

```cpp
// op.cpp
TORCH_LIBRARY_IMPL(myops, CPU, m) {
  m.impl("myadd", myadd_cpu);
}
```

### 2.2 CUDA Implementation

Similarly, create and register a CUDA implementation:

```cpp
// op.cpp
torch::Tensor myadd_cuda(const torch::Tensor& self, const torch::Tensor& other) {
  return self + other;
}

TORCH_LIBRARY_IMPL(myops, CUDA, m) {
  m.impl("myadd", myadd_cuda);
}
```

### 2.3 Registration Structure

The recommended structure for operator registrations is:
1. A single `TORCH_LIBRARY` block listing all operators in your namespace
2. Separate `TORCH_LIBRARY_IMPL` blocks per dispatch key (CPU, CUDA, etc.)

These blocks can be split across different files or libraries, allowing for modular compilation.

> **Note:** You can also write `TORCH_LIBRARY_IMPL` blocks for existing PyTorch operators. This is how libraries like `torch_xla` add support for new backends.

## 3. Handling Operators Without Autograd

For operators that don't require autograd support (PyTorch ≥ 1.10), register a fallback kernel to improve usability:

```cpp
TORCH_LIBRARY_IMPL(myops, Autograd, m) {
  m.impl("myadd", torch::CppFunction::makeFallthrough());
}
```

This registers an `Autograd` kernel that preserves gradient requirements during forward pass and raises errors on backward pass, helping debug gradient flow in complex models.

### 3.1 In-place or View Operations

If your operator mutates inputs in-place or returns tensor aliases, take these additional steps:

1. Register both `Autograd` and `ADInplaceOrView` kernels:

```cpp
TORCH_LIBRARY_IMPL(myops, Autograd, m) {
  m.impl("myadd", torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(myops, ADInplaceOrView, m) {
  m.impl("myadd", torch::CppFunction::makeFallthrough());
}
```

2. Ensure your operator schema properly annotates in-place mutations or aliasing. Refer to PyTorch's native operator documentation for schema annotation guidelines.

## 4. Adding Autograd Support

### 4.1 Creating a Dispatch Function

First, create a dispatching function that calls into the dispatcher:

```cpp
// op.cpp
torch::Tensor myadd(const torch::Tensor& self, const torch::Tensor& other) {
  static auto op = torch::Dispatcher::singleton()
    .findSchemaOrThrow("myops::myadd", "")
    .typed<decltype(myadd)>();
  return op.call(self, other);
}
```

This function:
- Looks up the operator handle from the dispatcher (cached for performance)
- Calls the appropriate kernel based on dispatch key

### 4.2 Implementing the Autograd Kernel

Create an autograd function using the dispatch function:

```cpp
// op.cpp
class MyAddFunction : public torch::autograd::Function<MyAddFunction> {
public:
  static torch::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor self,
    torch::Tensor other) {
    
    // Disable autograd to prevent infinite recursion
    at::AutoNonVariableTypeMode guard;
    
    // Call the dispatch function
    return myadd(self, other);
  }
  
  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs) {
    // Implement your backward pass here
    return {grad_outputs[0], grad_outputs[0]};
  }
};

torch::Tensor myadd_autograd(const torch::Tensor& self, const torch::Tensor& other) {
  return MyAddFunction::apply(self, other);
}
```

### 4.3 Registering the Autograd Kernel

Register the autograd implementation:

```cpp
// op.cpp
TORCH_LIBRARY_IMPL(myops, Autograd, m) {
  m.impl("myadd", myadd_autograd);
}
```

> **Note:** You can register backend-specific autograd kernels using `AutogradCPU` or `AutogradCUDA` dispatch keys for optimized implementations.

## 5. Adding Autocast Support (Automatic Mixed Precision)

For operators that benefit from mixed precision execution, add an autocast wrapper:

```cpp
// op.cpp
#include <ATen/autocast_mode.h>

torch::Tensor myadd_autocast(const torch::Tensor& self, const torch::Tensor& other) {
  // Exclude Autocast key to prevent recursion
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  
  // Cast inputs to preferred precision
  return myadd(
    at::autocast::cached_cast(at::kHalf, self),
    at::autocast::cached_cast(at::kHalf, other)
  );
}

TORCH_LIBRARY_IMPL(myops, Autocast, m) {
  m.impl("myadd", myadd_autocast);
}
```

The `cached_cast` function follows PyTorch's native autocast eligibility policy, casting CUDA float32 tensors to float16 while leaving other types unchanged.

### 5.1 Determining Execution Precision

For operations with multiple floating-point inputs, determine the widest type:

```cpp
#include <ATen/autocast_mode.h>

torch::Tensor my_multiple_input_op_autocast(
  const torch::Tensor& t0,
  const torch::Tensor& t1) {
  
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  
  // Find widest floating-point type
  auto exec_type = at::autocast::promote_type(at::kHalf, t0, t1);
  
  return my_multiple_input_op(
    at::autocast::cached_cast(exec_type, t0),
    at::autocast::cached_cast(exec_type, t1)
  );
}
```

## 6. Additional Dispatch Keys

The dispatcher supports many other dispatch keys for specialized functionality:

### 6.1 Batched Tensors (vmap support)
Once the batching API stabilizes, register kernels at the `Batched` dispatch key to support automatic vectorization via `vmap`.

### 6.2 Tracing (JIT support)
The `Tracer` dispatch key handles operator recording during `torch.jit.trace`. A boxed fallback for arbitrary operations is planned.

## Why Use the Dispatcher?

While you could implement manual if-statements for different backends, the dispatcher offers:

1. **Decentralization**: Operator components can be defined separately and combined automatically
2. **Extensibility**: Third parties can add implementations without modifying original code
3. **Comprehensive coverage**: Support for numerous dispatch keys beyond basic backends
4. **Fallback functions**: Default behavior implementations that apply to all operators

## Conclusion

By using the dispatcher system, you can create custom PyTorch operators that integrate seamlessly with PyTorch's ecosystem, supporting features like autograd, mixed precision, and JIT tracing. The modular approach allows for clean separation of concerns and easy extensibility.

Remember to structure your implementations with:
- Centralized schema definitions
- Separate implementations per dispatch key
- Proper handling of autograd and other cross-cutting concerns