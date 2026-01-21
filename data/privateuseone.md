# Integrating a New Backend with PyTorch's PrivateUse1 Mechanism

## Overview

This guide explains how to integrate a new hardware backend (e.g., a custom accelerator) into PyTorch using the `PrivateUse1` dispatch key. This mechanism allows you to prototype and deploy out-of-tree backend extensions without modifying the core PyTorch repository.

### Prerequisites

*   A working knowledge of PyTorch and C++.
*   A new hardware backend with custom operator implementations.
*   A build environment for PyTorch C++ extensions.

## What is PrivateUse1?

PyTorch provides reserved dispatch keys for prototyping out-of-tree backend extensions. Prior to PyTorch 2.0, three keys were available: `PrivateUse1`, `PrivateUse2`, and `PrivateUse3`. Each has a corresponding autograd key (e.g., `AutogradPrivateUse1`).

With PyTorch 2.1.0 and later, the `PrivateUse1` mechanism has been significantly enhanced to support full backend integration, including storage, automatic mixed precision (AMP), and distributed operations. The community now recommends using `PrivateUse1` for new backend integrations to avoid constant modifications to the main PyTorch codebase and to work within the 64-bit limit of the `DispatchKeySet`.

> **Note:** While multiple backends can use `PrivateUse1`, they typically aren't used simultaneously in practice.

## Step-by-Step Integration Guide

### Step 1: Register Kernels for the New Backend

Register your backend's high-performance operator implementations with the PyTorch dispatcher using the `TORCH_LIBRARY_IMPL` API.

#### 1.1 Register Forward Operators with CPU Fallback

Register all forward operators your backend supports. Implement a fallback mechanism so unsupported operators execute on the CPU.

```cpp
// Custom implementation of the add.Tensor operator
at::Tensor wrapper_Custom_Tensor_add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  // Your backend-specific implementation here
  // ...
}

// Register the operator implementation
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  // ... other operator registrations
  m.impl("add.Tensor", TORCH_FN(wrapper_Custom_Tensor_add));
  // ...
}

// Custom CPU fallback function with device-specific hints
void custom_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // Add hints about unsupported operations on your device
  at::native::cpu_fallback(op, stack);
}

// Register the fallback for all unhandled operators
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
}
```

#### 1.2 Register Autograd Operators

If your backend needs to override PyTorch's autograd layer, register kernels using `AutogradPrivateUse1`. The dispatcher will automatically call your forward and backward implementations.

```cpp
// Custom autograd function for the selu operator
class CustomSeluFunction : public torch::autograd::Function<CustomSeluFunction> {
  // Your backend-specific forward and backward implementations
  // ...
};

// Wrapper function for the autograd operator
at::Tensor wrapper_AutogradCustom__selu(const at::Tensor & self) {
  return CustomSeluFunction::apply(self);
}

// Register the autograd operator
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  // ... other autograd registrations
  m.impl("selu", TORCH_FN(wrapper_AutogradCustom__selu));
  // ...
}
```

#### 1.3 Register AMP-Compatible Kernels

To support Automatic Mixed Precision (AMP), register kernels with `AutocastPrivateUse1`. The autocast system will automatically use these kernels when AMP is enabled.

```cpp
// Register AMP-compatible operators
TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
  // ... other AMP registrations
  // Use the KERNEL_PRIVATEUSEONE macro for each operator
  KERNEL_PRIVATEUSEONE(<operator>, <policy>)
  // ...
}

// Register fallthrough for unhandled AMP operators
TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
```

#### 1.4 Implement AMP Support Module

To fully support AMP, you need to register a backend module with specific AMP-related APIs.

```python
import torch

# Create a backend module with AMP support
class BackendModule:
    @staticmethod
    def get_amp_supported_dtype():
        """Return list of dtypes supported for AMP on this backend."""
        return [torch.float16, torch.bfloat16]  # Example: support both
    
    @staticmethod
    def is_autocast_enabled():
        """Check if AMP is enabled for this backend."""
        # Your implementation here
        return False
    
    @staticmethod
    def get_autocast_dtype():
        """Get the current AMP dtype for this backend."""
        # Your implementation here
        return torch.float16
    
    @staticmethod
    def set_autocast_enabled(enabled):
        """Enable or disable AMP for this backend."""
        # Your implementation here
        pass
    
    @staticmethod
    def set_autocast_dtype(dtype):
        """Set the AMP dtype for this backend."""
        # Validate dtype is in get_amp_supported_dtype()
        # Your implementation here
        pass

# Register the backend module
torch._register_device_module("your_backend_name", BackendModule)
```

### Step 2: Register a Generator for the New Backend

Your backend needs a custom random number generator. Implement it by inheriting from `GeneratorImpl`.

```cpp
// Custom generator implementation
struct CustomGeneratorImpl : public c10::GeneratorImpl {
  // Implement generator methods for your backend
  // ...
};

// Generator builder function
at::Generator make_custom_generator(c10::DeviceIndex device_index) {
  return at::make_generator<CustomGeneratorImpl>(device_index);
}

// Register the generator
REGISTER_GENERATOR_PRIVATEUSE1(make_custom_generator)
```

### Step 3: Register Device Guard Implementation

Implement device, stream, and event switching functionality for your backend.

```cpp
// Custom device guard implementation
struct CustomGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  // Implement all required DeviceGuardImplInterface methods
  // ...
};

// Register the device guard
C10_REGISTER_GUARD_IMPL(PrivateUse1, CustomGuardImpl);
```

### Step 4: Register Serialization and Deserialization Functions

Support saving and loading models with your backend's tensor metadata.

```cpp
// Custom backend metadata structure
struct CustomBackendMetadata : public c10::BackendMeta {
  // Add backend-specific fields
  // ...
};

// Serialization function
void for_serialization(const at::Tensor& t, std::unordered_map<std::string, bool>& m) {
  // Serialize your backend metadata
  // ...
}

// Deserialization function
void for_deserialization(const at::Tensor& t, std::unordered_map<std::string, bool>& m) {
  // Deserialize your backend metadata
  // ...
}

// Register serialization/deserialization functions
TensorBackendMetaRegistry(c10::DeviceType::PrivateUse1, &for_serialization, &for_deserialization);
```

## Step 5: Improve User Experience

### 5.1 Register a Backend Module

Provide a user-friendly interface similar to `torch.cuda`.

```python
# Register your backend module
torch._register_device_module('your_backend', your_module)

# Users can now call:
# torch.your_backend.some_function()
```

### 5.2 Rename PrivateUse1 to a Custom Name

Make your backend more user-friendly by giving it a descriptive name.

**Python:**
```python
torch.rename_privateuse1_backend("your_backend")
```

**C++:**
```cpp
c10::register_privateuse1_backend("your_backend")
```

After renaming, users can specify devices as `'your_backend:0'` instead of `'privateuse1:0'`.

### 5.3 Generate Methods and Properties

Automatically generate tensor and storage methods for your backend.

```python
# Rename first
torch.rename_privateuse1_backend("npu")

# Define unsupported dtypes (if any)
unsupported_dtype = [torch.quint8]

# Generate methods and properties
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True,
    for_module=True,
    for_storage=True,
    unsupported_dtype=unsupported_dtype
)

# Now users can use:
# torch.Tensor.npu()
# torch.Tensor.is_npu
# torch.Storage.npu()
# torch.Storage.is_npu
```

## Future Work

The `PrivateUse1` mechanism continues to evolve. Future enhancements will include:

*   Integration methods for distributed collective communication
*   Integration methods for benchmark timers
*   Additional modules as needed by the community

## Conclusion

This guide has walked you through integrating a new backend into PyTorch using the `PrivateUse1` mechanism. You've learned how to:

1.  Register operator kernels with CPU fallback support
2.  Implement autograd and AMP compatibility
3.  Register custom generators and device guards
4.  Add serialization/deserialization support
5.  Improve the user experience with custom naming and generated methods

For a complete example, refer to the [Ascend NPU integration](https://github.com/ascend/pytorch). Remember that while this tutorial covers the essential components, you may only need to implement the modules relevant to your specific backend requirements.