# Extending the PyTorch Dispatcher for a New C++ Backend

## Overview

This guide walks you through extending the PyTorch dispatcher to add a new device backend that lives outside the main `pytorch/pytorch` repository. You'll learn how to register kernels, support autograd, build extensions, and maintain compatibility with native PyTorch.

> **Note**: This tutorial covers internal PyTorch components that are actively evolving. APIs may change, but we'll keep this guide updated with the latest information.

## Prerequisites

- Familiarity with [registering dispatched operators in C++](https://pytorch.org/tutorials/advanced/dispatcher.html)
- Knowledge of [writing custom autograd functions](https://pytorch.org/tutorials/advanced/cpp_autograd.html)
- A C++ development environment with PyTorch built from source

## Step 1: Determine If You Need a New Backend

Before proceeding, consider these common use cases:

| Use Case | Recommended Solution |
|----------|---------------------|
| New algorithms for existing operators | Submit a PR to PyTorch |
| Propose new operators | Submit feature request/PR to PyTorch |
| Support new hardware (TPU, custom chips) | **Follow this tutorial** (out-of-tree backend) |
| Support different tensor layouts (sparse, quantized) | **Follow this tutorial** (may require additional work) |

This tutorial focuses on adding a new out-of-tree device backend.

## Step 2: Acquire a Dispatch Key

Dispatch keys identify your backend in PyTorch's dispatcher system. PyTorch reserves three keys for prototyping:

- `PrivateUse1` / `AutogradPrivateUse1`
- `PrivateUse2` / `AutogradPrivateUse2`
- `PrivateUse3` / `AutogradPrivateUse3`

Choose one for your prototype. To create a tensor on your backend:

```cpp
// Example TensorImpl constructor
TensorImpl(
    Storage&& storage,
    DispatchKeySet ks,
    const caffe2::TypeMeta data_type);

// Create TensorImpl on PrivateUse1 backend
DispatchKeySet ks = c10::DispatchKeySet{
    c10::DispatchKey::PrivateUse1, 
    c10::DispatchKey::AutogradPrivateUse1
};
```

For backends without storage, use `OpaqueTensorImpl` and override methods as needed. See the [Vulkan TensorImpl](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/vulkan/VulkanOpaqueTensorImpl.h) for an example.

> **Note**: Once your prototype is complete and you plan regular releases, submit a PR to reserve a dedicated dispatch key.

## Step 3: Locate PyTorch Operators

After building PyTorch from source, find the operator list at `build/aten/src/ATen/RegistrationDeclarations.h`. Here's a sample:

```cpp
Tensor abs(const Tensor & self); // {"schema": "aten::abs(Tensor self) -> Tensor", "dispatch": "True", "default": "True"}
Tensor & abs_(Tensor & self); // {"schema": "aten::abs_(Tensor(a!) self) -> Tensor(a!)", "dispatch": "True", "default": "True"}
Tensor & abs_out(Tensor & out, const Tensor & self); // {"schema": "aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
```

For `abs_out`:
- `Tensor & abs_out(Tensor & out, const Tensor & self);` - C++ signature (must match exactly)
- `aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)` - Unique schema identifier
- `dispatch` and `default` fields indicate registration requirements

## Step 4: Register Kernels for Your Backend

Use `TORCH_LIBRARY_IMPL` to register kernels:

```cpp
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl(<schema_my_op1>, &my_op1);
  m.impl(<schema_my_op2>, &my_op2);
  m.impl(<schema_my_op2_backward>, &my_op2_backward);
}
```

### Understanding Operator Categories

PyTorch has 1600+ operators. You don't need to implement all of them:

| Category | Description | Metadata in `RegistrationDeclarations.h` |
|----------|-------------|------------------------------------------|
| **Ops requiring registration** | Backend-specific native implementation | `dispatch: True` AND `default: False` |
| **Optional registration** | Can use PyTorch's default kernel | `dispatch: False` OR `default: True` |

Focus on implementing operators with `dispatch: True` AND `default: False`. You can optionally override other operators for performance or special requirements.

## Step 5: Add Autograd Support

### Automatic Gradient Support

For most operators, PyTorch provides general gradient formulas. Just implement the forward pass:

```cpp
Tensor my_op1(const Tensor& self, const Tensor& other) {
  // Call backend-specific APIs
  // Ensure behavior matches PyTorch's native implementation
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl(<schema_my_op1>, &my_op1);
}
```

### Device-Specific Backward Kernels

Some operators require device-specific backward implementations. These appear as `*_backward` operators in `RegistrationDeclarations.h`:

```cpp
Tensor my_op2_backward(const Tensor& self, const Tensor& other) {
  // Backend-specific backward implementation
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl(<schema_my_op2>, &my_op2);
  m.impl(<schema_my_op2_backward>, &my_op2_backward);  // Note: Still registered to PrivateUse1
}
```

### Overriding Autograd (Rare Cases)

If PyTorch's gradient formula doesn't generalize to your backend, override it:

```cpp
class MyAddFunction : public torch::autograd::Function<MyAddFunction> {
public:
  static Tensor forward(AutogradContext *ctx, torch::Tensor self, torch::Tensor other) {
    at::AutoNonVariableTypeMode g;
    return myadd(self, other);
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grad_outputs) {
    auto grad_output = grad_outputs[0];
    return {grad_output, grad_output};
  }
};

Tensor myadd_autograd(const Tensor& self, const Tensor& other) {
  return MyAddFunction::apply(self, other)[0];
}

// Register autograd kernel
TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
  m.impl(<myadd_schema>, &myadd_autograd);
}

// Register inference kernel
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl(<myadd_schema>, &myadd);
}
```

See the [pytorch/xla example](https://github.com/pytorch/xla/blob/r1.7/torch_xla/csrc/aten_autograd_ops.h) for reference.

## Step 6: Build Your Extension

Create a C++ extension using `setuptools`. Here's a simplified example:

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='torch_xla',
    ext_modules=[
        CppExtension(
            '_XLAC',
            torch_xla_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            library_dirs=library_dirs,
            extra_link_args=extra_link_args + \
                [make_relative_rpath('torch_xla/lib')],
        ),
    ],
    cmdclass={
        'build_ext': Build,  # Custom Build class derived from BuildExtension
    }
    # Additional configuration...
)
```

Refer to the [C++ extension tutorial](https://pytorch.org/tutorials/advanced/cpp_extension.html#building-with-setuptools) for details.

## Step 7: Support Custom Operators

### Python Custom Operators
Your backend automatically supports Python custom operators composed of existing PyTorch operators that you've implemented.

### C++ Custom Operators
For C++ custom operators with backend-specific kernels (like [torchvision's NMS](https://github.com/pytorch/vision/blob/master/torchvision/csrc/ops/cuda/nms_kernel.cu)), you must:
1. Write a C++ kernel for your backend
2. Register it to the appropriate namespace
3. Alternatively, provide a custom API in your extension (e.g., `torch_xla.core.functions.nms`)

## Step 8: Test Against Native PyTorch

Use PyTorch's [generic device type testing framework](https://github.com/pytorch/pytorch/blob/master/torch/testing/_internal/common_device_type.py):

1. **Add your device type** following the [guide for new devices](https://github.com/pytorch/pytorch/blob/5a8198eb3c594aa18352930fd21f3c25bd7b7100/torch/testing/_internal/common_device_type.py#L369)
2. **Customize test execution** by blocking unsupported tests/dtypes or adjusting precision
3. **Reference XLA's implementation** for [test customization examples](https://github.com/pytorch/xla/blob/master/test/pytorch_test_base.py)

> **Note**: Not all test suites are device-generic yet. Search for `instantiate_device_type_tests` in the PyTorch codebase to find extensible test classes.

## Step 9: Maintain Backward Compatibility

PyTorch doesn't guarantee backward compatibility for registered operators. To minimize disruptions:

1. **Sync with major PyTorch releases** (quarterly cadence)
2. **Update your kernels** when operator signatures change
3. **Join the `#announcement` channel** on [pytorch.slack.com](http://pytorch.slack.com/) for release updates

## Known Issues and Limitations

1. **Tensor serialization**: No C++ extension point for serializing Python tensor objects. Currently requires modifying PyTorch's `Tensor.__reduce_ex__` method or monkey patching.
2. **View operations**: If your backend restricts direct memory access, implement view ops carefully to ensure proper storage sharing between tensors.
3. **Optimizer extension**: No C++ extension point for custom Optimizers that need to carry states through backward passes. Requires custom APIs or monkey patching.

## Future Improvements

The PyTorch team is working on:

- Better test coverage for generic testing frameworks
- Expanded `Math` kernel coverage with comprehensive testing
- Refactored `RegistrationDeclarations.h` with minimal information
- Backend fallback kernels for automatic CPU conversion

## Stay Connected

- **Questions**: Use [PyTorch dev discussions](https://dev-discuss.pytorch.org/)
- **Issues**: [File on GitHub](https://github.com/pytorch/pytorch/issues)
- **Contributions**: Reach out via GitHub or Slack if interested in helping with future work items

By following this guide, you can successfully extend PyTorch with your custom backend while maintaining compatibility with the evolving PyTorch ecosystem.