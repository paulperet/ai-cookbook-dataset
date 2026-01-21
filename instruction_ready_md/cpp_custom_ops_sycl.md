# Implementing Custom SYCL Operators for PyTorch on Intel GPUs

This guide walks you through creating a custom PyTorch operator written in SYCL, enabling high-performance execution on Intel GPUs. You'll learn how to integrate a custom fused `muladd` kernel into PyTorch's operator library.

## Prerequisites

Before starting, ensure you have:

- **PyTorch 2.8+** for Linux or **PyTorch 2.10+** for Windows
- **Intel Deep Learning Essentials** with the Intel Compiler (for SYCL compilation)
- **Intel GPU** with XPU drivers properly configured
- Basic understanding of SYCL programming

```bash
# Verify XPU availability
python -c "import torch; print('XPU available:', torch.xpu.is_available())"
```

## Project Structure

Create the following directory structure:

```
sycl_example/
├── setup.py
├── sycl_extension/
│   ├── __init__.py
│   ├── muladd.sycl
│   └── ops.py
└── test_sycl_extension.py
```

## Step 1: Configure the Build System

Create `setup.py` to handle the SYCL compilation. This file configures the build process for both Windows and Linux environments:

```python
import os
import torch
import glob
import platform
from setuptools import find_packages, setup
from torch.utils.cpp_extension import SyclExtension, BuildExtension

library_name = "sycl_extension"
py_limited_api = True

IS_WINDOWS = (platform.system() == 'Windows')

# Platform-specific compilation flags
if IS_WINDOWS:
    cxx_args = ["/O2", "/std:c++17", "/DPy_LIMITED_API=0x03090000"]
    sycl_args = ["/O2", "/std:c++17"]
else:
    cxx_args = ["-O3", "-fdiagnostics-color=always", "-DPy_LIMITED_API=0x03090000"]
    sycl_args = ["-O3"]

extra_compile_args = {"cxx": cxx_args, "sycl": sycl_args}

# Verify XPU availability
assert torch.xpu.is_available(), "XPU is not available, please check your environment"

# Collect source files
this_dir = os.path.dirname(os.path.curdir)
extensions_dir = os.path.join(this_dir, library_name)
sources = list(glob.glob(os.path.join(extensions_dir, "*.sycl")))

# Define the extension module
ext_modules = [
    SyclExtension(
        f"{library_name}._C",
        sources,
        extra_compile_args=extra_compile_args,
        py_limited_api=py_limited_api,
    )
]

setup(
    name=library_name,
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=["torch"],
    description="Simple Example of PyTorch Sycl extensions",
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
```

## Step 2: Implement the SYCL Kernel

Create `sycl_extension/muladd.sycl` with the custom operator implementation. This file contains the SYCL kernel, wrapper function, and PyTorch registration logic:

```cpp
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <Python.h>

namespace sycl_extension {

// ==========================================================
// 1. Kernel Definition
// ==========================================================
static void muladd_kernel(
    int numel, const float* a, const float* b, float c, float* result,
    const sycl::nd_item<1>& item) {
    int idx = item.get_global_id(0);
    if (idx < numel) {
        result[idx] = a[idx] * b[idx] + c;
    }
}

class MulAddKernelFunctor {
public:
    MulAddKernelFunctor(int _numel, const float* _a, const float* _b, float _c, float* _result)
        : numel(_numel), a(_a), b(_b), c(_c), result(_result) {}
    
    void operator()(const sycl::nd_item<1>& item) const {
        muladd_kernel(numel, a, b, c, result, item);
    }

private:
    int numel;
    const float* a;
    const float* b;
    float c;
    float* result;
};

// ==========================================================
// 2. Wrapper Function
// ==========================================================
at::Tensor mymuladd_xpu(const at::Tensor& a, const at::Tensor& b, double c) {
    // Input validation
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
    TORCH_CHECK(a.dtype() == at::kFloat, "a must be a float tensor");
    TORCH_CHECK(b.dtype() == at::kFloat, "b must be a float tensor");
    TORCH_CHECK(a.device().is_xpu(), "a must be an XPU tensor");
    TORCH_CHECK(b.device().is_xpu(), "b must be an XPU tensor");

    // Prepare contiguous tensors
    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor result = at::empty_like(a_contig);

    // Get raw pointers
    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* res_ptr = result.data_ptr<float>();
    int numel = a_contig.numel();

    // Launch SYCL kernel
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    constexpr int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for<MulAddKernelFunctor>(
            sycl::nd_range<1>(blocks * threads, threads),
            MulAddKernelFunctor(numel, a_ptr, b_ptr, static_cast<float>(c), res_ptr)
        );
    });

    return result;
}

// ==========================================================
// 3. PyTorch Operator Registration
// ==========================================================
TORCH_LIBRARY(sycl_extension, m) {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
}

TORCH_LIBRARY_IMPL(sycl_extension, XPU, m) {
    m.impl("mymuladd", &mymuladd_xpu);
}

} // namespace sycl_extension

// ==========================================================
// 4. Windows Compatibility Shim
// ==========================================================
extern "C" {
    #ifdef _WIN32
    __declspec(dllexport)
    #endif
    PyObject* PyInit__C(void) {
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "_C",
            "XPU Extension Shim",
            -1,
            NULL
        };
        return PyModule_Create(&moduledef);
    }
}
```

## Step 3: Create Python Interface

Create `sycl_extension/ops.py` to provide a clean Python API for your custom operator:

```python
import torch
from torch import Tensor

__all__ = ["mymuladd"]

def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Performs a * b + c in an efficient fused kernel.
    
    Args:
        a: First input tensor (must be on XPU device)
        b: Second input tensor (must be on XPU device)
        c: Scalar value to add
    
    Returns:
        Tensor containing a * b + c
    """
    return torch.ops.sycl_extension.mymuladd.default(a, b, c)
```

## Step 4: Initialize the Package

Create `sycl_extension/__init__.py` to make the extension importable and load the compiled library:

```python
import ctypes
import platform
from pathlib import Path

import torch

# Locate the compiled library
current_dir = Path(__file__).parent.parent
build_dir = current_dir / "build"

if platform.system() == 'Windows':
    file_pattern = "**/*.pyd"
else:
    file_pattern = "**/*.so"

lib_files = list(build_dir.glob(file_pattern))

# Fallback to package directory if not in build directory
if not lib_files:
    current_package_dir = Path(__file__).parent
    lib_files = list(current_package_dir.glob(file_pattern))

assert len(lib_files) > 0, f"Could not find any {file_pattern} file in {build_dir} or {current_dir}"
lib_file = lib_files[0]

# Load the library with PyTorch's guard
with torch._ops.dl_open_guard():
    loaded_lib = ctypes.CDLL(str(lib_file))

from . import ops

__all__ = ["loaded_lib", "ops"]
```

## Step 5: Build and Install

Build your extension using pip:

```bash
cd sycl_example
pip install -e .
```

This command compiles the SYCL code and installs the package in development mode.

## Step 6: Test Your Implementation

Create `test_sycl_extension.py` to verify correctness:

```python
import torch
from torch.testing._internal.common_utils import TestCase
import unittest
import sycl_extension

def reference_muladd(a, b, c):
    """Reference implementation for verification."""
    return a * b + c

class TestMyMulAdd(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        """Generate test inputs."""
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)

        return [
            [make_tensor(3), make_tensor(3), 1],
            [make_tensor(20), make_tensor(20), 3.14],
            [make_tensor(20), make_nondiff_tensor(20), -123],
            [make_nondiff_tensor(2, 3), make_tensor(2, 3), -0.3],
        ]

    def _test_correctness(self, device):
        """Test operator correctness on given device."""
        samples = self.sample_inputs(device)
        for args in samples:
            result = sycl_extension.ops.mymuladd(*args)
            expected = reference_muladd(*args)
            torch.testing.assert_close(result, expected)

    @unittest.skipIf(not torch.xpu.is_available(), "requires Intel GPU")
    def test_correctness_xpu(self):
        """Test on XPU device."""
        self._test_correctness("xpu")

if __name__ == "__main__":
    unittest.main()
```

Run the tests:

```bash
python test_sycl_extension.py
```

## Step 7: Use Your Custom Operator

Now you can use your custom SYCL operator in Python:

```python
import torch
import sycl_extension

# Create tensors on XPU device
a = torch.randn(100, 100, device="xpu")
b = torch.randn(100, 100, device="xpu")
c = 2.5

# Use the custom operator
result = sycl_extension.ops.mymuladd(a, b, c)
print(f"Result shape: {result.shape}")
print(f"Result device: {result.device}")
```

## Troubleshooting

If you encounter issues:

1. **XPU not available**: Ensure Intel GPU drivers and oneAPI toolkit are properly installed
2. **Compilation errors**: Verify SYCL compiler environment is activated
3. **Import errors**: Check that the library file was built in the correct location

## Next Steps

This tutorial covered implementing a custom inference operator. To extend functionality:

- Add backward support for training by implementing autograd functions
- Enable `torch.compile` compatibility for graph capture
- Implement additional operators following the same pattern
- Optimize kernel performance with advanced SYCL features

## Conclusion

You've successfully created a custom SYCL operator for PyTorch that runs efficiently on Intel GPUs. The key steps were:

1. Setting up the build system with `SyclExtension`
2. Implementing the SYCL kernel and wrapper function
3. Registering the operator with PyTorch's `TORCH_LIBRARY` API
4. Creating Python interfaces for easy usage
5. Testing correctness against reference implementations

This pattern can be extended to implement any custom operation that benefits from SYCL acceleration on Intel hardware.