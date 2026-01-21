# Custom C++ and CUDA Operators Tutorial

**Author:** [Richard Zou](https://github.com/zou3519)

## What You Will Learn
- How to integrate custom operators written in C++/CUDA with PyTorch
- How to test custom operators using `torch.library.opcheck`

## Prerequisites
- PyTorch 2.4 or later (or PyTorch 2.10 or later if using the stable ABI)
- Basic understanding of C++ and CUDA programming

> **Note:** This tutorial also works on AMD ROCm with no additional modifications.

PyTorch offers a large library of operators that work on Tensors (e.g., `torch.add`, `torch.sum`). However, you may wish to bring a new custom operator to PyTorch. This tutorial demonstrates the recommended path for authoring a custom operator written in C++/CUDA.

For our tutorial, we'll demonstrate how to author a fused multiply-add C++ and CUDA operator that composes with PyTorch subsystems. The semantics of the operation are as follows:

```python
def mymuladd(a: Tensor, b: Tensor, c: float):
    return a * b + c
```

You can find the end-to-end working example for this tutorial in the [extension-cpp](https://github.com/pytorch/extension-cpp) repository, which contains two parallel implementations:

- **extension_cpp/**: Uses the standard ATen/LibTorch API.
- **extension_cpp_stable/**: Uses APIs supported by the LibTorch Stable ABI (recommended for PyTorch 2.10+).

### Which API Should You Use?
- **ABI-Stable LibTorch API (recommended)**: If you are using PyTorch 2.10+, we recommend using the ABI-stable API. It allows you to build a single wheel that works across multiple PyTorch versions (2.10, 2.11, 2.12, etc.), reducing the maintenance burden of supporting multiple PyTorch releases.
- **Non-ABI-Stable LibTorch API**: Use this if you need APIs not yet available in the stable ABI, or if you are targeting PyTorch versions older than 2.10. Note that you will need to build separate wheels for each PyTorch version you want to support.

The code snippets below show both implementations using tabs, with the ABI-stable API shown by default.

---

## 1. Setting Up the Build System

If you are developing custom C++/CUDA code, it must be compiled. Use `torch.utils.cpp_extension` to compile custom C++/CUDA code for use with PyTorch. C++ extensions may be built either "ahead of time" with setuptools, or "just in time" via `load_inline`; we'll focus on the "ahead of time" flavor.

Using `cpp_extension` is as simple as writing a `setup.py`:

::::: tab-set
::: tab-item
ABI-Stable LibTorch API

```python
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="extension_cpp",
      ext_modules=[
          cpp_extension.CppExtension(
              "extension_cpp",
              ["muladd.cpp"],
              extra_compile_args={
                  "cxx": [
                      # define Py_LIMITED_API with min version 3.9 to expose only the stable
                      # limited API subset from Python.h
                      "-DPy_LIMITED_API=0x03090000",
                      # define TORCH_TARGET_VERSION with min version 2.10 to expose only the
                      # stable API subset from torch
                      "-DTORCH_TARGET_VERSION=0x020a000000000000",
                  ]
              },
              py_limited_api=True)],  # Build 1 wheel across multiple Python versions
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      options={"bdist_wheel": {"py_limited_api": "cp39"}}  # 3.9 is minimum supported Python version
)
```
:::

::: tab-item
Non-ABI-Stable LibTorch API

```python
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="extension_cpp",
      ext_modules=[
          cpp_extension.CppExtension(
              "extension_cpp",
              ["muladd.cpp"],
              extra_compile_args={
                  "cxx": [
                      "-DPy_LIMITED_API=0x03090000",
                  ]
              },
              py_limited_api=True)],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      options={"bdist_wheel": {"py_limited_api": "cp39"}}
)
```
:::
:::::

If you need to compile CUDA code (for example, `.cu` files), then instead use `torch.utils.cpp_extension.CUDAExtension`. Please see [extension-cpp](https://github.com/pytorch/extension-cpp) for an example for how this is set up.

### 1.1 CPython Agnosticism

The above examples represent what we refer to as a CPython agnostic wheel, meaning we are building a single wheel that can be run across multiple CPython versions (similar to pure Python packages). CPython agnosticism is desirable in minimizing the number of wheels your custom library needs to support and release. The minimum version we'd like to support is 3.9, since it is the oldest supported version currently, so we use the corresponding hexcode and specifier throughout the setup code.

To achieve this, there are three key lines to note.

First, specify `Py_LIMITED_API` in `extra_compile_args` to the minimum CPython version you would like to support:

```python
extra_compile_args={"cxx": ["-DPy_LIMITED_API=0x03090000"]},
```

Defining the `Py_LIMITED_API` flag helps verify that the extension is only using the [CPython Stable Limited API](https://docs.python.org/3/c-api/stable.html), which is a requirement for building a CPython agnostic wheel.

Second and third, specify `py_limited_api` to inform setuptools that you intend to build a CPython agnostic wheel:

```python
setup(name="extension_cpp",
      ext_modules=[
          cpp_extension.CppExtension(
            ...,
            py_limited_api=True)],  # Build 1 wheel across multiple Python versions
      ...,
      options={"bdist_wheel": {"py_limited_api": "cp39"}}  # 3.9 is minimum supported Python version
)
```

If your extension uses CPython APIs outside the stable limited set, then you cannot build a CPython agnostic wheel! You should build one wheel per CPython version instead:

```python
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="extension_cpp",
      ext_modules=[
          cpp_extension.CppExtension(
            "extension_cpp",
            ["muladd.cpp"])],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
)
```

### 1.2 LibTorch Stable ABI (PyTorch Agnosticism)

In addition to CPython agnosticism, there is a second axis of wheel compatibility: LibTorch agnosticism. While CPython agnosticism allows building a single wheel that works across multiple Python versions (3.9, 3.10, 3.11, etc.), LibTorch agnosticism allows building a single wheel that works across multiple PyTorch versions (2.10, 2.11, 2.12, etc.). These two concepts are orthogonal and can be combined.

To achieve LibTorch agnosticism, you must use the ABI stable LibTorch API, which provides a stable API for interacting with PyTorch tensors and operators. For example, instead of using `at::Tensor`, you must use `torch::stable::Tensor`. For comprehensive documentation on the stable ABI, including migration guides, supported types, and stack-based API conventions, see the [LibTorch Stable ABI documentation](https://pytorch.org/docs/main/notes/libtorch_stable_abi.html).

The stable ABI setup.py includes `TORCH_TARGET_VERSION=0x020a000000000000`, which indicates that the extension targets the LibTorch Stable ABI with a minimum supported PyTorch version of 2.10.

If the stable API/ABI does not contain what you need, you can use the Non-ABI-stable LibTorch API, but you will need to build separate wheels for each PyTorch version you want to support.

---

## 2. Defining the Custom Operator and Adding Backend Implementations

First, let's write a C++ function that computes `mymuladd`:

::::: tab-set
::: tab-item
ABI-Stable LibTorch API

```cpp
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

torch::stable::Tensor mymuladd_cpu(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    double c) {
  STD_TORCH_CHECK(a.sizes().equals(b.sizes()));
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CPU);
  STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CPU);

  torch::stable::Tensor a_contig = torch::stable::contiguous(a);
  torch::stable::Tensor b_contig = torch::stable::contiguous(b);
  torch::stable::Tensor result = torch::stable::empty_like(a_contig);

  const float* a_ptr = a_contig.const_data_ptr<float>();
  const float* b_ptr = b_contig.const_data_ptr<float>();
  float* result_ptr = result.mutable_data_ptr<float>();

  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
  }
  return result;
}
```
:::

::: tab-item
Non-ABI-Stable LibTorch API

```cpp
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

at::Tensor mymuladd_cpu(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = a_ptr[i] * b_ptr[i] + c;
  }
  return result;
}
```
:::
:::::

In order to use this from PyTorch's Python frontend, we need to register it as a PyTorch operator using the `TORCH_LIBRARY` (or `STABLE_TORCH_LIBRARY`) macro. This will automatically bind the operator to Python.

Operator registration is a two step-process:
1. **Defining the operator** - This step ensures that PyTorch is aware of the new operator.
2. **Registering backend implementations** - In this step, implementations for various backends, such as CPU and CUDA, are associated with the operator.

### 2.1 Defining an Operator

To define an operator, follow these steps:
1. Select a namespace for an operator. We recommend the namespace be the name of your top-level project; we'll use "extension_cpp" in our tutorial.
2. Provide a schema string that specifies the input/output types of the operator and if an input Tensors will be mutated. We support more types in addition to Tensor and float; please see [The Custom Operators Manual](https://pytorch.org/docs/main/notes/custom_operators.html) for more details.

::::: tab-set
::: tab-item
ABI-Stable LibTorch API

```cpp
STABLE_TORCH_LIBRARY(extension_cpp, m) {
  // Note that "float" in the schema corresponds to the C++ double type
  // and the Python float type.
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
}
```
:::

::: tab-item
Non-ABI-Stable LibTorch API

```cpp
TORCH_LIBRARY(extension_cpp, m) {
  // Note that "float" in the schema corresponds to the C++ double type
  // and the Python float type.
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
}
```
:::
:::::

This makes the operator available from Python via `torch.ops.extension_cpp.mymuladd`.

### 2.2 Registering Backend Implementations for an Operator

Use `TORCH_LIBRARY_IMPL` (or `STABLE_TORCH_LIBRARY_IMPL`) to register a backend implementation for the operator.

::::: tab-set
::: tab-item
ABI-Stable LibTorch API

Note that we wrap the function pointer with `TORCH_BOX()` - this is required for stable ABI functions to handle argument boxing/unboxing correctly.

```cpp
STABLE_TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("mymuladd", TORCH_BOX(&mymuladd_cpu));
}
```
:::

::: tab-item
Non-ABI-Stable LibTorch API

```cpp
TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("mymuladd", &mymuladd_cpu);
}
```
:::
:::::

If you also have a CUDA implementation of `mymuladd`, you can register it in a separate `TORCH_LIBRARY_IMPL` (or `STABLE_TORCH_LIBRARY_IMPL`) block:

::::: tab-set
::: tab-item
ABI-Stable LibTorch API

```cpp
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/c/shim.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void muladd_kernel(int numel, const float* a, const float* b, float c, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx] + c;
}

torch::stable::Tensor mymuladd_cuda(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    double c) {
  STD_TORCH_CHECK(a.sizes().equals(b.sizes()));
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(a.device().type() == torch::headeronly::DeviceType::CUDA);
  STD_TORCH_CHECK(b.device().type() == torch::headeronly::DeviceType::CUDA);

  torch::stable::Tensor a_contig = torch::stable::contiguous(a);
  torch::stable::Tensor b_contig = torch::stable::contiguous(b);
  torch::stable::Tensor result = torch::stable::empty_like(a_contig);

  const float* a_ptr = a_contig.const_data_ptr<float>();
  const float* b_ptr = b_contig.const_data_ptr<float>();
  float* result_ptr = result.mutable_data_ptr<float>();

  int numel = a_contig.numel();

  // For now, we rely on the raw shim API to get the current CUDA stream.
  // This will be improved in a future release.
  // When using a raw shim API, we need to use TORCH_ERROR_CODE_CHECK to
  // check the error code and throw an appropriate runtime_error otherwise.
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(a.get_device_index(), &stream_ptr));
  cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

  muladd_kernel<<<(numel+255)/256, 256, 0, stream>>>(numel, a_ptr, b_ptr, c, result_ptr);
  return result;
}

STABLE_TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("mymuladd", TORCH_BOX(&mymuladd_cuda));
}
```
:::

::: tab-item
Non-ABI-Stable LibTorch API

```cpp
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void muladd_kernel(int numel, const float* a, const float* b, float c, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx] + c;
}

at::Tensor mymuladd_cuda(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH