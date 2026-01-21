# TorchInductor C++ Wrapper Tutorial

**Author**: [Chunyuan Wu](https://github.com/chunyuan-w), [Bin Bao](https://github.com/desertfire), [Jiong Gong](https://github.com/jgong5)

## Prerequisites

Before starting, ensure you are familiar with:
- [torch.compile and TorchInductor concepts in PyTorch](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

## Introduction

When using `torch.compile` with the default **TorchInductor** backend, PyTorch generates Python wrapper code to manage memory allocation and kernel execution. While this approach offers flexibility and easier debugging, the interpreted nature of Python introduces runtime overhead in performance-critical applications.

To mitigate this overhead, TorchInductor provides a specialized mode that generates **C++ wrapper code** instead of Python. This mode reduces Python's involvement, leading to faster execution with minimal changes to your existing code.

## Step 1: Enable the C++ Wrapper Mode

To activate the C++ wrapper for TorchInductor, add the following configuration at the beginning of your script:

```python
import torch._inductor.config as config
config.cpp_wrapper = True
```

## Step 2: Define and Compile a Sample Model

Let's create a simple model to demonstrate the feature. This example performs an element-wise addition followed by a sum reduction.

```python
import torch
import torch._inductor.config as config

# Enable the C++ wrapper
config.cpp_wrapper = True

def fn(x, y):
    return (x + y).sum()

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(128, 128, device=device)
y = torch.randn(128, 128, device=device)

# Compile the function
opt_fn = torch.compile(fn)
result = opt_fn(x, y)
```

## Step 3: Understand the Generated Code

When you run the compiled function, TorchInductor generates wrapper code. The structure of this code differs significantly between the default Python wrapper and the C++ wrapper.

### For CPU: Python vs. C++ Wrapper

**Default Python Wrapper (CPU):**
The generated Python wrapper manages tensors and invokes the fused kernel.

```python
class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def call(self, args):
        arg0_1, arg1_1 = args
        args.clear()
        assert_size_stride(arg0_1, (128, 128), (128, 1))
        assert_size_stride(arg1_1, (128, 128), (128, 1))
        buf0 = empty_strided_cpu((), (), torch.float32)
        cpp_fused_add_sum_0(arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        return (buf0, )
```

**C++ Wrapper (CPU):**
With `cpp_wrapper = True`, the wrapper is generated as a C++ function named `inductor_entry_impl`. This function uses the AOTI (Ahead-Of-Time Inductor) C API for tensor operations.

```cpp
#include <torch/csrc/inductor/cpp_wrapper/cpu.h>

extern "C" void cpp_fused_add_sum_0(const float* in_ptr0,
                                    const float* in_ptr1,
                                    float* out_ptr0);

CACHE_TORCH_DTYPE(float32);
CACHE_TORCH_DEVICE(cpu);

void inductor_entry_impl(
    AtenTensorHandle* input_handles,  // Array of input handles (borrowed array, stolen handles)
    AtenTensorHandle* output_handles  // Array for output handles (borrowed array, caller steals handles)
) {
    py::gil_scoped_release_simple release;

    auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, 2);
    auto arg0_1 = std::move(inputs[0]);
    auto arg1_1 = std::move(inputs[1]);
    static constexpr int64_t *int_array_0 = nullptr;
    AtenTensorHandle buf0_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(0, int_array_0, int_array_0,
                                                         cached_torch_dtype_float32,
                                                         cached_torch_device_type_cpu,
                                                         0, &buf0_handle));
    RAIIAtenTensorHandle buf0(buf0_handle);
    cpp_fused_add_sum_0((const float*)(arg0_1.data_ptr()),
                        (const float*)(arg1_1.data_ptr()),
                        (float*)(buf0.data_ptr()));
    arg0_1.reset();
    arg1_1.reset();
    output_handles[0] = buf0.release();
} // inductor_entry_impl
```

The C++ code is then loaded via a Python binding:

```python
inductor_entry = CppWrapperCodeCache.load_pybinding(
    argtypes=["std::vector<AtenTensorHandle>"],
    main_code=cpp_wrapper_src,
    device_type="cpu",
    num_outputs=1,
    kernel_code=None,
)

call = _wrap_func(inductor_entry)
```

### For GPU: Python vs. C++ Wrapper

**Default Python Wrapper (GPU):**
The GPU wrapper manages CUDA streams and device context.

```python
def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)  # Ensure context
        buf0 = empty_strided((19, ), (1, ), device='cuda', dtype=torch.float32)
        stream0 = get_cuda_stream(0)
        triton_poi_fused_add_lift_fresh_0.run(constant0, arg0_1, buf0, 19,
                                              grid=grid(19), stream=stream0)
        run_intermediate_hooks('add', buf0)
        del arg0_1
        return (buf0, )
```

**C++ Wrapper (GPU):**
The C++ wrapper for GPU follows a similar pattern but uses a helper function to convert between Python tensors and C++ handles.

```python
inductor_entry = CppWrapperCodeCache.load_pybinding(
    argtypes=["std::vector<AtenTensorHandle>"],
    main_code=cpp_wrapper_src,  # Contains the C++ `inductor_entry_impl`
    device_type="cuda",
    num_outputs=1,
    kernel_code=None,
)

def _wrap_func(f):
    def g(args):
        # Convert input tensors to C handles
        input_tensors = [arg if isinstance(arg, torch.Tensor) else torch.tensor(arg, device='cpu') for arg in args]
        input_handles = torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(input_tensors)

        args.clear()
        del input_tensors

        # Call the C++ entry point
        output_handles = f(input_handles)
        # Convert output handles back to Python tensors
        output_tensors = torch._C._aoti.alloc_tensors_by_stealing_from_void_ptrs(output_handles)
        return output_tensors

    return g

call = _wrap_func(inductor_entry)
```

## Conclusion

This tutorial introduced the **C++ wrapper** feature in TorchInductor, a performance optimization that minimizes Python runtime overhead. You learned how to enable the feature with a single configuration flag and saw the structural differences between the default Python wrapper and the new C++ wrapper for both CPU and GPU backends. By adopting the C++ wrapper, you can achieve faster model execution with minimal code changes.