# Customizing Process Group Backends with C++ Extensions

**Author**: [Howard Huang](https://github.com/H-Huang), [Feng Tian](https://github.com/ftian1), [Shen Li](https://mrshenli.github.io/), [Min Si](https://minsii.github.io/)

> **Note**: You can view and edit this tutorial on [GitHub](https://github.com/pytorch/tutorials/blob/main/intermediate_source/process_group_cpp_extension_tutorial.rst).

## Prerequisites

Before starting, ensure you're familiar with:
- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch Collective Communication Package](https://pytorch.org/docs/stable/distributed.html)
- [PyTorch C++ Extensions](https://pytorch.org/docs/stable/cpp_extension.html)
- [Writing Distributed Applications with PyTorch](https://pytorch.org/tutorials/intermediate/dist_tuto.html)

## Introduction

PyTorch's distributed package provides powerful collective communication operations that enable features like `DistributedDataParallel` and `ZeroRedundancyOptimizer`. These operations work across different communication backends through an abstraction layer called the `Backend` class.

While PyTorch ships with three built-in backends (`ProcessGroupNCCL`, `ProcessGroupGloo`, and `ProcessGroupMPI`), you might need to:
- Support specialized hardware (TPU, Trainium, etc.)
- Integrate alternative communication libraries (UCC, OneCCL, etc.)
- Experiment with new communication algorithms

This tutorial demonstrates how to implement a custom backend using C++ extensions. We'll create a "dummy" backend that implements `all_reduce` and `all_gather` operations (simply setting tensors to zero) to illustrate the extension mechanism.

## Step 1: Implement the Backend Subclass

First, create a C++ header file that defines your custom backend class. This class must inherit from PyTorch's `Backend` base class and override the collective communication APIs you want to implement.

Create `dummy.hpp`:

```cpp
// dummy.hpp
#include <torch/python.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <pybind11/chrono.h>

namespace c10d {

class BackendDummy : public Backend {
  public:
    BackendDummy(int rank, int size);

    c10::intrusive_ptr<Work> allgather(
        std::vector<std::vector<at::Tensor>>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const AllgatherOptions& opts = AllgatherOptions()) override;

    c10::intrusive_ptr<Work> allreduce(
        std::vector<at::Tensor>& tensors,
        const AllreduceOptions& opts = AllreduceOptions()) override;

    // Note: APIs without custom implementations will error if called
};

class WorkDummy : public Work {
  public:
    WorkDummy(
      OpType opType,
      c10::intrusive_ptr<c10::ivalue::Future> future)
      : Work(-1, opType),  // rank is only used by recvAnySource
        future_(std::move(future)) {}
    
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
    virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

  private:
    c10::intrusive_ptr<c10::ivalue::Future> future_;
};

} // namespace c10d
```

Now implement the methods in `dummy.cpp`. For this demonstration, we'll create dummy implementations that simply zero out tensors:

```cpp
// dummy.cpp
#include "dummy.hpp"

namespace c10d {

// Dummy allgather that sets all output tensors to zero
c10::intrusive_ptr<Work> BackendDummy::allgather(
        std::vector<std::vector<at::Tensor>>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const AllgatherOptions& /* unused */) {
    
    // Zero out all output tensors
    for (auto& outputTensorVec : outputTensors) {
        for (auto& outputTensor : outputTensorVec) {
            outputTensor.zero_();
        }
    }

    // Create a future with the result
    auto future = c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
    future->markCompleted(c10::IValue(outputTensors));
    
    return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

// Dummy allreduce that sets all tensors to zero
c10::intrusive_ptr<Work> BackendDummy::allreduce(
        std::vector<at::Tensor>& tensors,
        const AllreduceOptions& opts) {
    
    // Zero out all input tensors
    for (auto& tensor : tensors) {
        tensor.zero_();
    }

    // Create a future with the result
    auto future = c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()));
    future->markCompleted(c10::IValue(tensors));
    
    return c10::make_intrusive<WorkDummy>(OpType::ALLGATHER, std::move(future));
}

} // namespace c10d
```

## Step 2: Expose Python APIs

The backend needs to be accessible from Python. Add a static constructor method and registration code to your header file:

Update `dummy.hpp` with the following additions:

```cpp
// Add to dummy.hpp
class BackendDummy : public Backend {
    // ... existing code from Step 1 ...
    
    static c10::intrusive_ptr<Backend> createBackendDummy(
        const c10::intrusive_ptr<::c10d::Store>& store,
        int rank,
        int size,
        const std::chrono::duration<float>& timeout);

    static void BackendDummyConstructor() __attribute__((constructor)) {
        py::object module = py::module::import("torch.distributed");
        py::object register_backend =
            module.attr("Backend").attr("register_backend");
        
        // Register "dummy" as a valid backend
        register_backend("dummy", py::cpp_function(createBackendDummy));
    }
};
```

Now implement the constructor in `dummy.cpp`:

```cpp
// Add to dummy.cpp
c10::intrusive_ptr<Backend> BackendDummy::createBackendDummy(
        const c10::intrusive_ptr<::c10d::Store>& /* unused */,
        int rank,
        int size,
        const std::chrono::duration<float>& /* unused */) {
    return c10::make_intrusive<BackendDummy>(rank, size);
}

// Add PYBIND11_MODULE at the end of dummy.cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("createBackendDummy", &BackendDummy::createBackendDummy);
}
```

## Step 3: Build the Extension

Create a `setup.py` file to build your C++ extension:

```python
# setup.py
import os
import sys
import torch
from setuptools import setup
from torch.utils import cpp_extension

sources = ["src/dummy.cpp"]
include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/include/"]

# Choose appropriate extension based on CUDA availability
if torch.cuda.is_available():
    module = cpp_extension.CUDAExtension(
        name = "dummy_collectives",
        sources = sources,
        include_dirs = include_dirs,
    )
else:
    module = cpp_extension.CppExtension(
        name = "dummy_collectives",
        sources = sources,
        include_dirs = include_dirs,
    )

setup(
    name = "Dummy-Collectives",
    version = "0.0.1",
    ext_modules = [module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
```

Build and install the extension:

```bash
python setup.py develop
```

> **Note**: For real-world extensions that depend on third-party libraries, specify `library_dirs` and `libraries` in the extension configuration. See the [torch-ucc](https://github.com/openucx/torch-ucc) project for a complete example.

## Step 4: Use the Custom Backend in Applications

After installation, you can use your `dummy` backend just like any built-in backend. Here's an example application:

```python
# app.py
import os
import torch

# Importing dummy_collectives registers "dummy" as a valid backend
import dummy_collectives
import torch.distributed as dist

# Set up distributed environment
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

# Initialize process group with backend dispatching
# CPU tensors use gloo, CUDA tensors use dummy
dist.init_process_group("cpu:gloo,cuda:dummy", rank=0, world_size=1)

# This operation goes through gloo (CPU tensor)
x = torch.ones(6)
dist.all_reduce(x)
print(f"CPU allreduce result: {x}")

# This operation goes through dummy (CUDA tensor)
if torch.cuda.is_available():
    y = x.cuda()
    dist.all_reduce(y)
    print(f"CUDA allreduce result: {y}")
    
    # Unimplemented operations will raise RuntimeError
    try:
        dist.broadcast(y, 0)
    except RuntimeError:
        print("RuntimeError: broadcast not implemented in dummy backend")
```

Alternatively, to send all tensors through the dummy backend:

```python
# Use dummy backend for all operations
dist.init_process_group("dummy", rank=0, world_size=1)
```

## Summary

You've successfully created and integrated a custom PyTorch distributed backend. The key steps were:

1. **Implement a Backend subclass** in C++ that overrides collective communication APIs
2. **Expose Python APIs** using pybind11 for backend registration
3. **Build the extension** using PyTorch's C++ extension utilities
4. **Use the custom backend** in your distributed applications

For production use, replace the dummy implementations with actual communication logic using your preferred libraries or hardware interfaces. The full example code is available in the [dummy collectives repository](https://github.com/H-Huang/torch_collective_extension).