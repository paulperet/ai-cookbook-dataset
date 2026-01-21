# Guide: Supporting Custom C++ Classes in torch.compile and torch.export

## Overview

This guide builds upon the foundation of creating custom C++ classes for PyTorch. It details the additional steps required to make those classes compatible with `torch.compile` and `torch.export`. These steps involve implementing methods to inspect and reconstruct the object's state, and registering a Python "fake" class to enable safe, efficient tracing.

> **Note:** This feature is a prototype as of PyTorch 2.8 and is subject to breaking changes. Please report any issues on GitHub.

## Prerequisites

Ensure you have a working custom C++ class built as a shared library. This guide uses a thread-safe tensor queue as an example.

## Step 1: Implement `__obj_flatten__` in C++

The `__obj_flatten__` method allows PyTorch's compiler to inspect the object's internal state for guarding purposes. It must return a tuple of `(attribute_name, value)` pairs.

### 1.1 Add the Method to Your C++ Class

Add the `__obj_flatten__` method to your class definition. It should return the object's key states.

```cpp
// Thread-safe Tensor Queue
struct TensorQueue : torch::CustomClassHolder {
    // ... existing constructors and methods ...

    std::tuple<std::tuple<std::string, std::vector<at::Tensor>>, std::tuple<std::string, at::Tensor>> __obj_flatten__() {
        return std::tuple(
            std::tuple("queue", this->get_raw_queue()),
            std::tuple("init_tensor_", this->init_tensor_.clone())
        );
    }

private:
    std::deque<at::Tensor> queue_;
    std::mutex mutex_;
    at::Tensor init_tensor_;
};
```

### 1.2 Register the Method in the Library

Update your `TORCH_LIBRARY` registration to expose the new method to Python.

```cpp
TORCH_LIBRARY(MyCustomClass, m) {
    m.class_<TensorQueue>("TensorQueue")
        .def(torch::init<at::Tensor>())
        .def("push", &TensorQueue::push)
        .def("pop", &TensorQueue::pop)
        .def("get_raw_queue", &TensorQueue::get_raw_queue)
        .def("__obj_flatten__", &TensorQueue::__obj_flatten__); // Register the new method
}
```

## Step 2: Create and Register a Python Fake Class

A fake class allows `torch.compile` and `torch.export` to trace operations without executing the real, potentially expensive or side-effect-heavy, C++ code.

### 2.1 Implement the Fake Class

Create a Python class that mirrors your C++ class's interface. It must be decorated with `@torch._library.register_fake_class`.

```python
import torch
from typing import List

# The decorator argument must match your C++ namespace and class name.
@torch._library.register_fake_class("MyCustomClass::TensorQueue")
class FakeTensorQueue:
    def __init__(self, queue: List[torch.Tensor], init_tensor_: torch.Tensor) -> None:
        # Store the object's state as simple Python/Torch objects.
        self.queue = queue
        self.init_tensor_ = init_tensor_

    # Mirror the C++ methods with fake implementations.
    def push(self, tensor: torch.Tensor) -> None:
        self.queue.append(tensor)

    def pop(self) -> torch.Tensor:
        if len(self.queue) > 0:
            return self.queue.pop(0)
        return self.init_tensor_
```

### 2.2 Implement the `__obj_unflatten__` Class Method

This method tells PyTorch how to reconstruct a fake object from the flattened state returned by `__obj_flatten__`.

```python
@torch._library.register_fake_class("MyCustomClass::TensorQueue")
class FakeTensorQueue:
    # ... __init__, push, pop ...

    @classmethod
    def __obj_unflatten__(cls, flattened_state):
        # The flattened_state is the tuple returned by __obj_flatten__.
        # Convert it to a dictionary and pass it to the constructor.
        return cls(**dict(flattened_state))
```

## Step 3: Using Your Class with torch.compile and torch.export

With the fake class registered, you can now use your custom object in compilable code.

### 3.1 Load the Library and Create an Instance

```python
import torch

# Load your compiled C++ library.
torch.classes.load_library("build/libcustom_class.so")

# Create an instance of your custom class.
tq = torch.classes.MyCustomClass.TensorQueue(torch.empty(0).fill_(-1))
```

### 3.2 Define and Compile a Module

Create a PyTorch module that uses your custom object.

```python
class Mod(torch.nn.Module):
    def forward(self, tq, x):
        tq.push(x.sin())
        tq.push(x.cos())
        popped_t = tq.pop()
        # Verify the operation worked as expected.
        assert torch.allclose(popped_t, x.sin())
        return tq, popped_t

# Compile the module. The fake class will be used during tracing.
compiled_mod = torch.compile(Mod(), backend="eager", fullgraph=True)
tq_result, popped_t_result = compiled_mod(tq, torch.randn(2, 3))

print(f"Queue size after operations: {tq_result.size()}")
```

### 3.3 Export the Module

You can also export the module to a static graph.

```python
# Export the model for later use.
exported_program = torch.export.export(Mod(), (tq, torch.randn(2, 3)), strict=False)

# Run the exported graph.
exported_program.module()(tq, torch.randn(2, 3))
```

## Advanced: Supporting Custom Ops with Fake Implementations

If your C++ class has associated custom ops, you must also provide fake implementations for them in Python.

### 1. Add a Custom Op in C++

First, add a method to your class and register it as a custom op.

```cpp
struct TensorQueue : torch::CustomClassHolder {
    // ... existing code ...
    void for_each_add_(at::Tensor inc) {
        for (auto& t : queue_) {
            t.add_(inc);
        }
    }
};

TORCH_LIBRARY_FRAGMENT(MyCustomClass, m) {
    m.class_<TensorQueue>("TensorQueue")
        .def("for_each_add_", &TensorQueue::for_each_add_);
    m.def("for_each_add_(TensorQueue foo, Tensor inc) -> ()");
}

// Implement the dispatch logic.
void for_each_add_(c10::intrusive_ptr<TensorQueue> tq, at::Tensor inc) {
    tq->for_each_add_(inc);
}

TORCH_LIBRARY_IMPL(MyCustomClass, CPU, m) {
    m.impl("for_each_add_", for_each_add_);
}
```

### 2. Register a Fake Implementation in Python

Use `@torch.library.register_fake` to provide a Python implementation for tracing.

```python
@torch.library.register_fake("MyCustomClass::for_each_add_")
def fake_for_each_add_(tq, inc):
    # Call the corresponding method on the fake object.
    tq.for_each_add_(inc)
```

### 3. Use the Custom Op in Export

After recompiling your C++ library, you can use the custom op in an exportable module.

```python
class ForEachAdd(torch.nn.Module):
    def forward(self, tq: torch.ScriptObject, a: torch.Tensor) -> torch.ScriptObject:
        torch.ops.MyCustomClass.for_each_add_(tq, a)
        return tq

mod = ForEachAdd()
tq = empty_tensor_queue() # Assume a helper function to create a queue
for i in range(10):
    tq.push(torch.zeros(1))

# Export the module using the custom op.
ep = torch.export.export(mod, (tq, torch.ones(1)), strict=False)
```

## Important Considerations

### Why Fake Classes Are Necessary

Tracing with real custom objects has significant drawbacks:
1.  **Performance:** Real operations (e.g., network/disk I/O) are slow and unsuitable for repeated tracing runs.
2.  **Side Effects:** Tracing should not mutate real objects or the environment.
3.  **Dynamic Shapes:** Real objects often cannot support the shape analysis required for compilation.

### Opting Out of Fakification

If writing a fake class is impractical (e.g., due to complex third-party dependencies), you can force the tracer to use the real object by implementing a `tracing_mode` method in your C++ class.

```cpp
std::string tracing_mode() {
    return "real";
}
```

> **Warning:** Using `"real"` mode reintroduces the downsides mentioned above and is not recommended for production compilation.

### Tensor Aliasing Caveat

The fakification process assumes **no tensor aliasing** between tensors inside the custom object and tensors outside of it. Mutating a tensor that aliases an external tensor will lead to undefined behavior during tracing. Ensure your object's design respects this constraint.

## Summary

To make a custom C++ class compatible with `torch.compile` and `torch.export`:
1.  Implement `__obj_flatten__` in C++ to expose object state.
2.  Create and register a Python fake class with `@torch._library.register_fake_class`, implementing all methods and `__obj_unflatten__`.
3.  For any custom ops, register corresponding fake implementations with `@torch.library.register_fake`.

Following these steps allows PyTorch's compiler to safely and efficiently trace programs that use your custom classes, unlocking the benefits of graph compilation.