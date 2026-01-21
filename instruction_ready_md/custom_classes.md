# Extending PyTorch with Custom C++ Classes: A Step-by-Step Guide

This guide walks you through binding a custom C++ class into PyTorch, enabling you to seamlessly integrate C++ logic with your PyTorch workflows. The API closely resembles pybind11, making it familiar if you have experience with that system.

## Prerequisites

Before starting, ensure you have:
- PyTorch installed (with C++ extensions support)
- CMake (version 3.10 or higher)
- A C++ compiler (GCC/Clang)

## Step 1: Implement Your C++ Class

Create a file called `class.cpp` with the following content. This example defines a simple stack-like class that maintains persistent state.

```cpp
#include <torch/custom_class.h>
#include <torch/script.h>
#include <vector>
#include <string>

// BEGIN class
class MyStackClass : public torch::CustomClassHolder {
 private:
  std::vector<std::string> stack_;

 public:
  MyStackClass(std::vector<std::string> init) : stack_(init.begin(), init.end()) {}

  void push(std::string x) {
    stack_.push_back(x);
  }

  std::string pop() {
    auto val = stack_.back();
    stack_.pop_back();
    return val;
  }

  c10::intrusive_ptr<MyStackClass> clone() const {
    return c10::make_intrusive<MyStackClass>(stack_);
  }

  std::vector<std::string> get_stack() const {
    return stack_;
  }
};
// END class
```

Key points to note:
- Include `torch/custom_class.h` to access PyTorch's custom class API
- Your class must inherit from `torch::CustomClassHolder` to support reference counting
- Use `c10::intrusive_ptr<>` for managing instances (similar to `std::shared_ptr` but with embedded reference counting)

## Step 2: Bind the Class to PyTorch

Add the binding code to the same `class.cpp` file, after your class definition:

```cpp
// BEGIN binding
TORCH_LIBRARY(my_classes, m) {
  m.class_<MyStackClass>("MyStackClass")
    .def(torch::init<std::vector<std::string>>())
    .def("push", &MyStackClass::push)
    .def("pop", &MyStackClass::pop)
    .def("clone", &MyStackClass::clone)
    .def("get_stack", &MyStackClass::get_stack);
}
// END binding
```

This code:
- Creates a library namespace `my_classes`
- Registers `MyStackClass` with its constructor and methods
- Makes the class accessible from Python

## Step 3: Build with CMake

Create a `CMakeLists.txt` file in the same directory:

```cmake
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(custom_class_project)

find_package(Torch REQUIRED)

add_library(custom_class SHARED class.cpp)
target_link_libraries(custom_class "${TORCH_LIBRARIES}")
set_property(TARGET custom_class PROPERTY CXX_STANDARD 14)
```

Create a build directory and compile:

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
make -j
```

After successful compilation, you should find `libcustom_class.so` (or similar) in your build directory.

## Step 4: Use the Class from Python

Create a Python script `custom_test.py` to test your bound class:

```python
import torch

# Load the compiled library
torch.ops.load_library("build/libcustom_class.so")

# Create an instance of your custom class
s = torch.classes.my_classes.MyStackClass(["foo", "bar"])

# Use the class methods
print(s.get_stack())  # ['foo', 'bar']

s.push("pushed")
print(s.pop())  # 'pushed'
print(s.get_stack())  # ['foo', 'bar']
```

Run the script to verify everything works:

```bash
python custom_test.py
```

## Step 5: Add Serialization Support (Optional)

If you need to save models containing your custom class, you must define serialization methods. Update your binding code in `class.cpp`:

```cpp
// Add this inside your TORCH_LIBRARY block, after the method definitions
// BEGIN def_pickle
.def_pickle(
  // __getstate__
  [](const c10::intrusive_ptr<MyStackClass>& self) -> std::vector<std::string> {
    return self->get_stack();
  },
  // __setstate__
  [](std::vector<std::string> state) -> c10::intrusive_ptr<MyStackClass> {
    return c10::make_intrusive<MyStackClass>(std::move(state));
  }
);
// END def_pickle
```

Now you can save and load models containing your custom class:

```python
import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = torch.classes.my_classes.MyStackClass(["foo", "bar"])
    
    def forward(self):
        return self.stack.get_stack()

# Create and save the module
m = MyModule()
torch.jit.script(m).save("model.pt")

# Load it back
loaded = torch.jit.load("model.pt")
print(loaded.stack.get_stack())  # ['foo', 'bar']
```

## Step 6: Define Custom Operators (Optional)

You can also create operators that work with your custom class. Add this to your `class.cpp`:

```cpp
// BEGIN free_function
c10::intrusive_ptr<MyStackClass> manipulate_instance(
    c10::intrusive_ptr<MyStackClass> instance) {
  instance->push("manipulated");
  return instance;
}
// END free_function
```

Register the operator in your `TORCH_LIBRARY` block:

```cpp
// BEGIN def_free
m.def("manipulate_instance", &manipulate_instance);
// END def_free
```

Use it in Python:

```python
class TryCustomOp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.classes.my_classes.MyStackClass(["foo", "bar"])

    def forward(self):
        return torch.ops.my_classes.manipulate_instance(self.f)

module = TryCustomOp()
result = module()
print(result.get_stack())  # ['foo', 'bar', 'manipulated']
```

## Conclusion

You've successfully learned how to:
1. Implement a custom C++ class for PyTorch
2. Bind it to the PyTorch ecosystem
3. Build it using CMake
4. Use it from Python
5. Add serialization support for model saving
6. Create custom operators that work with your class

This enables you to extend PyTorch with high-performance C++ code while maintaining seamless integration with Python workflows. For further questions, consult the [PyTorch forums](https://discuss.pytorch.org/) or [GitHub issues](https://github.com/pytorch/pytorch/issues).