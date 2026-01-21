# PyTorch Out-of-Tree Extension Autoloading Guide

**Author:** [Yuanhao Ji](https://github.com/shink)

## Overview

PyTorch's extension autoloading mechanism enables automatic loading of out-of-tree backend extensions without requiring explicit import statements. This feature enhances the user experience by maintaining the familiar PyTorch device programming model and allows existing PyTorch applications to work with new hardware devices with zero code changes.

## Prerequisites

- **PyTorch v2.5 or later**
- Basic understanding of Python package structure and entry points

## Quick Start: How Autoloading Works

Autoloading is implemented using Python's [Entrypoints](https://packaging.python.org/en/latest/specifications/entry-points/) mechanism. When you install an out-of-tree extension package, PyTorch automatically discovers and loads it through defined entry points in `torch/__init__.py`.

### Important Note

This feature is enabled by default. To disable it, set the environment variable:
```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

If you encounter "Failed to load the backend extension" errors, disable autoloading and contact the extension maintainer for assistance.

## Step-by-Step Implementation Guide

### Step 1: Create Your Extension Package Structure

Suppose you're creating a backend named `foo` with package name `torch_foo`. Your package must be compatible with PyTorch 2.5 or later.

### Step 2: Add Autoload Function to `__init__.py`

In your `torch_foo/__init__.py` file, include an autoload function:

```python
def _autoload():
    print("Check things are working with `torch.foo.is_available()`.")
```

### Step 3: Define Entry Point in `setup.py`

Configure your package's entry point in the setup configuration:

```python
from setuptools import setup

setup(
    name="torch_foo",
    version="1.0",
    entry_points={
        "torch.backends": [
            "torch_foo = torch_foo:_autoload",
        ],
    }
)
```

### Step 4: Test Your Implementation

After installing your package, test the autoloading:

```python
>>> import torch
Check things are working with `torch.foo.is_available()`.
>>> torch.foo.is_available()
True
```

Notice that you only need `import torch` - the `torch_foo` module loads automatically!

## Handling Circular Imports: Real-World Examples

Circular imports can occur when your extension interacts with PyTorch during initialization. Here are two proven approaches from existing implementations.

### Example 1: Intel Gaudi HPU (Habana Frameworks)

The `habana_frameworks.torch` package enables PyTorch programs on Intel Gaudi HPU devices.

#### Step 1: Add Entry Point to Setup

In `habana_frameworks/setup.py`:

```python
setup(
    name="habana_frameworks",
    version="2.5",
    entry_points={
        'torch.backends': [
            "device_backend = habana_frameworks:__autoload",
        ],
    }
)
```

#### Step 2: Implement Autoload with State Tracking

In `habana_frameworks/__init__.py`:

```python
import os

is_loaded = False  # Track if module has been imported

def __autoload():
    # Entry point for PyTorch autoload mechanism
    global is_loaded
    if is_loaded:
        return  # Skip to avoid circular imports
    import habana_frameworks.torch
```

#### Step 3: Update State in Submodule

In `habana_frameworks/torch/__init__.py`:

```python
import os
import habana_frameworks

# Prevent circular imports by updating global state
habana_frameworks.is_loaded = True
```

### Example 2: Huawei Ascend NPU (torch_npu)

The `torch_npu` package enables PyTorch programs on Huawei Ascend NPU using the `PrivateUse1` device key.

#### Step 1: Define Entry Point

In `torch_npu/setup.py`:

```python
setup(
    name="torch_npu",
    version="2.5",
    entry_points={
        'torch.backends': [
            'torch_npu = torch_npu:_autoload',
        ],
    }
)
```

#### Step 2: Control Autoloading with Environment Variable

This implementation uses environment variables to prevent circular imports:

```python
import os

# Disable autoloading before importing torch
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import torch
# Your extension logic here
```

## Implementation Details

The autoloading mechanism works as follows:

1. **Entry Point Discovery**: PyTorch scans for all packages with `torch.backends` entry points
2. **Automatic Loading**: When `import torch` is executed, all discovered entry points are loaded
3. **Extension Initialization**: Each extension's autoload function is called, performing necessary setup
4. **Device Availability**: The extension makes its device available through PyTorch's device API

For technical implementation details, see the [original pull request](https://github.com/pytorch/pytorch/pull/127074).

## Best Practices

1. **Version Compatibility**: Ensure your extension supports PyTorch 2.5+
2. **Error Handling**: Implement graceful fallbacks if autoloading fails
3. **State Management**: Use global variables or environment variables to track loading state
4. **Minimal Initialization**: Keep autoload functions lightweight to avoid startup delays

## Conclusion

PyTorch's out-of-tree extension autoloading mechanism simplifies hardware integration by eliminating explicit import requirements. By defining proper entry points and handling circular imports appropriately, you can create seamless backend extensions that automatically integrate with PyTorch's device ecosystem.

This approach has been successfully implemented by major hardware vendors including Intel (Gaudi HPU) and Huawei (Ascend NPU), demonstrating its robustness for production use cases.