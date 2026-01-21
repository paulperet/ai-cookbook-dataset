# Boost PyTorch Model Performance on CPU with Max-Autotune Compilation

**Author**: [Jiong Gong](https://github.com/jgong5), [Leslie Fang](https://github.com/leslie-fang-intel), [Chunyuan Wu](https://github.com/chunyuan-w)

## Overview

This guide demonstrates how to use PyTorch's `max-autotune` compilation mode to accelerate model inference on CPU. This mode profiles multiple operation implementations at compile time, selecting the fastest one—trading longer compilation for superior runtime performance. It is especially effective for GEMM (General Matrix Multiply) operations, where a new C++ template-based implementation can outperform traditional ATen kernels that rely on oneDNN/MKL.

## Prerequisites

- Basic understanding of `torch.compile` and TorchInductor. Review the [PyTorch torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) if needed.

## Setup & Configuration

Before you begin, configure your environment to enable the necessary features.

### 1. Enable Max-Autotune Mode

To activate the `max-autotune` mode, pass `mode="max-autotune"` to `torch.compile`. This instructs the Inductor backend to profile and select the optimal kernel implementations.

### 2. (Optional) Force C++ Template Usage

If you want to bypass autotuning and always use the C++ template-based GEMM implementation, set the following environment variable:
```bash
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=CPP
```

### 3. Enable Model Freezing for Inference

The C++ template implementation pre-packs constant model weights for optimal cache usage. This requires the model to be frozen (i.e., in inference mode). Ensure both compilation and execution happen within a `torch.no_grad()` context and set:
```bash
export TORCHINDUCTOR_FREEZING=1
```

## Step-by-Step Tutorial

Follow these steps to compile and run a simple neural network using `max-autotune`.

### Step 1: Import Libraries and Configure Logging

First, import PyTorch and enable logging for autotuning results to see which kernels are selected.

```python
import torch
from torch._inductor import config

# Enable logging of autotuning results
config.trace.log_autotuning_results = True
```

### Step 2: Define a Simple Model

Create a small neural network with a linear layer followed by a ReLU activation. This is a common pattern where GEMM optimization can provide significant benefits.

```python
class SimpleModel(torch.nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x
```

### Step 3: Prepare Input and Model

Instantiate the model and create a sample input tensor.

```python
# Configuration
amp_enabled = True  # Enable automatic mixed precision
batch_size = 64
in_features = 16
out_features = 32
bias = True

# Create input and model
x = torch.randn(batch_size, in_features)
model = SimpleModel(in_features, out_features, bias)
```

### Step 4: Compile and Run with Max-Autotune

Wrap the compilation and execution inside `torch.no_grad()` to ensure the model is frozen for inference. Use `torch.cpu.amp.autocast` if you want to leverage mixed precision.

```python
with torch.no_grad(), torch.cpu.amp.autocast(enabled=amp_enabled):
    # Compile the model with max-autotune mode
    compiled_model = torch.compile(model, mode="max-autotune")
    
    # Run inference
    y = compiled_model(x)
```

## Understanding the Output

When you run the code, the autotuning system will profile available kernels and log the results. For our example, you might see output similar to:

```
AUTOTUNE linear_unary(64x16, 32x16, 32)
cpp_packed_gemm_0 0.2142 ms 100.0%
_linear_pointwise 0.2441 ms 87.7%
```

This log shows that the C++ template GEMM kernel (`cpp_packed_gemm_0`) was faster than the ATen-based kernel (`_linear_pointwise`) and was therefore selected.

### Inspecting the Generated Code

To see the actual generated kernel code, set the environment variable `TORCH_LOGS="+output_code"` before running your script. When the C++ template is selected, the generated code will not contain calls like `torch.ops.mkldnn._linear_pointwise.default` (for BF16) or `torch.ops.mkl._mkl_linear.default` (for FP32). Instead, you'll find a fused kernel (e.g., `cpp_fused__to_copy_relu_1`) that incorporates the linear transformation, bias addition, and ReLU activation within a single, optimized GEMM template.

The exact generated code is hardware-specific and may change, but it will be highly optimized for your CPU architecture, using techniques like loop tiling, vectorization, and OpenMP parallelism.

## Key Takeaways

1.  **Performance Trade-off**: `max-autotune` increases compilation time but can significantly improve runtime performance, especially for compute-bound operations like GEMM.
2.  **Inference-Only**: The current C++ template implementation requires a frozen model. Always run compilation and inference inside `torch.no_grad()` with `TORCHINDUCTOR_FREEZING=1`.
3.  **Kernel Selection**: The autotuner automatically chooses the fastest kernel. You can monitor this selection via logging.
4.  **Architecture-Specific Code**: The generated kernels are tailored to your CPU, leveraging advanced instruction sets (like AMX on newer Intel CPUs) for maximum speed.

## Conclusion

You have successfully used PyTorch's `max-autotune` mode to compile a model for faster CPU inference. This feature is under active development. If you encounter issues or have feature requests, please file a report on the [PyTorch GitHub Issues](https://github.com/pytorch/pytorch/issues) page.