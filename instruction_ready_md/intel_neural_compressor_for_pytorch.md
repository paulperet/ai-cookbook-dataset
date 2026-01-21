# Intel® Neural Compressor for PyTorch: A Practical Guide to Model Quantization

## Overview

While most deep learning inference runs at 32-bit floating-point precision, lower precision formats like FP8 offer significant performance gains. The challenge lies in maintaining accuracy while transitioning to these efficient data types.

Intel® Neural Compressor extends PyTorch with accuracy-driven automatic tuning strategies, helping you find the optimal quantized model for Intel hardware with minimal accuracy loss. This open-source tool provides a streamlined approach to model optimization.

## Key Features

- **Familiar API:** Reuses PyTorch's `prepare` and `convert` methods for intuitive usage
- **Accuracy-First Tuning:** Automatic tuning process that prioritizes maintaining model accuracy
- **Multiple Quantization Methods:** Supports INT8, weight-only, FP8, and MX data type emulation quantization
- **Hardware Optimization:** Automatically detects and optimizes for available accelerators (HPU, Intel GPU, CUDA, CPU)

## Prerequisites

### Installation

Install the PyTorch version of Intel Neural Compressor:

```bash
pip install neural-compressor-pt
```

**Note:** For device-specific optimization, set the target device environment variable:
```bash
export INC_TARGET_DEVICE=cpu  # Options: cpu, hpu, cuda, xpu
```

## Practical Examples

### 1. FP8 Quantization for HPU Acceleration

FP8 quantization is specifically supported on Intel® Gaudi®2&3 AI Accelerators (HPU). Ensure you have the [Intel® Gaudi® environment](https://docs.habana.ai/en/latest/index.html) properly configured before running this example.

```python
from neural_compressor.torch.quantization import FP8Config, prepare, convert
import torch
import torchvision.models as models

# Load a pre-trained model
model = models.resnet18()

# Configure FP8 quantization (E4M3 format)
qconfig = FP8Config(fp8_config="E4M3")
model = prepare(model, qconfig)

# Calibrate with sample data
calibration_data = torch.randn(1, 3, 224, 224).to("hpu")
model(calibration_data)

# Convert to FP8 precision
model = convert(model)

# Run inference
input_data = torch.randn(1, 3, 224, 224).to("hpu")
output = model(input_data).to("cpu")
print(output)
```

### 2. Weight-Only Quantization

Weight-only quantization reduces model size while maintaining activation precision. This example demonstrates loading a pre-quantized model from HuggingFace:

```python
from neural_compressor.torch.quantization import load
import torch

# Load a GPTQ model and convert it to HPU format
model_name = "TheBloke/Llama-2-7B-GPTQ"
model = load(
    model_name_or_path=model_name,
    format="huggingface",
    device="hpu",
    torch_dtype=torch.bfloat16,
)
```

**Important:** The first load converts the model from AutoGPTQ to HPU format and caches it locally as `hpu_model.safetensors`. Subsequent loads will be significantly faster.

### 3. Static Quantization with PT2E Backend

The PT2E backend uses TorchDynamo for graph capture and TorchCompile for optimized operator replacement. Follow these four steps:

```python
import torch
from neural_compressor.torch.export import export
from neural_compressor.torch.quantization import StaticQuantConfig, prepare, convert

# Step 1: Export eager model to FX graph
model = UserFloatModel()
example_inputs = ...  # Your model inputs
exported_model = export(model=model, example_inputs=example_inputs)

# Step 2: Prepare for quantization
quant_config = StaticQuantConfig()
prepared_model = prepare(exported_model, quant_config=quant_config)

# Step 3: Calibrate with your data
run_fn(prepared_model)  # Your calibration function

# Step 4: Convert and compile
q_model = convert(prepared_model)

# Optimize with TorchCompile
from torch._inductor import config
config.freezing = True
opt_model = torch.compile(q_model)
```

### 4. Accuracy-Driven Automatic Tuning

For scenarios where maintaining accuracy is critical, use the `autotune` function to automatically find the best quantization configuration:

```python
from neural_compressor.torch.quantization import RTNConfig, TuningConfig, autotune

def eval_fn(model) -> float:
    # Your evaluation function returning a accuracy metric
    return accuracy_score

# Define tuning parameters
tune_config = TuningConfig(
    config_set=RTNConfig(use_sym=[False, True], group_size=[32, 128]),
    tolerable_loss=0.2,  # Maximum acceptable accuracy drop
    max_trials=10,       # Stop after 10 configurations
)

# Run automatic tuning
q_model = autotune(model, tune_config=tune_config, eval_fn=eval_fn)
```

The tuner will:
1. Test different configurations from your parameter space
2. Evaluate each using your `eval_fn`
3. Stop when it finds a configuration within the tolerable loss threshold or reaches the maximum trial limit
4. Return the best-performing quantized model

## Next Steps

For comprehensive tutorials and advanced usage, visit the [official Intel® Neural Compressor documentation](https://intel.github.io/neural-compressor/latest/docs/source/Welcome.html).

The tool supports various quantization methods across different hardware platforms, with detailed configuration options for fine-tuning your optimization strategy.