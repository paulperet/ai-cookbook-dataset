# PyTorch 2 Export Quantization for OpenVINO torch.compile Backend: A Step-by-Step Guide

**Authors**: Daniil Lyakhov, Aamir Nazir, Alexander Suslov, Yamini Nimmagadda, Alexander Kozlov

## Introduction

This guide demonstrates how to use the `OpenVINOQuantizer` from the Neural Network Compression Framework (NNCF) within the PyTorch 2 Export Quantization flow. This process generates a quantized model optimized for the **OpenVINO torch.compile backend**, enabling you to leverage high-performance, low-precision kernels on Intel hardware.

> **Note**: This is an experimental feature. The quantization API is subject to change.

The workflow consists of four main steps:
1.  Capture the model's computational graph using `torch.export`.
2.  Apply post-training quantization using `OpenVINOQuantizer`.
3.  Lower the quantized model into an optimized OpenVINO representation using `torch.compile`.
4.  *(Optional)* Improve model accuracy using NNCF's advanced quantization algorithms.

## Prerequisites

Ensure you have the following installed:
*   PyTorch (2.x or later)
*   OpenVINO
*   NNCF (Neural Network Compression Framework)

You can install the required packages via pip:

```bash
pip install -U pip
pip install openvino nncf
```

Familiarity with the following concepts is helpful:
*   [PyTorch 2 Export Post Training Quantization](https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html)
*   [How to Write a Quantizer for PyTorch 2 Export Quantization](https://pytorch.org/tutorials/prototype/pt2e_quantizer.html)

## Step 1: Capture the FX Graph

First, you need to capture your eager model into a static graph representation using PyTorch's export mechanism. This graph will be the foundation for the quantization process.

```python
import copy
import openvino.torch
import torch
import torchvision.models as models
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
import nncf.torch

# 1. Load your model and set it to evaluation mode
model_name = "resnet18"
model = models.__dict__[model_name](pretrained=True)
model = model.eval()

# 2. Prepare example input data (using dummy data here)
traced_bs = 50
x = torch.randn(traced_bs, 3, 224, 224)
example_inputs = (x,)

# 3. Capture the FX Graph
with torch.no_grad(), nncf.torch.disable_patching():
    exported_model = torch.export.export(model, example_inputs).module()
```

## Step 2: Apply Post-Training Quantization

With the FX graph captured, you can now apply quantization. This involves preparing the model, calibrating it with sample data, and finally converting it to a quantized version.

### 2.1 Initialize the OpenVINOQuantizer

Import and create an instance of `OpenVINOQuantizer`. This quantizer is specifically designed to place quantization operations optimally for the OpenVINO backend.

```python
from nncf.experimental.torch.fx import OpenVINOQuantizer

# Create the quantizer with default settings
quantizer = OpenVINOQuantizer()
```

The `OpenVINOQuantizer` accepts several key parameters to fine-tune the quantization process for better accuracy or performance:

*   **`preset`**: Defines the quantization scheme.
    *   `nncf.QuantizationPreset.PERFORMANCE` (Default): Uses symmetric quantization for both weights and activations.
    *   `nncf.QuantizationPreset.MIXED`: Uses symmetric quantization for weights and asymmetric quantization for activations. Recommended for models with non-ReLU activation functions (e.g., ELU, GELU).

    ```python
    quantizer = OpenVINOQuantizer(preset=nncf.QuantizationPreset.MIXED)
    ```

*   **`model_type`**: Specifies a model-specific quantization scheme. Currently, `nncf.ModelType.Transformer` is supported for models like BERT or Llama.
    ```python
    quantizer = OpenVINOQuantizer(model_type=nncf.ModelType.Transformer)
    ```

*   **`ignored_scope`**: Excludes specific layers or operations from quantization to preserve accuracy.
    ```python
    # Exclude by layer name
    OpenVINOQuantizer(ignored_scope=nncf.IgnoredScope(names=['layer_1', 'layer_2']))
    # Exclude by layer type
    OpenVINOQuantizer(ignored_scope=nncf.IgnoredScope(types=['Linear']))
    # Exclude via regex pattern
    OpenVINOQuantizer(ignored_scope=nncf.IgnoredScope(patterns='.*attention.*'))
    ```

*   **`target_device`**: Optimizes quantization for a specific device (`CPU`, `GPU`, `NPU`, etc.).
    ```python
    quantizer = OpenVINOQuantizer(target_device=nncf.TargetDevice.CPU)
    ```

For a complete parameter list, refer to the [OpenVINOQuantizer documentation](https://openvinotoolkit.github.io/nncf/autoapi/nncf/experimental/torch/fx/index.html#nncf.experimental.torch.fx.OpenVINOQuantizer).

### 2.2 Prepare, Calibrate, and Convert the Model

Now, apply the standard PyTorch 2 Export quantization steps using your quantizer.

```python
# 1. Prepare the model: Inserts observers and folds BatchNorm layers.
prepared_model = prepare_pt2e(exported_model, quantizer)

# 2. Calibrate the model: Run inference to collect statistics for quantization.
# Use your calibration dataset here. We use the dummy input for demonstration.
prepared_model(*example_inputs)

# 3. Convert the model: Produces the final quantized model.
quantized_model = convert_pt2e(prepared_model, fold_quantize=False)
```

You now have a quantized model ready for the OpenVINO backend.

## Step 3: Lower into OpenVINO Representation

To execute the model using optimized OpenVINO kernels, you must compile it with the `"openvino"` backend via `torch.compile`. This step transforms the FX graph into a highly optimized OpenVINO model.

```python
with torch.no_grad(), nncf.torch.disable_patching():
    # Compile the quantized model for the OpenVINO backend
    optimized_model = torch.compile(quantized_model, backend="openvino")

    # Run inference (this triggers compilation and execution)
    result = optimized_model(*example_inputs)
```

The `optimized_model` now uses low-precision OpenVINO kernels, which should provide a significant inference speedup compared to the original eager model, especially on Intel CPUs.

## Step 4 (Optional): Improve Quantized Model Accuracy

NNCF provides advanced algorithms like **SmoothQuant** and **Bias Correction** to improve the accuracy of quantized models. You can access these via the `quantize_pt2e` convenience function, which bundles the prepare, calibrate, and convert steps.

```python
from nncf.experimental.torch.fx import quantize_pt2e
import nncf

# 1. Prepare your calibration dataset
# calibration_loader = torch.utils.data.DataLoader(...)
# def transform_fn(data_item):
#     images, _ = data_item
#     return images
# calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)

# 2. Apply quantization with advanced algorithms
# quantized_model = quantize_pt2e(
#     exported_model,
#     quantizer,
#     calibration_dataset,
#     smooth_quant=True,       # Enable SmoothQuant
#     fast_bias_correction=False
# )
```

For a complete, runnable example using these techniques, see the [NNCF ResNet18 quantization example](https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/torch_fx/resnet18/README.md).

## Conclusion

You have successfully quantized a PyTorch model using the PyTorch 2 Export flow with the OpenVINO-specific quantizer and compiled it for high-performance inference using the OpenVINO backend via `torch.compile`.

**Next Steps & Resources:**
*   **NNCF Quantization Guide**: Dive deeper into quantization options in the [NNCF documentation](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html).
*   **OpenVINO torch.compile**: Learn more about deploying models with `torch.compile` in the [OpenVINO documentation](https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html).