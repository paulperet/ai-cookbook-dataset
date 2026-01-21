# Quantifying the Phi Family: A Guide to Model Quantization

## Introduction

Model quantization is the process of mapping neural network parameters—such as weights and activation values—from a large, continuous value range to a smaller, finite set. This technique reduces model size and computational complexity, enabling efficient operation in resource-constrained environments like mobile devices, AI PCs, and IoT systems. While quantization introduces some precision loss, it provides a crucial balance between model size, computational efficiency, and accuracy.

This guide explains common quantization precisions and demonstrates practical approaches for quantizing Phi-3.x models using various hardware-optimized frameworks.

## Understanding Quantization Precisions

Different quantization levels offer trade-offs between model size, speed, and accuracy.

### **INT4**
- **Description:** Aggressive quantization using 4-bit integers.
- **Advantages:** Maximum reduction in storage and computational requirements.
- **Considerations:** Significant precision loss; not all hardware supports INT4 operations.
- **Use Case:** Mobile and IoT devices where extreme efficiency is critical.

### **INT8**
- **Description:** Standard quantization using 8-bit integers.
- **Advantages:** Substantial memory and computation savings with acceptable accuracy for most applications.
- **Considerations:** Requires quantization/dequantization steps during inference.
- **Use Case:** Balanced performance for edge deployment and AI PCs.

### **FP16**
- **Description:** 16-bit floating-point representation.
- **Advantages:** Half the memory footprint of FP32 with good hardware acceleration support.
- **Considerations:** Potential numerical instability in precision-sensitive operations.
- **Use Case:** Large model deployment where memory constraints exist.

### **FP32**
- **Description:** Full-precision 32-bit floating-point.
- **Advantages:** Highest accuracy and numerical stability.
- **Considerations:** Maximum memory usage and computational cost.
- **Use Case:** Training and scenarios requiring maximum precision.

## Hardware and Framework Ecosystem

Different hardware manufacturers provide optimized frameworks for generative model deployment:

| Framework | Manufacturer | Key Features |
|-----------|--------------|--------------|
| OpenVINO | Intel | Cross-platform optimization for Intel hardware |
| QNN | Qualcomm | Optimized for Snapdragon platforms |
| MLX | Apple | Native Apple Silicon acceleration |
| CUDA | NVIDIA | GPU acceleration for NVIDIA hardware |

## Quantization Formats

After quantization, models can be exported in various formats:

- **PyTorch/TensorFlow Formats:** Native framework formats
- **GGUF:** llama.cpp format optimized for CPU inference
- **ONNX:** Cross-platform format with broad hardware support

For this guide, we focus on **ONNX** due to its excellent framework compatibility and hardware support across multiple platforms.

## Practical Quantization Tutorials

Follow these step-by-step guides to quantize Phi-3.5/4 models using different frameworks:

### 1. [Quantizing Phi-3.5/4 using llama.cpp](./UsingLlamacppQuantifyingPhi.md)
Learn how to use the popular llama.cpp library to quantize Phi models for CPU-optimized inference.

### 2. [Quantizing Phi-3.5/4 using Generative AI extensions for ONNX Runtime](./UsingORTGenAIQuantifyingPhi.md)
Use Microsoft's ONNX Runtime with Generative AI extensions for optimized quantization and inference.

### 3. [Quantizing Phi-3.5/4 using Intel OpenVINO](./UsingIntelOpenVINOQuantifyingPhi.md)
Optimize Phi models for Intel hardware using the OpenVINO toolkit.

### 4. [Quantizing Phi-3.5/4 using Apple MLX Framework](./UsingAppleMLXQuantifyingPhi.md)
Leverage Apple's MLX framework for native quantization and inference on Apple Silicon.

## Getting Started

Before beginning any quantization tutorial, ensure you have:

1. **Python 3.8+** installed
2. **Sufficient disk space** for model downloads and conversions
3. **Hardware requirements** specific to each framework
4. **Basic understanding** of command-line operations

## Next Steps

Choose the tutorial that matches your target deployment platform. Each guide provides complete, step-by-step instructions with code examples and best practices for successful quantization.

**Have a better approach?** We welcome community contributions! Submit a PR with your quantization method to help improve this guide.