# Quantizing the Phi Model Family with ONNX Runtime Generative AI Extensions

## Introduction

This guide walks you through quantizing Microsoft's Phi-3.5 models using the ONNX Runtime Generative AI extensions. Quantization reduces model size and improves inference speed, making deployment on resource-constrained devices feasible.

The Generative AI extensions provide a high-level API for running generative models with ONNX Runtime, handling the complete generation loop including inference, logits processing, search/sampling, and KV cache management.

## Prerequisites

Before starting, ensure you have the required packages installed:

```bash
pip install torch transformers onnx onnxruntime
pip install --pre onnxruntime-genai
```

## Understanding the Model Builder

The `onnxruntime-genai` package includes a Model Builder tool that simplifies creating optimized, quantized ONNX models compatible with the ONNX Runtime `generate()` API.

### Model Builder Command Structure

The basic command format is:

```bash
python3 -m onnxruntime_genai.models.builder -m model_name -o output_folder -p precision -e execution_provider -c cache_dir
```

**Key Parameters:**
- `-m model_name`: Hugging Face model identifier (e.g., `microsoft/Phi-3.5-mini-instruct`) or local model path
- `-o output_folder`: Directory to save the converted model
- `-p precision`: Quantization precision (`int4`, `int8`, `fp16`, `fp32`)
- `-e execution_provider`: Hardware acceleration backend (`cpu`, `cuda`, `DirectML`)
- `-c cache_dir`: Local directory to cache downloaded Hugging Face files

## Step 1: Quantizing Phi-3.5-Instruct Models

### CPU Acceleration with INT4 Quantization

To create an INT4-quantized model optimized for CPU execution:

```bash
python3 -m onnxruntime_genai.models.builder \
  -m microsoft/Phi-3.5-mini-instruct \
  -o ./onnx-cpu \
  -p int4 \
  -e cpu \
  -c ./Phi-3.5-mini-instruct
```

### CUDA Acceleration with INT4 Quantization

For GPU-accelerated inference with NVIDIA CUDA:

```bash
python3 -m onnxruntime_genai.models.builder \
  -m microsoft/Phi-3.5-mini-instruct \
  -o ./onnx-cuda \
  -p int4 \
  -e cuda \
  -c ./Phi-3.5-mini-instruct
```

## Step 2: Quantizing Phi-3.5-Vision Models

The Phi-3.5-Vision model requires additional preparation steps due to its multimodal architecture.

### Preparation Steps

1. **Create a working directory:**
   ```bash
   mkdir models
   cd models
   ```

2. **Download the base model** from Hugging Face: [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)

3. **Download required support files** to your Phi-3.5-vision-instruct folder:
   - [config.json](https://huggingface.co/lokinfey/Phi-3.5-vision-instruct-onnx-cpu/resolve/main/onnx/config.json)
   - [image_embedding_phi3_v_for_onnx.py](https://huggingface.co/lokinfey/Phi-3.5-vision-instruct-onnx-cpu/blob/main/onnx/image_embedding_phi3_v_for_onnx.py)
   - [modeling_phi3_v.py](https://huggingface.co/lokinfey/Phi-3.5-vision-instruct-onnx-cpu/blob/main/onnx/modeling_phi3_v.py)

4. **Download the build script** to your models folder:
   - [build.py](https://huggingface.co/lokinfey/Phi-3.5-vision-instruct-onnx-cpu/blob/main/onnx/build.py)

### CPU Conversion with FP32 Precision

Run the build script to create an FP32 model for CPU execution:

```bash
python build.py \
  -i ./Your-Phi-3.5-vision-instruct-Path/ \
  -o ./vision-cpu-fp32 \
  -p f32 \
  -e cpu
```

Replace `./Your-Phi-3.5-vision-instruct-Path/` with the actual path to your downloaded model.

## Important Considerations

1. **Model Support**: Model Builder currently supports Phi-3.5-Instruct and Phi-3.5-Vision models, but not Phi-3.5-MoE.

2. **Using Quantized Models**: After quantization, use the Generative AI extensions SDK to run inference on the converted ONNX models.

3. **Responsible AI Testing**: Always conduct thorough testing after quantization to ensure model performance and accuracy meet your requirements.

4. **Edge Deployment**: INT4 quantization is particularly valuable for edge device deployment where memory and compute resources are limited.

## Next Steps

Once you have quantized models, you can:

1. Integrate them into your applications using the ONNX Runtime Generative AI SDK
2. Deploy to edge devices for local inference
3. Benchmark performance against the original models

## Resources

- [ONNX Runtime Generative AI Documentation](https://onnxruntime.ai/docs/genai/)
- [ONNX Runtime Generative AI GitHub Repository](https://github.com/microsoft/onnxruntime-genai)
- [Microsoft Olive](https://github.com/microsoft/Olive) - Alternative tool that incorporates Generative AI extensions functionality

---

*Note: The Generative AI extensions for ONNX Runtime are currently in preview but provide stable functionality for model quantization and deployment.*