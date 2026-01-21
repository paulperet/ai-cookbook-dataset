# Optimizing OpenAI GPT-OSS Models with NVIDIA TensorRT-LLM

This guide provides a step-by-step tutorial on optimizing OpenAI's `gpt-oss` models using NVIDIA's TensorRT-LLM for high-performance inference. TensorRT-LLM offers a Python API to define Large Language Models (LLMs) and includes state-of-the-art optimizations for efficient execution on NVIDIA GPUs.

TensorRT-LLM supports the following models:
*   `gpt-oss-20b`
*   `gpt-oss-120b`

This tutorial will use the `gpt-oss-20b` model. For instructions on the larger model or more advanced customization, refer to the [official deployment guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog9_Deploying_GPT_OSS_on_TRTLLM.md).

> **Note on Prompt Format:** For the model to work correctly, your input prompts should adhere to the [OpenAI Harmony response format](http://cookbook.openai.com/articles/openai-harmony), though this tutorial does not enforce it.

## Prerequisites

### Hardware
To run the `gpt-oss-20b` model, you need an NVIDIA GPU with at least **20 GB of VRAM**.

**Recommended GPUs:** NVIDIA Hopper (e.g., H100, H200), NVIDIA Blackwell (e.g., B100, B200), NVIDIA RTX PRO, or NVIDIA RTX 50 Series (e.g., RTX 5090).

### Software
*   **CUDA Toolkit:** Version 12.8 or later.
*   **Python:** Version 3.12 or later.

## Step 1: Install TensorRT-LLM

There are multiple ways to install TensorRT-LLM. This guide covers using a pre-built Docker container and building from source. If you are using [NVIDIA Brev](https://developer.nvidia.com/brev), you can skip this step.

### Option A: Using the Pre-built NVIDIA NGC Container
This is the easiest method, as it includes all dependencies.

1.  Pull the pre-built TensorRT-LLM container for GPT-OSS from NVIDIA NGC:
    ```bash
    docker pull nvcr.io/nvidia/tensorrt-llm/release:gpt-oss-dev
    ```
2.  Run the container, mounting your current directory to `/workspace`:
    ```bash
    docker run --gpus all -it --rm -v $(pwd):/workspace nvcr.io/nvidia/tensorrt-llm/release:gpt-oss-dev
    ```

### Option B: Building from Source (Docker)
This approach is useful if you need to modify the source code or use a custom branch. For detailed instructions, refer to the [official TensorRT-LLM Docker documentation](https://github.com/NVIDIA/TensorRT-LLM/tree/feat/gpt-oss/docker).

> **Note on GPU Architecture:** The first time you run a model, TensorRT-LLM will build an optimized engine for your specific GPU (e.g., Hopper, Ada, Blackwell). If you encounter warnings about your GPU's CUDA capability (e.g., `sm_90`, `sm_120`), ensure you have the latest NVIDIA drivers and a CUDA Toolkit version that matches your PyTorch installation.

## Step 2: Verify the Installation

First, verify that TensorRT-LLM is installed correctly by importing the necessary modules.

```python
from tensorrt_llm import LLM, SamplingParams
```

If no error occurs, your installation is successful.

## Step 3: Download, Build, and Run the Model

Now, you will use the TensorRT-LLM Python API to:
1.  Download the `gpt-oss-20b` model weights.
2.  Automatically build a TensorRT engine optimized for your GPU.
3.  Load the model and run a text generation example.

> **Important:** The first execution may take several minutes as it downloads the model and builds the engine. Subsequent runs will be significantly faster because the engine is cached.

### 3.1 Initialize the Model
Create an `LLM` object, specifying the model identifier from Hugging Face.

```python
llm = LLM(model="openai/gpt-oss-20b")
```

### 3.2 Define Prompts and Sampling Parameters
Prepare your input prompts and set the parameters that control the randomness and creativity of the model's output.

```python
prompts = ["Hello, my name is", "The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```

### 3.3 Generate Text
Pass the prompts and sampling parameters to the model's `generate` method and print the results.

```python
for output in llm.generate(prompts, sampling_params):
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
```

You should see output similar to the following:
```
Prompt: 'Hello, my name is', Generated text: ' John and I am a software engineer.'
Prompt: 'The capital of France is', Generated text: ' Paris, a city known for its art, fashion, and culture.'
```

## Conclusion and Next Steps

Congratulations! You have successfully optimized and run a large language model using NVIDIA TensorRT-LLM.

In this tutorial, you learned how to:
*   Set up your environment with the necessary dependencies.
*   Use the `tensorrt_llm.LLM` API to download a model and build a high-performance TensorRT engine.
*   Run inference with the optimized model.

To further improve performance and efficiency, consider exploring these advanced features:

*   **Benchmarking:** Run a [benchmark](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/benchmarking-default-performance.html#benchmarking-with-trtllm-bench) to compare the latency and throughput of your TensorRT-LLM engine against the original model. You can do this by iterating over a large set of prompts and measuring execution time.
*   **Quantization:** TensorRT-LLM [supports](https://github.com/NVIDIA/TensorRT-Model-Optimizer) quantization techniques (like INT8 or FP8) to reduce model size and accelerate inference with minimal accuracy loss. This is especially useful for deployment on resource-constrained hardware.
*   **Production Deployment:** For scalable, multi-model serving in production, you can deploy your TensorRT-LLM engine using [NVIDIA Dynamo](https://docs.nvidia.com/dynamo/latest/).