# Running Phi-3 with Apple's MLX Framework

## Overview

This guide demonstrates how to run the Phi-3-mini language model locally on Apple Silicon (M1/M2/M3) using Apple's MLX framework. MLX is an array framework designed for efficient machine learning research on Apple hardware, offering a user-friendly interface for model inference and quantization.

## Prerequisites

Before you begin, ensure you have:

*   **Python 3.11.x** installed.
*   An **Apple Silicon Mac** (M1, M2, or M3 series).

## Step 1: Install the MLX Library

First, install the `mlx-lm` package, which provides tools for running large language models with MLX.

```bash
pip install mlx-lm
```

## Step 2: Run Phi-3-mini from the Terminal

You can immediately start generating text with the pre-trained model using the command line. The `mlx_lm.generate` module handles downloading the model and running inference.

The following command asks the model to introduce itself. The `--max-token` flag limits the response length, and the `--prompt` uses the specific chat template required by the Phi-3 instruct model.

```bash
python -m mlx_lm.generate \
  --model microsoft/Phi-3-mini-4k-instruct \
  --max-tokens 2048 \
  --prompt "<|user|>\nCan you introduce yourself<|end|>\n<|assistant|>"
```

**Expected Output:**
You should see the model's response printed in your terminal, which will be a polite introduction from the AI assistant.

## Step 3: Quantize the Model for Faster Inference

To improve performance and reduce memory usage, you can quantize the model. Quantization converts the model's weights to a lower-precision format (like 4-bit integers). The `mlx_lm.convert` tool handles this process.

Run the following command to download and quantize the Phi-3-mini model to 4-bit (INT4) by default. The quantized model will be saved in a new `./mlx_model/` directory.

```bash
python -m mlx_lm.convert --hf-path microsoft/Phi-3-mini-4k-instruct
```

## Step 4: Run the Quantized Model

Once quantization is complete, you can run inference using the local, quantized model. Point the `--model` argument to the `./mlx_model/` directory.

```bash
python -m mlx_lm.generate \
  --model ./mlx_model/ \
  --max-tokens 2048 \
  --prompt "<|user|>\nCan you introduce yourself<|end|>\n<|assistant|>"
```

**Expected Output:**
You will receive a similar introduction from the model, but now it's running from your local, optimized `mlx_model` directory. The response should be faster and consume less memory.

## Next Steps & Resources

You can integrate MLX and the quantized model into your own Python scripts or Jupyter notebooks for more complex applications. The core steps remain the same: load the model from the `./mlx_model` path and use the appropriate chat template for prompts.

*   **Official MLX Documentation:** [https://ml-explore.github.io/mlx/build/html/index.html](https://ml-explore.github.io/mlx/build/html/index.html)
*   **MLX GitHub Repository:** [https://github.com/ml-explore](https://github.com/ml-explore)