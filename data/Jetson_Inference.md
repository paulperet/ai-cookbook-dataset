# Running Phi-3 on NVIDIA Jetson: A Step-by-Step Guide

This guide walks you through deploying the efficient Phi-3 Mini model on an NVIDIA Jetson device for edge AI applications. We'll use LlamaEdge, a WebAssembly runtime, to run a quantized GGUF model, enabling cross-platform deployment from the cloud to the edge.

## Prerequisites

Ensure your development environment meets the following requirements:

*   **Hardware:** Jetson Orin NX or Jetson NX.
*   **Software:**
    *   JetPack 5.1.2 or later.
    *   CUDA 11.8.
    *   Python 3.8 or later.

## Step 1: Install WasmEdge with GGML Plugin

First, you need to install the WasmEdge runtime, which includes the necessary plugin for running GGML-based models (like GGUF).

Run the following command in your terminal:

```bash
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | bash -s -- --plugin wasi_nn-ggml
```

This script downloads and installs WasmEdge along with the `wasi_nn-ggml` plugin required for AI inference.

## Step 2: Download the LlamaEdge Server and UI

Next, download the components that will serve the model and provide a web interface.

1.  Download the LlamaEdge API server (a WebAssembly module):
    ```bash
    curl -LO https://github.com/LlamaEdge/LlamaEdge/releases/latest/download/llama-api-server.wasm
    ```

2.  Download and extract the chatbot user interface:
    ```bash
    curl -LO https://github.com/LlamaEdge/chatbot-ui/releases/latest/download/chatbot-ui.tar.gz
    tar xzf chatbot-ui.tar.gz
    ```

**Important:** After extraction, ensure the `llama-api-server.wasm` file and the `chatbot-ui` directory are located in the same folder.

## Step 3: Prepare Your Phi-3 Model

You need a quantized Phi-3 model in the GGUF format. You can obtain one from community sources like Hugging Face.

1.  Download a Phi-3 GGUF model (e.g., `Phi-3-mini-4k-instruct-q4.gguf`) and place it in your working directory.
2.  Take note of the full path to this `.gguf` file. You will need it for the next step.

## Step 4: Start the Inference Server

Now, you will start the WebAssembly server, preloading your Phi-3 model for inference.

In your terminal, navigate to the directory containing the downloaded files and run the following command. Replace `{Your gguf path}` with the actual path to your `.gguf` model file.

```bash
wasmedge --dir .:. --nn-preload default:GGML:AUTO:{Your gguf path} llama-api-server.wasm -p phi-3-chat
```

**Command Breakdown:**
*   `--dir .:.`: Grants the server permission to access the current directory.
*   `--nn-preload default:GGML:AUTO:...`: Preloads the specified GGUF model into the neural network interface.
*   `-p phi-3-chat`: Specifies the prompt template to use for the Phi-3 chat model.

The server will start and indicate it is running, typically on a local port (e.g., `http://0.0.0.0:8080`).

## Step 5: Interact with the Model

With the server running, you can now interact with Phi-3.

1.  Open a web browser on your Jetson device.
2.  Navigate to `http://localhost:8080`. This will load the Chatbot UI you extracted earlier.
3.  Use the web interface to send prompts and receive responses from the Phi-3 model running locally on your Jetson.

## Summary and Further Resources

You have successfully deployed a quantized Phi-3 model on an NVIDIA Jetson using LlamaEdge and WasmEdge. This approach leverages WebAssembly for a portable, secure, and efficient runtime ideal for edge devices.

*   **Explore a Sample Notebook:** For a different workflow, you can review the [Phi-3 mini WASM Notebook Sample](https://github.com/Azure-Samples/Phi-3MiniSamples/tree/main/wasm).
*   **Consider TensorRT-LLM:** For maximum performance on NVIDIA hardware, explore deploying Phi-3 using [NVIDIA's TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), which applies advanced optimizations like FP8 quantization and inflight batching.

Phi-3 Mini, combined with edge-optimized tooling, provides a powerful tool for building responsive and intelligent applications directly on embedded devices and robots.