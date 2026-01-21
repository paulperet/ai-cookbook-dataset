# Running Phi-3-mini on Android: A Practical Guide

This guide walks you through the practical steps to run the Phi-3-mini large language model (LLM) on Android devices. Phi-3-mini is a lightweight, powerful model from Microsoft designed for deployment on edge and IoT devices.

## Prerequisites

Before you begin, ensure you have:
*   An Android device.
*   Basic familiarity with running applications from APK files (you may need to enable "Install from Unknown Sources" in your device settings).

## Method 1: Using the MLC Chat App (Simplest Method)

The most straightforward way to run Phi-3-mini on your Android phone is via the pre-packaged **MLC Chat** application.

1.  **Download the App:** Obtain the MLC Chat APK file (approximately 148 MB) from a trusted source.
2.  **Install the APK:** On your Android device, locate the downloaded `.apk` file and tap to install it. You may need to grant permission to install applications from unknown sources.
3.  **Launch and Select Model:** Open the MLC Chat app. You will be presented with a list of available AI models. Select **Phi-3-mini** from this list to begin using it.

This method provides a quick, user-friendly interface for inference without any additional setup.

## Method 2: Advanced Deployment with Semantic Kernel

For developers looking to integrate Phi-3-mini into custom applications, [Semantic Kernel](https://github.com/microsoft/semantic-kernel) is a powerful application framework. It allows you to build AI apps that can use models from Azure OpenAI, OpenAI, or local models like Phi-3-mini.

### Connecting to Phi-3-mini

You can connect Semantic Kernel to Phi-3-mini using the Hugging Face connector. This typically involves pointing to the model's ID on the Hugging Face Hub. You can also configure it to connect to a local model server if you have one running.

**Sample Code:** Refer to the [Semantic Kernel sample repository](https://github.com/Azure-Samples/Phi-3MiniSamples/tree/main/semantickernel) for implementation details.

## Method 3: Using Quantized Models with Ollama or LlamaEdge

To improve performance and reduce resource usage on devices, quantized model formats (like `.gguf`) are often used. Both Ollama and LlamaEdge provide ways to run these optimized models.

### Option A: Using Ollama

[Ollama](https://ollama.com/) is a popular tool for running LLMs locally. To use a quantized Phi-3 `.gguf` file with Ollama, you need to create a `Modelfile`.

1.  **Create a Modelfile:** Create a text file named `Modelfile` with the following content, replacing `{Add your gguf file path}` with the actual path to your `.gguf` file.

    ```dockerfile
    FROM {Add your gguf file path}
    TEMPLATE """<|user|> .Prompt<|end|> <|assistant|>"""
    PARAMETER stop <|end|>
    PARAMETER num_ctx 4096
    ```

2.  **Run the Model:** Use Ollama commands to create and run the model defined in your `Modelfile`.

**Sample Code:** Explore the [Ollama sample project](https://github.com/Azure-Samples/Phi-3MiniSamples/tree/main/ollama) for a complete example.

### Option B: Using LlamaEdge

[LlamaEdge](https://llamaedge.com) is an excellent choice if you need to run `.gguf` models consistently across cloud and edge environments, including via WebAssembly (Wasm).

**Getting Started:** The [LlamaEdge (Wasm) sample code](https://github.com/Azure-Samples/Phi-3MiniSamples/tree/main/wasm) provides a foundation for deploying Phi-3-mini in this versatile runtime.

## Summary

Phi-3-mini makes advanced generative AI accessible on Android devices. You can start experimenting immediately with the **MLC Chat app**, or choose a more developer-centric path using **Semantic Kernel**, **Ollama**, or **LlamaEdge** for deeper integration and optimized performance.