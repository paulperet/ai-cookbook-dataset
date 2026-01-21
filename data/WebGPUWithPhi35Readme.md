# Building a Phi-3.5-Instruct RAG Chatbot with WebGPU

## Overview

This guide demonstrates how to build a Retrieval-Augmented Generation (RAG) chatbot using the Phi-3.5-Instruct model optimized with ONNX Runtime and accelerated via WebGPU. This pattern combines efficient model deployment with modern browser-based GPU acceleration for high-performance AI applications.

## Prerequisites

### Browser Requirements

WebGPU is supported in the following browsers:

- **Google Chrome 113+**
- **Microsoft Edge 113+**
- **Safari 18** (macOS 15)
- **Firefox Nightly**

### Enable WebGPU

#### For Chrome/Microsoft Edge

1. **Open your browser** and navigate to `chrome://flags`
2. **Search for** `enable-unsafe-webgpu` in the search box
3. **Enable the flag** by selecting "Enabled" from the dropdown
4. **Restart your browser** using the "Relaunch" button

#### For Linux Users
Launch Chrome/Edge with the flag: `--enable-features=Vulkan`

#### For Safari 18 (macOS 15)
WebGPU is enabled by default.

#### For Firefox Nightly
1. Navigate to `about:config`
2. Set `dom.webgpu.enabled` to `true`

### Optimize GPU Settings for Microsoft Edge (Windows)

To ensure Edge uses your high-performance GPU:

1. **Open Windows Settings** → **System** → **Display**
2. **Click** "Graphics settings"
3. **Under** "Choose an app to set preference," select "Desktop app"
4. **Click** "Browse" and navigate to: `C:\Program Files (x86)\Microsoft\Edge\Application`
5. **Select** `msedge.exe`
6. **Click** "Options," choose "High performance," then "Save"
7. **Restart your machine** for changes to take effect

## Understanding the Components

### What is WebGPU?

WebGPU is a modern web graphics API that provides efficient, low-level access to your device's GPU directly from web browsers. It's designed as the successor to WebGL with significant improvements:

- **Modern GPU Compatibility**: Works seamlessly with Vulkan, Metal, and Direct3D 12 APIs
- **Enhanced Performance**: Supports general-purpose GPU computations for both graphics and machine learning
- **Advanced Features**: Enables complex computational workloads with reduced JavaScript overhead
- **Cross-Platform**: Currently supported in major browsers with expanding platform support

### The RAG Pattern

Retrieval-Augmented Generation combines:
1. **Retrieval**: Fetching relevant information from a knowledge base
2. **Generation**: Using a language model to synthesize responses based on retrieved content

This approach is particularly effective for domain-specific applications where you need accurate, up-to-date information combined with the language understanding capabilities of models like Phi-3.5.

## Next Steps

Now that you've configured your environment for WebGPU acceleration, you're ready to implement the RAG chatbot. The implementation will involve:

1. Setting up the ONNX Runtime with WebGPU backend
2. Loading and optimizing the Phi-3.5-Instruct model
3. Implementing the retrieval component for your knowledge base
4. Creating the generation pipeline that combines retrieved context with user queries
5. Building the chat interface that leverages GPU acceleration

For complete code samples and implementation details, refer to the [GitHub repository](https://github.com/microsoft/aitour-exploring-cutting-edge-models/tree/main/src/02.ONNXRuntime/01.WebGPUChatRAG).

---

*Note: This tutorial focuses on environment setup and conceptual understanding. The actual implementation code will be covered in subsequent sections where you'll build the complete RAG chatbot pipeline.*