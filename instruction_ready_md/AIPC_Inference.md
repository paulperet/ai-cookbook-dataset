# Running Phi-3 on an Intel AI PC: A Practical Guide

This guide walks you through three methods for running the Phi-3-mini model on an Intel AI PC equipped with a Core™ Ultra processor. You'll learn how to leverage the dedicated Neural Processing Unit (NPU), DirectML with ONNX Runtime, and OpenVINO for hardware-accelerated inference.

## Prerequisites

Ensure you have a compatible Intel AI PC (with a Core™ Ultra processor) and Python installed. You will install specific libraries for each method in the sections below.

## Method 1: Using the Intel NPU Acceleration Library

The Intel NPU Acceleration Library is designed to offload AI computations to the efficient Neural Processing Unit.

### Step 1: Install the Library
Install the required package using pip.
```bash
pip install intel-npu-acceleration-library
```

### Step 2: Load and Quantize the Model
The library allows you to load the Phi-3 model and quantize it (e.g., to INT4) for efficient execution on the NPU.

```python
from transformers import AutoTokenizer, pipeline, TextStreamer
from intel_npu_acceleration_library import NPUModelForCausalLM, int4
from intel_npu_acceleration_library.compiler import CompilerConfig
import warnings

# Define the model ID
model_id = "microsoft/Phi-3-mini-4k-instruct"

# Configure the compiler for INT4 quantization
compiler_conf = CompilerConfig(dtype=int4)

# Load and quantize the model for NPU execution
model = NPUModelForCausalLM.from_pretrained(
    model_id,
    use_cache=True,
    config=compiler_conf,
    attn_implementation="sdpa"
).eval()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize a text streamer for real-time output
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
```

### Step 3: Configure and Run the Generation Pipeline
Set up the text-generation pipeline with your desired parameters and execute the model.

```python
# Define generation parameters
generation_args = {
    "max_new_tokens": 1024,
    "return_full_text": False,
    "temperature": 0.3,
    "do_sample": False,
    "streamer": text_streamer,
}

# Create the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define your prompt
query = "<|system|>You are a helpful AI assistant.<|end|><|user|>Can you introduce yourself?<|end|><|assistant|>"

# Run inference (suppressing warnings for cleaner output)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pipe(query, **generation_args)
```

When this code runs, you can monitor the NPU utilization in your system's Task Manager to confirm the hardware is being used.

## Method 2: Using DirectML with ONNX Runtime

This method uses Microsoft's DirectML API via ONNX Runtime for broad GPU hardware support, including NPUs.

### Step 1: Build the ONNX Runtime Generative AI Library
This process compiles the necessary libraries with DirectML support.

```bash
# Install CMake
winget install --id=Kitware.CMake -e

# Clone and build ONNX Runtime
git clone https://github.com/microsoft/onnxruntime.git
cd .\onnxruntime\
./build.bat --build_shared_lib --skip_tests --parallel --use_dml --config Release
cd ../

# Clone the Generative AI extension
git clone https://github.com/microsoft/onnxruntime-genai.git
cd .\onnxruntime-genai\

# Prepare the directory structure and copy necessary files
mkdir ort
cd ort
mkdir include
mkdir lib
copy ..\onnxruntime\include\onnxruntime\core\providers\dml\dml_provider_factory.h ort\include
copy ..\onnxruntime\include\onnxruntime\core\session\onnxruntime_c_api.h ort\include
copy ..\onnxruntime\build\Windows\Release\Release\*.dll ort\lib
copy ..\onnxruntime\build\Windows\Release\Release\onnxruntime.lib ort\lib

# Build the Python package with DirectML support
python build.py --use_dml
```

### Step 2: Install the Generated Wheel File
Install the custom-built Python package.

```bash
pip install .\onnxruntime_genai_directml-0.3.0.dev0-cp310-cp310-win_amd64.whl
```

You can now use the `onnxruntime-genai` Python API to load and run quantized ONNX models. Refer to the [sample repository](https://github.com/Azure-Samples/Phi-3MiniSamples/tree/main/onnx) for complete usage examples.

## Method 3: Using Intel OpenVINO

OpenVINO is a toolkit for optimizing deep learning models across Intel hardware (CPU, integrated GPU). *Note: NPU support is not yet available.*

### Step 1: Install OpenVINO and Required Libraries
Install the necessary packages, including the Optimum-Intel integration.

```bash
pip install git+https://github.com/huggingface/optimum-intel.git
pip install git+https://github.com/openvinotoolkit/nncf.git
pip install openvino-nightly
```

### Step 2: Quantize and Export the Phi-3 Model
Use the `optimum-cli` tool to convert the Hugging Face model to an optimized OpenVINO format. Choose your preferred quantization.

**For INT4 quantization:**
```bash
optimum-cli export openvino --model "microsoft/Phi-3-mini-4k-instruct" --task text-generation-with-past --weight-format int4 --group-size 128 --ratio 0.6 --sym --trust-remote-code ./openvinomodel/phi3/int4
```

**For FP16 quantization:**
```bash
optimum-cli export openvino --model "microsoft/Phi-3-mini-4k-instruct" --task text-generation-with-past --weight-format fp16 --trust-remote-code ./openvinomodel/phi3/fp16
```

This creates a directory (e.g., `./openvinomodel/phi3/int4/`) containing the optimized model.

### Step 3: Load and Run the Model with OpenVINO
Load the exported model, targeting the integrated GPU for acceleration.

```python
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoConfig

# Path to your exported model
model_dir = "./openvinomodel/phi3/int4"

# Configuration for low-latency inference
ov_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}

# Load the model onto the GPU
ov_model = OVModelForCausalLM.from_pretrained(
    model_dir,
    device='GPU.0',  # Target the integrated GPU
    ov_config=ov_config,
    config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
    trust_remote_code=True,
)
```

Once loaded, you can use `ov_model` with a Hugging Face `pipeline` for text generation, similar to the NPU example. Monitor GPU utilization in Task Manager to confirm hardware acceleration.

## Summary and Recommendation

You have explored three pathways to run Phi-3 on an Intel AI PC:
1.  **Intel NPU Acceleration Library:** Directly targets the NPU for power-efficient AI inference.
2.  **DirectML with ONNX Runtime:** Offers broad hardware compatibility across GPUs and NPUs using a standardized API.
3.  **Intel OpenVINO:** Provides optimized execution on Intel CPUs and integrated GPUs.

**For AI PC inference, using the NPU via the Intel NPU Acceleration Library is recommended.** The NPU is purpose-built for AI tasks, offering an optimal balance of performance and power efficiency, freeing up the CPU and GPU for other workloads.