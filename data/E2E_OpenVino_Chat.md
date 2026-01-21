# OpenVINO Chat with Phi-3-mini-4k-instruct: A Step-by-Step Guide

This guide walks you through exporting, loading, and running inference with Microsoft's Phi-3-mini-4k-instruct model using OpenVINO for optimized performance. You'll learn how to quantize the model to INT4 precision and generate conversational responses.

## Prerequisites

Before starting, ensure you have the necessary Python packages installed:

```bash
pip install optimum[openvino] transformers
```

## Step 1: Export the Model to OpenVINO Format

First, export the Phi-3-mini-4k-instruct model to OpenVINO's Intermediate Representation (IR) format with INT4 quantization for efficient inference:

```bash
optimum-cli export openvino \
  --model "microsoft/Phi-3-mini-4k-instruct" \
  --task text-generation-with-past \
  --weight-format int4 \
  --group-size 128 \
  --ratio 0.6 \
  --sym \
  --trust-remote-code \
  ./model/phi3-instruct/int4
```

**What this command does:**
- `--model`: Specifies the Hugging Face model identifier
- `--task text-generation-with-past`: Configures the model for text generation with past key/values caching
- `--weight-format int4`: Quantizes weights to 4-bit integers for reduced memory usage
- `--group-size 128`: Sets group size for quantization (affects accuracy/speed trade-off)
- `--ratio 0.6`: Compression ratio for mixed-precision quantization
- `--sym`: Uses symmetric quantization (balanced positive/negative ranges)
- `--trust-remote-code`: Allows execution of custom model code
- The final argument specifies the output directory

## Step 2: Import Required Libraries

Now, import the necessary Python modules for loading and running the model:

```python
from transformers import AutoConfig, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
```

## Step 3: Configure Model Settings

Define the model directory and OpenVINO configuration parameters:

```python
# Path to the exported model
model_dir = './model/phi3-instruct/int4'

# OpenVINO runtime configuration
ov_config = {
    "PERFORMANCE_HINT": "LATENCY",  # Optimize for low latency
    "NUM_STREAMS": "1",             # Use single inference stream
    "CACHE_DIR": ""                 # Disable model caching
}
```

The `LATENCY` performance hint tells OpenVINO to prioritize response speed over throughput, ideal for interactive applications.

## Step 4: Load the Quantized Model

Load the INT4-quantized model with the specified configuration:

```python
ov_model = OVModelForCausalLM.from_pretrained(
    model_dir,
    device='GPU.0',  # Use GPU for inference (change to 'CPU' if needed)
    ov_config=ov_config,
    config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
    trust_remote_code=True,
)
```

**Note:** The `trust_remote_code=True` parameter is required for models with custom architectures like Phi-3.

## Step 5: Initialize the Tokenizer

Load the tokenizer that matches the model:

```python
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
```

Configure tokenizer settings to match the model's expected format:

```python
tokenizer_kwargs = {
    "add_special_tokens": False  # We'll handle special tokens manually
}
```

## Step 6: Prepare the Conversation Prompt

Phi-3 uses a specific chat template format. Construct a prompt with system, user, and assistant roles:

```python
prompt = "<|system|>You are a helpful AI assistant.<|end|><|user|>can you introduce yourself?<|end|><|assistant|>"
```

This prompt structure includes:
- System message defining the assistant's role
- User query asking for an introduction
- Assistant tag indicating where the model should start generating

## Step 7: Tokenize the Input

Convert the text prompt into token IDs that the model can process:

```python
input_tokens = tok(prompt, return_tensors="pt", **tokenizer_kwargs)
```

The `return_tensors="pt"` parameter ensures the output is in PyTorch tensor format, which OpenVINO expects.

## Step 8: Generate the Response

Run inference to generate the assistant's response:

```python
answer = ov_model.generate(**input_tokens, max_new_tokens=1024)
```

The `max_new_tokens=1024` parameter limits the response length to prevent excessively long outputs.

## Step 9: Decode and Display the Result

Convert the generated token IDs back to human-readable text:

```python
decoded_answer = tok.batch_decode(answer, skip_special_tokens=True)[0]
print(decoded_answer)
```

The `skip_special_tokens=True` parameter removes special formatting tokens, leaving only the natural language response.

## Complete Example Script

Here's the complete code in a single script:

```python
from transformers import AutoConfig, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

# Configuration
model_dir = './model/phi3-instruct/int4'
ov_config = {
    "PERFORMANCE_HINT": "LATENCY",
    "NUM_STREAMS": "1",
    "CACHE_DIR": ""
}

# Load model and tokenizer
ov_model = OVModelForCausalLM.from_pretrained(
    model_dir,
    device='GPU.0',
    ov_config=ov_config,
    config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
    trust_remote_code=True,
)

tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
tokenizer_kwargs = {"add_special_tokens": False}

# Prepare and tokenize prompt
prompt = "<|system|>You are a helpful AI assistant.<|end|><|user|>can you introduce yourself?<|end|><|assistant|>"
input_tokens = tok(prompt, return_tensors="pt", **tokenizer_kwargs)

# Generate response
answer = ov_model.generate(**input_tokens, max_new_tokens=1024)

# Decode and display
decoded_answer = tok.batch_decode(answer, skip_special_tokens=True)[0]
print("Assistant:", decoded_answer)
```

## Expected Output

When you run this script, you should see a response similar to:

```
Assistant: Hello! I'm an AI assistant based on Microsoft's Phi-3-mini model. I'm here to help answer questions, provide information, and assist with various tasks. I'm designed to be helpful, harmless, and honest in my responses. How can I assist you today?
```

## Next Steps

- Experiment with different `ov_config` settings for throughput-optimized scenarios
- Try varying the `max_new_tokens` parameter for longer or shorter responses
- Test the model with different conversation prompts and system instructions
- Compare performance between CPU and GPU execution by changing the `device` parameter

This implementation demonstrates how to leverage OpenVINO's optimizations to run large language models efficiently while maintaining conversational quality.