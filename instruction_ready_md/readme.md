# Chat with Phi-4-mini ONNX: A Step-by-Step Guide

This guide walks you through converting the Phi-4-mini model to a quantized ONNX format and running inference using ONNX Runtime GenAI. ONNX (Open Neural Network Exchange) is an open format that enables you to run machine learning models across various frameworks and hardware, making it ideal for deploying models on edge devices or in resource-constrained environments.

## Prerequisites

Ensure you have Python installed. We'll install the necessary packages in the following steps.

## Step 1: Install Required Tools

First, install the Microsoft Olive SDK for model conversion and the Hugging Face Transformers library.

```bash
pip install olive-ai
pip install transformers
```

## Step 2: Convert Phi-4-mini to Quantized ONNX

Use Microsoft Olive to convert your Phi-4-mini model to a quantized ONNX format optimized for CPU execution. This command performs automatic optimization (`auto-opt`).

```bash
olive auto-opt \
  --model_name_or_path "Your Phi-4-mini location" \
  --output_path "Your onnx output location" \
  --device cpu \
  --provider CPUExecutionProvider \
  --precision int4 \
  --use_model_builder \
  --log_level 1
```

**Note:** Replace the placeholder paths with your actual model directory and desired output directory. This example uses CPU execution and 4-bit integer quantization (`int4`) to reduce model size and improve inference speed on limited hardware.

## Step 3: Install ONNX Runtime GenAI

ONNX Runtime GenAI is a library for running generative AI models in the ONNX format. Install the latest pre-release version.

```bash
pip install --pre onnxruntime-genai
```

## Step 4: Run Inference with ONNX Runtime GenAI

Now, you can load the converted model and run text generation. Below are two examples for different library versions.

### Option A: Using ONNX Runtime GenAI 0.5.2

This version uses a simple loop to generate tokens and decode them with a streaming tokenizer.

```python
import onnxruntime_genai as og

# 1. Load the converted model
model_folder = "Your Phi-4-mini-onnx-cpu-int4 location"
model = og.Model(model_folder)

# 2. Prepare the tokenizer
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# 3. Configure generation parameters
search_options = {
    'max_length': 2048,
    'past_present_share_buffer': False
}

# 4. Define the chat template and user input
chat_template = "<|user|>\n{input}</s>\n<|assistant|>"
text = "Can you introduce yourself"
prompt = chat_template.format(input=text)

# 5. Tokenize the input
input_tokens = tokenizer.encode(prompt)

# 6. Set up the generator
params = og.GeneratorParams(model)
params.set_search_options(**search_options)
params.input_ids = input_tokens
generator = og.Generator(model, params)

# 7. Generate the response token by token
while not generator.is_done():
    generator.compute_logits()
    generator.generate_next_token()
    new_token = generator.get_next_tokens()[0]
    print(tokenizer_stream.decode(new_token), end='', flush=True)
```

### Option B: Using ONNX Runtime GenAI 0.6.0

This updated version includes latency measurement and uses a slightly different API for appending tokens.

```python
import onnxruntime_genai as og
import time

# 1. Load the converted model
model_folder = "Your Phi-4-mini-onnx model path"
model = og.Model(model_folder)

# 2. Prepare the tokenizer
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# 3. Configure generation parameters
search_options = {
    'max_length': 1024,
    'past_present_share_buffer': False
}

# 4. Define the chat template and user input
chat_template = "<|user|>{input}<|assistant|>"
text = "can you introduce yourself"
prompt = chat_template.format(input=text)

# 5. Tokenize the input
input_tokens = tokenizer.encode(prompt)

# 6. Set up the generator and start timing
params = og.GeneratorParams(model)
params.set_search_options(**search_options)
generator = og.Generator(model, params)

start_time = time.time()
token_count = 0

# 7. Append tokens and generate the response
generator.append_tokens(input_tokens)

while not generator.is_done():
    generator.generate_next_token()
    new_token = generator.get_next_tokens()[0]
    token_text = tokenizer.decode(new_token)

    # Measure and print first token latency
    if token_count == 0:
        first_token_time = time.time()
        first_response_latency = first_token_time - start_time
        print(f"First token delay: {first_response_latency:.4f} s")

    print(token_text, end='', flush=True)
    token_count += 1
```

## Summary

You have successfully converted the Phi-4-mini model to a quantized ONNX format and run inference using ONNX Runtime GenAI. This workflow enables efficient deployment of generative AI models on edge devices or in environments with limited computing resources. Remember to adjust the model paths, generation parameters, and chat templates based on your specific use case and model version.