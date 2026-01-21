# Quantizing a Model with AWQ: A Step-by-Step Guide

This guide walks you through quantizing a Mistral 7B model using AWQ (Activation-aware Weight Quantization). AWQ is a post-training quantization method optimized for GPU inference. It identifies and protects the most critical weights (approximately 1%) that significantly impact model accuracy, resulting in a compressed model that maintains high performance.

## Prerequisites

Ensure you have a GPU-enabled environment (e.g., Colab, an AWS instance, or a local machine with a CUDA-capable GPU). You will also need a Hugging Face account and a read token for accessing gated models.

## Step 1: Install AutoAWQ

Begin by installing the `autoawq` library, which provides tools for quantization and inference.

```bash
pip install autoawq
```

## Step 2: Authenticate with Hugging Face

Log in to Hugging Face using your read token. This grants access to download the base model.

```python
from huggingface_hub import login

# Replace "read_token" with your actual token
login("read_token")
```

**Note:** If you are using a gated model like Mistral, you must first accept its terms of use on the model's Hugging Face page.

## Step 3: Load the Model and Tokenizer

Specify the model you want to quantize and load it along with its tokenizer.

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

pretrained_model_dir = "mistralai/Mistral-7B-Instruct-v0.3"
quantized_model_dir = "mistral_awq_quant"

# Load the base model
model = AutoAWQForCausalLM.from_pretrained(
    pretrained_model_dir,
    low_cpu_mem_usage=True,
    use_cache=False
)
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, trust_remote_code=True)
```

## Step 4: Configure and Run Quantization

Define the quantization settings and run the quantization process. This step analyzes the model's activations and quantizes the weights to 4-bit precision.

```python
# Define quantization configuration
quant_config = {
    "zero_point": True,   # Use zero-point quantization
    "q_group_size": 128,  # Group size for quantization
    "w_bit": 4,           # Quantize to 4-bit
    "version": "GEMM"     # Use GEMM version for computation
}

# Quantize the model
model.quantize(tokenizer, quant_config=quant_config)
```

Quantization may take several minutes, depending on your hardware.

## Step 5: Save the Quantized Model

Once quantization is complete, save the model and tokenizer to disk for future use.

```python
# Save the quantized model
model.save_quantized(quantized_model_dir)
# Save the tokenizer
tokenizer.save_pretrained(quantized_model_dir)
```

Your model is now quantized and saved in the directory `mistral_awq_quant`.

## Step 6: Load and Run Inference with the Quantized Model

You can now load the quantized model and use it for inference.

```python
# Load the quantized model onto the GPU
model = AutoAWQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)

# Prepare a conversation prompt
conversation = [{"role": "user", "content": "How are you today?"}]
prompt = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize and move inputs to GPU
inputs = tokenizer(prompt, return_tensors="pt")
inputs = inputs.to("cuda:0")

# Generate a response
outputs = model.generate(**inputs, max_new_tokens=32)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

**Expected Output:**
```
How are you today? I'm an AI and don't have feelings, but I'm here and ready to help you with your questions! How can I assist you today
```

## Summary

You have successfully quantized a Mistral 7B model to 4-bit using AWQ, significantly reducing its memory footprint while preserving accuracy. The quantized model can be loaded and run efficiently on a GPU for inference tasks. Remember to save your quantized models to avoid re-running the quantization process.