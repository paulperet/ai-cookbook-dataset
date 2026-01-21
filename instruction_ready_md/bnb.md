# Quantizing Models with BitsAndBytes: A Quick Start Guide

BitsAndBytes (BnB) is a fast, straightforward quantization method that quantizes models during loading. It's ideal for quick experiments and scenarios where speed is prioritized over optimal quality. This guide walks you through quantizing the Mistral 7B Instruct model to 8-bit.

## Prerequisites

First, install the required libraries. We'll use the latest versions from the Hugging Face repositories.

```bash
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
```

## Step 1: Authenticate with Hugging Face Hub

To download the model, you need to authenticate with a Hugging Face read token. Ensure you have accepted the model's terms of use on its repository page first.

```python
from huggingface_hub import login

# Replace "read_token" with your actual token
login("read_token")
```

## Step 2: Configure Quantization

We'll configure the model to load in 8-bit precision using the `BitsAndBytesConfig`. This configuration object will be passed to the model loader.

```python
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

# Define model paths
pretrained_model_dir = "mistralai/Mistral-7B-Instruct-v0.3"

# Create the quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)
```

## Step 3: Load and Quantize the Model

Unlike other methods, BnB performs quantization on-the-fly during model loading, making the process very efficient.

```python
# Load the model with quantization
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_dir,
    quantization_config=quantization_config
)

# Load the corresponding tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
```

## Step 4: Generate a Response

Now you can use the quantized model for inference. We'll format a simple conversation prompt and generate a response.

```python
# Define a conversation
conversation = [{"role": "user", "content": "Tell me a joke."}]

# Format the conversation using the model's chat template
prompt = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True,
)

print("Formatted Prompt:")
print(prompt)
print("\n" + "-"*50 + "\n")

# Tokenize the input and move to GPU
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
inputs = inputs.to("cuda:0")

# Generate a response
outputs = model.generate(**inputs, max_new_tokens=64)

# Decode the generated tokens
response = tokenizer.decode(outputs[0], skip_special_tokens=False)

print("Model Response:")
print(response)
```

### Expected Output

The model will generate a response similar to the following:

```
Formatted Prompt:
<s>[INST] Tell me a joke.[/INST]

--------------------------------------------------

Model Response:
<s>[INST] Tell me a joke.[/INST] Sure, here's a classic one for you:

Why don't scientists trust atoms?

Because they make up everything!

I hope that made you smile! If you'd like, I can tell you another one. Just let me know!</s>
```

## Summary

You have successfully quantized the Mistral 7B Instruct model to 8-bit using BitsAndBytes and generated a response. This method provides a quick way to reduce model memory footprint with minimal setup. For production use cases requiring higher quality, consider more advanced quantization techniques, but for prototyping and quick tests, BnB is an excellent choice.