# Guide: Running OpenAI's GPT-OSS 20B in Google Colab

This guide walks you through setting up and running OpenAI's `gpt-oss-20b` model in a free Google Colab environment. The model is Apache 2.0 licensed and optimized for lower latency, making it suitable for local or specialized use cases.

## Prerequisites

Before you begin, ensure you have a Google Colab notebook open with GPU acceleration enabled:
1.  Go to **Runtime** > **Change runtime type**.
2.  Select **T4 GPU** (or any available GPU) from the "Hardware accelerator" dropdown.
3.  Click **Save**.

## Step 1: Install Required Packages

Since support for the MXFP4 quantization used by the model is bleeding-edge, we need specific versions of PyTorch, CUDA, and the `transformers` library. Run the following commands in a single code cell.

```bash
!pip install -q --upgrade torch
!pip install -q transformers triton==3.4 kernels
!pip uninstall -q torchvision torchaudio -y
```

**Important:** After running the cell above, you **must restart your Colab runtime** for the changes to take effect. Go to **Runtime** > **Restart runtime**.

## Step 2: Load the Model and Tokenizer

Once your runtime has restarted, you can load the `gpt-oss-20b` model and its tokenizer directly from Hugging Face.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cuda",
)
```

This code downloads the model and automatically places it on your Colab's GPU.

## Step 3: Create a Chat and Generate a Response

Now, let's create a simple chat interaction. You can define a system message to guide the model's behavior and a user message as the prompt.

```python
messages = [
    {"role": "system", "content": "Always respond in riddles"},
    {"role": "user", "content": "What is the weather like in Madrid?"},
]

# Format the messages into the model's expected input format
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

# Generate a response
generated = model.generate(**inputs, max_new_tokens=500)

# Decode and print only the new tokens (the model's response)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

You should see a riddle-like response about the weather in Madrid.

## Step 4: Control the Model's Reasoning Effort

The `gpt-oss` models allow you to specify a `reasoning_effort` level, which influences the depth of the model's internal processing. You can set it to `"low"`, `"medium"` (the default), or `"high"`.

Here's how to use it with a more complex prompt:

```python
messages = [
    {"role": "system", "content": "Always respond in riddles"},
    {"role": "user", "content": "Explain why the meaning of life is 42"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    reasoning_effort="high",  # Specify the reasoning effort here
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

Setting `reasoning_effort="high"` will typically result in a more detailed and thoughtful response.

## Next Steps

You can now experiment with different system prompts, user questions, and reasoning effort levels. For more inspiration and ideas on using the `gpt-oss` models, check out the [Hugging Face blog post](https://hf.co/blog/welcome-openai-gpt-oss).