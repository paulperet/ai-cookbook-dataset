# Guide: Running GPT-OSS Models with Hugging Face Transformers

## Overview
This guide provides step-by-step instructions for running OpenAI's GPT-OSS models (20B and 120B parameter versions) using the Hugging Face Transformers library. You'll learn how to set up your environment, perform inference using both high-level and low-level APIs, and deploy a local server compatible with the OpenAI Responses API format.

## Prerequisites

### 1. Install Required Packages
Create a fresh Python environment and install the necessary dependencies:

```bash
pip install -U transformers accelerate torch triton==3.4 kernels
```

**Note:** The Triton kernels package is required for MXFP4 quantization support on compatible hardware (Hopper architecture or later, including H100, GB200, and RTX 50xx series GPUs).

### 2. Choose Your Model
Two GPT-OSS models are available on Hugging Face:

- **`openai/gpt-oss-20b`**: ~16GB VRAM requirement with MXFP4 quantization
- **`openai/gpt-oss-120b`**: ≥60GB VRAM requirement with MXFP4 quantization

Both models use MXFP4 quantization by default. If you use `bfloat16` instead, memory consumption will be significantly higher (~48GB for the 20B model).

## Method 1: Quick Inference with Pipeline API

The simplest way to run GPT-OSS models is using the high-level `pipeline` abstraction:

```python
from transformers import pipeline

# Initialize the text generation pipeline
generator = pipeline(
    "text-generation",
    model="openai/gpt-oss-20b",
    torch_dtype="auto",
    device_map="auto"  # Automatically places model on available GPUs
)

# Define your conversation messages
messages = [
    {"role": "user", "content": "Explain what MXFP4 quantization is."},
]

# Generate a response
result = generator(
    messages,
    max_new_tokens=200,
    temperature=1.0,
)

# Print the generated text
print(result[0]["generated_text"])
```

## Method 2: Advanced Inference with `.generate()`

For more control over the generation process, load the model and tokenizer directly:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "openai/gpt-oss-20b"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Prepare your conversation
messages = [
    {"role": "user", "content": "Explain what MXFP4 quantization is."},
]

# Apply the chat template to format messages correctly
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

# Generate the response
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7
)

# Decode and print the output
print(tokenizer.decode(outputs[0]))
```

## Method 3: Local Server Deployment

To create a local server compatible with the OpenAI Responses API format:

### Start the Server
```bash
transformers serve
```

### Interact with the Server
You can use the built-in chat CLI:

```bash
transformers chat localhost:8000 --model-name-or-path openai/gpt-oss-20b
```

Or send HTTP requests directly:

```bash
curl -X POST http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "system", "content": "hello"}],
    "temperature": 0.9,
    "max_tokens": 1000,
    "stream": true,
    "model": "openai/gpt-oss-20b"
  }'
```

## Working with Chat Templates and Tool Calling

GPT-OSS models use the Harmony response format for structured messages. You can use either the built-in Transformers chat template or the dedicated `openai-harmony` library.

### Option A: Using Built-in Chat Template

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
)

# Define your conversation
messages = [
    {"role": "system", "content": "Always respond in riddles"},
    {"role": "user", "content": "What is the weather like in Madrid?"},
]

# Format messages using the chat template
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

# Generate response
generated = model.generate(**inputs, max_new_tokens=100)

# Decode only the new tokens (excluding the prompt)
response = tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :])
print(response)
```

### Option B: Using the `openai-harmony` Library

First, install the library:

```bash
pip install openai-harmony
```

Then use it to build and parse conversations:

```python
import json
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the Harmony encoding
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

# Build a conversation
convo = Conversation.from_messages([
    Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
    Message.from_role_and_content(
        Role.DEVELOPER,
        DeveloperContent.new().with_instructions("Always respond in riddles")
    ),
    Message.from_role_and_content(Role.USER, "What is the weather like in SF?")
])

# Render the prompt for completion
prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
stop_token_ids = encoding.stop_tokens_for_assistant_actions()

# Load the model
model_name = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype="auto", 
    device_map="auto"
)

# Generate the response
outputs = model.generate(
    input_ids=[prefill_ids],
    max_new_tokens=128,
    eos_token_id=stop_token_ids
)

# Parse the completion tokens
completion_ids = outputs[0][len(prefill_ids):]
entries = encoding.parse_messages_from_completion_tokens(
    completion_ids, 
    Role.ASSISTANT
)

# Print the parsed messages
for message in entries:
    print(json.dumps(message.to_dict(), indent=2))
```

**Note:** The `Developer` role in Harmony maps to the `system` prompt in the chat template.

## Multi-GPU and Distributed Inference

For the larger GPT-OSS-120B model, you may need to distribute the workload across multiple GPUs:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig
import torch

model_path = "openai/gpt-oss-120b"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

# Configure distributed settings
device_map = {
    "distributed_config": DistributedConfig(enable_expert_parallel=1),
    "tp_plan": "auto",  # Enable automatic tensor parallelism
}

# Load the model with distributed configuration
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    attn_implementation="kernels-community/vllm-flash-attn3",  # Use optimized attention
    **device_map,
)

# Prepare the conversation
messages = [
    {"role": "user", "content": "Explain how expert parallelism works in large language models."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

# Generate the response
outputs = model.generate(**inputs, max_new_tokens=1000)

# Decode and extract the final response
response = tokenizer.decode(outputs[0])
final_response = response.split("<|channel|>final<|message|>")[-1].strip()
print("Model response:", final_response)
```

Save this script as `generate.py` and run it with:

```bash
torchrun --nproc_per_node=4 generate.py
```

This command distributes the workload across 4 GPUs using PyTorch's distributed backend.

## Next Steps

For more advanced use cases, including fine-tuning GPT-OSS models with Transformers, refer to the [fine-tuning guide](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transformers).

Additional documentation on serving models and integration with development tools is available in the [Transformers serving documentation](https://huggingface.co/docs/transformers/main/serving).