# Guide: Solving a Math Problem with Microsoft's Phi-4-mini-flash-reasoning Model

This tutorial walks you through using Microsoft's `Phi-4-mini-flash-reasoning` model, a powerful language model optimized for reasoning tasks, to solve a quadratic equation. You'll learn how to load the model, format a prompt, and generate a step-by-step solution.

## Prerequisites

Ensure you have the necessary Python libraries installed. This guide uses PyTorch and the Hugging Face Transformers library.

```bash
pip install torch transformers
```

## Step 1: Import Required Libraries

Begin by importing PyTorch and the necessary components from the Transformers library.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
```

## Step 2: Set the Random Seed

To ensure reproducible results across different runs, set a manual seed for PyTorch's random number generator.

```python
torch.random.manual_seed(0)
```

## Step 3: Load the Model and Tokenizer

Next, load the `Phi-4-mini-flash-reasoning` model and its corresponding tokenizer from Hugging Face. The model is configured to run on a CUDA-enabled GPU for faster inference and uses automatic data type detection.

```python
model_id = "microsoft/Phi-4-mini-flash-reasoning"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

## Step 4: Format the Input Prompt

Models often require prompts to be formatted in a specific chat-like structure. Here, you will create a user message containing a quadratic equation problem and apply the model's chat template to format it correctly for generation.

```python
messages = [{
    "role": "user",
    "content": "How to solve 3*x^2+4*x+5=1?"
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
```

## Step 5: Generate the Model's Response

Now, instruct the model to generate a response. The parameters `temperature` and `top_p` control the randomness and creativity of the output, while `max_new_tokens` sets a limit for the response length.

```python
outputs = model.generate(
    **inputs.to(model.device),
    max_new_tokens=32768,
    temperature=0.6,
    top_p=0.95,
    do_sample=True,
)
```

## Step 6: Decode and Display the Answer

Finally, decode the generated token IDs back into human-readable text. The decoding step excludes the original input tokens, showing only the model's new response.

```python
response = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
print(response[0])
```

Running this code will output the model's step-by-step solution to the quadratic equation `3*x^2+4*x+5=1`. The response will detail the algebraic steps, such as rearranging the equation, applying the quadratic formula, and providing the final solutions for `x`.

You have now successfully used a state-of-the-art language model to solve a mathematical problem. This workflow can be adapted for other reasoning tasks by modifying the user prompt in Step 4.