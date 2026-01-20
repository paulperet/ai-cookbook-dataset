[Open in Colab](https://colab.research.google.com/github/openai/openai-cookbook/blob/main/articles/gpt-oss/run-colab.ipynb)

# Run OpenAI gpt-oss 20B in a FREE Google Colab

OpenAI released `gpt-oss` [120B](https://hf.co/openai/gpt-oss-120b) and [20B](https://hf.co/openai/gpt-oss-20b). Both models are Apache 2.0 licensed.

Specifically, `gpt-oss-20b` was made for lower latency and local or specialized use cases (21B parameters with 3.6B active parameters).

Since the models were trained in native MXFP4 quantization it makes it easy to run the 20B even in resource constrained environments like Google Colab.

Authored by: [Pedro](https://huggingface.co/pcuenq) and [VB](https://huggingface.co/reach-vb)

## Setup environment

Since support for mxfp4 in transformers is bleeding edge, we need a recent version of PyTorch and CUDA, in order to be able to install the `mxfp4` triton kernels.

We also need to install transformers from source, and we uninstall `torchvision` and `torchaudio` to remove dependency conflicts.

```python
!pip install -q --upgrade torch
```

```python
!pip install -q transformers triton==3.4 kernels
```

```python
!pip uninstall -q torchvision torchaudio -y
```

Please, restart your Colab runtime session after installing the packages above.

## Load the model from Hugging Face in Google Colab

We load the model from here: [openai/gpt-oss-20b](https://hf.co/openai/gpt-oss-20b)

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

## Setup messages/ chat

You can provide an optional system prompt or directly the input.

```python
messages = [
    {"role": "system", "content": "Always respond in riddles"},
    {"role": "user", "content": "What is the weather like in Madrid?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

## Specify Reasoning Effort

Simply pass it as an additional argument to `apply_chat_template()`. Supported values are `"low"`, `"medium"` (default), or `"high"`.

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
    reasoning_effort="high",
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

## Try out other prompts and ideas!

Check out our blogpost for other ideas: [https://hf.co/blog/welcome-openai-gpt-oss](https://hf.co/blog/welcome-openai-gpt-oss)

```python

```