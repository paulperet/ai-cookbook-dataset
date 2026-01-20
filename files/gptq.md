# GPTQ
GPTQ is a quantization format optimized for GPU inference. It makes use of a calibration dataset to improve its quantizations.

### Quantizing with [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)

Let's do a short demo and quantize Mistral 7B.

First, we install `auto-gptq`. It will allow us to easily quantize and infer GPTQ models.

```python
!pip install auto-gptq --no-build-isolation
```

Once we're done, we can download the model we want to quantize. First, let's log in with a read access token so we have access to the models.

Note: You need to first accept the terms in the repo.

```python
from huggingface_hub import login

login("read_token")
```

Now everything is ready, so we can load the model and quantize it! Here, we will quantize the model to 4-bit!

```python
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging

pretrained_model_dir = "mistralai/Mistral-7B-Instruct-v0.3"
quantized_model_dir = "mistral_gptq_quant"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)

examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on the GPTQ algorithm."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may be slightly bad, feel free to change
)

model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples)
```

[INFO - Start quantizing layer 1/32, ..., INFO - Quantizing mlp.down_proj in layer 32/32...]

Now that the model is quantized, we can save it so we can share it or load it later! Since quantizing with GPTQ takes a while and some resources, it's advised to always save them.

```python
model.save_quantized(quantized_model_dir)

tokenizer.save_pretrained(quantized_model_dir)

model.save_quantized(quantized_model_dir, use_safetensors=True)
```

Model quantized and saved to GPTQ 4-bit precision!

You can also load it for inference using `auto-gptq` as follows:

```python
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0") # loads quantized model to the first GPU
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)

conversation = [{"role": "user", "content": "How are you today?"}]

prompt = tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt")
inputs.to("cuda:0") # loads tensors to the first GPU

outputs = model.generate(**inputs, max_new_tokens=32)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```