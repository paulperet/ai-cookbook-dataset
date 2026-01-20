# Bits and Bytes
Bits-and-bytes is a very fast and straightforward approach to quantization, quantizing while loading. However, speed and quality are not optimal, useful for quick quantization and loading of models quantizing in the fly.
### Quantizing with [transformers](https://github.com/huggingface/transformers)

Lets do a short demo and quantize Mistral 7B!

First, we install `transformers` and all dependencies required.


```python
!pip install -q -U bitsandbytes
!pip install -q -U git+https://github.com/huggingface/transformers.git
!pip install -q -U git+https://github.com/huggingface/peft.git
!pip install -q -U git+https://github.com/huggingface/accelerate.git
```

[Installation logs...]

Once we're done, we can download the model we want to quantize. First, let's log in with a read access token so we have access to the models.

Note: You need to first accept the terms in the repo.


```python
from huggingface_hub import login

login("read_token")
```

Login successful

Now everything is ready, so we can load the model and quantize it! Here, we will quantize the model to 8-bit!


```python
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

pretrained_model_dir = "mistralai/Mistral-7B-Instruct-v0.3"
quantized_model_dir = "mistral_bnb_quant"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)
```

Contrary to other methods, BnB is pretty fast and efficient. We do not necessarily need to quantize it beforehand, we can do it on the fly!


```python
model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
```

[Model loading logs...]

Once ready you can use the model as follows:


```python
conversation = [{"role": "user", "content": "Tell me a joke."}]

prompt = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
)

print(prompt)

inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
inputs.to("cuda:0")

outputs = model.generate(**inputs, max_new_tokens=64)
response = tokenizer.decode(outputs[0], skip_special_tokens=False)

print(response)
```

Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.

<s>[INST] Tell me a joke.[/INST]
<s>[INST] Tell me a joke.[/INST] Sure, here's a classic one for you:

Why don't scientists trust atoms?

Because they make up everything!

I hope that made you smile! If you'd like, I can tell you another one. Just let me know!</s>