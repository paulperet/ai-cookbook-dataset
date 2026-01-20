## Import Required Libraries
Import PyTorch and transformers libraries needed for loading and using the Phi-4 model.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
```

## Set Random Seed
Set the random seed to ensure reproducible results across different runs.

```python
torch.random.manual_seed(0)
```

## Load Phi-4-mini-flash-reasoning Model and Tokenizer
Load the Microsoft Phi-4-mini-flash-reasoningmodel and its corresponding tokenizer from Hugging Face. The model will be loaded on CUDA for faster inference.

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

## Prepare Input Message
Create a conversation message with a quadratic equation math problem and format it using the chat template for the model.

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

## Generate Response
Generate a response from the model using specified parameters like temperature and top_p for controlled randomness in the output.

```python
outputs = model.generate(
   **inputs.to(model.device),
   max_new_tokens=32768,
   temperature=0.6,
   top_p=0.95,
   do_sample=True,
)
```

## Decode Output to Text
Convert the generated token sequences back to human-readable text, excluding the original input tokens to show only the model's response.

```python
outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
```