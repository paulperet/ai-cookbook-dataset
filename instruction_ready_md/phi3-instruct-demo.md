# Guide: Running a Local Phi-3 Model with Hugging Face Transformers

This guide walks you through loading and running a local Phi-3 model using the Hugging Face Transformers library. You'll learn how to set up the model, tokenizer, and text-generation pipeline to produce responses in Chinese.

## Prerequisites

Ensure you have the required libraries installed. If you don't have them, run the following command in your environment:

```bash
pip install torch transformers
```

**Note:** This guide assumes you have a local directory named `../phi-3-instruct` containing the Phi-3 model files. If you are using a GPU, ensure your CUDA drivers are properly configured. The warnings about `libcudart` are non-critical if you are running on a CPU-only machine.

## Step 1: Import Libraries and Set Seed

First, import the necessary modules and set a manual seed for PyTorch to ensure reproducible results.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set seed for reproducibility
torch.random.manual_seed(0)
```

## Step 2: Load the Model

Load the Phi-3 model from your local directory. The `device_map="cuda"` argument will utilize your GPU if available. The `torch_dtype="auto"` setting allows the library to automatically select the optimal data type (e.g., `float16` for GPU).

```python
model = AutoModelForCausalLM.from_pretrained(
    "../phi-3-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
```

**Note:** You may see a progress bar as the model checkpoint shards are loaded. This is normal.

## Step 3: Load the Tokenizer

Load the tokenizer from the same local directory. The tokenizer converts your text input into tokens the model can understand.

```python
tokenizer = AutoTokenizer.from_pretrained("../phi-3-instruct")
```

You might see a message regarding the "legacy" behavior of the `LlamaTokenizerFast`. This is a standard informational message from the Transformers library and does not affect functionality. You can safely ignore it.

## Step 4: Prepare the Input Prompt

Phi-3 models use a specific chat template format. Instead of a list of dictionaries, you need to construct the prompt as a single string using special tokens.

The format is:
```
<|system|>
System message here.
<|end|>
<|user|>
User question here.
<|end|>
<|assistant|>
```

Let's construct a prompt where the system is a helpful Chinese assistant and the user asks about the city of Changsha.

```python
messages = "<|system|>\n你是我的人工智能助手，协助我用中文解答问题.\n<|end|><|user|>\n你知道长沙吗？\n<|end|><|assistant|>"
```

## Step 5: Create the Text-Generation Pipeline

The Hugging Face `pipeline` API provides a high-level interface for running inference. We'll create a text-generation pipeline using our loaded model and tokenizer.

```python
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
```

## Step 6: Configure Generation Parameters

Define the arguments that control how the model generates text. These parameters are crucial for balancing response quality, length, and determinism.

```python
generation_args = {
    "max_new_tokens": 1024,   # Limits the length of the generated response
    "return_full_text": False, # Returns only the newly generated text, not the input prompt
    "temperature": 0.3,       # Lower temperature makes outputs more deterministic
    "do_sample": False,       # Disables sampling for deterministic (greedy) generation
}
```

## Step 7: Generate a Response

Pass your formatted prompt and generation arguments to the pipeline.

```python
output = pipe(messages, **generation_args)
```

## Step 8: Extract and View the Result

The pipeline returns a list of dictionaries. Extract the generated text from the first result.

```python
generated_text = output[0]['generated_text']
print(generated_text)
```

**Expected Output:**
The model should generate a coherent response in Chinese about the city of Changsha.

```
是的，我知道长沙。长沙是中国南部的一座大城市，位于湖南省中部，是中国的国际性大都市。它拥有丰富的历史和文化，曾是明朝的都城。长沙以其繁华的商业、独特的自然风光和丰富的历史遗迹而闻名，如橘子洲、岳麓山和长沙博物院。此外，长沙也是中国重要的科技和教育中心，包括中国工程院和中国科学院长沙分院。
```

## Summary

You have successfully loaded a local Phi-3 model and used it to generate a text response. The key steps were:
1.  Loading the model and tokenizer.
2.  Formatting the input prompt according to the model's specific template.
3.  Configuring the text-generation pipeline with appropriate parameters.
4.  Executing the pipeline and extracting the result.

You can now modify the `messages` string to ask different questions or adjust the `generation_args` (like `temperature` or `max_new_tokens`) to change the style and length of the responses.