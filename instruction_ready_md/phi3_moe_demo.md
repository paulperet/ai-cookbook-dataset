# Guide: Implementing a Multi-Agent Workflow with a Phi-3 MOE Model

This guide walks you through setting up a Mixture of Experts (MOE) language model and using it to execute a structured, multi-step agent workflow. You will load a Phi-3 MOE model, configure a text-generation pipeline, and prompt it to perform a sequence of tasks (writing a blog post and translating it) by following a strict JSON-based instruction format.

## Prerequisites

Ensure you have the necessary libraries installed. Run the following commands in your environment.

```bash
pip install transformers torch torchvision torchaudio -U
pip install flash-attn --no-build-isolation
```

## Step 1: Import Libraries and Load the Model

First, import the required modules and load the Phi-3 MOE model. We'll use `bfloat16` for efficient memory usage and enable automatic device mapping to leverage available GPUs.

```python
from torch import bfloat16
import transformers

# Define the model ID (adjust the path if your model is stored elsewhere)
model_id = "../Phi3MOE"

# Load the model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=bfloat16,
    device_map='auto'
)

# Set the model to evaluation mode
model.eval()
```

The model architecture will be printed, confirming it's a `PhiMoEForCausalLM` with 32 decoder layers and a sparse MOE block containing 16 experts.

## Step 2: Load the Tokenizer and Create a Pipeline

Next, load the corresponding tokenizer and create a text-generation pipeline. This pipeline will handle prompt formatting and generation.

```python
# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

# Create a text-generation pipeline
pipe = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
```

**Note:** You may see a warning stating the model `'PhiMoEForCausalLM'` is not officially supported for text-generation. This is often a false positive; the pipeline will typically function correctly with models that support causal language modeling.

## Step 3: Configure Generation Arguments

Define the parameters that will control the text generation process, such as the maximum length of the output and sampling behavior.

```python
generation_args = {
    "max_new_tokens": 512,      # Limit the length of the generated text
    "return_full_text": False,  # Return only the newly generated text, not the input prompt
    "temperature": 0.3,         # Lower temperature for more deterministic outputs
    "do_sample": False,         # Use greedy decoding (not sampling)
}
```

**Important:** The combination of `do_sample=False` and `temperature=0.3` is contradictory. The `temperature` parameter is ignored when not sampling. We'll keep the configuration as provided, but note that the model will use greedy decoding.

## Step 4: Define the System Prompt and Instruction Format

The core of this workflow is instructing the model to act as a multi-step agent. You must provide a detailed system message that defines the available tools and the required JSON output format.

```python
sys_msg = """You are a helpful AI assistant, you are an agent capable of using a variety of tools to answer a question. Here are a few of the tools available to you:

- Blog: This tool helps you describe a certain knowledge point and content, and finally write it into Twitter or Facebook style content
- Translate: This is a tool that helps you translate into any language, using plain language as required

To use these tools you must always respond in JSON format containing `"tool_name"` and `"input"` key-value pairs. For example, to answer the question, "Build Multi Agents with MOE models" you must use the calculator tool like so:

```json
{
    "tool_name": "Blog",
    "input": "Build Multi Agents with MOE models"
}
```

Or to translate the question "can you introduce yourself in Chinese" you must respond:

```json
{
    "tool_name": "Search",
    "input": "can you introduce yourself in Chinese"
}
```

Remember just output the final result, output in JSON format containing `"agentid"`,`"tool_name"` , `"input"` and `"output"`  key-value pairs .:

```json
[
{   "agentid": "step1",
    "tool_name": "Blog",
    "input": "Build Multi Agents with MOE models",
    "output": "........."
},
{   "agentid": "step2",
    "tool_name": "Search",
    "input": "can you introduce yourself in Chinese",
    "output": "........."
},
{
    "agentid": "final"
    "tool_name": "Result",
    "output": "........."
}
]
```

The users answer is as follows.
"""
```

Now, create a helper function to format the conversation correctly for the model. This uses the specific chat template tokens (`<|system|>`, `<|user|>`, `<|assistant|>`) required by the Phi-3 model.

```python
def instruction_format(sys_message: str, query: str):
    # Formats the system message and user query into the model's expected chat template.
    # Note: Do not add "</s>" to the end.
    return f'<|system|> {sys_message} <|end|>\n<|user|> {query} <|end|>\n<|assistant|>'
```

## Step 5: Prepare the User Query and Input Prompt

Define the task you want the agent to perform. In this case, we ask it to write about a topic and then translate the result.

```python
query = 'Write something about Generative AI with MOE , translate it to Chinese'
```

Combine the system message and the user query using the formatting function.

```python
input_prompt = instruction_format(sys_msg, query)
```

The `input_prompt` variable now contains the fully formatted instruction for the model, including all tool definitions and the user's request.

## Step 6: Optimize Memory and Generate the Response

Before generation, it's good practice to clear any cached GPU memory and configure PyTorch for expandable memory segments to avoid out-of-memory errors.

```python
import torch
import os

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

Now, pass the formatted prompt to the text-generation pipeline with the arguments defined earlier.

```python
output = pipe(input_prompt, **generation_args)
```

You may see a warning about the `temperature` and `do_sample` mismatch, which you can safely ignore for this run.

## Step 7: Extract and Review the Result

The pipeline's output is a list of dictionaries. Extract the generated text from the first element.

```python
generated_response = output[0]['generated_text']
print(generated_response)
```

### Expected Output

The model should return a well-structured JSON array following the instructions precisely:

```json
[
{   "agentid": "step1",
    "tool_name": "Blog",
    "input": "Generative AI with MOE",
    "output": "Generative AI with MOE (Mixture of Experts) is a powerful approach that combines the strengths of generative models and the flexibility of MOE architecture. This hybrid model can generate high-quality, diverse, and contextually relevant content, making it suitable for various applications such as content creation, data augmentation, and more."
},
{   "agentid": "step2",
    "tool_name": "Translate",
    "input": "Generative AI with MOE is a powerful approach that combines the strengths of generative models and the flexibility of MOE architecture. This hybrid model can generate high-quality, diverse, and contextually relevant content, making it suitable for various applications such as content creation, data augmentation, and more.",
    "output": "基于生成AI的MOE（Mixture of Experts）是一种强大的方法，它结合了生成模型的优势和MOE架构的灵活性。这种混合模型可以生成高质量、多样化且上下文相关的内容，使其适用于各种应用，如内容创建、数据增强等。"
},
{
    "agentid": "final",
    "tool_name": "Result",
    "output": "基于生成AI的MOE（Mixture of Experts）是一种强大的方法，它结合了生成模型的优势和MOE架构的灵活性。这种混合模型可以生成高质量、多样化且上下文相关的内容，使其适用于各种应用，如内容创建、数据增强等。"
}
]
```

## Summary

You have successfully implemented a multi-agent workflow using a Phi-3 MOE model. The process involved:
1.  Loading a specialized MOE model and tokenizer.
2.  Creating a generation pipeline with specific parameters.
3.  Crafting a detailed system prompt that defines a JSON-based agent protocol.
4.  Formatting a user query and generating a response where the model decomposed the task into sequential steps (Blog -> Translate) and provided the final result.

This pattern is highly extensible. You can modify the `sys_msg` to define different tools, complex workflows, or integrate external APIs by having the model output structured calls that another system can execute.