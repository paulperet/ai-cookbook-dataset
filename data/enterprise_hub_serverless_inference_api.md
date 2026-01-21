# A Practical Guide to the Hugging Face Serverless Inference API

_Authored by: [Andrew Reed](https://huggingface.co/andrewrreed)_

## Introduction

Hugging Face provides a [Serverless Inference API](https://huggingface.co/docs/api-inference/index) that allows you to quickly test and evaluate thousands of publicly accessible machine learning models with simple API calls—completely free.

This guide demonstrates several ways to query the Serverless Inference API while exploring various AI tasks, including:
- Generating text with open LLMs
- Creating images with Stable Diffusion
- Reasoning over images with Vision-Language Models (VLMs)
- Generating speech from text

> **Tip:** The free Serverless Inference API has rate limits (~few hundred requests per hour). For higher limits, you can [upgrade to a PRO account](https://huggingface.co/subscribe/pro). For high-volume production workloads, consider [Dedicated Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index).

## Prerequisites

Before you begin, ensure you have:
1. A [Hugging Face Hub profile](https://huggingface.co/join) (or [login](https://huggingface.co/login)).
2. A [User Access Token](https://huggingface.co/docs/hub/security-tokens) with `read` or `write` permissions. For this guide, you'll need a fine-grained token with:
   - `Inference > Make calls to the serverless Inference API` permissions
   - Read access to the `meta-llama/Meta-Llama-3-8B-Instruct` and `HuggingFaceM4/idefics2-8b-chatty` repositories.

## Setup

First, install the required packages and authenticate with your Hugging Face token.

```bash
pip install -U huggingface_hub transformers
```

```python
import os
from huggingface_hub import interpreter_login, whoami, get_token

# This will prompt you to enter your Hugging Face credentials
interpreter_login()
```

> **Tip:** As an alternative to `interpreter_login()`, you can use `notebook_login()` from the Hub Python Library or the `login` command from the Hugging Face CLI.

Verify your login by checking your active username and organizations:

```python
whoami()
```

## How to Query the Serverless Inference API

The Serverless Inference API exposes models via a simple URL pattern:
`https://api-inference.huggingface.co/models/<MODEL_ID>`

Where `<MODEL_ID>` is the model's repository name on the Hub. For example, `codellama/CodeLlama-7b-hf` becomes `https://api-inference.huggingface.co/models/codellama/CodeLlama-7b-hf`.

### Method 1: Using HTTP Requests Directly

You can call the API with a simple `POST` request using the `requests` library.

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/codellama/CodeLlama-7b-hf"
HEADERS = {"Authorization": f"Bearer {get_token()}"}

def query(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

output = query(
    payload={
        "inputs": "A HTTP POST request is used to ",
        "parameters": {"temperature": 0.8, "max_new_tokens": 50, "seed": 42},
    }
)
print(output)
```

**How it works:** The API dynamically loads the requested model onto shared compute infrastructure. It uses the model's `pipeline_tag` from its Model Card to determine the appropriate inference task. You can reference the [task](https://huggingface.co/tasks) or [pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines) documentation for allowed arguments.

> **Tip:** If a model isn't already loaded in memory, the API may initially return a 503 response. Wait a moment and try again. You can check currently loaded models using `InferenceClient().list_deployed_models()`.

### Method 2: Using the `huggingface_hub` Python Library

The `InferenceClient` utility simplifies API calls.

```python
from huggingface_hub import InferenceClient

client = InferenceClient()
response = client.text_generation(
    prompt="A HTTP POST request is used to ",
    model="codellama/CodeLlama-7b-hf",
    temperature=0.8,
    max_new_tokens=50,
    seed=42,
    return_full_text=True,
)
print(response)
```

You can inspect the function signature for details on parameters:

```python
# help(client.text_generation)
```

> **Tip:** You can also use JavaScript via [huggingface.js](https://huggingface.co/docs/huggingface.js/index) for JS or Node.js applications.

## Practical Applications

Now let's explore specific use cases with the Serverless Inference API.

### 1. Generating Text with Open LLMs

Text generation is a common use case, but it's important to understand the difference between model types:
- **Base models** (e.g., `codellama/CodeLlama-7b-hf`): Good at continuing generation from a prompt but not fine-tuned for conversation.
- **Instruction-tuned models** (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`): Trained to follow instructions and often fine-tuned for multi-turn chat.

Instruction-tuned models require specific **chat templates** for proper formatting. Using the wrong format won't cause errors but will degrade output quality.

#### Step 1: Format Your Prompt with the Correct Chat Template

Use the model's tokenizer to apply the proper chat template automatically.

```python
from transformers import AutoTokenizer

# Define your conversation
system_input = "You are an expert prompt engineer with artistic flair."
user_input = "Write a concise prompt for a fun image containing a llama and a cookbook. Only return the prompt."
messages = [
    {"role": "system", "content": system_input},
    {"role": "user", "content": user_input},
]

# Load the tokenizer and apply the chat template
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
print(f"\nFormatted Prompt:\n{prompt}")
```

#### Step 2: Query the Model

Pass the formatted prompt to the API.

```python
llm_response = client.text_generation(
    prompt, model=model_id, max_new_tokens=250, seed=42
)
print(llm_response)
```

#### Step 3: Compare with Incorrect Formatting

See what happens without proper formatting:

```python
bad_prompt = system_input + " " + user_input
out = client.text_generation(
    bad_prompt, model=model_id, max_new_tokens=250, seed=42
)
print(out)
```

#### Step 4: Simplify with `chat_completion`

The `InferenceClient` provides a `chat_completion` method that handles formatting automatically.

```python
for token in client.chat_completion(
    messages, model=model_id, max_tokens=250, stream=True, seed=42
):
    print(token.choices[0].delta.content, end="")
```

> **Learn More:**
> 1. [How to generate text](https://huggingface.co/blog/how-to-generate)
> 2. [Text generation strategies](https://huggingface.co/docs/transformers/generation_strategies)
> 3. [Inference for PROs](https://huggingface.co/blog/inference-pro)
> 4. [Inference Client Docs](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client#inference)

### 2. Creating Images with Stable Diffusion

The Serverless Inference API supports many tasks, including image generation.

#### Step 1: Generate an Image

Use the text-to-image capability with Stable Diffusion.

```python
image = client.text_to_image(
    prompt=llm_response,
    model="stabilityai/stable-diffusion-xl-base-1.0",
    guidance_scale=8,
    seed=42,
)
print("PROMPT:", llm_response)
# The image object is now available for display or saving
```

#### Step 2: Understand Caching

The API caches responses by default. Identical requests return identical results.

```python
# This will return the same image as above due to caching
image_cached = client.text_to_image(
    prompt=llm_response,
    model="stabilityai/stable-diffusion-xl-base-1.0",
    guidance_scale=8,
    seed=42,
)
```

#### Step 3: Disable Caching for New Generations

Use the `x-use-cache` header to force new generations.

```python
# Turn caching off
client.headers["x-use-cache"] = "0"

# Generate a new image with the same prompt
image_new = client.text_to_image(
    prompt=llm_response,
    model="stabilityai/stable-diffusion-xl-base-1.0",
    guidance_scale=8,
    seed=42,
)
```

### 3. Reasoning Over Images with Idefics2

Vision-Language Models (VLMs) can process both text and images. Let's use Idefics2 to write a poem about our generated image.

#### Step 1: Prepare the Image

Convert the PIL image to a base64-encoded string for transmission.

```python
import base64
from io import BytesIO

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

image_b64 = pil_image_to_base64(image_new)
```

#### Step 2: Format the VLM Prompt

Use the model's processor to apply the correct chat template for multimodal input.

```python
from transformers import AutoProcessor

vlm_model_id = "HuggingFaceM4/idefics2-8b-chatty"
processor = AutoProcessor.from_pretrained(vlm_model_id)

# Define the multimodal message
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Write a short limerick about this image."},
        ],
    },
]

# Apply the chat template
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

# Insert the base64-encoded image
image_input = f"data:image/jpeg;base64,{image_b64}"
image_input = f"![]({image_input})"
prompt = prompt.replace("<image>", image_input)
```

#### Step 3: Query the VLM

Get a poetic description of your image.

```python
limerick = client.text_generation(
    prompt, model=vlm_model_id, max_new_tokens=200, seed=42
)
print(limerick)
```

### 4. Generating Speech from Text

Finally, let's convert the generated limerick into speech using the Bark text-to-audio model.

#### Step 1: Generate Speech

```python
tts_model_id = "suno/bark"
speech_out = client.text_to_speech(text=limerick, model=tts_model_id)
```

#### Step 2: Play the Audio (in a notebook environment)

```python
from IPython.display import Audio

Audio(speech_out, rate=24000)
print(limerick)
```

## Conclusion

In this guide, you've learned how to use the Hugging Face Serverless Inference API to:
- Query models via direct HTTP requests or the `InferenceClient`
- Generate text with proper chat template formatting
- Create images with Stable Diffusion and manage caching
- Perform multimodal reasoning with Vision-Language Models
- Convert text to speech

This is just the beginning. Explore the [API documentation](https://huggingface.co/docs/api-inference/en/index) to discover more capabilities and models available through the Serverless Inference API.