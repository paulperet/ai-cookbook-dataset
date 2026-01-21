# Getting Started with the Gemini API: A Comprehensive Guide

This guide provides a comprehensive introduction to using the Google Gen AI SDK to interact with Gemini models via the Gemini Developer API. You will learn how to set up the client, send various types of prompts, configure model behavior, and leverage advanced features like function calling and grounding.

## Prerequisites & Setup

### 1. Install the SDK
First, install the latest version of the Google Gen AI SDK from PyPI.

```bash
pip install -U 'google-genai>=1.51.0'
```

### 2. Configure Your API Key
You need a Gemini API key. Store it securely. This example assumes the key is stored in an environment variable or a secure secret manager. For demonstration, we retrieve it from a hypothetical `userdata` module.

```python
# Example: Retrieving an API key (adjust for your environment)
# from google.colab import userdata
# GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')

# For local development, you might use an environment variable
import os
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
```

### 3. Initialize the Client
Create a client instance using your API key.

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=GEMINI_API_KEY)
```

### 4. Select a Model
Choose a model for this tutorial. You can select from popular options or specify your own.

```python
# Example: Using Gemini 2.5 Flash
MODEL_ID = "gemini-2.5-flash"

# Other options include:
# "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3-pro-preview"
```

---

## 1. Sending Your First Text Prompt

Use the `generate_content` method to send a prompt and get a response. Access the text output via the `.text` property.

```python
from IPython.display import Markdown

response = client.models.generate_content(
    model=MODEL_ID,
    contents="What's the largest planet in our solar system?"
)

print(response.text)
```

**Output:**
```
The largest planet in our solar system is Jupiter.
```

## 2. Adding System Instructions

Guide the model's persona and response style by providing a `system_instruction`.

```python
system_instruction = "You are a pirate and are explaining things to 5 years old kids."

response = client.models.generate_content(
    model=MODEL_ID,
    contents="What's the largest planet in our solar system?",
    config=types.GenerateContentConfig(
        system_instruction=system_instruction,
    )
)

print(response.text)
```

**Output (example snippet):**
```
Ahoy there, me hearties! ... The most enormous, the most ginormous ... It be the mighty JUPITER!
```

## 3. Counting Tokens

Estimate the token cost of a prompt before sending it using `count_tokens`.

```python
response = client.models.count_tokens(
    model=MODEL_ID,
    contents="What's the highest mountain in Africa?",
)

print(f"This prompt was worth {response.total_tokens} tokens.")
```

**Output:**
```
This prompt was worth 10 tokens.
```

## 4. Configuring Model Parameters

Fine-tune the generation by adjusting parameters like `temperature`, `top_p`, and `stop_sequences`.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Tell me how the internet works, but pretend I'm a puppy who only understands squeaky toys.",
    config=types.GenerateContentConfig(
        temperature=0.4,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        seed=5,
        stop_sequences=["STOP!"],
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
)

print(response.text)
```

**Output (example snippet):**
```
*Woof!* Sit! Stay! ... Your Squeak runs to a little blinking box in the corner...
```

## 5. Controlling the Thinking Process

Gemini 2.5+ models are "thinking" models. They perform internal reasoning before generating a final answer, which is beneficial for complex tasks.

### 5.1 Inspect the Thought Process
Enable `include_thoughts` to see the model's internal reasoning.

```python
prompt = "A man moves his car to an hotel and tells the owner he’s bankrupt. Why?"

response = client.models.generate_content(
  model=MODEL_ID,
  contents=prompt,
  config=types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(
      include_thoughts=True
    )
  )
)

for part in response.parts:
  if not part.text:
    continue
  if part.thought:
    print("### Thought Summary:")
    print(part.text)
    print()
  else:
    print("### Final Answer:")
    print(part.text)
    print()

print(f"Thought tokens: {response.usage_metadata.thoughts_token_count}")
print(f"Output tokens: {response.usage_metadata.prompt_token_count}")
```

**Output (example snippet):**
```
### Thought Summary:
Unraveling the Riddle: A Monopoly-Focused Thought Process ...
The key is recognizing the symbolism. It's a classic lateral thinking puzzle...

### Final Answer:
He is playing Monopoly.

Thought tokens: 575
Output tokens: 20
```

### 5.2 Disable Thinking (Flash Models)
For faster, simpler queries, you can disable thinking on Flash models by setting `thinking_budget=0`.

```python
if "-pro" not in MODEL_ID:  # Check if it's a Flash model
  response = client.models.generate_content(
    model=MODEL_ID,
    contents="Quicky tell me a joke about unicorns.",
    config=types.GenerateContentConfig(
      thinking_config=types.ThinkingConfig(
        thinking_budget=0
      )
    )
  )
  print(response.text)
```

**Output:**
```
Why did the unicorn run across the road?
To get to the other rainbow!
```

## 6. Sending Multimodal Prompts

Gemini models accept mixed media inputs. Here's an example using an image.

### 6.1 Prepare an Image
Download and save an image locally.

```python
import requests
import pathlib
from PIL import Image

IMG_URL = "https://storage.googleapis.com/generativeai-downloads/data/jetpack.png"
img_bytes = requests.get(IMG_URL).content

img_path = pathlib.Path('jetpack.png')
img_path.write_bytes(img_bytes)

image = Image.open(img_path)
image.thumbnail([512, 512])
```

### 6.2 Send the Image with a Text Prompt
Pass the image object along with your text instruction.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        image,
        "Write a short and engaging blog post based on this picture."
    ]
)

print(response.text)
```

**Output (example snippet):**
```
Title: Beat the Traffic in Style: Introducing the Jetpack Backpack Concept
We’ve all been there. You’re five minutes late... This design changes the game...
```

## 7. Generating Images (Image-Out)

Certain Gemini models can generate images directly. Use the dedicated image generation model.

```python
from IPython.display import Image, Markdown

response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents='Hi, can you create a 3D render of a cute robot holding a flower?'
)

# The response for image generation would contain image data or a reference.
# Display logic depends on the output format (e.g., base64, URL).
# This is a placeholder for the image generation call.
```

---

## Next Steps

This guide covered the foundational operations. The Gemini API supports many more advanced features:

*   **Function Calling:** Enable models to trigger external tools.
*   **Grounding:** Enhance responses with context from uploaded files, Google Search, Google Maps, YouTube, or web URLs.
*   **Context Caching:** Improve performance for long conversations.
*   **Text Embeddings:** Generate vector representations of text.
*   **Asynchronous Requests & Streaming:** Handle long-running tasks and receive partial outputs.

For detailed guides on these topics and model-specific capabilities (like Gemini 3 Pro's thinking levels), refer to the official [Gemini API documentation](https://ai.google.dev/gemini-api/docs).