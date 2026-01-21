# Guide: Using the Gemini API with the OpenAI Python Library

This guide demonstrates how to interact with Google's Gemini API using the familiar OpenAI Python client library. You'll learn to perform text generation, multimodal analysis, and structured output tasks through a unified interface.

## Prerequisites

Before you begin, ensure you have the following:

1.  A **Gemini API Key**. You can generate one on the [Google AI Studio API key page](https://aistudio.google.com/app/apikey).
2.  Python installed on your system.

## Setup

### 1. Install Required Libraries

Install the necessary Python packages using pip. You'll need the OpenAI library and some utilities for handling PDFs and images.

```bash
pip install -U openai pillow pdf2image pdfminer.six
```

**Note for Linux Users:** If you plan to process PDFs, you may also need to install `poppler-utils` via your system's package manager (e.g., `sudo apt install poppler-utils`).

### 2. Configure the OpenAI Client for Gemini

Import the OpenAI client and configure it to point to the Gemini API endpoint using your API key.

```python
from openai import OpenAI

# Replace with your actual Gemini API key
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

# Initialize the client with the Gemini base URL
client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
```

### 3. Select a Model

First, let's see which Gemini models are available through this interface.

```python
models = client.models.list()
for model in models:
    if 'gemini-2' in model.id:
        print(model.id)
```

For this tutorial, we'll use the `gemini-2.0-flash` model. You can explore other models on the [official Gemini models page](https://ai.google.dev/gemini-api/docs/models/gemini).

```python
MODEL_ID = "gemini-2.0-flash"
```

---

## Part 1: Basic Text Generation

Let's start with a simple text completion to verify our setup works.

### Step 1: Generate a Text Response

We'll ask the model a fundamental question about AI.

```python
from IPython.display import Markdown

prompt = "What is generative AI?"

response = client.chat.completions.create(
  model=MODEL_ID,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {
      "role": "user",
      "content": prompt
    }
  ]
)

# Display the response nicely formatted
Markdown(response.choices[0].message.content)
```

**Expected Output:**
The model will return a detailed explanation of generative AI, covering its definition, how it works (training, learning patterns, generation), key model types (GANs, VAEs, Transformers, Diffusion Models), applications, and limitations.

### Step 2: Generate Code

The Gemini models are excellent at code generation. Let's request a C program.

```python
prompt = """
    Write a C program that takes two IP addresses, representing the start and end of a range
    (e.g., 192.168.1.1 and 192.168.1.254), as input arguments. The program should convert this
    IP address range into the minimal set of CIDR notations that completely cover the given
    range. The output should be a comma-separated list of CIDR blocks.
"""

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

Markdown(response.choices[0].message.content)
```

**What the Code Does:**
The model generates a complete C program that:
*   Converts IP address strings to 32-bit integers.
*   Implements an algorithm to find the largest possible CIDR blocks covering the range.
*   Handles errors for invalid IPs and incorrect usage.
*   Uses standard C network libraries (`arpa/inet.h`) for proper byte order conversion.

You can compile and run this program directly.

---

## Part 2: Multimodal Interactions

Gemini models can process multiple data types, including images and audio. In this section, you'll learn to analyze images.

**Important Note:** The OpenAI SDK compatibility layer currently supports inline images and audio. For video support, use the [native Gemini Python SDK](https://ai.google.dev/gemini-api/docs/sdks).

### Step 1: Prepare an Image

We'll download and display a sample image to analyze.

```python
from PIL import Image as PImage
import requests

# Define the image URL
image_url = "https://storage.googleapis.com/generativeai-downloads/images/Japanese_Bento.png"
image_filename = image_url.split("/")[-1]

# Download the image
response = requests.get(image_url)
with open(image_filename, 'wb') as f:
    f.write(response.content)

# Display the image
im = PImage.open(image_filename)
im.thumbnail([620,620], PImage.Resampling.LANCZOS)
im.show()  # Or display in your notebook environment
```

### Step 2: Encode and Analyze the Image

To send the image to the model, we need to encode it in base64 format.

```python
import base64

def encode_image(image_path):
    """Helper function to encode an image from a URL or local path."""
    if image_path.startswith('http'):
        image = requests.get(image_path)
        content = image.content
    else:
        with open(image_path, 'rb') as f:
            content = f.read()
    return base64.b64encode(content).decode('utf-8')

# Encode our downloaded image
encoded_image = encode_image(image_filename)
```

Now, let's ask the model to describe the image's contents and translate any text it finds.

```python
response = client.chat.completions.create(
  model=MODEL_ID,
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Describe the items on this image. If there is any non-English text, translate it as well"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/png;base64,{encoded_image}",
          },
        },
      ],
    }
  ]
)

Markdown(response.choices[0].message.content)
```

**Expected Output:**
The model will provide a detailed description of the Japanese bento box items, identifying each food item (e.g., Matcha Swiss Roll, Anpan, Umeboshi) and translating the Japanese labels into English.

---

## Next Steps

You've successfully used the OpenAI client to interact with Gemini models for:
1.  **Text Generation:** Answering questions and generating code.
2.  **Multimodal Analysis:** Describing and translating content within images.

To explore further, consider:
*   **Structured Outputs:** Use the `response_format` parameter to request JSON or specific field outputs.
*   **Function Calling:** Implement tools/function calling for more interactive agent-like behavior.
*   **Embeddings:** Generate text embeddings using Gemini's embedding models via the same client.

For advanced features like video processing or streaming, refer to the [official Gemini API documentation](https://ai.google.dev/gemini-api/docs).