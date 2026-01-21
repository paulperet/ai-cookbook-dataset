# Gemini API Prompting Quickstart Guide

This guide walks you through the fundamentals of prompting the Gemini API, from generating simple text responses to handling multi-turn conversations and image inputs.

## Prerequisites

Before you begin, ensure you have the necessary library installed and your API key configured.

### 1. Install the Python SDK

Install the `google-genai` Python SDK. Version 1.4.0 or higher is required for chat history features.

```bash
pip install -U "google-genai>=1.4.0"
```

### 2. Configure Your API Key

Your API key must be stored securely. This example uses Google Colab secrets. If you're not using Colab, you can set the `GOOGLE_API_KEY` as an environment variable.

```python
from google.colab import userdata
from google import genai

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

### 3. Select a Model

Choose a Gemini model for this tutorial. The `gemini-3-flash-preview` model is a good balance of speed and capability for these examples.

```python
MODEL_ID = "gemini-3-flash-preview"
```

---

## Step 1: Run Your First Text Prompt

The primary method for generating content is `client.models.generate_content()`. You pass your prompt as text and can access the response's text content.

Let's ask the model for Python code to sort a list.

```python
from IPython.display import Markdown

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Give me python code to sort a list"
)

display(Markdown(response.text))
```

The model will return a detailed explanation and examples of Python's `list.sort()` and `sorted()` functions.

## Step 2: Use Images in Your Prompt

Gemini models are multimodal and can process images. Let's download an image and ask the model to analyze it.

First, download an image and load it using the Python Imaging Library (PIL).

```bash
curl -o image.jpg "https://storage.googleapis.com/generativeai-downloads/images/jetpack.jpg"
```

```python
import PIL.Image
img = PIL.Image.open('image.jpg')
```

Now, craft a prompt that asks the model to describe the product in the image and output the result in a structured JSON format.

```python
prompt = """
    This image contains a sketch of a potential product along with some notes.
    Given the product sketch, describe the product as thoroughly as possible based on what you
   see in the image, making sure to note all of the product features. Return output in json format:
   {description: description, features: [feature1, feature2, feature3, etc]}
"""
```

Pass both the text prompt and the image object as a list to the `generate_content` method.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, img],
)

print(response.text)
```

You will receive a JSON-structured response similar to this:

```json
{
  "description": "The \"Jetpack Backpack\" is a conceptual personal flight device disguised as a conventional backpack...",
  "features": [
    "Looks like a normal backpack",
    "Lightweight",
    "Fits 18\" laptop",
    "Padded strap support",
    "Retractable boosters",
    "Steam-powered",
    "Green/Clean (environmental friendly propulsion)",
    "USB-C charging",
    "15-min battery life"
  ]
}
```

## Step 3: Have a Multi-Turn Conversation

For interactive dialogues, use the `ChatSession` class, which automatically manages conversation history.

### 3.1 Start a Chat Session

Create a new chat session with your chosen model.

```python
chat = client.chats.create(model=MODEL_ID)
```

### 3.2 Send Your First Message

Use the `send_message` method to interact with the chat.

```python
response = chat.send_message(
    message="In one sentence, explain how a computer works to a young child."
)

print(response.text)
# Output: A computer is a super-smart helper that follows your instructions very, very fast...
```

### 3.3 View the Chat History

The session stores the conversation. You can retrieve and print the history.

```python
messages = chat.get_history()
for message in messages:
  print(f"{message.role}: {message.parts[0].text}")
```

### 3.4 Continue the Conversation

Send a follow-up message. The model's response will be contextual, based on the previous exchange.

```python
response = chat.send_message("Okay, how about a more detailed explanation to a high schooler?")

print(response.text)
# Output: A computer works by taking your input, translating it into binary (0s and 1s)...
```

## Step 4: Configure Generation Parameters

You can control the model's behavior using a `types.GenerateContentConfig` object. Key parameters include:
*   `temperature`: Controls randomness (higher = more creative, lower = more deterministic).
*   `max_output_tokens`: Limits the length of the response.
*   `stop_sequences`: Stops generation when a specified sequence is encountered.

Let's generate a short, creative list of cat facts.

```python
from google.genai import types

response = client.models.generate_content(
    model=MODEL_ID,
    contents='Give me a numbered list of cat facts.',
    config=types.GenerateContentConfig(
        max_output_tokens=2000,
        temperature=1.9,  # High temperature for varied responses
        stop_sequences=['\n6']  # Stop after the 5th fact
    )
)

display(Markdown(response.text))
```

This will yield a creative list, stopping before fact number 6.

## Next Steps

You've learned the basics of prompting the Gemini API. To dive deeper, explore these resources:

*   **Advanced Examples:** Check out the [Market a Jetpack](https://github.com/google-gemini/cookbook/blob/main/examples/Market_a_Jet_Backpack.ipynb) notebook for more complex prompting techniques.
*   **Safety Settings:** Learn about configurable safety filters in the [safety quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Safety.ipynb).
*   **Detailed SDK Guide:** For comprehensive details on the Python SDK, refer to the [get started notebook](./Get_started.ipynb) or the official [Python quickstart documentation](https://ai.google.dev/tutorials/python_quickstart).