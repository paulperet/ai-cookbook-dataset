# Guide: How to Pass Images to Claude 3 Models

This guide demonstrates how to send images to Anthropic's Claude 3 model family via the API. You'll learn two methods: using local image files and using images from URLs.

## Prerequisites

First, install the required packages:

```bash
pip install anthropic httpx
```

## Method 1: Using a Local Image File

### Step 1: Prepare Your Image

Ensure you have an image file available. In this example, we'll use a file named `sunset.jpeg` located in a relative `../images/` directory.

### Step 2: Encode the Image

Claude's API requires images to be base64-encoded. The following code reads a local file and converts it to the required format.

```python
import base64
from anthropic import Anthropic

# Initialize the Anthropic client
client = Anthropic()
MODEL_NAME = "claude-3-opus-20240229"  # Or your preferred Claude 3 model

# Read and encode the image
with open("../images/sunset.jpeg", "rb") as image_file:
    binary_data = image_file.read()
    base64_string = base64.b64encode(binary_data).decode("utf-8")
```

### Step 3: Construct the Message

Create a message list following the Anthropic API format. The `content` field is a list that can contain both image and text objects.

```python
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_string
                }
            },
            {
                "type": "text",
                "text": "Write a sonnet based on this image."
            }
        ]
    }
]
```

### Step 4: Call the API and Print the Response

Send the message to Claude and extract the text response.

```python
response = client.messages.create(
    model=MODEL_NAME,
    max_tokens=2048,
    messages=message_list
)

print(response.content[0].text)
```

**Example Output:**
```
Upon the rocky shore, a beacon bright,
Its steadfast light a guide through darkest night.
While sun descends in hues of pink and red,
The lighthouse stands, a stalwart figure head.

The waves crash 'gainst the weathered stone below,
A ceaseless rhythm, ancient ebb and flow.
Yet still the tower remains, resolute,
A guardian watching, ever vigilant mute.

The vast expanse of sea and sky surround,
Horizon's line where heaven meets the ground.
This timeless scene, a testament to might,
Of nature's power and the human fight.

The lighthouse, proud amid the fading day,
Eternal symbol, showing safe the way.
```

## Method 2: Using an Image from a URL

If your image is hosted online, you can fetch and encode it directly.

### Step 1: Define the Image URL

```python
IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Machu_Picchu%2C_Peru_%282018%29.jpg/2560px-Machu_Picchu%2C_Peru_%282018%29.jpg"
```

### Step 2: Download and Encode the Image

Use the `httpx` library to download the image and encode it.

```python
import httpx

# Fetch the image and encode it
response = httpx.get(IMAGE_URL)
IMAGE_DATA = base64.b64encode(response.content).decode("utf-8")
```

### Step 3: Construct and Send the Message

The message structure is identical to the local file method.

```python
message_list = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": IMAGE_DATA
                }
            },
            {
                "type": "text",
                "text": "Describe this image in two sentences."
            }
        ]
    }
]

response = client.messages.create(
    model=MODEL_NAME,
    max_tokens=2048,
    messages=message_list
)

print(response.content[0].text)
```

**Example Output:**
```
The image depicts the ancient Inca city of Machu Picchu, perched high in the Andes Mountains of Peru. The well-preserved stone ruins, including terraces, plazas, and buildings, are set against a stunning backdrop of steep, verdant mountains under a partly cloudy sky.
```

## Key Takeaways

1.  **Encoding is Required:** All images must be base64-encoded before being sent to the Claude API.
2.  **Message Structure:** The `content` field is a list that can mix `image` and `text` objects, allowing you to provide context alongside the visual input.
3.  **Media Type:** Remember to specify the correct `media_type` (e.g., `image/jpeg`, `image/png`) in the image source.
4.  **Model Support:** Ensure you are using a Claude 3 model (e.g., `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`), as earlier model versions do not support image inputs.

You can now integrate image analysis into your applications by following these patterns.