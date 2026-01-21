# Gemini Native Image Generation Guide

This guide demonstrates how to use Gemini's native image generation models to create, edit, and iterate on images through multimodal prompts. You'll learn to generate images, maintain character consistency, control aspect ratios, and use chat mode for iterative refinement.

## Prerequisites

### 1. Install the SDK
Ensure you have the latest `google-genai` SDK installed.

```bash
pip install -U -q "google-genai>=1.40.0"
```

### 2. Set Up Your API Key
Store your Google AI API key in an environment variable named `GOOGLE_API_KEY`. If you're using Google Colab, you can store it as a secret.

```python
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

### 3. Initialize the Client
Create a client instance with your API key.

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)
```

### 4. Choose a Model
Select the model you want to use. The `gemini-2.5-flash-image` model is recommended for most tasks due to its speed and cost-effectiveness. The `gemini-3-pro-image-preview` offers advanced features like reasoning and higher resolution outputs.

```python
MODEL_ID = "gemini-2.5-flash-image"  # or "gemini-3-pro-image-preview"
```

### 5. Utility Functions
Define helper functions to display and save the model's responses, which can contain both text and image parts.

```python
from IPython.display import display, Markdown
import pathlib

def display_response(response):
    """Display all text and image parts from a response."""
    for part in response.parts:
        if part.thought:  # Skip internal reasoning
            continue
        if part.text:
            display(Markdown(part.text))
        elif image := part.as_image():
            image.show()

def save_image(response, path):
    """Save the last image from a response to a file."""
    for part in response.parts:
        if image := part.as_image():
            image.save(path)
```

## Step 1: Generate Your First Image

Start by creating a simple image. The `generate_content` method works like any Gemini model call. You can specify the expected response modalities, but it's optional for image-generation models.

```python
prompt = 'Create a photorealistic image of a siamese cat with a green left eye and a blue right one and red patches on his face and a black and pink nose'

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
        response_modalities=['Text', 'Image']  # Use ['Image'] for images only
    )
)

display_response(response)
save_image(response, 'cat.png')
```

The model will generate an image matching your description and display it. The image is also saved as `cat.png`.

## Step 2: Edit an Existing Image

You can edit an image by providing it as part of the prompt. The model maintains character consistency, allowing you to place your subject in new scenes.

```python
import PIL

text_prompt = "Create a side view picture of that cat, in a tropical forest, eating a nano-banana, under the stars"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        text_prompt,
        PIL.Image.open('cat.png')
    ]
)

display_response(response)
save_image(response, 'cat_tropical.png')
```

The output will feature the same cat with its distinctive features, now in a tropical setting.

## Step 3: Control the Aspect Ratio

You can specify the aspect ratio of the generated image via the `image_config` parameter. The model defaults to a 1:1 square if no input image is provided.

**Available Aspect Ratios & Resolutions:**

| Aspect Ratio | Resolution | Tokens |
|--------------|------------|--------|
| 1:1          | 1024x1024  | 1290   |
| 2:3          | 832x1248   | 1290   |
| 3:2          | 1248x832   | 1290   |
| 3:4          | 864x1184   | 1290   |
| 4:3          | 1184x864   | 1290   |
| 4:5          | 896x1152   | 1290   |
| 5:4          | 1152x896   | 1290   |
| 9:16         | 768x1344   | 1290   |
| 16:9         | 1344x768   | 1290   |
| 21:9         | 1536x672   | 1290   |

**Example: Generate a 16:9 image**

```python
text_prompt = "Now the cat should keep the same attitude, but be well dressed in fancy restaurant and eat a fancy nano banana."
aspect_ratio = "16:9"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        text_prompt,
        PIL.Image.open('cat_tropical.png')
    ],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
        )
    )
)

display_response(response)
save_image(response, 'cat_restaurant.png')
```

## Step 4: Generate Multiple Images in a Single Call

The model can produce multiple images in one response, useful for creating step-by-step guides or visual stories.

```python
prompt = "Show me how to bake macarons with images"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

display_response(response)
```

The output will include a textual guide interspersed with illustrative images for each step.

## Step 5: Use Chat Mode for Iterative Editing

Chat mode is ideal for iterative image refinement, as it maintains context across turns.

```python
# Start a new chat session
chat = client.chats.create(
    model=MODEL_ID,
)

# First message: create the initial image
message = "create a image of a plastic toy fox figurine in a kid's bedroom, it can have accessories but no weapon"
response = chat.send_message(message)
display_response(response)
save_image(response, "figurine.png")

# Second message: modify the image
message = "Add a blue planet on the figuring's helmet or hat (add one if needed)"
response = chat.send_message(message)
display_response(response)

# Third message: change the scene
message = 'Move that figurine on a beach'
response = chat.send_message(message)
display_response(response)

# Continue iterating...
message = 'Now it should be base-jumping from a spaceship with a wingsuit'
response = chat.send_message(message)
display_response(response)
```

You can also control the aspect ratio within chat mode:

```python
message = "Bring it back to the bedroom"
response = chat.send_message(
    message,
    config=types.GenerateContentConfig(
        image_config=types.ImageConfig(aspect_ratio="16:9"),
    ),
)
display_response(response)
```

## Step 6: Combine Multiple Images

Merge elements from up to three images (with the Flash model) to create a composite scene.

```python
import PIL

text_prompt = "Create a picture of that figurine riding that cat in a fantasy world."

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        text_prompt,
        PIL.Image.open('cat.png'),
        PIL.Image.open('figurine.png')
    ],
)

display_response(response)
```

The model will generate a new image blending the two provided subjects into a single fantasy scene.

## Advanced: Using the Pro Model (`gemini-3-pro-image-preview`)

The Pro model offers enhanced capabilities:
*   **Thinking**: Handles more complex, nuanced requests.
*   **Search Grounding**: Accesses up-to-date information via Google Search.
*   **Higher Resolutions**: Generates images up to 4K.
*   **Extended Language Support**.

Switch your `MODEL_ID` to `"gemini-3-pro-image-preview"` to leverage these features. Note the Pro model has different [pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-flash-image).

## Summary

You've learned how to:
1.  Generate images from text prompts.
2.  Edit existing images while preserving character details.
3.  Control the aspect ratio of generated images.
4.  Produce multiple images in a single call for stories or tutorials.
5.  Use chat mode for an interactive, iterative image creation process.
6.  Combine elements from multiple source images.

For detailed model capabilities and best practices, refer to the official [Gemini API documentation](https://ai.google.dev/gemini-api/docs/image-generation).