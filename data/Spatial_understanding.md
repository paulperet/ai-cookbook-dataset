# 2D Spatial Understanding with Gemini

This guide demonstrates how to use the Gemini API for object detection and spatial reasoning, similar to the [Spatial Understanding example](https://aistudio.google.com/starter-apps/spatial) in AI Studio. You will learn to detect objects, search within images, and leverage multilingual capabilities.

## Prerequisites

### 1. Install the SDK
First, install the required Python SDK.

```bash
pip install -U "google-genai>=1.16.0"
```

### 2. Set Up Your API Key
Store your Gemini API key in an environment variable named `GOOGLE_API_KEY`.

```python
import os

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
```

### 3. Initialize the Client
Create a client instance with your API key.

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)
```

### 4. Select a Model
For spatial understanding, Gemini 2.0 Flash models are recommended. Some advanced features like segmentation require Gemini 2.5 models.

```python
MODEL_ID = "gemini-2.5-flash-lite"  # Or "gemini-2.5-pro", "gemini-3-flash-preview", etc.
```

### 5. Define System Instructions and Safety Settings
System instructions help format the model's output consistently. We also configure safety settings.

```python
bounding_box_system_instructions = """
    Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
    If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
"""

safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]
```

**Note:** Disable the model's "thinking" feature for this task, as it adds latency without improving detection results.

### 6. Import Required Libraries

```python
from PIL import Image
import io
import json
import random
from io import BytesIO
from PIL import ImageDraw, ImageFont, ImageColor
```

### 7. Download Example Images
We'll use several sample images for demonstration.

```python
import requests

image_urls = {
    "Socks.jpg": "https://storage.googleapis.com/generativeai-downloads/images/socks.jpg",
    "Vegetables.jpg": "https://storage.googleapis.com/generativeai-downloads/images/vegetables.jpg",
    "Japanese_bento.png": "https://storage.googleapis.com/generativeai-downloads/images/Japanese_Bento.png",
    "Cupcakes.jpg": "https://storage.googleapis.com/generativeai-downloads/images/Cupcakes.jpg",
    "Origamis.jpg": "https://storage.googleapis.com/generativeai-downloads/images/origamis.jpg",
    "Fruits.jpg": "https://storage.googleapis.com/generativeai-downloads/images/fruits.jpg",
    "Cat.jpg": "https://storage.googleapis.com/generativeai-downloads/images/cat.jpg",
    "Pumpkins.jpg": "https://storage.googleapis.com/generativeai-downloads/images/pumpkins.jpg",
    "Breakfast.jpg": "https://storage.googleapis.com/generativeai-downloads/images/breakfast.jpg",
    "Bookshelf.jpg": "https://storage.googleapis.com/generativeai-downloads/images/bookshelf.jpg",
    "Spill.jpg": "https://storage.googleapis.com/generativeai-downloads/images/spill.jpg",
}

# Download images
for filename, url in image_urls.items():
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded: {filename}")
```

### 8. Utility Functions
We need helper functions to parse the model's JSON output and draw bounding boxes on images.

```python
def parse_json(json_output: str):
    """Extract JSON from markdown code fences."""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

def plot_bounding_boxes(im, bounding_boxes):
    """
    Draw bounding boxes and labels on an image.
    Coordinates are normalized [y1, x1, y2, x2] where y comes before x.
    """
    img = im.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)

    # Define a color palette
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple',
        'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta', 'lime',
        'navy', 'maroon', 'teal', 'olive', 'coral', 'lavender', 'violet',
        'gold', 'silver'
    ] + [c for c in ImageColor.colormap.keys()]

    # Parse the JSON string
    bounding_boxes = parse_json(bounding_boxes)
    data = json.loads(bounding_boxes)

    # Load a font (install system fonts or provide a .ttf file)
    try:
        font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    except:
        font = ImageFont.load_default()

    for i, box in enumerate(data):
        color = colors[i % len(colors)]

        # Convert normalized coordinates to absolute pixels
        y1 = int(box["box_2d"][0] / 1000 * height)
        x1 = int(box["box_2d"][1] / 1000 * width)
        y2 = int(box["box_2d"][2] / 1000 * height)
        x2 = int(box["box_2d"][3] / 1000 * width)

        # Ensure correct ordering
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        # Draw rectangle and label
        draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=4)
        if "label" in box:
            draw.text((x1 + 8, y1 + 6), box["label"], fill=color, font=font)

    return img
```

---

## Step 1: Basic Object Detection with Overlaid Information

Let's start with a simple example: detecting and labeling cupcakes in an image.

1.  **Load and prepare the image.**

    ```python
    image_path = "Cupcakes.jpg"
    im = Image.open(image_path)
    im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
    ```

2.  **Define your detection prompt.** Be specific about what you want to detect and how to label it.

    ```python
    prompt = "Detect the 2d bounding boxes of the cupcakes (with 'label' as topping description)"
    ```

3.  **Call the Gemini API.** Pass the prompt, image, system instructions, and configuration.

    ```python
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[prompt, im],
        config=types.GenerateContentConfig(
            system_instruction=bounding_box_system_instructions,
            temperature=0.5,  # Adds variability to prevent repetitive outputs
            safety_settings=safety_settings,
            thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disable thinking
        )
    )
    ```

4.  **Examine the raw output.** The model returns a JSON array with `box_2d` coordinates (normalized, y-first) and labels.

    ```json
    [
      {"box_2d": [390, 64, 574, 203], "label": "red sprinkles"},
      {"box_2d": [382, 250, 537, 369], "label": "pink and blue sprinkles"},
      ...
    ]
    ```

5.  **Visualize the results.** Use the utility function to draw the bounding boxes.

    ```python
    annotated_image = plot_bounding_boxes(im, response.text)
    annotated_image.show()  # or save with annotated_image.save("output.jpg")
    ```

## Step 2: Searching Within an Image

You can use natural language queries to find specific objects.

1.  **Choose a different image and a search prompt.**

    ```python
    image_path = "Socks.jpg"
    im = Image.open(image_path)
    im.thumbnail([640, 640], Image.Resampling.LANCZOS)

    prompt = "Show me the positions of the socks with the face"
    ```

2.  **Generate the detection.** Use the same API call pattern.

    ```python
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[prompt, im],
        config=types.GenerateContentConfig(
            system_instruction=bounding_box_system_instructions,
            temperature=0.5,
            safety_settings=safety_settings,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    print(response.text)
    ```

    The output will be a JSON array pinpointing the requested objects.

3.  **Plot the bounding boxes.**

    ```python
    annotated_image = plot_bounding_boxes(im, response.text)
    annotated_image.show()
    ```

**Experiment:** Try different prompts like "Detect all rainbow socks" or "Find the sock that goes with the one at the top."

## Step 3: Multilingual Spatial Understanding

Gemini can understand and generate labels in multiple languages. Let's analyze a Japanese bento box image.

1.  **Load the image.**

    ```python
    image_path = "Japanese_bento.png"
    im = Image.open(image_path)
    im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
    ```

2.  **Create a multilingual prompt.** Ask the model to label items in both Japanese and English.

    ```python
    prompt = """
    Detect all food items in this bento box.
    For each item, provide a label in Japanese characters followed by its English translation in parentheses.
    """
    ```

3.  **Run the detection.**

    ```python
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[prompt, im],
        config=types.GenerateContentConfig(
            system_instruction=bounding_box_system_instructions,
            temperature=0.5,
            safety_settings=safety_settings,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
    )
    print(response.text)
    ```

    Example output:
    ```json
    [
      {"box_2d": [120, 150, 280, 300], "label": "たまごやき (Tamagoyaki)"},
      {"box_2d": [350, 200, 500, 350], "label": "ブロッコリー (Broccoli)"},
      ...
    ]
    ```

4.  **Visualize the multilingual labels.**

    ```python
    annotated_image = plot_bounding_boxes(im, response.text)
    annotated_image.show()
    ```

    **Note:** For proper rendering of Japanese characters, ensure your system has appropriate CJK fonts installed. The `plot_bounding_boxes` function attempts to use the Noto Sans CJK font.

## Summary

You've learned how to use the Gemini API for 2D spatial understanding:
1.  **Basic Detection:** Extract objects with bounding boxes and custom labels.
2.  **Visual Search:** Use natural language queries to locate specific items within an image.
3.  **Multilingual Analysis:** Combine spatial reasoning with translation and understanding across languages.

**Key Tips:**
*   Use a `temperature > 0` (e.g., 0.5) to encourage varied outputs.
*   Limit the number of objects in your system instructions to prevent excessive processing.
*   Always set `thinking_budget=0` for object detection tasks to minimize latency.
*   Experiment with different prompts and system instructions to optimize for your specific use case.

You can now apply these techniques to your own images and detection scenarios.