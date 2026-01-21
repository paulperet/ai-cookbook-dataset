# Gemini Robotics-ER 1.5: A Quickstart Guide

This guide introduces the **Gemini Robotics-ER 1.5** model, a vision-language model (VLM) designed to bring Gemini's agentic capabilities to robotics. It enables advanced reasoning in the physical world, allowing robots to interpret complex visual data, perform spatial reasoning, and plan actions from natural language commands.

**Key Features:**
*   **Enhanced Autonomy:** Robots can reason, adapt, and respond to changes in open-ended environments.
*   **Natural Language Interaction:** Enables complex task assignments using natural language.
*   **Task Orchestration:** Deconstructs commands into subtasks and integrates with existing robot controllers.
*   **Versatile Capabilities:** Locates and identifies objects, understands object relationships, plans grasps and trajectories, and interprets dynamic scenes.

## Prerequisites & Setup

Before you begin, ensure you have a Google API key. This guide assumes you are working in a Google Colab environment.

### 1. Install the SDK
First, install the Google Generative AI Python SDK.

```bash
%pip install -U -q google-genai
```

### 2. Configure Your API Key
Store your API key in a Colab Secret named `GOOGLE_API_KEY`. The following code retrieves it.

```python
from google.colab import userdata

GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
```

### 3. Initialize the Client
Initialize the Gemini SDK client with your API key.

```python
from google import genai

client = genai.Client(api_key=GOOGLE_API_KEY)
```

### 4. Select and Test the Model
Define the model ID and run a simple test to verify the connection.

```python
MODEL_ID = "gemini-robotics-er-1.5-preview"

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Are you there?"
)
print(response.text)
```

### 5. Import Helper Libraries and Utilities
Import the necessary libraries and define helper functions for parsing JSON outputs and handling images.

```python
import json
import textwrap
import time
from PIL import Image
import base64
import dataclasses
from io import BytesIO
import numpy as np
from PIL import ImageColor, ImageDraw, ImageFont
from typing import Tuple
import IPython
from IPython import display

def parse_json(json_output):
    """Parses JSON output, removing any surrounding markdown code fences."""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i + 1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

def get_image_resized(img_path):
    """Resizes an image for faster processing and smaller API payloads."""
    img = Image.open(img_path)
    img = img.resize(
        (800, int(800 * img.size[1] / img.size[0])),
        Image.Resampling.LANCZOS
    )
    return img
```

## Visualization Utilities

The following functions help visualize model outputs like points, bounding boxes, and segmentation masks. You can copy these into your notebook.

### Point Visualization (HTML)
This function generates an interactive HTML visualization for points overlaid on an image.

```python
def generate_point_html(pil_image, points_json):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    points_json = parse_json(points_json)

    # The function returns a large HTML string for visualization.
    # For brevity, the full HTML is omitted here but is included in the source.
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Point Visualization</title>
        <style>...</style>
    </head>
    <body>...</body>
    </html>
    """
    return html_template
```

### Bounding Box Visualization
This function draws bounding boxes and labels directly onto a PIL Image.

```python
def plot_bounding_boxes(img, bounding_boxes):
    """Plots bounding boxes on an image with labels and distinct colors."""
    width, height = img.size
    draw = ImageDraw.Draw(img)

    colors = [
        "red", "green", "blue", "yellow", "orange", "pink", "purple",
        "brown", "gray", "beige", "turquoise", "cyan", "magenta", "lime",
        "navy", "maroon", "teal", "olive", "coral", "lavender", "violet",
        "gold", "silver"
    ] + [c for c in ImageColor.colormap.keys()]

    bounding_boxes = parse_json(bounding_boxes)
    font = ImageFont.truetype("LiberationSans-Regular.ttf", size=14)

    for i, bbox in enumerate(json.loads(bounding_boxes)):
        color = colors[i % len(colors)]
        abs_y1 = int(bbox["box_2d"][0] / 1000 * height)
        abs_x1 = int(bbox["box_2d"][1] / 1000 * width)
        abs_y2 = int(bbox["box_2d"][2] / 1000 * height)
        abs_x2 = int(bbox["box_2d"][3] / 1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
        if "label" in bbox:
            draw.text((abs_x1 + 8, abs_y1 + 6), bbox["label"], fill=color, font=font)
    img.show()
```

### Segmentation Mask Parsing and Overlay
These utilities handle segmentation mask data, parsing it from the model's JSON output and overlaying it on images.

```python
@dataclasses.dataclass(frozen=True)
class SegmentationMask:
    y0: int
    x0: int
    y1: int
    x1: int
    mask: np.array  # Shape: [img_height, img_width], values 0..255
    label: str

def parse_segmentation_masks(predicted_str: str, *, img_height: int, img_width: int) -> list[SegmentationMask]:
    """Parses the model's JSON string output into a list of SegmentationMask objects."""
    items = json.loads(parse_json(predicted_str))
    masks = []
    for item in items:
        abs_y0 = int(item["box_2d"][0] / 1000 * img_height)
        abs_x0 = int(item["box_2d"][1] / 1000 * img_width)
        abs_y1 = int(item["box_2d"][2] / 1000 * img_height)
        abs_x1 = int(item["box_2d"][3] / 1000 * img_width)

        if abs_y0 >= abs_y1 or abs_x0 >= abs_x1:
            continue

        label = item["label"]
        png_str = item["mask"]
        if not png_str.startswith("data:image/png;base64,"):
            continue

        png_str = png_str.removeprefix("data:image/png;base64,")
        png_str = base64.b64decode(png_str)
        mask_img = Image.open(BytesIO(png_str))

        bbox_height = abs_y1 - abs_y0
        bbox_width = abs_x1 - abs_x0
        mask_img = mask_img.resize((bbox_width, bbox_height), resample=Image.Resampling.BILINEAR)

        np_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        np_mask[abs_y0:abs_y1, abs_x0:abs_x1] = mask_img
        masks.append(SegmentationMask(abs_y0, abs_x0, abs_y1, abs_x1, np_mask, label))
    return masks

def overlay_mask_on_img(img: Image, mask: np.ndarray, color: str, alpha: float = 0.7) -> Image.Image:
    """Overlays a single segmentation mask onto a PIL Image with a specified color and transparency."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("Alpha must be between 0.0 and 1.0")

    try:
        color_rgb: Tuple[int, int, int] = ImageColor.getrgb(color)
    except ValueError as e:
        raise ValueError(f"Invalid color name '{color}'. Error: {e}")

    img_rgba = img.convert("RGBA")
    width, height = img_rgba.size

    alpha_int = int(alpha * 255)
    overlay_color_rgba = color_rgb + (alpha_int,)
    colored_mask_layer_np = np.zeros((height, width, 4), dtype=np.uint8)

    mask_np_logical = mask > 127
    # Apply the overlay color where the mask is active
    colored_mask_layer_np[mask_np_logical] = overlay_color_rgba
    colored_mask_layer = Image.fromarray(colored_mask_layer_np, mode="RGBA")

    return Image.alpha_composite(img_rgba, colored_mask_layer)
```

You are now ready to use the Gemini Robotics-ER 1.5 model. The subsequent sections of the original notebook provide specific examples of tasks like object detection, spatial reasoning, and task planning using these utilities.