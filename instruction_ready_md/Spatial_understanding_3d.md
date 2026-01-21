# Pointing and 3D Spatial Understanding with Gemini (Experimental)

This guide demonstrates how to use Gemini's advanced spatial understanding capabilities, including pointing (2D point prediction) and 3D object detection. These experimental features allow the model to precisely refer to entities in an image and understand their position in three-dimensional space.

**Key Concepts:**
*   **Pointing:** The model can predict 2D points (in `[y, x]` format) on an image to identify specific items.
*   **3D Bounding Boxes:** The model can detect objects in 3D space, returning their center, size, and orientation.
*   **Coordinate System:** All 2D coordinates are normalized to a `1000x1000` grid, where the top-left is `(0,0)` and the bottom-right is `(1000,1000)`.

## Prerequisites & Setup

### 1. Install the SDK
Begin by installing the required Python client library.

```bash
pip install -U -q google-genai
```

### 2. Configure Your API Key
You need a Gemini API key. Store it in an environment variable or a secure secret manager. This example assumes it's stored in a Colab Secret named `GOOGLE_API_KEY`.

```python
from google.colab import userdata
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

### 3. Initialize the Client and Model
Import the necessary modules and create a client instance. We'll use a Gemini 2.0 Flash model for its spatial understanding capabilities.

```python
from google import genai
from google.genai import types
from PIL import Image

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_ID = "gemini-2.5-flash-preview"  # Or "gemini-3-flash-preview"
```

### 4. Download Sample Images
Download the images used in the following examples.

```python
# Download sample images
!wget https://storage.googleapis.com/generativeai-downloads/images/tool.png -O tool.png -q
!wget https://storage.googleapis.com/generativeai-downloads/images/kitchen.jpg -O kitchen.jpg -q
```

## Tutorial: Pointing to Items with Gemini

Instead of drawing bounding boxes, you can ask Gemini to pinpoint specific locations on an image. This is useful for creating less cluttered visualizations.

### Step 1: Analyze an Image and Request Points
Load an image and send it to the model with a prompt that requests points in a specific JSON format. Using a `temperature` above 0 (e.g., 0.5) and limiting the number of items helps prevent repetitive output.

```python
# Load and resize the image
img = Image.open("tool.png")
img = img.resize((800, int(800 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS)

# Analyze the image using Gemini
image_response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        img,
        """
        Point to no more than 10 items in the image, include spill.
        The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...].
        The points are in [y, x] format normalized to 0-1000.
        """
    ],
    config=types.GenerateContentConfig(temperature=0.5)
)

print(image_response.text)
```

The model will return a JSON array like this:
```json
[
  {"point": [130, 760], "label": "handle"},
  {"point": [427, 517], "label": "screw"},
  {"point": [472, 201], "label": "clamp arm"}
]
```

### Step 2: Visualize the Points
To make the results interactive, you can overlay the points on the original image. The following helper function generates an HTML visualization.

```python
import IPython
import base64
from io import BytesIO

def parse_json(json_output):
    """Helper to extract JSON from a markdown code fence."""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

def generate_point_html(pil_image, points_json):
    """Generates HTML to display an image with interactive points overlaid."""
    # Convert PIL image to base64
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    points_json = parse_json(points_json)

    # Return the HTML string (see full function in the source content)
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Point Visualization</title>
        <style>/* CSS styles for points and labels */</style>
    </head>
    <body>
        <div id="container" style="position: relative;">
            <canvas id="canvas" style="background: #000;"></canvas>
            <div id="pointOverlay" class="point-overlay"></div>
        </div>
        <script>
            const pointsData = {points_json};
            // JavaScript to render points and handle hover interactions
        </script>
    </body>
    </html>
    """
    return html_template

# Display the visualization
IPython.display.HTML(generate_point_html(img, image_response.text))
```

This creates an interactive display where you can hover over points to see their labels and highlight them.

## Tutorial: Combining Pointing with Reasoning

You can enhance the basic pointing task by asking the model to reason about the identified items. This combines spatial understanding with descriptive analysis.

### Step 1: Request Detailed Explanations
Modify the prompt to ask *how* to use each part, integrating the explanation into the label.

```python
img = Image.open("tool.png")
img = img.resize((800, int(800 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS)

image_response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        img,
        """
        Pinpoint no more than 10 items in the image.
        The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...].
        The points are in [y, x] format normalized to 0-1000. One element a line.
        Explain how to use each part, put them in the label field, remove duplicated parts and instructions.
        """
    ],
    config=types.GenerateContentConfig(temperature=0.5)
)

IPython.display.HTML(generate_point_html(img, image_response.text))
```

The model's response will now contain labels with instructional text, such as `"handle: Turn to adjust clamping pressure"` instead of just `"handle"`.

### Step 2: Apply to a Safety Use Case
Let's apply this combined approach to a different scenario: identifying potential hazards in a kitchen.

```python
img = Image.open("kitchen.jpg")
img = img.resize((800, int(800 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS)

image_response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        img,
        """
        Point to no more than 10 items in the image.
        The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...].
        The points are in [y, x] format normalized to 0-1000. One element a line.
        Explain how to prevent kids from getting hurt, put them in the label field, remove duplicated parts and instructions.
        """
    ],
    config=types.GenerateContentConfig(temperature=0.5)
)

IPython.display.HTML(generate_point_html(img, image_response.text))
```

The model will identify items like stove knobs or sharp corners and provide safety advice directly in the label (e.g., `"Stove knob: Use safety covers to prevent accidental turning"`).

## Summary

You've learned how to:
1.  **Set up** the Gemini Python client for spatial understanding tasks.
2.  **Request 2D points** from the model to pinpoint items in an image.
3.  **Visualize** these points with an interactive HTML overlay.
4.  **Combine pointing with reasoning** to get descriptive, actionable labels.

These pointing capabilities are experimental and work best with the Gemini 2.0 Flash family of models. For production use-cases requiring higher accuracy, consider using 2D bounding boxes. To explore 3D bounding box detection, refer to the model's documentation and ensure your prompts request the appropriate 3D format.