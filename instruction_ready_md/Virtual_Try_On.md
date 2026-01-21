# Virtual Try-On with Gemini 2.5 & Imagen 3: A Step-by-Step Guide

This guide walks you through building a virtual try-on application. You'll use Gemini 2.5 to create precise segmentation masks of clothing items and then use Imagen 3 to edit the image, replacing the item with a new one based on your description.

## Prerequisites

Before you begin, ensure you have the following:

1.  **Gemini API Access:** You need a valid Gemini API key.
2.  **Google Cloud Platform (GCP) Account:** An active GCP account with a billing project and the Vertex AI API enabled is required for Imagen 3.
3.  **Environment:** This tutorial is designed for **Google Colab**. Some commands are specific to this environment.

## Step 1: Setup and Installation

First, install the necessary Python library and authenticate your environment.

```bash
pip install -U -q google-genai
```

```python
# Authenticate within Google Colab
from google.colab import auth
auth.authenticate_user()
```

## Step 2: Import Required Libraries

Import all the libraries you'll need for image processing, API calls, and model interactions.

```python
import cv2
import numpy as np
from PIL import Image as PILImage
from matplotlib import pyplot as plt
from google import genai
from google.genai import types
import base64
import json
import io

import vertexai
from vertexai.preview.vision_models import (
    Image,
    ImageGenerationModel,
    MaskReferenceImage,
    RawReferenceImage,
)
```

## Step 3: Configure API Keys and Model

Configure your Gemini API key and GCP Project ID. These should be stored as secrets in Colab named `GOOGLE_API_KEY` and `GCP_PROJECT_ID`, respectively.

```python
from google.colab import userdata

# Configure Gemini API
GEMINI_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)

# Configure GCP Project for Vertex AI (Imagen 3)
GCP_PROJECT_ID = userdata.get('GCP_PROJECT_ID')

# Select the Gemini model for segmentation
MODEL_ID = "gemini-2.5-flash-preview"
```

## Step 4: Define Helper Functions for Mask Generation

You'll need functions to parse the model's JSON response and convert it into a usable image mask.

```python
def parse_json(text: str) -> str:
    """Cleans JSON string from potential code fence markers."""
    return text.strip().removeprefix("```json").removesuffix("```")

def generate_mask(predicted_str: str, *, img_height: int, img_width: int) -> list[tuple[np.ndarray, str]]:
    """
    Converts Gemini's JSON mask output into a list of (mask_array, label) tuples.
    """
    try:
        items = json.loads(parse_json(predicted_str))
        if not isinstance(items, list):
            print("Error: Parsed JSON is not a list.")
            return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during JSON parsing: {e}")
        return []

    segmentation_data = []
    for item in items:
        if not isinstance(item, dict) or "box_2d" not in item or "mask" not in item:
            continue

        label = item.get("label", "unknown")
        png_str = item["mask"]
        if not png_str.startswith("data:image/png;base64,"):
            continue

        # Decode the base64 mask image
        png_str = png_str.removeprefix("data:image/png;base64,")
        try:
            png_bytes = base64.b64decode(png_str)
            bbox_mask = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if bbox_mask is None:
                continue
        except Exception as e:
            print(f"Error decoding mask: {e}")
            continue

        # Process bounding box coordinates (normalized 0-1000 to pixel values)
        box = item["box_2d"]
        y0_norm, x0_norm, y1_norm, x1_norm = map(float, box)
        abs_y0 = max(0, min(int(y0_norm / 1000.0 * img_height), img_height - 1))
        abs_x0 = max(0, min(int(x0_norm / 1000.0 * img_width), img_width - 1))
        abs_y1 = max(0, min(int(y1_norm / 1000.0 * img_height), img_height))
        abs_x1 = max(0, min(int(x1_norm / 1000.0 * img_width), img_width))

        bbox_height = abs_y1 - abs_y0
        bbox_width = abs_x1 - abs_x0
        if bbox_height <= 0 or bbox_width <= 0:
            continue

        # Resize the mask to match the bounding box dimensions
        resized_bbox_mask = cv2.resize(
            bbox_mask, (bbox_width, bbox_height), interpolation=cv2.INTER_LINEAR
        )

        # Place the resized mask into a full-image-sized canvas
        full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        full_mask[abs_y0:abs_y1, abs_x0:abs_x1] = resized_bbox_mask
        segmentation_data.append((full_mask, label))

    return segmentation_data

def create_binary_mask_overlay(
    img: PILImage.Image,
    segmentation_data: list[tuple[np.ndarray, str]],
) -> np.ndarray:
    """
    Combines multiple segmentation masks into a single binary mask image.
    """
    binary_mask = np.zeros(img.size[::-1], dtype=np.uint8)
    for mask, label in segmentation_data:
        if mask is not None and mask.shape == binary_mask.shape:
            binary_mask = np.maximum(binary_mask, mask)
    result = np.zeros_like(binary_mask, dtype=np.uint8)
    result[binary_mask > 0] = 255
    return result
```

## Step 5: Generate a Segmentation Mask with Gemini 2.5

Now, use Gemini 2.5 to identify and create a mask for a specific clothing item in an image.

First, download a sample image.

```bash
wget -q https://storage.googleapis.com/generativeai-downloads/images/Virtual_try_on_person.png -O /content/image_01.png
```

Next, define the target object and generate the mask.

```python
# Configuration
input_image = 'image_01.png'
object_to_segment = 'hoodie'  # The clothing item you want to replace
image_path = f"/content/{input_image}"

# Load the image
img = PILImage.open(image_path)
img_height, img_width = img.size[1], img.size[0]

# Craft the prompt for Gemini
prompt = f"Give the segmentation masks for {object_to_segment}. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key 'box_2d', the segmentation mask in key 'mask', and the text label in the key 'label'."

# Call the Gemini API
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[prompt, img],
    config=types.GenerateContentConfig(
        temperature=0.5,
        safety_settings=[types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH")],
    )
)

# Process the response to create the mask
result = response.text
segmentation_data = generate_mask(result, img_height=img_height, img_width=img_width)

# Save the final binary mask
if segmentation_data:
    binary_mask = create_binary_mask_overlay(img, segmentation_data)
    mask_file_path = f"/content/annotation_mask_{input_image}"
    cv2.imwrite(mask_file_path, binary_mask)
    print(f"Mask saved to: {mask_file_path}")
else:
    print("No segmentation mask found.")
```

**Visualize the Result:** Let's display the original image alongside the generated mask to verify accuracy.

```python
img_array = np.array(img)
binary_array = np.array(binary_mask)
# Convert the 2D mask to 3D (RGB) for display
binary_array_3d = np.repeat(np.expand_dims(binary_array, axis=2), 3, axis=2)

# Create a side-by-side comparison
gap = np.full((img_array.shape[0], 20, 3), 255, dtype=img_array.dtype)
combined_image = np.hstack((img_array, gap, binary_array_3d))

plt.figure(figsize=(15, 10))
plt.imshow(combined_image)
plt.axis('off')
plt.show()
```

## Step 6: Configure Imagen 3 for Inpainting

With the mask ready, configure the parameters for Imagen 3's image editing (inpainting) task.

```python
# Paths
mask_file = f"/content/annotation_mask_{input_image}"
output_file = f"/content/output_{input_image}"

# Inpainting Parameters
prompt = "A dark green jacket, white shirt inside it"  # Describe the new clothing item
edit_mode = 'inpainting-insert'  # Options: 'inpainting-insert', 'outpainting', 'inpainting-remove'
mask_mode = 'foreground'          # We are editing the area defined by the mask (the hoodie)
dilation = 0.01                   # Slightly expand the mask boundary for a smoother blend
```

## Step 7: Generate the Final Image with Imagen 3

Finally, use Imagen 3 to perform the inpainting, replacing the old clothing item with the new one described in your prompt.

```python
# Initialize Vertex AI
vertexai.init(project=GCP_PROJECT_ID, location="us-central1")

# Load the Imagen 3 model
edit_model = ImageGenerationModel.from_pretrained("imagen-3.0-capability-001")

# Load the base image and the mask
base_img = Image.load_from_file(location=image_path)
mask_img = Image.load_from_file(location=mask_file)

# Create reference images for the model
raw_ref_image = RawReferenceImage(image=base_img, reference_id=0)
mask_ref_image = MaskReferenceImage(
    reference_id=1, image=mask_img, mask_mode=mask_mode, dilation=dilation
)

# Generate the edited image
edited_image = edit_model.edit_image(
    prompt=prompt,
    edit_mode=edit_mode,
    reference_images=[raw_ref_image, mask_ref_image],
    number_of_images=1,
    safety_filter_level="block_some",
    person_generation="allow_adult",
)

# Save the result
edited_image[0].save(output_file)
print(f"Final image saved to: {output_file}")
```

**View Your Result:**

```python
output_img = PILImage.open(output_file)
display(output_img)
```

## Conclusion

Congratulations! You have successfully built a virtual try-on pipeline. You used **Gemini 2.5** for advanced visual understanding to create a precise segmentation mask and then leveraged **Imagen 3**'s powerful inpainting capabilities to edit the image realistically.

**Key Takeaways:**
*   The quality of the final output heavily depends on the accuracy of the initial mask.
*   Detailed and specific prompts for Imagen 3 yield better results.
*   Experimenting with parameters like `dilation` and `edit_mode` can help refine the output.

You can now experiment with different images, clothing items, and descriptive prompts to create various virtual try-on scenarios.