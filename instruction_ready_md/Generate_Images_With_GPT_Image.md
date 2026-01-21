# Generate and Edit Images with GPT Image

In this guide, you'll learn how to use GPT Image, a large language model with advanced image generation capabilities. This model excels at instruction-following and can produce photorealistic images based on detailed descriptions.

## Prerequisites

Before you begin, ensure you have the necessary Python packages installed.

```bash
pip install pillow openai -U
```

## Setup

Import the required libraries and initialize the OpenAI client.

```python
import base64
import os
from openai import OpenAI
from PIL import Image
from io import BytesIO
from IPython.display import Image as IPImage, display

# Initialize the OpenAI client
client = OpenAI()
# Alternatively, if your API key is not set globally:
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your-key>"))

# Create a directory to store generated images
folder_path = "imgs"
os.makedirs(folder_path, exist_ok=True)
```

## Step 1: Generate an Image

GPT Image is highly adept at following detailed instructions. Let's start by generating an image of a fictional character.

### 1.1 Define the Prompt

Create a detailed prompt describing the character.

```python
prompt1 = """
Render a realistic image of this character:
Blobby Alien Character Spec Name: Glorptak (or nickname: "Glorp")
Visual Appearance Body Shape: Amorphous and gelatinous. Overall silhouette resembles a teardrop or melting marshmallow, shifting slightly over time. Can squish and elongate when emotional or startled.
Material Texture: Semi-translucent, bio-luminescent goo with a jelly-like wobble. Surface occasionally ripples when communicating or moving quickly.
Color Palette:
- Base: Iridescent lavender or seafoam green
- Accents: Subsurface glowing veins of neon pink, electric blue, or golden yellow
- Mood-based color shifts (anger = dark red, joy = bright aqua, fear = pale gray)
Facial Features:
- Eyes: 3–5 asymmetrical floating orbs inside the blob that rotate or blink independently
- Mouth: Optional—appears as a rippling crescent on the surface when speaking or emoting
- No visible nose or ears; uses vibration-sensitive receptors embedded in goo
- Limbs: None by default, but can extrude pseudopods (tentacle-like limbs) when needed for interaction or locomotion. Can manifest temporary feet or hands.
Movement & Behavior Locomotion:
- Slides, bounces, and rolls.
- Can stick to walls and ceilings via suction. When scared, may flatten and ooze away quickly.
Mannerisms:
- Constant wiggling or wobbling even at rest
- Leaves harmless glowing slime trails
- Tends to absorb nearby small objects temporarily out of curiosity
"""

img_path1 = "imgs/glorptak.jpg"
```

### 1.2 Generate and Save the Image

Call the API to generate the image, then save and display it.

```python
# Generate the image
result1 = client.images.generate(
    model="gpt-image-1",
    prompt=prompt1,
    size="1024x1024"
)

# Decode and save the image
image_base64 = result1.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

image = Image.open(BytesIO(image_bytes))
image = image.resize((300, 300), Image.LANCZOS)
image.save(img_path1, format="JPEG", quality=80, optimize=True)

# Display the result
display(IPImage(img_path1))
```

## Step 2: Customize the Output

You can customize several properties of the generated image, including quality, size, compression, and background.

### 2.1 Adjust Quality and Size

Let's generate a pixel-art portrait with specific settings.

```python
prompt2 = "generate a portrait, pixel-art style, of a grey tabby cat dressed as a blond woman on a dark background."
img_path2 = "imgs/cat_portrait_pixel.jpg"

# Generate with custom settings
result2 = client.images.generate(
    model="gpt-image-1",
    prompt=prompt2,
    quality="low",
    output_compression=50,
    output_format="jpeg",
    size="1024x1536"
)

# Save and display
image_base64 = result2.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

image = Image.open(BytesIO(image_bytes))
image = image.resize((250, 375), Image.LANCZOS)
image.save(img_path2, format="JPEG", quality=80, optimize=True)

display(IPImage(img_path2))
```

### 2.2 Create an Image with a Transparent Background

You can request a transparent background by specifying the format as PNG or WEBP.

```python
prompt3 = "generate a pixel-art style picture of a green bucket hat with a pink quill on a transparent background."
img_path3 = "imgs/hat.png"

result3 = client.images.generate(
    model="gpt-image-1",
    prompt=prompt3,
    quality="low",
    output_format="png",
    size="1024x1024"
)

image_base64 = result3.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

image = Image.open(BytesIO(image_bytes))
image = image.resize((250, 250), Image.LANCZOS)
image.save(img_path3, format="PNG")

display(IPImage(img_path3))
```

## Step 3: Edit Existing Images

GPT Image can accept input images to create new compositions. You can provide up to 10 images.

### 3.1 Combine Multiple Images

Let's combine the cat and hat images into a new scene.

```python
prompt_edit = """
Combine the images of the cat and the hat to show the cat wearing the hat while being perched in a tree, still in pixel-art style.
"""
img_path_edit = "imgs/cat_with_hat.jpg"

# Open the input images
img1 = open(img_path2, "rb")
img2 = open(img_path3, "rb")

# Generate the edited image
result_edit = client.images.edit(
    model="gpt-image-1",
    image=[img1, img2],
    prompt=prompt_edit,
    size="1024x1536"
)

# Save and display
image_base64 = result_edit.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

image = Image.open(BytesIO(image_bytes))
image = image.resize((250, 375), Image.LANCZOS)
image.save(img_path_edit, format="JPEG", quality=80, optimize=True)

display(IPImage(img_path_edit))
```

## Step 4: Edit an Image with a Mask

You can use a mask to protect specific parts of an image from being edited. The mask should contain an alpha channel.

### 4.1 Generate a Mask

First, let's generate a mask for our Glorptak character.

```python
img_path_mask = "imgs/mask.png"
prompt_mask = "generate a mask delimiting the entire character in the picture, using white where the character is and black for the background. Return an image in the same size as the input image."

img_input = open(img_path1, "rb")

# Generate the mask
result_mask = client.images.edit(
    model="gpt-image-1",
    image=img_input,
    prompt=prompt_mask
)

# Save the mask
image_base64 = result_mask.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

image = Image.open(BytesIO(image_bytes))
image = image.resize((300, 300), Image.LANCZOS)
image.save(img_path_mask, format="PNG")

display(IPImage(img_path_mask))
```

### 4.2 Add an Alpha Channel to the Mask

Convert the black-and-white mask to an RGBA image with an alpha channel.

```python
# Load the mask as a grayscale image
mask = Image.open(img_path_mask).convert("L")

# Convert to RGBA to add an alpha channel
mask_rgba = mask.convert("RGBA")

# Use the mask itself as the alpha channel
mask_rgba.putalpha(mask)

# Save the mask with alpha channel
buf = BytesIO()
mask_rgba.save(buf, format="PNG")
mask_bytes = buf.getvalue()

img_path_mask_alpha = "imgs/mask_alpha.png"
with open(img_path_mask_alpha, "wb") as f:
    f.write(mask_bytes)
```

### 4.3 Edit the Image Using the Mask

Now, apply the mask to edit only the background of the original image.

```python
prompt_mask_edit = "A strange character on a colorful galaxy background, with lots of stars and planets."
mask = open(img_path_mask_alpha, "rb")

result_mask_edit = client.images.edit(
    model="gpt-image-1",
    prompt=prompt_mask_edit,
    image=img_input,
    mask=mask,
    size="1024x1024"
)

# Save and display the result
img_path_mask_edit = "imgs/mask_edit.png"

image_base64 = result_mask_edit.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

image = Image.open(BytesIO(image_bytes))
image = image.resize((300, 300), Image.LANCZOS)
image.save(img_path_mask_edit, format="JPEG", quality=80, optimize=True)

display(IPImage(img_path_mask_edit))
```

## Conclusion

In this tutorial, you learned how to:

1. Generate images from detailed text prompts using GPT Image.
2. Customize output properties like quality, size, and background.
3. Edit images by combining multiple inputs.
4. Use masks to control which parts of an image are edited.

You can use these techniques as a foundation for more complex image generation and editing workflows. For further inspiration, explore the [image gallery](https://platform.openai.com/docs/guides/image-generation?image-generation-model=gpt-image-1&gallery=open#generate-images) in the OpenAI documentation.

Happy building!