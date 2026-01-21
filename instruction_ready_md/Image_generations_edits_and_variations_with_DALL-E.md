# Guide: Using the DALL·E API for Image Generation, Variation, and Editing

This guide walks you through using OpenAI's DALL·E API to generate, vary, and edit images programmatically. You'll learn how to call the three main endpoints: **Generations**, **Variations**, and **Edits**.

## Prerequisites

Before you begin, ensure you have the following:

1.  An OpenAI API key. Set it as an environment variable:
    ```bash
    export OPENAI_API_KEY="your-api-key-here"
    ```
2.  The required Python packages. Install them using pip:
    ```bash
    pip install openai requests pillow
    ```

## Step 1: Initial Setup

First, import the necessary libraries and set up your environment.

```python
# Import required libraries
from openai import OpenAI  # For making API calls
import requests  # For downloading images from URLs
import os  # For file and directory operations
from PIL import Image  # For image manipulation

# Initialize the OpenAI client
# The client will automatically use the OPENAI_API_KEY environment variable.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Define a directory to save your generated images
image_dir_name = "images"
image_dir = os.path.join(os.curdir, image_dir_name)

# Create the directory if it doesn't exist
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

print(f"Images will be saved to: {image_dir}")
```

## Step 2: Generate an Image

The `generations` endpoint creates an image from a text prompt. Here, you'll generate a single image using the DALL·E 3 model.

```python
# Define your creative prompt
prompt = "A cyberpunk monkey hacker dreaming of a beautiful bunch of bananas, digital art"

# Call the DALL·E API
generation_response = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    n=1,  # Number of images to generate
    size="1024x1024",
    response_format="url",  # Returns an image URL
)

# The response contains a URL for the generated image
generated_image_url = generation_response.data[0].url
print(f"Image generated! URL: {generated_image_url}")
```

Now, download and save the generated image to your local directory.

```python
# Define a filename and path for the new image
generated_image_name = "generated_image.png"
generated_image_filepath = os.path.join(image_dir, generated_image_name)

# Download the image from the URL
generated_image = requests.get(generated_image_url).content

# Save the image to a file
with open(generated_image_filepath, "wb") as image_file:
    image_file.write(generated_image)

print(f"Image saved to: {generated_image_filepath}")
```

## Step 3: Create Image Variations

The `variations` endpoint generates new images that are variations of an existing image. You'll create two variations of the image you just generated.

```python
# Call the variations endpoint
variation_response = client.images.create_variation(
    image=generated_image,  # Use the image data downloaded in the previous step
    n=2,  # Create two variations
    size="1024x1024",
    response_format="url",
)

# Extract the URLs for the new variation images
variation_urls = [datum.url for datum in variation_response.data]
```

Download and save each variation.

```python
variation_images = [requests.get(url).content for url in variation_urls]
variation_image_names = [f"variation_image_{i}.png" for i in range(len(variation_images))]

for img, name in zip(variation_images, variation_image_names):
    filepath = os.path.join(image_dir, name)
    with open(filepath, "wb") as f:
        f.write(img)
    print(f"Variation saved: {filepath}")
```

## Step 4: Edit an Image with a Mask

The `edits` endpoint allows you to regenerate a specific part of an image. You need to provide the original image, a text prompt, and a **mask**. The mask is a transparent PNG where transparent areas (alpha=0) indicate the parts of the image to be regenerated.

### Step 4.1: Create a Mask

First, you'll programmatically create a mask. This example creates a mask where the bottom half of the image is transparent.

```python
# Define image dimensions (must match your generated image size)
width = 1024
height = 1024

# Create a new, initially opaque (non-transparent) RGBA image
mask = Image.new("RGBA", (width, height), (0, 0, 0, 255))

# Set the bottom half of the image to be fully transparent
for x in range(width):
    for y in range(height // 2, height):  # Loop over bottom half
        mask.putpixel((x, y), (0, 0, 0, 0))  # Set alpha channel to 0

# Save the mask
mask_name = "bottom_half_mask.png"
mask_filepath = os.path.join(image_dir, mask_name)
mask.save(mask_filepath)
print(f"Mask saved to: {mask_filepath}")
```

### Step 4.2: Perform the Edit

Now, use the original image, the mask, and a prompt to call the edit API.

```python
# Call the edits endpoint
edit_response = client.images.edit(
    image=open(generated_image_filepath, "rb"),  # Original image file
    mask=open(mask_filepath, "rb"),  # Mask file
    prompt=prompt,  # The same prompt used for generation
    n=1,
    size="1024x1024",
    response_format="url",
)

# Get the URL of the edited image
edited_image_url = edit_response.data[0].url
```

Finally, download and save the edited image.

```python
edited_image_name = "edited_image.png"
edited_image_filepath = os.path.join(image_dir, edited_image_name)

edited_image = requests.get(edited_image_url).content
with open(edited_image_filepath, "wb") as image_file:
    image_file.write(edited_image)

print(f"Edited image saved to: {edited_image_filepath}")
```

## Summary

You have successfully used the DALL·E API to:
1.  **Generate** a new image from a text prompt.
2.  Create **variations** of an existing image.
3.  **Edit** a portion of an image by applying a custom mask.

Your `images/` directory should now contain:
-   `generated_image.png`
-   `variation_image_0.png`
-   `variation_image_1.png`
-   `bottom_half_mask.png`
-   `edited_image.png`

You can extend this workflow by experimenting with different prompts, mask shapes, and API parameters like `size`, `style`, and `quality`. For detailed parameter information, refer to the [official DALL·E API documentation](https://platform.openai.com/docs/api-reference/images).