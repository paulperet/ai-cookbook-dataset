# Guide: High-Fidelity Image Editing with the OpenAI API

This guide demonstrates how to use the `input_fidelity="high"` parameter in the OpenAI Image API to preserve distinctive features from your input images. This setting is essential for editing tasks involving faces, logos, or any detail that must remain consistent in the final output.

## Prerequisites

First, ensure you have the necessary libraries installed.

```bash
pip install pillow openai -U
```

Now, import the required modules and set up your client.

```python
import base64
import os
from io import BytesIO
from PIL import Image
from IPython.display import display, Image as IPImage
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI()
# If your API key isn't set globally, use:
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your-key>"))

# Create a directory to store images
folder_path = "imgs"
os.makedirs(folder_path, exist_ok=True)
```

## Core Utility Functions

We'll define two helper functions: one to resize images for display and another to handle the API call for editing.

```python
def resize_img(image, target_w):
    """Resize an image to a target width while maintaining aspect ratio."""
    w, h = image.size
    target_h = int(round(h * (target_w / float(w))))
    resized_image = image.resize((target_w, target_h), Image.LANCZOS)
    return resized_image

def edit_img(input_img, prompt):
    """
    Edit an image using the OpenAI API with high input fidelity.
    `input_img` can be a file path, bytes, or a tuple for multiple images.
    """
    result = client.images.edit(
        model="gpt-image-1",
        image=input_img,
        prompt=prompt,
        input_fidelity="high",
        quality="high",
        output_format="jpeg"
    )

    # Decode and open the returned image
    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes))
    return image
```

## 1. Precise Editing for Isolated Elements

High input fidelity excels at making subtle, localized changes without affecting the rest of the image. This is perfect for controlled edits.

### 1.1 Edit an Item
Change a specific object, like the color of a mug.

```python
# Assume `edit_input_img` is a valid image file path or bytes object
edit_prompt = "Make the mug olive green"
edit_result = edit_img(edit_input_img, edit_prompt)

# Display the result
edit_resized_result = resize_img(edit_result, 300)
display(edit_resized_result)
```

### 1.2 Remove an Item
Cleanly delete an object from the scene.

```python
remove_prompt = "Remove the mug from the desk"
remove_result = edit_img(edit_input_img, remove_prompt)

# Display the result
remove_resized_result = resize_img(remove_result, 300)
display(remove_resized_result)
```

### 1.3 Add an Item
Insert a new object seamlessly.

```python
add_prompt = "Add a post-it note saying 'Be right back!' to the monitor"
add_result = edit_img(edit_input_img, add_prompt)

# Display the result
add_resized_result = resize_img(add_result, 300)
display(add_resized_result)
```

## 2. Face Preservation

When editing images with people, high input fidelity ensures facial features remain recognizable.

**Important Note:** While all input images benefit from high fidelity, the *first* image you provide retains the richest texture detail. For multiple faces, combine them into a single composite image first.

### 2.1 Edit a Portrait
Apply stylistic changes while preserving the subject's identity.

```python
# Assume `face_input_img` is a valid image file path or bytes object
edit_face_prompt = "Add soft neon purple and lime green lighting and glowing backlighting."
edit_face_result = edit_img(face_input_img, edit_face_prompt)

# Display the result
edit_face_resized_result = resize_img(edit_face_result, 300)
display(edit_face_resized_result)
```

### 2.2 Create an Avatar
Generate a stylized avatar that still resembles the original person.

```python
avatar_prompt = "Generate an avatar of this person in digital art style, with vivid splash of colors."
avatar_result = edit_img(face_input_img, avatar_prompt)

# Display the result
avatar_resized_result = resize_img(avatar_result, 300)
display(avatar_resized_result)
```

### 2.3 Combine Multiple Faces into One Image
To edit multiple faces with optimal fidelity, first merge them into a single image.

```python
def combine_imgs(left_path, right_path, bg_color=(255, 255, 255)):
    """Combine two images side-by-side on a white background."""
    left_img = Image.open(open(left_path, "rb"))
    right_img = Image.open(open(right_path, "rb"))

    # Convert to RGBA to handle transparency
    left = left_img.convert("RGBA")
    right = right_img.convert("RGBA")

    # Resize the right image to match the left image's height
    target_h = left.height
    scale = target_h / float(right.height)
    target_w = int(round(right.width * scale))
    right = right.resize((target_w, target_h), Image.LANCZOS)

    # Create a new canvas
    total_w = left.width + right.width
    canvas = Image.new("RGBA", (total_w, target_h), bg_color + (255,))

    # Paste images onto the canvas
    canvas.paste(left, (0, 0), left)
    canvas.paste(right, (left.width, 0), right)

    return canvas

# Combine the images
combined_img = combine_imgs(second_woman_input_path, face_input_path)
display(combined_img)
```

Now, convert the combined image to bytes and send it to the API.

```python
import io

def pil_to_bytes(img, fmt="PNG"):
    """Convert a PIL Image to bytes."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf

combined_img_bytes = pil_to_bytes(combined_img)

# Edit the combined image
combined_prompt = "Put these two women in the same picture, holding shoulders, as if part of the same photo."
combined_result = edit_img(("combined.png", combined_img_bytes, "image/png"), combined_prompt)

# Display the result
combined_resized_result = resize_img(combined_result, 300)
display(combined_resized_result)
```

## 3. Maintaining Brand Consistency

Keep logos and brand elements intact while generating new marketing materials.

### 3.1 Create a Marketing Banner
Generate a banner featuring your logo.

```python
# Assume `logo_input_img` is a valid image file path or bytes object
marketing_prompt = "Generate a beautiful, modern hero banner featuring this logo in the center. It should look futuristic, with blue & violet hues."
marketing_result = edit_img(logo_input_img, marketing_prompt)

# Display the result
marketing_resized_result = resize_img(marketing_result, 300)
display(marketing_resized_result)
```

### 3.2 Generate a Product Mockup
Place your logo into a realistic scene.

```python
mockup_prompt = "Generate a highly realistic picture of a hand holding a tilted iphone, with an app on the screen that showcases this logo in the center with a loading animation below"
mockup_result = edit_img(logo_input_img, mockup_prompt)

# Display the result
mockup_resized_result = resize_img(mockup_result, 300)
display(mockup_resized_result)
```

### 3.3 Product Photography
Showcase a product in a new setting.

```python
# Assume `bag_input_img` is a valid image file path or bytes object
product_prompt = "Generate a beautiful ad with this bag in the center, on top of a dark background with a glowing halo emanating from the center, behind the bag."
product_result = edit_img(bag_input_img, product_prompt)

# Display the result
product_resized_result = resize_img(product_result, 300)
display(product_resized_result)
```

## 4. Fashion and Product Retouching

Edit clothing, accessories, and products while preserving textures and details.

### 4.1 Change an Outfit
Alter the clothing on a model.

```python
# Assume `model_input_img` is a valid image file path or bytes object
variation_prompt = "Edit this picture so that the model wears a blue tank top instead of the coat and sweater."
variation_result = edit_img(model_input_img, variation_prompt)

# Display the result
variation_resized_result = resize_img(variation_result, 300)
display(variation_resized_result)
```

### 4.2 Add an Accessory
Insert an accessory (like a bag) into a model photo. Provide the face image first for best detail retention.

```python
# Prepare a list of input images. The first image retains the most detail.
input_imgs = [
    ('model.png', open('imgs/model.png', 'rb'), 'image/png'),
    ('bag.png', open('imgs/bag.png', 'rb'), 'image/png'),
]

accessory_prompt = "Add the crossbody bag to the outfit."
accessory_result = edit_img(input_imgs, accessory_prompt)

# Display the result
accessory_resized_result = resize_img(accessory_result, 300)
display(accessory_resized_result)
```

### 4.3 Extract a Product
Isolate a product on a clean background.

```python
extraction_prompt = "Generate a picture of this exact same jacket on a white background"
extraction_result = edit_img(model_input_img, extraction_prompt)

# Display the result
extraction_resized_result = resize_img(extraction_result, 300)
display(extraction_resized_result)
```

## Conclusion

You've learned how to use `input_fidelity="high"` to preserve critical details in images across various use cases: precise editing, face preservation, brand consistency, and fashion retouching.

**Key Reminders:**
1.  High input fidelity consumes more image input tokens than the default setting.
2.  The *first* image in your input list preserves the finest detail and richest texture. Prioritize faces or your most important asset here.

Use the examples above as a starting point and experiment with your own images to see the impact of high-fidelity editing.