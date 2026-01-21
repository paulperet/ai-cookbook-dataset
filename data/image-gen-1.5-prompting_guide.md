# GPT-Image 1.5: A Practical Prompting Guide

## Introduction

`gpt-image-1.5` is a state-of-the-art image generation model designed for production-quality visuals and highly controllable creative workflows. It offers significant improvements in realism, accuracy, and editability, making it suitable for professional design tasks and iterative content creation.

This guide provides practical prompting patterns, best practices, and example workflows drawn from real production use cases.

## Prerequisites & Setup

Before you begin, ensure you have the OpenAI Python client installed and your API key configured. You'll also need to set up directories for your input and output images.

1.  **Install the OpenAI client** (if not already installed):
    ```bash
    pip install openai
    ```

2.  **Configure your API key** in your environment.

3.  **Run the following setup script** to create the necessary directories and a helper function for saving images.

```python
import os
import base64
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI()

# Create directories for input and output images
os.makedirs("../../images/input_images", exist_ok=True)
os.makedirs("../../images/output_images", exist_ok=True)

def save_image(result, filename: str) -> None:
    """
    Saves the first returned image to the given filename inside the output_images folder.
    """
    image_base64 = result.data[0].b64_json
    out_path = os.path.join("../../images/output_images", filename)
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(image_base64))
```

**Note:** Place any reference images you plan to use for edits into the `../../images/input_images/` directory.

## Core Prompting Principles

Before diving into specific use cases, let's review the fundamental principles for crafting effective prompts for `gpt-image-1.5`.

*   **Structure Your Prompt:** Write prompts in a logical order: describe the background/scene, then the main subject, followed by key details, and finally any constraints. For complex requests, use line breaks or short, labeled segments.
*   **Be Specific and Concrete:** Use precise language for materials, textures, shapes, and the visual medium (e.g., "photo," "watercolor," "3D render"). For photorealism, use photography terms like lens type, aperture, and lighting rather than generic terms like "8K."
*   **Control Composition:** Specify framing (close-up, wide shot), perspective (eye-level, low-angle), and lighting (soft diffuse, golden hour) to direct the final image.
*   **State Constraints Explicitly:** Clearly define what should *not* change. Use phrases like "no watermark," "preserve identity," or "keep everything else the same" to maintain control.
*   **Handle Text Carefully:** Place literal text in **quotes** or **ALL CAPS**. Specify font style, size, color, and placement. For tricky words, spell them out letter-by-letter.
*   **Iterate, Don't Overload:** Start with a clean base prompt. Refine with small, single-change follow-ups like "make lighting warmer" or "remove the extra tree."

## Use Case 1: Generate (Text → Image)

### 1.1 Creating Detailed Infographics

Infographics are excellent for explaining structured information. For layouts with dense text, it's recommended to set the generation quality to `"high"`.

```python
prompt = """
Create a detailed Infographic of the functioning and flow of an automatic coffee machine like a Jura.
From bean basket, to grinding, to scale, water tank, boiler, etc.
I'd like to understand technically and visually the flow.
"""

result = client.images.generate(
    model="gpt-image-1.5",
    prompt=prompt
)

save_image(result, "infographic_coffee_machine.png")
```

### 1.2 Generating Photorealistic Images

To achieve believable photorealism, prompt as if you are directing a real photoshoot. Use photography language and ask for authentic textures and imperfections.

```python
prompt = """
Create a photorealistic candid photograph of an elderly sailor standing on a small fishing boat.
He has weathered skin with visible wrinkles, pores, and sun texture, and a few faded traditional sailor tattoos on his arms.
He is calmly adjusting a net while his dog sits nearby on the deck. Shot like a 35mm film photograph, medium close-up at eye level, using a 50mm lens.
Soft coastal daylight, shallow depth of field, subtle film grain, natural color balance.
The image should feel honest and unposed, with real skin texture, worn materials, and everyday detail. No glamorization, no heavy retouching.
"""

result = client.images.generate(
    model="gpt-image-1.5",
    prompt=prompt,
    quality="high"  # Use high quality for critical detail
)

save_image(result, "photorealism.png")
```

### 1.3 Leveraging World Knowledge

The model possesses strong world knowledge and reasoning. You can describe a scene contextually, and it will infer the appropriate details.

```python
prompt = """
Create a realistic outdoor crowd scene in Bethel, New York on August 16, 1969.
Photorealistic, period-accurate clothing, staging, and environment.
"""

result = client.images.generate(
    model="gpt-image-1.5",
    prompt=prompt
)

save_image(result, "world_knowledge.png")
```

### 1.4 Generating Logo Variations

For logo generation, focus on brand personality and scalability. Use the `n` parameter to request multiple variations.

```python
prompt = """
Create an original, non-infringing logo for a company called Field & Flour, a local bakery.
The logo should feel warm, simple, and timeless. Use clean, vector-like shapes, a strong silhouette, and balanced negative space.
Favor simplicity over detail so it reads clearly at small and large sizes. Flat design, minimal strokes, no gradients unless essential.
Plain background. Deliver a single centered logo with generous padding. No watermark.
"""

result = client.images.generate(
    model="gpt-image-1.5",
    prompt=prompt,
    n=4  # Generate 4 different logo concepts
)

# Save all generated logos
for i, item in enumerate(result.data, start=1):
    image_base64 = item.b64_json
    image_bytes = base64.b64decode(image_base64)
    with open(f"../../images/output_images/logo_generation_{i}.png", "wb") as f:
        f.write(image_bytes)
```

### 1.5 Creating a Comic Strip from a Story

Define your narrative as a sequence of clear visual beats, one per panel, to create a coherent comic strip.

```python
prompt = """
Create a short vertical comic-style reel with 4 equal-sized panels.
Panel 1: The owner leaves through the front door. The pet is framed in the window behind them, small against the glass, eyes wide, paws pressed high, the house suddenly quiet.
Panel 2: The door clicks shut. Silence breaks. The pet slowly turns toward the empty house, posture shifting, eyes sharp with possibility.
Panel 3: The house transformed. The pet sprawls across the couch like it owns the place, crumbs nearby, sunlight cutting across the room like a spotlight.
Panel 4: The door opens. The pet is seated perfectly by the entrance, alert and composed, as if nothing happened.
"""

result = client.images.generate(
    model="gpt-image-1.5",
    prompt=prompt
)

save_image(result, "comic_reel.png")
```

## Use Case 2: Edit (Text + Image → Image)

### 2.1 Performing Style Transfer

Apply the visual style (palette, texture, brushwork) of one image to a new subject or scene.

```python
prompt = """
Use the same style from the input image and generate a man riding a motorcycle on a white background.
"""

result = client.images.edit(
    model="gpt-image-1.5",
    image=[
        open("../../images/input_images/pixels.png", "rb"),
    ],
    prompt=prompt
)

save_image(result, "motorcycle.png")
```

### 2.2 Virtual Clothing Try-On

This is ideal for e-commerce. The key is to lock the person's identity (face, pose, body shape) and change only the garments.

```python
prompt = """
Edit the image to dress the woman using the provided clothing images. Do not change her face, facial features, skin tone, body shape, pose, or identity in any way. Preserve her exact likeness, expression, hairstyle, and proportions. Replace only the clothing, fitting the garments naturally to her existing pose and body geometry with realistic fabric behavior. Match lighting, shadows, and color temperature to the original photo so the outfit integrates photorealistically, without looking pasted on. Do not change the background, camera angle, framing, or image quality, and do not add accessories, text, logos, or watermarks.
"""

result = client.images.edit(
    model="gpt-image-1.5",
    image=[
        open("../../images/input_images/woman_in_museum.png", "rb"),
        open("../../images/input_images/tank_top.png", "rb"),
        open("../../images/input_images/jacket.png", "rb"),
        open("../../images/input_images/tank_top.png", "rb"),
        open("../../images/input_images/boots.png", "rb"),
    ],
    prompt=prompt
)

save_image(result, "outfit.png")
```

### 2.3 Converting a Sketch to a Rendered Image

Turn a rough drawing into a photorealistic concept while preserving the original layout and intent.

```python
prompt = """
Turn this drawing into a photorealistic image.
Preserve the exact layout, proportions, and perspective.
Choose realistic materials and lighting consistent with the sketch intent.
Do not add new elements or text.
"""

result = client.images.edit(
    model="gpt-image-1.5",
    image=[
        open("../../images/input_images/drawings.png", "rb"),
    ],
    prompt=prompt
)

save_image(result, "realistic_valley.png")
```

### 2.4 Creating Marketing Creatives with Text

For ads with in-image text, be explicit about the exact copy, typography, and placement.

```python
prompt = """
Create a realistic billboard mockup of the shampoo on a highway scene during sunset.
Billboard text (EXACT, verbatim, no extra characters):
"Fresh and clean"
Typography: bold sans-serif, high contrast, centered, clean kerning.
Ensure text appears once and is perfectly legible.
No watermarks, no logos.
"""

result = client.images.edit(
    model="gpt-image-1.5",
    image=[
        open("../../images/input_images/shampoo.png", "rb"),
    ],
    prompt=prompt
)

save_image(result, "billboard.png")
```

### 2.5 Transforming Lighting and Weather

Change the environmental conditions of a scene while preserving its core composition.

```python
prompt = """
Make it look like a winter evening with snowfall.
"""

result = client.images.edit(
    model="gpt-image-1.5",
    input_fidelity="high",  # Use high input fidelity to preserve details
    quality="high",
    image=[
        open("../../images/output_images/billboard.png", "rb"),
    ],
    prompt=prompt
)

save_image(result, "billboard_winter.png")
```

## Summary

`gpt-image-1.5` is a powerful tool for generating and editing images across a wide range of professional use cases. By following the structured prompting principles and examples in this guide—being specific, stating constraints clearly, and iterating thoughtfully—you can reliably produce high-quality, controllable visual content. Start with the basic generation examples and experiment with edits to integrate the model into your creative workflow.