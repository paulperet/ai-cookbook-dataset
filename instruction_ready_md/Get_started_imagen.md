# Getting Started with Image Generation using the Gemini API

This guide will walk you through using Google's Imagen 4 family of models to generate high-quality images from text descriptions. You'll learn how to set up the Python SDK, configure generation parameters, and create images with text.

## Prerequisites

### ⚠️ Important Note
Image generation is a paid-only feature and will not work on the free tier. Please review the [pricing page](https://ai.google.dev/pricing#imagen-4) for details.

## Setup

### Step 1: Install the Python SDK

First, install the Google Generative AI Python SDK:

```bash
pip install -q -U "google-genai>=1.0.0"
```

### Step 2: Configure Your API Key

You'll need a Google AI API key. Store it in an environment variable or directly in your code:

```python
# Replace with your actual API key
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
```

### Step 3: Initialize the Client

Create a client instance to interact with the Gemini API:

```python
from google import genai

client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Generating Your First Image

### Step 4: Choose a Model

Select from the available Imagen models:

- **`imagen-4.0-generate-001`**: The standard model for most use cases
- **`imagen-4.0-ultra-generate-001`**: Highest quality, especially good with text (generates one image at a time)
- **`imagen-4.0-fast-generate-001`**: Faster and more cost-effective
- **`imagen-3.0-generate-002`**: Previous generation (for backward compatibility)

```python
MODEL_ID = "imagen-4.0-generate-001"
```

### Step 5: Craft Your Prompt

Create a descriptive prompt. Imagen 4 models perform better with longer, more detailed descriptions. Here are the key parameters you can configure:

- `number_of_images`: How many images to generate (1-4)
- `person_generation`: Control generation of adult figures (`DONT_ALLOW` or `ALLOW_ADULT`)
- `aspect_ratio`: Image dimensions (`1:1`, `3:4`, `4:3`, `16:9`, `9:16`)
- `image_size`: Output resolution (`1k` or `2k`)

```python
prompt = "A cat lounging lazily on a sunny windowsill playing with a kid toy."
number_of_images = 1
person_generation = "ALLOW_ADULT"
aspect_ratio = "1:1"
image_size = "1k"
```

### Step 6: Generate Images

Call the API with your configuration:

```python
result = client.models.generate_images(
    model=MODEL_ID,
    prompt=prompt,
    config=dict(
        number_of_images=number_of_images,
        output_mime_type="image/jpeg",
        person_generation=person_generation,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
    )
)
```

### Step 7: Display the Results

View your generated images:

```python
for generated_image in result.generated_images:
    (image := generated_image.image).show()

# For non-Colab environments, use:
# from PIL import Image
# from io import BytesIO
# for generated_image in result.generated_images:
#     image = Image.open(BytesIO(generated_image.image.image_bytes))
```

## Advanced: Generating Images with Text

Imagen models excel at incorporating text into images. Here's an example creating a comic strip:

### Step 8: Create a Text-Rich Prompt

```python
prompt = """A 3-panel cosmic epic comic. 
Panel 1: Tiny 'Stardust' in nebula; radar shows anomaly (text 'ANOMALY DETECTED'), hull text 'stardust'. Pilot whispers. 
Panel 2: Bioluminescent leviathan emerges; console red text 'WARNING!'. 
Panel 3: Leviathan chases ship through asteroids; console text 'SHIELD CRITICAL!', screen text 'EVADE!'. Pilot screams, SFX 'CRUNCH!', 'ROOOOAAARR!'."""

number_of_images = 1
person_generation = "ALLOW_ADULT"
aspect_ratio = "1:1"
```

### Step 9: Generate the Comic

```python
result = client.models.generate_images(
    model=MODEL_ID,
    prompt=prompt,
    config=dict(
        number_of_images=number_of_images,
        output_mime_type="image/jpeg",
        person_generation=person_generation,
        aspect_ratio=aspect_ratio
    )
)

for generated_image in result.generated_images:
    (image := generated_image.image).show()
```

### Step 10: Generate Poetic Text on a Wall

Here's another example with literary text:

```python
prompt = """a wall on which a colorful tag is drawn and that can be read as the first verse of Charles Baudelaire's poem "l'invitation au voyage": 
Mon enfant, ma sœur,    
Songe à la douceur  
D'aller là-bas vivre ensemble !    
Aimer à loisir,    
Aimer et mourir  
Au pays qui te ressemble !"""

number_of_images = 1
person_generation = "ALLOW_ADULT"
aspect_ratio = "9:16"

result = client.models.generate_images(
    model=MODEL_ID,
    prompt=prompt,
    config=dict(
        number_of_images=number_of_images,
        output_mime_type="image/jpeg",
        person_generation=person_generation,
        aspect_ratio=aspect_ratio
    )
)

for generated_image in result.generated_images:
    (image := generated_image.image).show()
```

## Next Steps

### Improve Your Prompts
Check the [Imagen prompt guide](https://ai.google.dev/gemini-api/docs/imagen-prompt-guide) for advanced prompting techniques.

### Explore More Examples
- [Illustrate a book](../examples/Book_illustration.ipynb): Combine Gemini and Imagen to create book illustrations
- [Spatial understanding](./Spatial_understanding.ipynb): Learn about Gemini's image analysis capabilities
- [Video understanding](./Video_understanding.ipynb): Explore video processing examples

### Experiment with Different Models
Try switching between `imagen-4.0-generate-001`, `imagen-4.0-ultra-generate-001`, and `imagen-4.0-fast-generate-001` to compare quality, speed, and cost for your specific use case.

Remember that all generated images include a non-visible digital SynthID watermark for authenticity verification.