# Guide: Generate Images with the Gemini Imagen REST API

This guide walks you through generating high-quality images using Google's Imagen models via the REST API. You will learn how to set up your environment, send a generation request, and inspect the results.

## Prerequisites

You will need:
1.  A Google AI Studio API key with access to the Imagen models (paid tier required).
2.  The API key stored in a Colab Secret named `GOOGLE_API_KEY`.

## Step 1: Environment Setup

First, import the necessary libraries and load your API key.

```python
import os
import requests
import base64
import io
from PIL import Image
from google.colab import userdata

# Load your API key from the Colab Secret
GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
```

## Step 2: Configure Your Generation Request

Define the parameters for your image generation. The Imagen models perform best with detailed, descriptive prompts.

```python
# Select your model
model_name = "imagen-4.0-generate-001"  # Options: imagen-3.0-generate-002, imagen-4.0-generate-001, imagen-4.0-ultra-generate-001

# Define your prompt
prompt = "A hairy bunny in my kitchen playing with a tomato."

# Set generation parameters
sampleCount = 1          # Number of images to generate (1-4)
aspectRatio = "1:1"      # Options: "1:1", "3:4", "4:3", "16:9", "9:16"
personGeneration = "allow_adult"  # Options: "dont_allow", "allow_adult"
sampleImageSize = "1k"   # Output resolution: "1k" or "2k"
```

## Step 3: Send the API Request

Construct and send a POST request to the Imagen API endpoint.

```python
# Construct the API endpoint URL
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:predict"

# Define request headers and payload
headers = {'Content-Type': 'application/json'}

data = {
    "instances": [{"prompt": prompt}],
    "parameters": {
        "sampleCount": sampleCount,
        "personGeneration": personGeneration,
        "aspectRatio": aspectRatio,
        "sampleImageSize": sampleImageSize,
    }
}

# Make the API call
response = requests.post(
    f"{url}?key={GOOGLE_API_KEY}",
    headers=headers,
    json=data
)

# Check the response status
if response.status_code != 200:
    print(f"Request failed with status code: {response.status_code}")
    print("Response:", response.text)
else:
    print("Request successful!")
```

## Step 4: Examine the Response Metadata

The API response contains metadata for each generated image. Let's inspect it.

```python
response_data = response.json()
predictions = response_data.get('predictions', [])

print(f"Number of predictions: {len(predictions)}")

for index, prediction in enumerate(predictions):
    # For the last prediction, label it as the main output
    if index == len(predictions) - 1:
        print("Positive prompt:")
    else:
        print(f"Index: {index}")
    
    # Check the MIME type (e.g., image/png)
    if "mimeType" in prediction:
        print(f"  mimeType: {prediction['mimeType']}")
    
    # Check for any Responsible AI filtering reasons
    if "raiFilteredReason" in prediction:
        print(f"  raiFilteredReason: {prediction['raiFilteredReason']}")
    
    # Review safety attributes
    if "safetyAttributes" in prediction:
        print(f"  safetyAttributes: {prediction['safetyAttributes']}")
```

## Step 5: Decode and Display the Generated Images

The generated images are returned as Base64-encoded strings. Decode and display them using the Python Imaging Library (PIL).

```python
for prediction in predictions:
    if "bytesBase64Encoded" in prediction:
        # Decode the Base64 image data
        decoded_bytes = base64.b64decode(prediction["bytesBase64Encoded"])
        
        # Create an in-memory buffer and load the image
        image_buffer = io.BytesIO(decoded_bytes)
        img = Image.open(image_buffer)
        
        # Display the image and print its dimensions
        display(img)
        print(f"Image size: {img.size}")
```

## Next Steps & Resources

*   **Prompting Guide:** For best results, consult the official [Imagen Prompt Guide](https://ai.google.dev/gemini-api/docs/imagen-prompt-guide) to learn how to craft effective prompts.
*   **Example Projects:** Explore creative applications, such as the [Book Illustration example](../examples/Book_illustration.ipynb), which combines Gemini and Imagen.
*   **Expand Your Skills:** The Gemini API is also powerful for understanding images and videos. Explore the [Spatial Understanding](./Spatial_understanding.ipynb) and [Video Understanding](./Video_understanding.ipynb) guides.

---
**Note:** Image generation is a paid feature. Please review the [pricing details](https://ai.google.dev/pricing#imagen-4) for the Imagen models. All generated images include a non-visible digital SynthID watermark.