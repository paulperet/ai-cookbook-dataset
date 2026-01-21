# Getting Started with Gemini Native Image Generation (Nano-Banana Models)

Welcome to this guide on using Gemini's native image generation capabilities, often referred to as the "Nano-Banana" models. This tutorial will walk you through setting up your environment and generating images with these models.

## Prerequisites & Setup

Before we begin, you need to install the required Python library and import the necessary modules.

First, install the `google-generativeai` package using pip:

```bash
pip install -q -U google-generativeai
```

Next, import the library and configure your API key. Replace `"YOUR_API_KEY"` with your actual Google AI Studio API key.

```python
import google.generativeai as genai

# Configure the API
genai.configure(api_key="YOUR_API_KEY")
```

## Step 1: List Available Models

Let's start by exploring which generative models are available to us through the API. We'll filter the list to focus on models capable of image generation.

```python
# List all available models
for m in genai.list_models():
    # Filter for models that support 'generateContent'
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
```

This will output a list of model names. Look for models with "generation" or similar terms in their name, which are typically the image generation models.

## Step 2: Initialize the Image Generation Model

Now, select an image generation model from the list and initialize it. In this example, we'll use a model like `models/gemini-1.5-flash-exp-image-generation`. The exact model name may vary, so use the one available in your list.

```python
# Specify the model for image generation
generation_model = 'models/gemini-1.5-flash-exp-image-generation'

# Create the model instance
model = genai.GenerativeModel(generation_model)
```

## Step 3: Generate Your First Image

With the model initialized, you can now generate an image by providing a text prompt. The `generate_content` method will take your description and create a corresponding image.

```python
# Define your image prompt
prompt = "A serene landscape with a mountain reflected in a calm lake at sunset."

# Generate the image
response = model.generate_content(prompt)
```

## Step 4: Handle and Save the Image Response

The model's response contains the generated image data. We need to extract this data and save it as a viewable image file.

```python
# Check if the generation was successful and images were returned
if response.candidates[0].content.parts[0].inline_data:
    # Extract the image data and MIME type
    image_data = response.candidates[0].content.parts[0].inline_data.data
    mime_type = response.candidates[0].content.parts[0].inline_data.mime_type
    
    # Determine the file extension from the MIME type
    if mime_type == "image/png":
        file_extension = ".png"
    elif mime_type == "image/jpeg":
        file_extension = ".jpg"
    else:
        file_extension = ".bin"  # Fallback for unknown types
    
    # Define a filename and save the image
    filename = "generated_image" + file_extension
    with open(filename, "wb") as f:
        f.write(image_data)
    
    print(f"Image successfully saved as: {filename}")
else:
    print("No image data was returned in the response.")
```

**Important Note on Output:** The code block above shows the logical process. When you run the generation, the API response object will be printed, which contains metadata and the base64-encoded image data. The saving code extracts this data and writes it to a file.

## Step 5: Generate Multiple Images (Optional)

You can also request multiple images in a single call by setting the `candidate_count` parameter in the generation configuration.

```python
# Configure the request to generate multiple candidates (images)
generation_config = {
    "candidate_count": 3,  # Request 3 different images
    # You can add other parameters like 'temperature' here if the model supports it
}

# Generate multiple images based on the prompt
response = model.generate_content(prompt, generation_config=generation_config)

# Save each generated image
for i, candidate in enumerate(response.candidates):
    if candidate.content.parts[0].inline_data:
        image_data = candidate.content.parts[0].inline_data.data
        mime_type = candidate.content.parts[0].inline_data.mime_type
        file_extension = ".png" if mime_type == "image/png" else ".jpg"
        
        filename = f"generated_image_{i+1}{file_extension}"
        with open(filename, "wb") as f:
            f.write(image_data)
        print(f"Saved: {filename}")
```

## Conclusion

You've successfully set up the Gemini API and generated images using native models. You can experiment with different detailed prompts to guide the image generation. Remember to refer to the official [Google AI Studio documentation](https://ai.google.dev/gemini-api/docs) for the latest model names, best practices, and advanced configuration options.

**Next Steps:** Try modifying the prompts for different styles (e.g., "in the style of a watercolor painting" or "a cyberpunk cityscape"). Explore other models in the list for potential variations in style and quality.