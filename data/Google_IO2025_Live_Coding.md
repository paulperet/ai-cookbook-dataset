# A Hands-On Guide to the Gemini API: GenMedia, TTS, and Tool Use

Welcome to the practical guide for the Gemini API, based on the Google I/O 2025 live coding session. This tutorial will walk you through building applications with cutting-edge generative media, text-to-speech, and intelligent tool use. You'll learn to generate images and videos, create natural-sounding audio, and empower models with real-time search and code execution.

## Prerequisites & Setup

Before you begin, ensure you have a Google Cloud project with the Gemini API enabled and an API key. This key is required for all API calls.

### Step 1: Install the SDK
Install the official Python SDK for the Gemini API.

```bash
pip install -U 'google-genai>=1.16'
```

### Step 2: Configure Your API Key
Securely set your API key. In a Colab environment, you can use `userdata`. For local development, use environment variables.

```python
# Example for Google Colab
from google.colab import userdata
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

# Alternative for local scripts using an environment variable
# import os
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
```

### Step 3: Initialize the Client
Import the necessary libraries and initialize the client with your API key.

```python
import time
from google import genai
from google.genai import types
from IPython.display import Video, HTML, Markdown, Image
import PIL
import io
import pathlib

client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Part 1: Working with GenMedia Models

The Gemini API provides access to powerful generative media models for creating images and videos.

### Generating Images with Imagen 3

Imagen 3 is Google's state-of-the-art text-to-image model, producing highly detailed and artistic images.

**Note:** Imagen is a paid feature and requires a billing account.

#### Step 1.1: Select the Model
Specify the Imagen 3 model ID.

```python
IMAGEN_MODEL_ID = "imagen-3.0-generate-002"
```

#### Step 1.2: Generate an Image
Define your prompt and configuration, then call the API. The `config` parameter lets you control the number of images, aspect ratio, and safety settings.

```python
prompt = """
Dynamic anime illustration: A happy Brazilian man with short grey hair and a
grey beard, mid-presentation at a tech conference. He's wearing a fun blue
short-sleeve shirt covered in mini avocado prints. Capture a funny, energetic
moment where he's clearly enjoying himself, perhaps with an exaggerated joyful
expression or a humorous gesture, stage background visible.
"""

result = client.models.generate_images(
    model=IMAGEN_MODEL_ID,
    prompt=prompt,
    config=dict(
        number_of_images=1,
        output_mime_type="image/jpeg",
        person_generation="ALLOW_ADULT",
        aspect_ratio="1:1"
    )
)
```

#### Step 1.3: Display the Result
The API returns a list of generated images. You can display them directly.

```python
for generated_image in result.generated_images:
    display(generated_image.image.show())
```

### Generating and Editing Images with Gemini 2.0 Flash (Experimental)

The `gemini-2.0-flash-preview-image-generation` model can generate images interleaved with text in a single response, and also edit existing images.

#### Step 2.1: Select the Model
```python
GEMINI_IMAGE_MODEL_ID = "gemini-2.0-flash-preview-image-generation"
```

#### Step 2.2: Generate Interleaved Text and Images
To get both text and images, you must explicitly set `response_modalities`.

```python
contents = """
Show me how to cook a Brazilian cuscuz with coconut milk and grated coconut.
Include detailed step by step guidance with images.
"""

response = client.models.generate_content(
    model=GEMINI_IMAGE_MODEL_ID,
    contents=contents,
    config=types.GenerateContentConfig(
        response_modalities=['Text', 'Image']
    )
)

# Display the response parts
for part in response.candidates[0].content.parts:
    if part.text is not None:
        display(Markdown(part.text))
    elif part.inline_data is not None:
        # Display the generated image
        display(Image(data=part.inline_data.data, width=512, height=512))
```

#### Step 2.3: Generate a Standalone Image
You can also use the model for pure image generation.

```python
prompt = """
Dynamic anime illustration: A happy Brazilian man with short grey hair and a
grey beard, mid-presentation at a tech conference. He's wearing a fun blue
short-sleeve shirt covered in mini avocado prints.
"""

response = client.models.generate_content(
    model=GEMINI_IMAGE_MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
        response_modalities=['Text', 'Image']
    )
)

# Extract and save the image
for part in response.candidates[0].content.parts:
    if part.inline_data is not None:
        data = part.inline_data.data
        pathlib.Path("generated_image.png").write_bytes(data)
        display(Image(data=data))
```

#### Step 2.4: Edit an Existing Image
Provide an image and a text prompt describing the edits.

```python
edit_prompt = """
make the image background in full white and add a wireless presentation
clicker on the hand of the person
"""

# Load the previously saved image
input_image = PIL.Image.open('generated_image.png')

response = client.models.generate_content(
    model=GEMINI_IMAGE_MODEL_ID,
    contents=[
        edit_prompt,
        input_image
    ],
    config=types.GenerateContentConfig(
        response_modalities=['Text', 'Image']
    )
)

# Display the edited image
for part in response.candidates[0].content.parts:
    if part.inline_data is not None:
        display(Image(data=part.inline_data.data))
        # Save the edited version
        pathlib.Path("edited_image.png").write_bytes(part.inline_data.data)
```

### Generating Videos with Veo 2

Veo 2 is an advanced model for generating videos from text or an image.

**Note:** Veo is a paid feature and requires a billing account.

#### Step 3.1: Select the Model
```python
VEO_MODEL_ID = "veo-2.0-generate-001"
```

#### Step 3.2: Generate a Video from Text
Video generation is an asynchronous operation. You start the generation and poll for completion.

```python
prompt = """
Dynamic anime scene: A happy Brazilian man with short grey hair and a
grey beard, mid-presentation at a tech conference.
"""

operation = client.models.generate_videos(
    model=VEO_MODEL_ID,
    prompt=prompt,
    config=types.GenerateVideosConfig(
        person_generation="allow_adult",
        aspect_ratio="16:9",
        number_of_videos=1,
        duration_seconds=8,
    ),
)

# Poll for completion
while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)
    print("Operation status:", operation.metadata)

# Download and display the video
for n, generated_video in enumerate(operation.result.generated_videos):
    client.files.download(file=generated_video.video)
    generated_video.video.save(f'generated_video_{n}.mp4')
    display(generated_video.video.show())
```

#### Step 3.3: Generate a Video from an Image
You can use an image as the starting frame for video generation.

```python
# Load the image
input_image = PIL.Image.open('edited_image.png')

# Convert to bytes
image_bytes_io = io.BytesIO()
input_image.save(image_bytes_io, format=input_image.format)
image_bytes = image_bytes_io.getvalue()

operation = client.models.generate_videos(
    model=VEO_MODEL_ID,
    prompt=prompt, # Use the same prompt as above
    image=types.Image(image_bytes=image_bytes, mime_type=input_image.format),
    config=types.GenerateVideosConfig(
        aspect_ratio="16:9",
        number_of_videos=1,
        duration_seconds=8,
    ),
)

# Poll and display as before
while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)

for n, generated_video in enumerate(operation.result.generated_videos):
    generated_video.video.save(f'image_to_video_{n}.mp4')
    display(generated_video.video.show())
```

## Part 2: Generating Text-to-Speech (TTS)

The Gemini API includes native text-to-speech capabilities with control over voice, style, and language.

### Step 4.1: Select the TTS Model
```python
TTS_MODEL_ID = "gemini-2.0-flash-exp"
```

### Step 4.2: Generate Speech from Text
Configure the audio output with your desired voice and parameters.

```python
response = client.models.generate_content(
    model=TTS_MODEL_ID,
    contents="Hello! Welcome to Google I/O 2025. Let's build the future together!",
    config=types.GenerateContentConfig(
        audio_config=types.AudioConfig(
            voice_config=types.VoiceConfig(
                predefined_voice_config=types.PredefinedVoiceConfig(
                    voice_name="andrea",
                )
            ),
            audio_encoding="MP3",
            speaking_rate=1.0,
        )
    )
)

# The response contains the audio data
audio_data = response.candidates[0].content.parts[0].inline_data.data

# Save to a file
pathlib.Path("welcome.mp3").write_bytes(audio_data)
print("Audio file saved as 'welcome.mp3'")
```

## Part 3: Intelligent Tool Use

Gemini models can be equipped with built-in tools to perform actions like web search and code execution.

### Grounding with Google Search
Provide the model with real-time, factual information from the web.

```python
response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents="What are the latest announcements from Google I/O 2025?",
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())]
    )
)

display(Markdown(response.text))
```

### Code Execution for Problem Solving
The model can write and execute Python code in a secure sandbox to solve complex problems.

```python
response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents="Write a Python function to calculate the nth Fibonacci number and run it for n=10.",
    config=types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.CodeExecution())]
    )
)

display(Markdown(response.text))
```

## Next Steps

You've successfully explored key capabilities of the Gemini API:
- Generated high-quality images with Imagen 3 and Gemini 2.0 Flash.
- Created and edited videos with Veo 2.
- Synthesized natural-sounding speech.
- Enhanced model responses with real-time search and code execution.

To dive deeper, experiment with different prompts, adjust configuration parameters, and combine these features to build sophisticated multimodal applications. Refer to the [official Gemini API documentation](https://ai.google.dev/gemini-api) for more details on models, parameters, and best practices.