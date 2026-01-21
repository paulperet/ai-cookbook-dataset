# Gemini API Cookbook: Video Summarization

This guide demonstrates how to use the Gemini API's multimodal capabilities to generate a summary of a video's content.

## Prerequisites

First, install the required Python client library.

```bash
pip install -U "google-genai>=1.0.0"
```

## 1. Configure the API Client

To use the Gemini API, you need an API key. Store it securely and initialize the client.

```python
from google import genai

# Replace with your actual API key. For Colab, you could use:
# from google.colab import userdata
# API_KEY = userdata.get('GOOGLE_API_KEY')
API_KEY = "YOUR_API_KEY_HERE"

client = genai.Client(api_key=API_KEY)
```

## 2. Obtain a Video File

This example uses the short film "Wing It!" by Blender Studio, available under a Creative Commons license. We'll download it to our local environment.

```python
import subprocess

# Define the download URL and local file path
url = "https://upload.wikimedia.org/wikipedia/commons/3/38/WING_IT%21_-_Blender_Open_Movie-full_movie.webm"
path = "wingit.webm"

# Download the video file
subprocess.run(["wget", url, "-O", path], check=True)
```

## 3. Upload the Video to Gemini

The Gemini API requires video files to be uploaded to its service before processing.

```python
video_file = client.files.upload(file=path)
```

The file enters a `PROCESSING` state. You must wait for this to complete before proceeding.

```python
import time

# Poll the file status until it's ready
while video_file.state.name == "PROCESSING":
    print('.', end='')
    time.sleep(5)
    video_file = client.files.get(name=video_file.name)

# Check for upload failure
if video_file.state.name == "FAILED":
    raise ValueError(f"File upload failed: {video_file.state.name}")

print("\nVideo file is ready.")
```

## 4. Generate a Video Summary

Now, instruct a Gemini model to analyze the uploaded video and provide a concise summary.

### 4.1. Define the System Prompt

First, set the context for the model to ensure it provides the desired output format.

```python
system_prompt = "You should provide a quick 2 or 3 sentence summary of what is happening in the video."
```

### 4.2. Call the Model

Use the `generate_content` method, passing the video file and your instruction.

```python
from google.genai import types

MODEL_ID = "gemini-1.5-flash"  # You can also use "gemini-1.5-pro"

response = client.models.generate_content(
    model=f"models/{MODEL_ID}",
    contents=[
        "Summarise this video please.",
        video_file
    ],
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
    ),
)

print(response.text)
```

**Example Output:**
```
Okay, here is a quick summary of what's happening in the video:

A cat and a dog are building a spaceship out of various objects lying around in a barn. They launch into space, but the ship starts falling apart almost immediately. They barely make it through the atmosphere before crash landing back into their barn.
```

The model successfully captures the core plot of the short film.

## 5. Clean Up

Once you have your summary, you can delete the uploaded video file from the Gemini servers to manage your storage.

```python
# Delete the uploaded video file
client.files.delete(name=video_file.name)
print("File deleted.")
```

## Important Consideration

The Gemini API samples videos at a rate of approximately one frame per second. This means very brief visual events (those lasting less than a second) might not be captured in the model's analysis. For tasks requiring fine-grained temporal understanding, consider this limitation.

## Summary

You have now successfully used the Gemini API to:
1.  Upload a video file.
2.  Instruct a multimodal model to analyze its visual content.
3.  Generate a natural language summary of the video's plot.
4.  Clean up the uploaded resource.

This is just one application of video understanding with Gemini. You can extend this workflow for tasks like scene description, action recognition, or content moderation.