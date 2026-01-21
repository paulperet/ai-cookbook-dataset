# Video Analysis with the Gemini API: Animal Classification Guide

This guide demonstrates how to use the multimodal capabilities of the Gemini model to classify animal species in a video. You will learn to upload a video, analyze its content, and extract structured information.

## Prerequisites & Setup

First, install the required Python client library.

```bash
pip install -U -q "google-genai>=1.0.0"
```

### Configure Your API Key

To authenticate with the Gemini API, you need a valid API key. Store it securely and load it into your environment.

```python
from google.colab import userdata
from google import genai

# Retrieve your API key (adjust this based on your environment)
API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=API_KEY)
```

> **Note:** If you don't have an API key or need help setting up a secret, refer to the [Authentication guide](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb).

## Step 1: Download a Sample Video

For this tutorial, we'll use a video of an American black bear published under a Creative Commons license.

```python
# Define the download path and URL
path = "black_bear.webm"
url = "https://upload.wikimedia.org/wikipedia/commons/8/81/American_black_bears_%28Ursus_americanus%29.webm"

# Download the video using wget
!wget $url -O $path
```

## Step 2: Upload the Video File

To analyze the video with the Gemini API, you must first upload it using the File API.

```python
# Upload the video file
video_file = client.files.upload(file=path)
```

The upload process runs asynchronously. Wait for it to complete before proceeding.

```python
import time

# Poll until the file is ready
while video_file.state.name == "PROCESSING":
    print('.', end='')
    time.sleep(5)
    video_file = client.files.get(name=video_file.name)

# Check for upload failure
if video_file.state.name == "FAILED":
    raise ValueError(video_file.state.name)
```

## Step 3: Preview the Video Content (Optional)

You can inspect the first frame of the video to confirm its content.

```python
import cv2
import matplotlib.pyplot as plt

# Read the first frame
cap = cv2.VideoCapture(path)
_, frame = cap.read()

# Convert from BGR to RGB for display
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Display the frame
plt.imshow(frame_rgb)
plt.axis('off')
plt.show()

# Release the video capture
cap.release()
```

## Step 4: Configure the Model Prompt

Define a system instruction to guide the model's response. This prompt instructs the model to act as a zoologist and provide both common and scientific names.

```python
system_prompt = """
You are a zoologist whose job is to name animals in videos.
You should always provide an english and latin name.
"""
```

## Step 5: Analyze the Video with Gemini

Now, send the video and your query to a Gemini model for analysis.

```python
from google.genai import types

# Select your model
MODEL_ID = "gemini-3-flash-preview"

# Generate content
response = client.models.generate_content(
    model=f"models/{MODEL_ID}",
    contents=[
        "Please identify the animal(s) in this video",
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
Okay! Here are the animals in the video.

- American Black Bear (Ursus americanus)

Hope this helps!
```

The model correctly identifies the animal and provides its Latin name.

## Step 6: Clean Up

Delete the uploaded video file to free up storage.

```python
# Delete the video file
client.files.delete(name=video_file.name)
```

## Summary

You have successfully used the Gemini API to classify animal species in a video. This workflow demonstrates how to:
1.  Prepare and upload a video file.
2.  Configure a model with a specific system prompt.
3.  Perform multimodal analysis to extract structured information.

For more advanced video analysis examples, explore the [Video understanding notebook](../quickstarts/Video_understanding.ipynb).