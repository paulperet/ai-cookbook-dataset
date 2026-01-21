# Guide: Analyzing Videos for Historic Event Recognition with the Gemini API

This guide demonstrates how to use the multimodal capabilities of Google's Gemini models to analyze video content and identify significant historical events.

## Prerequisites & Setup

First, ensure you have the required Python package installed.

```bash
pip install -U "google-genai>=1.0.0"
```

### 1. Configure Your API Key

To use the Gemini API, you need a valid API key. This guide assumes you have stored your key securely. If you need to create an API key or learn how to manage secrets, refer to the official [Authentication guide](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb).

```python
from google import genai
from google.colab import userdata  # For Colab environments. Use your preferred secret manager elsewhere.

API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=API_KEY)
```

## Tutorial: Analyzing a Historic Speech

We will analyze a famous historical video: President Ronald Reagan's "Tear Down This Wall" speech at the Berlin Wall on June 12, 1987.

### Step 1: Download the Video File

First, we need to obtain the video. We'll download it from a public URL.

```python
# Define the download path and URL
path = "berlin.mp4"
url = "https://s3.amazonaws.com/NARAprodstorage/opastorage/live/16/147/6014716/content/presidential-libraries/reagan/5730544/6-12-1987-439.mp4"

# Use wget to download the file (run this in a shell)
!wget $url -O $path
```

### Step 2: Upload the Video to the Gemini API

The Gemini API requires video files to be uploaded to its service before processing.

```python
# Upload the video file
video_file = client.files.upload(file=path)
```

### Step 3: Wait for Video Processing

After uploading, the file enters a processing state. We must wait until it's ready for analysis.

```python
import time

# Poll the file status until it is no longer "PROCESSING"
while video_file.state.name == "PROCESSING":
    print('.', end='')
    time.sleep(5)
    video_file = client.files.get(name=video_file.name)

# Check for upload failure
if video_file.state.name == "FAILED":
    raise ValueError(video_file.state.name)
```

Once the loop finishes, your video is ready.

### Step 4: Define the System Prompt and Safety Settings

We will instruct the model to act as a historian. For analyzing historical content that may involve sensitive topics, it's often necessary to adjust the default safety filters.

```python
# Define a system instruction to guide the model's role and output
system_prompt = """
You are a historian who specializes in events caught on film.
When you receive a video, answer the following questions:
1. When did it happen?
2. Who is the most important person in the video?
3. What is the event called?
"""

# Configure safety settings to avoid blocks on historical content
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
```

### Step 5: Generate Content with the Video

Now, we can send the video to the Gemini model along with our prompt and configuration.

```python
from google.genai import types

# Select your preferred Gemini model
MODEL_ID = "gemini-1.5-flash"  # You can change this to other available models like "gemini-1.5-pro"

# Generate the analysis
response = client.models.generate_content(
    model=f"models/{MODEL_ID}",
    contents=[
        "Analyze the video please",  # User prompt
        video_file                    # The uploaded video file
    ],
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        safety_settings=safety_settings,
    ),
)

# Print the model's response
print(response.text)
```

**Example Output:**
```
Certainly! Here's the information about the video:

Based on the video, here are the answers to your questions:

- **When did it happen?** This event happened on June 12, 1987.
- **Who is the most important person in the video?** The most important person in the video is Ronald Reagan, who was the President of the United States at the time.
- **What is the event called?** The event is commonly referred to as President Reagan's "Tear Down This Wall" speech.
```

The model successfully identified the date, key figure, and the common name for the historical event.

### Step 6: Clean Up (Optional)

To manage your storage quota, you can delete the uploaded video file from the Gemini API.

```python
# Delete the uploaded video file
client.files.delete(name=video_file.name)
```

## Summary

You have now learned how to use the Gemini API to analyze video content for historical context. The process involves:
1.  Setting up your API client.
2.  Uploading and processing a video file.
3.  Crafting a specific system prompt to guide the model's analysis.
4.  Adjusting safety settings for historical content.
5.  Generating and receiving a structured analysis.

This is just one application of video understanding with Gemini. For more examples and advanced techniques, explore the [Video understanding notebook](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Video_understanding.ipynb) in the official cookbook.