# Getting Started with GPT-4o and GPT-4o mini: A Multimodal Guide

## Introduction

GPT-4o ("o" for "omni") and its lightweight sibling, GPT-4o mini, represent a new generation of natively multimodal models. They are designed to process a combination of text, audio, and image inputs, and can generate outputs in text, audio, and image formats.

This guide walks you through the core capabilities of these models via the OpenAI API, focusing on practical implementations for text, image, and video processing.

### Prerequisites

Before you begin, ensure you have:
1.  An OpenAI API key. You can create one in your [OpenAI project settings](https://platform.openai.com/api-keys).
2.  Python installed on your system.
3.  (Optional, for video processing) `ffmpeg` installed. You can install it via `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Linux).

---

## 1. Setup and Initial Text Request

First, install the OpenAI Python SDK and configure your client.

### Step 1.1: Install the OpenAI Library

```bash
pip install --upgrade openai
```

### Step 1.2: Configure the Client and Make Your First Request

Create a Python script or notebook cell and set up the client with your API key. We'll start with a simple text-based interaction.

```python
from openai import OpenAI
import os

# Configuration
MODEL = "gpt-4o-mini"
# It's best practice to store your API key in an environment variable.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your-api-key>"))
```

Now, let's send a basic chat completion request. We'll use a `system` message to set the assistant's behavior and a `user` message to ask a question.

```python
completion = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Help me with my math homework!"
        },
        {
            "role": "user",
            "content": "Hello! Could you solve 2+2?"
        }
    ]
)

print("Assistant:", completion.choices[0].message.content)
```

**Expected Output:**
```
Assistant: Of course! 2 + 2 = 4.
```

---

## 2. Processing Images with GPT-4o mini

GPT-4o mini can understand and reason about images. You can provide images in two ways: as a Base64-encoded string or via a public URL.

For this example, we'll use a simple image of a triangle. Ensure you have a file named `triangle.png` in a `data/` directory relative to your script.

### Step 2.1: Encode an Image as Base64

We'll create a helper function to read an image file and encode it.

```python
import base64

def encode_image(image_path):
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

IMAGE_PATH = "data/triangle.png"
base64_image = encode_image(IMAGE_PATH)
```

### Step 2.2: Send the Base64 Image to the Model

Now, construct a message that includes both text and the image. The `content` field for a multimodal message is a list of dictionaries.

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that responds in Markdown. Help me with my math homework!"
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's the area of the triangle?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    temperature=0.0,  # Set to 0 for deterministic, factual responses
)

print(response.choices[0].message.content)
```

The model will analyze the image, identify the base and height of the triangle, calculate the area, and return a detailed, formatted response.

### Step 2.3: Process an Image from a URL

You can also use a publicly accessible image URL. Here's an example using an image from Wikimedia Commons.

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant that responds in Markdown. Help me with my math homework!"
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's the area of the triangle?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/e/e2/The_Algebra_of_Mohammed_Ben_Musa_-_page_82b.png"
                    }
                }
            ]
        }
    ],
    temperature=0.0,
)

print(response.choices[0].message.content)
```

---

## 3. Processing Videos with Frame Sampling and Audio

While the API doesn't accept video files directly, you can process videos by extracting frames (images) and audio. This section shows you how to create a video summary using different modalities: visual only, audio only, and combined.

### Step 3.1: Install Video Processing Libraries

We'll use `opencv-python` for frame extraction and `moviepy` for audio extraction.

```bash
pip install opencv-python moviepy
```

### Step 3.2: Extract Frames and Audio from a Video

Create a function that samples frames at a specified interval and extracts the audio track as an MP3 file.

```python
import cv2
from moviepy.editor import VideoFileClip
import os
import base64

def process_video(video_path, seconds_per_frame=2):
    """
    Extracts frames and audio from a video file.
    Args:
        video_path: Path to the video file.
        seconds_per_frame: Interval in seconds between sampled frames.
    Returns:
        base64Frames: List of base64-encoded JPEG images.
        audio_path: Path to the extracted MP3 audio file.
    """
    base64_frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame = 0

    # Loop through and sample frames
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    # Extract audio
    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.close()

    print(f"Extracted {len(base64_frames)} frames")
    print(f"Extracted audio to {audio_path}")
    return base64_frames, audio_path

# Process the video (extract 1 frame per second)
VIDEO_PATH = "data/keynote_recap.mp4"
base64_frames, audio_path = process_video(VIDEO_PATH, seconds_per_frame=1)
```

### Step 3.3: Generate a Summary Using Only Visual Frames

Send the extracted frames to the model and ask for a summary.

```python
# Create a list of image message objects from the frames
image_messages = [
    {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpg;base64,{frame}", "detail": "low"}
    }
    for frame in base64_frames
]

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "You are generating a video summary. Please provide a summary of the video. Respond in Markdown."
        },
        {
            "role": "user",
            "content": ["These are the frames from the video."] + image_messages
        }
    ],
    temperature=0,
)
print(response.choices[0].message.content)
```

This summary will be based solely on the visual content (slides, presenters) and will miss any spoken details.

### Step 3.4: Generate a Summary Using Only Audio

First, we need to transcribe the audio. As of July 2024, audio input is in preview, so we use the `gpt-4o-audio-preview` model for transcription.

```python
# Transcribe the audio
with open(audio_path, 'rb') as audio_file:
    audio_content = base64.b64encode(audio_file.read()).decode('utf-8')

response = client.chat.completions.create(
    model='gpt-4o-audio-preview',
    modalities=["text"],
    messages=[
        {
            "role": "system",
            "content": "You are generating a transcript. Create a transcript of the provided audio."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "This is the audio."},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_content,
                        "format": "mp3"
                    }
                }
            ]
        }
    ],
    temperature=0,
)
transcription = response.choices[0].message.content
```

Now, summarize the transcription using the standard `gpt-4o-mini` model.

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "You are generating a transcript summary. Create a summary of the provided transcription. Respond in Markdown."
        },
        {
            "role": "user",
            "content": f"Summarize this text: {transcription}"
        }
    ],
    temperature=0,
)
transcription_summary = response.choices[0].message.content
print(transcription_summary)
```

This summary will be rich in spoken content but lack the context provided by the visual slides.

### Step 3.5: Generate a Combined Audio + Visual Summary

For the most comprehensive summary, provide both the frames and the audio transcript to the model in a single request.

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "You are generating a video summary. Create a summary of the provided video and its transcript. Respond in Markdown."
        },
        {
            "role": "user",
            "content": [
                "These are the frames from the video.",
                *image_messages,  # Unpacks the list of image messages
                {"type": "text", "text": f"Transcript: {transcription}"}
            ]
        }
    ],
    temperature=0,
)
print(response.choices[0].message.content)
```

This final summary leverages the full multimodal context, combining what was shown (frames) with what was said (transcript) to produce the most accurate and detailed overview.

## Conclusion

You've now learned how to interact with GPT-4o mini for text completion, image analysis, and complex video processing by combining frame sampling and audio transcription. This multimodal approach allows you to build applications that can understand and reason about rich, real-world content.

**Key Takeaways:**
1.  **Images:** Can be provided via Base64 encoding or URL.
2.  **Videos:** Process by extracting frames (as images) and audio separately.
3.  **Multimodal Context:** Combining different input types (text, image, audio) in a single request leads to more accurate and context-aware responses.

Experiment with different sampling rates, detail levels for images, and prompt engineering to tailor these techniques to your specific use case.