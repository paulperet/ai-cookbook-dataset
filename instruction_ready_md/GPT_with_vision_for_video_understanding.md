# Video Processing and Narration with GPT-4.1-mini and GPT-4o TTS

This guide demonstrates how to use GPT-4.1-mini's visual capabilities to analyze video frames and generate a descriptive voiceover, which is then synthesized into speech using the GPT-4o Text-to-Speech (TTS) API.

You will learn how to:
1. Extract and encode frames from a video file.
2. Use GPT-4.1-mini to generate a compelling video description.
3. Create a narrated script in a specific style (e.g., David Attenborough).
4. Generate a high-quality voiceover audio file using the TTS API.

## Prerequisites

Ensure you have the following installed and configured:

```bash
pip install opencv-python openai
```

You will also need an OpenAI API key. Set it as an environment variable or configure it directly in the code.

```python
import cv2
import base64
import time
from openai import OpenAI
import os

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
```

## Step 1: Extract Frames from a Video

We'll use OpenCV to read a video file and extract its frames, encoding each frame as a base64 string for processing.

```python
# Load the video file
video = cv2.VideoCapture("data/bison.mp4")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    # Encode the frame as a JPEG and convert to base64
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(f"{len(base64Frames)} frames read.")
```

**Output:**
```
618 frames read.
```

## Step 2: Generate a Video Description with GPT-4.1-mini

We can now use GPT-4.1-mini's vision capabilities to analyze the video. To manage token usage, we'll sample frames (e.g., every 25th frame) rather than sending every single frame.

```python
# Prepare the prompt and sampled frames
sampled_frames = base64Frames[0::25]

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "These are frames from a video that I want to upload. "
                        "Generate a compelling description that I can upload along with the video."
                    )
                },
                *[
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{frame}"
                    }
                    for frame in sampled_frames
                ]
            ]
        }
    ],
)

print(response.output_text)
```

**Output:**
```
Witness the raw power and strategy of nature in this intense wildlife encounter captured in stunning detail. A determined pack of wolves surrounds a lone bison on a snowy plain, showcasing the relentless dynamics of predator and prey in the wild. As the wolves close in, the bison stands its ground amidst the swirling snow, illustrating a gripping battle for survival. This rare footage offers an up-close look at the resilience and instincts that govern life in the animal kingdom, making it a must-watch for nature enthusiasts and wildlife lovers alike. Experience the drama, tension, and beauty of this extraordinary moment frozen in time.
```

## Step 3: Create a Voiceover Script in a Specific Style

Next, we'll prompt GPT-4.1-mini to generate a narration script in the style of David Attenborough, using the same sampled frames.

```python
result = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "These are frames of a video. Create a short voiceover script in the style of David Attenborough. "
                        "Only include the narration."
                    )
                },
                *[
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{frame}"
                    }
                    for frame in sampled_frames
                ]
            ]
        }
    ]
)

print(result.output_text)
```

**Output:**
```
In the frozen expanse of the winter landscape, a coordinated pack of wolves moves with calculated precision. Their target, a lone bison, is powerful but vulnerable when isolated. The wolves encircle their prey, their numbers overwhelming, displaying the brutal reality of survival in the wild. As the bison struggles to break free, reinforcements from the herd arrive just in time, charging into the pack. A dramatic clash unfolds, where strength meets strategy in the perpetual battle for life. Here, in the heart of nature’s harshest conditions, every moment is a testament to endurance and the delicate balance of predator and prey.
```

## Step 4: Synthesize the Voiceover with GPT-4o TTS

Finally, we'll use the GPT-4o TTS API to convert the generated script into speech. We'll provide detailed instructions to shape the voice's affect, tone, and pacing to match the desired style.

```python
instructions = """
Voice Affect: Calm, measured, and warmly engaging; convey awe and quiet reverence for the natural world.

Tone: Inquisitive and insightful, with a gentle sense of wonder and deep respect for the subject matter.

Pacing: Even and steady, with slight lifts in rhythm when introducing a new species or unexpected behavior; natural pauses to allow the viewer to absorb visuals.

Emotion: Subtly emotive—imbued with curiosity, empathy, and admiration without becoming sentimental or overly dramatic.

Emphasis: Highlight scientific and descriptive language (“delicate wings shimmer in the sunlight,” “a symphony of unseen life,” “ancient rituals played out beneath the canopy”) to enrich imagery and understanding.

Pronunciation: Clear and articulate, with precise enunciation and slightly rounded vowels to ensure accessibility and authority.

Pauses: Insert thoughtful pauses before introducing key facts or transitions (“And then... with a sudden rustle...”), allowing space for anticipation and reflection.
"""

audio_response = client.audio.speech.create(
  model="gpt-4o-mini-tts",
  voice="echo",
  instructions=instructions,
  input=result.output_text,
  response_format="wav"
)

# Save the audio to a file
with open("voiceover.wav", "wb") as f:
    f.write(audio_response.content)

print("Voiceover saved as 'voiceover.wav'")
```

**Output:**
```
Voiceover saved as 'voiceover.wav'
```

You can now play the generated `voiceover.wav` file to hear the narration in the specified style.

## Summary

In this tutorial, you learned how to:
1. Extract and encode video frames for AI processing.
2. Leverage GPT-4.1-mini's vision capabilities to generate descriptive text from video content.
3. Create a stylized narration script.
4. Use the GPT-4o TTS API to produce a high-quality voiceover with custom vocal characteristics.

This workflow enables you to automatically generate engaging descriptions and professional voiceovers for video content, enhancing accessibility and viewer engagement.