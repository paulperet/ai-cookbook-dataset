# Introduction to GPT-4o and GPT-4o mini

GPT-4o ("o" for "omni") and GPT-4o mini are natively multimodal models designed to handle a combination of text, audio, and video inputs, and can generate outputs in text, audio, and image formats. GPT-4o mini is the lightweight version of GPT-4o.

### Background

Before GPT-4o, users could interact with ChatGPT using Voice Mode, which operated with three separate models. GPT-4o integrates these capabilities into a single model that's trained across text, vision, and audio. This unified approach ensures that all inputs — whether text, visual, or auditory — are processed cohesively by the same neural network.

GPT-4o mini is the next iteration of this omni model family, available in a smaller and cheaper version. This model offers higher accuracy than GPT-3.5 Turbo while being just as fast and supporting multimodal inputs and outputs.

### Current API Capabilities

Currently, the `gpt-4o-mini` model supports `{text, image}`, with `{text}` outputs, the same modalities as `gpt-4-turbo`.  As a preview, we will also be using the `gpt-4o-audio-preview` model to showcase transcription though the GPT4o model.

## Getting Started

### Install OpenAI SDK for Python

```python
%pip install --upgrade openai
```

### Configure the OpenAI client and submit a test request
To setup the client for our use, we need to create an API key to use with our request. Skip these steps if you already have an API key for usage. 

You can get an API key by following these steps:
1. [Create a new project](https://help.openai.com/en/articles/9186755-managing-your-work-in-the-api-platform-with-projects)
2. [Generate an API key in your project](https://platform.openai.com/api-keys)
3. (RECOMMENDED, BUT NOT REQUIRED) [Setup your API key for all projects as an env var](https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key)

Once we have this setup, let's start with a simple {text} input to the model for our first request. We'll use both `system` and `user` messages for our first request, and we'll receive a response from the `assistant` role.

```python
from openai import OpenAI 
import os

## Set the API key and model name
MODEL="gpt-4o-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>"))
```

```python
completion = client.chat.completions.create(
  model=MODEL,
  messages=[
    {"role": "system", "content": "You are a helpful assistant. Help me with my math homework!"}, # <-- This is the system message that provides context to the model
    {"role": "user", "content": "Hello! Could you solve 2+2?"}  # <-- This is the user message for which the model will generate a response
  ]
)

print("Assistant: " + completion.choices[0].message.content)
```

    Assistant: Of course! \( 2 + 2 = 4 \).

## Image Processing
GPT-4o mini can directly process images and take intelligent actions based on the image. We can provide images in two formats:
1. Base64 Encoded
2. URL

Let's first view the image we'll use, then try sending this image as both Base64 and as a URL link to the API

```python
from IPython.display import Image, display, Audio, Markdown
import base64

IMAGE_PATH = "data/triangle.png"

# Preview image for context
display(Image(IMAGE_PATH))
```

#### Base64 Image Processing

```python
# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image(IMAGE_PATH)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me with my math homework!"},
        {"role": "user", "content": [
            {"type": "text", "text": "What's the area of the triangle?"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64_image}"}
            }
        ]}
    ],
    temperature=0.0,
)

print(response.choices[0].message.content)
```

    To find the area of the triangle, you can use the formula:
    
    \[
    \text{Area} = \frac{1}{2} \times \text{base} \times \text{height}
    \]
    
    In the triangle you provided:
    
    - The base is \(9\) (the length at the bottom).
    - The height is \(5\) (the vertical line from the top vertex to the base).
    
    Now, plug in the values:
    
    \[
    \text{Area} = \frac{1}{2} \times 9 \times 5
    \]
    
    Calculating this:
    
    \[
    \text{Area} = \frac{1}{2} \times 45 = 22.5
    \]
    
    Thus, the area of the triangle is **22.5 square units**.

#### URL Image Processing

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that responds in Markdown. Help me with my math homework!"},
        {"role": "user", "content": [
            {"type": "text", "text": "What's the area of the triangle?"},
            {"type": "image_url", "image_url": {
                "url": "https://upload.wikimedia.org/wikipedia/commons/e/e2/The_Algebra_of_Mohammed_Ben_Musa_-_page_82b.png"}
            }
        ]}
    ],
    temperature=0.0,
)

print(response.choices[0].message.content)
```

    To find the area of the triangle, you can use the formula:
    
    \[
    \text{Area} = \frac{1}{2} \times \text{base} \times \text{height}
    \]
    
    In the triangle you provided:
    
    - The base is \(9\) (the length at the bottom).
    - The height is \(5\) (the vertical line from the top vertex to the base).
    
    Now, plug in the values:
    
    \[
    \text{Area} = \frac{1}{2} \times 9 \times 5
    \]
    
    Calculating this gives:
    
    \[
    \text{Area} = \frac{1}{2} \times 45 = 22.5
    \]
    
    Thus, the area of the triangle is **22.5 square units**.

## Video Processing
While it's not possible to directly send a video to the API, GPT-4o can understand videos if you sample frames and then provide them as images. 

Since GPT-4o mini in the API does not yet support audio-in (as of July 2024), we'll use a combination of GPT-4o mini and Whisper to process both the audio and visual for a provided video, and showcase two usecases:
1. Summarization
2. Question and Answering

### Setup for Video Processing
We'll use two python packages for video processing - opencv-python and moviepy. 

These require [ffmpeg](https://ffmpeg.org/about.html), so make sure to install this beforehand. Depending on your OS, you may need to run `brew install ffmpeg` or `sudo apt install ffmpeg`

```python
%pip install opencv-python
%pip install moviepy
```

### Process the video into two components: frames and audio

```python
import cv2
from moviepy import *
import time
import base64

# We'll be using the OpenAI DevDay Keynote Recap video. You can review the video here: https://www.youtube.com/watch?v=h02ti0Bl6zk
VIDEO_PATH = "data/keynote_recap.mp4"
```

```python
def process_video(video_path, seconds_per_frame=2):
    base64Frames = []
    base_video_path, _ = os.path.splitext(video_path)

    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()

    # Extract audio from video
    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()

    print(f"Extracted {len(base64Frames)} frames")
    print(f"Extracted audio to {audio_path}")
    return base64Frames, audio_path

# Extract 1 frame per second. You can adjust the `seconds_per_frame` parameter to change the sampling rate
base64Frames, audio_path = process_video(VIDEO_PATH, seconds_per_frame=1)

```

    MoviePy - Writing audio in data/keynote_recap.mp3
    MoviePy - Done.
    Extracted 218 frames
    Extracted audio to data/keynote_recap.mp3

```python
## Display the frames and audio for context
display_handle = display(None, display_id=True)
for img in base64Frames:
    display_handle.update(Image(data=base64.b64decode(img.encode("utf-8")), width=600))
    time.sleep(0.025)

Audio(audio_path)
```

### Example 1: Summarization
Now that we have both the video frames and the audio, let's run a few different tests to generate a video summary to compare the results of using the models with different modalities. We should expect to see that the summary generated with context from both visual and audio inputs will be the most accurate, as the model is able to use the entire context from the video.

1. Visual Summary
2. Audio Summary
3. Visual + Audio Summary

#### Visual Summary
The visual summary is generated by sending the model only the frames from the video. With just the frames, the model is likely to capture the visual aspects, but will miss any details discussed by the speaker.

```python
response = client.chat.completions.create(
    model=MODEL,
    messages=[
    {"role": "system", "content": "You are generating a video summary. Please provide a summary of the video. Respond in Markdown."},
    {"role": "user", "content": [
        "These are the frames from the video.",
        *map(lambda x: {"type": "image_url", 
                        "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames)
        ],
    }
    ],
    temperature=0,
)
print(response.choices[0].message.content)
```

    # OpenAI Dev Day Summary
    
    ## Overview
    The video captures highlights from OpenAI's Dev Day, showcasing new advancements and features in AI technology, particularly focusing on the latest developments in the GPT-4 model and its applications.
    
    ## Key Highlights
    
    ### Event Introduction
    - The event is branded as "OpenAI Dev Day," setting the stage for discussions on AI advancements.
    
    ### Keynote Recap
    - The keynote features a recap of significant updates and innovations in AI, particularly around the GPT-4 model.
    
    ### New Features
    - **GPT-4 Turbo**: Introduction of a faster and more efficient version of GPT-4, emphasizing improved performance and reduced costs.
    - **DALL-E 3**: Updates on the image generation model, showcasing its capabilities and integration with other tools.
    - **Custom Models**: Introduction of features allowing users to create tailored AI models for specific tasks.
    
    ### Technical Innovations
    - **Function Calling**: Demonstration of how the model can handle complex instructions and execute functions based on user queries.
    - **JSON Mode**: A new feature that allows for structured data handling, enhancing the model's ability to process and respond to requests.
    
    ### User Experience Enhancements
    - **Threading and Retrieval**: New functionalities that improve how users can interact with the model, making it easier to manage conversations and retrieve information.
    - **Code Interpreter**: Introduction of a tool that allows the model to execute code, expanding its utility for developers.
    
    ### Community Engagement
    - The event emphasizes community involvement, encouraging developers to explore and utilize the new features in their applications.
    
    ### Conclusion
    - The event wraps up with a call to action for developers to engage with the new tools and features, fostering innovation in AI applications.
    
    ## Closing Remarks
    The OpenAI Dev Day serves as a platform for showcasing the latest advancements in AI technology, encouraging developers to leverage these innovations for enhanced applications and user experiences.

The results are as expected - the model is able to capture the high level aspects of the video visuals, but misses the details provided in the speech.

#### Audio Summary
The audio summary is generated by sending the model the audio transcript. With just the audio, the model is likely to bias towards the audio content, and will miss the context provided by the presentations and visuals.

`{audio}` input for GPT-4o is currently in preview, but will be incorporated into the base model in the near future.  Because of this, we will use the `gpt-4o-audio-preview` model to process the audio.

```python
#transcribe the audio
with open(audio_path, 'rb') as audio_file:
    audio_content = base64.b64encode(audio_file.read()).decode('utf-8')

response = client.chat.completions.create(
            model='gpt-4o-audio-preview',
            modalities=["text"],
            messages=[
                    {   "role": "system", 
                        "content":"You are generating a transcript. Create a transcript of the provided audio."
                    },
                    {
                        "role": "user",
                        "content": [
                            { 
                                "type": "text",
                                "text": "this is the audio."
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_content,
                                    "format": "mp3"
                                }
                            }
                        ]
                    },
                ],
            temperature=0,
        )

# Extract and return the transcription
transcription = response.choices[0].message.content
print (transcription)
```

Looking good.  Now let's summarize this and format in markdown.

```python
#summarize the transcript
response = client.chat.completions.create(
            model=MODEL,
            modalities=["text"],
            messages=[
                {"role": "system", "content": "You are generating a transcript summary. Create a summary of the provided transcription. Respond in Markdown."},
                {"role": "user", "content": f"Summarize this text: {transcription}"},
            ],
            temperature=0,
        )
transcription_summary = response.choices[0].message.content
print (transcription_summary)
```

    # OpenAI Dev Day Summary
    
    On the inaugural OpenAI Dev Day, several significant updates and features were announced:
    
    - **Launch of GPT-4 Turbo**: This new model supports up to 128,000 tokens of context and is designed to follow instructions more effectively.
      
    - **JSON Mode**: A new feature that ensures the model responds with valid JSON.
    
    - **Function Calling**: Users can now call multiple functions simultaneously, enhancing the model's capabilities.
    
    - **Retrieval Feature**: This allows models to access external knowledge from documents or databases, improving their contextual understanding.
    
    - **Knowledge Base**: GPT-4 Turbo has knowledge up to April 2023, with plans for ongoing improvements.
    
    - **Dolly 3 and New Models**: The introduction of Dolly 3, GPT-4 Turbo with Vision, and a new Text-to-Speech model, all available via the API.
    
    - **Custom Models Program**: A new initiative where researchers collaborate with companies to create tailored models for specific use cases.
    
    - **Increased Rate Limits**: Established GPT-4 customers will see a doubling of tokens per minute, with options to request further changes in API settings.
    
    - **Cost Efficiency**: GPT-4 Turbo is significantly cheaper than its predecessor, with a 3x reduction for prompt tokens and 2x for completion tokens.
    
    - **Introduction of GPTs**: Tailored versions of ChatGPT designed for specific purposes, allowing users to create and share private or public GPTs easily, even without coding skills.
    
    - **Upcoming GPT Store**: A platform for users to share their GPT creations.
    
    - **Assistance API**: Features persistent threads, built-in retrieval, a code interpreter, and improved function calling to streamline user interactions.
    
    The event concluded with excitement about the future of AI technology and an invitation for attendees to return next year to see further advancements.

The audio summary is biased towards the content discussed during the speech, but comes out with much less structure than the video summary.

#### Audio + Visual Summary
The Audio + Visual summary is generated by sending the model both the visual and the audio from the video at once. When sending both of these, the model is expected to better summarize since it can perceive the entire video at once.

```python
## Generate a summary with visual and audio
response = client.chat.completions.create(
    model=MODEL,
    messages=[
    {"role": "system", "content":"""You are generating a video summary. Create a summary of the provided video and its transcript. Respond in Markdown"""},
    {"role": "user", "content": [
        "These are the frames from the video.",
        *map(lambda x: {"type": "image_url", 
                        "image_url": {"url": f'data:image/jpg;base64,{