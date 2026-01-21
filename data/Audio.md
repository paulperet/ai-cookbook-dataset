# Gemini API Audio Guide: Process Audio and Video with AI

This guide demonstrates how to use the Gemini API to analyze audio files and YouTube videos. You'll learn to upload files, generate transcripts, extract summaries from specific timestamps, and process video content.

## Prerequisites

Before you begin, ensure you have the following:

1.  A **Google AI API Key**. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Python 3.7+ installed on your system.

## Step 1: Install Required Libraries

First, install the necessary Python packages. The `google-genai` library is the official SDK, and `pydub` is useful for manipulating audio files.

```bash
pip install -U "google-genai>=1.0.0" pydub
```

## Step 2: Configure the Gemini Client

Import the libraries and initialize the client with your API key. Replace `'YOUR_API_KEY'` with your actual key.

```python
from google import genai

# Initialize the client
GOOGLE_API_KEY = 'YOUR_API_KEY'
client = genai.Client(api_key=GOOGLE_API_KEY)

# Select a model
MODEL_ID = "gemini-2.0-flash-exp" # A capable, fast model for audio tasks
```

## Step 3: Download and Upload an Audio File

To analyze an audio file, you must first upload it using the Gemini File API. Let's start by downloading a sample file.

```python
import requests

# URL for a sample audio file (JFK's 1961 State of the Union Address)
URL = "https://storage.googleapis.com/generativeai-downloads/data/State_of_the_Union_Address_30_January_1961.mp3"
local_filename = "sample.mp3"

# Download the file
response = requests.get(URL)
with open(local_filename, 'wb') as f:
    f.write(response.content)
print(f"Downloaded audio file to {local_filename}")

# Upload the file to the Gemini API
your_audio_file = client.files.upload(file=local_filename)
print("File uploaded successfully.")
```

## Step 4: Generate a Summary from Audio

Now, prompt the model to listen to the audio and provide a summary.

```python
response = client.models.generate_content(
  model=MODEL_ID,
  contents=[
    'Listen carefully to the following audio file. Provide a brief summary.',
    your_audio_file,
  ]
)

print("Summary of the audio:")
print(response.text)
```

**Expected Output:**
The model will return a concise summary of the key points from President Kennedy's address, covering domestic economic challenges, foreign policy, and calls for national unity.

## Step 5: Process a Short Audio Clip Inline

For small audio segments, you can embed the audio data directly in the request without uploading. This example uses `pydub` to extract the first 10 seconds.

```python
from pydub import AudioSegment
from google.genai import types

# Load the audio file
sound = AudioSegment.from_mp3("sample.mp3")

# Extract the first 10 seconds (10,000 milliseconds)
audio_clip = sound[:10000]

# Convert the clip to bytes
audio_bytes = audio_clip.export().read()

# Generate a description of the clip
response = client.models.generate_content(
  model=MODEL_ID,
  contents=[
    'Describe this audio clip',
    types.Part.from_bytes(
      data=audio_bytes,
      mime_type='audio/mp3',
    )
  ]
)

print("Description of the audio clip:")
print(response.text)
```

**Note on File Size:** The maximum total request size (including prompts and inline files) is 100 MB. For larger files or repeated use, the File API (Step 3) is more efficient.

## Step 6: Generate a Transcript from Specific Timestamps

You can direct the model to transcribe only a specific portion of the audio by providing timestamps in the format `MM:SS`.

```python
# Create a prompt targeting a specific time range
prompt = "Provide a transcript of the speech between the timestamps 02:30 and 03:29."

response = client.models.generate_content(
  model=MODEL_ID,
  contents=[
    prompt,
    your_audio_file,
  ]
)

print("Transcript for 02:30 to 03:29:")
print(response.text)
```

## Step 7: Analyze a YouTube Video

The Gemini API can also process YouTube videos directly via their URL. Provide a structured prompt to get a detailed analysis.

```python
from google.genai import types

youtube_url = "https://www.youtube.com/watch?v=RDOMKIw1aF4" # Replace with your video URL

prompt = """
    Analyze the following YouTube video content. Provide a concise summary covering:
    1.  **Main Thesis/Claim:** What is the central point the creator is making?
    2.  **Key Topics:** List the main subjects discussed.
    3.  **Call to Action:** Identify any explicit requests made to the viewer.
    4.  **Summary:** Provide a concise summary of the video content.
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=types.Content(
        parts=[
            types.Part(text=prompt),
            types.Part(
                file_data=types.FileData(file_uri=youtube_url)
            )
        ]
    )
)

print("YouTube Video Analysis:")
print(response.text)
```

## Step 8: Count Tokens in Your Audio File

Understanding token usage helps manage costs and API limits. Audio files consume tokens at a fixed rate per second.

```python
count_tokens_response = client.models.count_tokens(
    model=MODEL_ID,
    contents=[your_audio_file],
)

print(f"Audio file tokens: {count_tokens_response.total_tokens}")
```

## Next Steps & Resources

You've successfully used the Gemini API to process audio and video. Here are ways to continue exploring:

*   **API Documentation:** Dive deeper into [Audio capabilities](https://ai.google.dev/gemini-api/docs/audio) and the [File API reference](https://ai.google.dev/api/files).
*   **Related Examples:** Explore the [Voice Memos notebook](https://github.com/google-gemini/cookbook/blob/main/examples/Voice_memos.ipynb) for creative applications.
*   **Prompting Guide:** Learn best practices for [prompting with media files](https://ai.google.dev/tutorials/prompting_with_media).

This guide provides a foundation for building applications that leverage AI-powered audio and video analysis, from automated note-taking to content moderation.