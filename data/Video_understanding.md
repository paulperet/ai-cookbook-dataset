# Video Understanding with Gemini: A Step-by-Step Guide

Gemini models are inherently multimodal, capable of analyzing various media types, including video, by leveraging their long context window. This guide demonstrates how to use the Gemini API to perform advanced video analysis tasks such as scene description, text extraction, information structuring, and screen recording analysis.

## Prerequisites & Setup

Before you begin, ensure you have the necessary libraries installed and your environment configured.

### 1. Install the SDK

Install the latest `google-genai` Python SDK.

```bash
pip install -U -q "google-genai>=1.16.0"
```

### 2. Set Up Your API Key

Store your Gemini API key in an environment variable named `GOOGLE_API_KEY`. If you're using Google Colab, you can store it as a secret.

```python
from google.colab import userdata
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

### 3. Initialize the Client

Create a client instance using your API key.

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)
```

### 4. Select a Model

Video understanding works best with the Gemini 2.5 or 3.0 series. Define your model ID.

```python
MODEL_ID = "gemini-3-flash-preview"  # Recommended for video tasks
```

### 5. Download Sample Videos

You'll work with sample videos for this tutorial. Download them to your local environment.

```bash
wget https://storage.googleapis.com/generativeai-downloads/videos/Pottery.mp4 -O Pottery.mp4 -q
wget https://storage.googleapis.com/generativeai-downloads/videos/Jukin_Trailcam_Videounderstanding.mp4 -O Trailcam.mp4 -q
wget https://storage.googleapis.com/generativeai-downloads/videos/post_its.mp4 -O Post_its.mp4 -q
wget https://storage.googleapis.com/generativeai-downloads/videos/user_study.mp4 -O User_study.mp4 -q
```

### 6. Upload Videos to Gemini

For analysis, videos must be uploaded and processed via the Gemini Files API. This function handles the upload and waits for processing to complete.

```python
import time

def upload_video(video_file_name):
    video_file = client.files.upload(file=video_file_name)

    while video_file.state == "PROCESSING":
        print('Waiting for video to be processed.')
        time.sleep(10)
        video_file = client.files.get(name=video_file.name)

    if video_file.state == "FAILED":
        raise ValueError(video_file.state)
    print(f'Video processing complete: ' + video_file.uri)
    return video_file

# Upload all sample videos
pottery_video = upload_video('Pottery.mp4')
trailcam_video = upload_video('Trailcam.mp4')
post_its_video = upload_video('Post_its.mp4')
user_study_video = upload_video('User_study.mp4')
```

### 7. Import Helper Libraries

Import the libraries needed for displaying results.

```python
import json
from IPython.display import display, Markdown, HTML
```

Now you're ready to start analyzing videos.

## Tutorial 1: Search and Describe Scenes in a Video

A common task is to generate a searchable index of scenes within a video. You'll use a trail camera video to identify and describe animal sightings.

**Step 1: Define your prompt and select the video.**

The prompt instructs the model to generate time-coded captions for each scene.

```python
prompt = """
For each scene in this video, generate captions that describe the scene along with any spoken text placed in quotation marks.
Place each caption into an object with the timecode of the caption in the video.
"""
video = trailcam_video
```

**Step 2: Generate the content analysis.**

Pass the video and prompt to the model.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[video, prompt]
)
```

**Step 3: Display the structured results.**

The model returns a JSON array of scenes with timestamps and descriptions.

```python
Markdown(response.text)
```

**Example Output:**

```json
[
  {
    "time": "00:00 - 00:17",
    "caption": "Two gray foxes in the wild, foraging. One comes into view from the right, followed by another. They are sniffing the ground, and one climbs onto a rock."
  },
  {
    "time": "00:17 - 00:34",
    "caption": "A mountain lion is seen sniffing the ground in a forest, then briefly looking up and walking off. (Night vision)"
  }
  // ... more scenes
]
```

You can customize the prompt for specific searches, like asking "Find all scenes containing foxes." The structured output can be post-processed to create interactive applications, such as clicking a timestamp to jump to that scene.

## Tutorial 2: Extract and Organize Text from Video

Gemini can read text within videos and organize it logically. You'll analyze a video of sticky notes to transcribe and categorize the content.

**Step 1: Define a prompt for transcription and organization.**

```python
prompt = "Transcribe the sticky notes, organize them and put it in a table. Can you come up with a few more ideas?"
video = post_its_video
```

**Step 2: Generate the analysis.**

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[video, prompt]
)
Markdown(response.text)
```

**Example Output:**

The model returns a Markdown table of transcribed project names and suggests additional ideas.

```
## Brainstorm: Project Names

| Project Name         | Project Name         |
| :------------------- | :------------------- |
| Aether               | Leo Minor            |
| Andromeda's Reach    | Lunar Eclipse        |
| ...                  | ...                  |

---

## A Few More Project Name Ideas:
1.  **Pulsar:** (Astronomical, suggests powerful and rhythmic energy)
2.  **Axiom:** (Mathematical/logical, implies a fundamental truth or starting point)
...
```

This demonstrates Gemini's ability to not only extract visible text but also use its reasoning to generate new, related content.

## Tutorial 3: Structure Information About Physical Objects

Gemini can analyze videos of physical objects, interpret handwritten notes, and structure the information. You'll analyze a video of pottery items with price tags and notes.

**Step 1: Define the prompt and add a system instruction.**

The system instruction ensures special characters (like `$`) are handled correctly in the output.

```python
prompt = "Give me a table of my items and notes"
video = pottery_video

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[video, prompt],
    config=types.GenerateContentConfig(
        system_instruction="Don't forget to escape the dollar signs",
    )
)
Markdown(response.text)
```

**Example Output:**

The model creates a detailed table, correctly associating each handwritten note with its corresponding pottery item.

```
| Category          | Item                | Description                                                                 | Dimensions      | Price   | Additional Notes          |
| :---------------- | :------------------ | :-------------------------------------------------------------------------- | :-------------- | :------ | :------------------------ |
| Drinkware         | Tumblers            | Stacked tumblers with earthy brown base and light blue wavy glaze...        | 4"h x 3"d (approx.) | \$20    | \#5 Artichoke double dip  |
| Bowls             | Small Bowls         | Two bowls with speckled, rustic exterior and darker, iridescent interior... | 3.5"h x 6.5"d   | \$35    |                           |
```

This shows Gemini's advanced capability to understand context and correlate disparate pieces of information within a video.

## Tutorial 4: Analyze Screen Recordings for Key Moments

Screen recordings from user studies or software tutorials can be automatically summarized to identify key actions and moments.

**Step 1: Define a summarization prompt.**

```python
prompt = """
Generate a paragraph that summarizes this video.
Keep it to 3 to 5 sentences with corresponding timecodes.
"""
video = user_study_video
```

**Step 2: Generate the summary.**

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[video, prompt]
)
Markdown(response.text)
```

**Example Output:**

```
(00:00 - 00:15) The video begins with a user opening a design application and creating a new project file. (00:15 - 00:45) They then use the shape tool to draw a rectangle and the text tool to add a label. (00:45 - 01:20) The user experiments with the color palette, changing the fill of the shape several times before settling on a blue shade. (01:20 - 01:50) Finally, they use the alignment tools to center the text within the shape and export the final image.
```

This automated analysis can save hours of manual review by quickly providing a structured overview of user interactions.

## Next Steps

You've learned how to use the Gemini API for several practical video analysis tasks. To explore further:

1.  **Experiment with Prompts:** Customize the prompts for your specific use case (e.g., "List all tools used in the screen recording").
2.  **Process Your Own Videos:** Upload your own video files using the `upload_video` function.
3.  **Build an Application:** Use the JSON outputs to build interactive applications, like a searchable video library. Check the [Video Starter Applet code](https://github.com/google-gemini/starter-applets/tree/main/video) for inspiration.
4.  **Explore Other Models:** Try the same prompts with different `MODEL_ID` values (like `gemini-2.5-pro`) to compare performance and output style.

Gemini's video understanding capabilities provide a powerful tool for automating content analysis, extracting structured data, and gaining insights from visual media.