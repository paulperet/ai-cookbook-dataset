# Gemini API: Analyze a Video - Historic Event Recognition

This notebook shows how you can use Gemini models' multimodal capabilities to recognize which historic event is happening in the video.

```
%pip install -U -q "google-genai>=1.0.0"
```

[First Entry, ..., Last Entry]

## Configure your API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.

```
from google import genai
from google.colab import userdata

API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=API_KEY)
```

## Example

This example uses [video of President Ronald Reagan's Speech at the Berlin Wall](https://s3.amazonaws.com/NARAprodstorage/opastorage/live/16/147/6014716/content/presidential-libraries/reagan/5730544/6-12-1987-439.mp4) taken on June 12 1987.

```
# Download video
path = "berlin.mp4"
url = "https://s3.amazonaws.com/NARAprodstorage/opastorage/live/16/147/6014716/content/presidential-libraries/reagan/5730544/6-12-1987-439.mp4"
!wget $url -O $path
```

[First Entry, ..., Last Entry]

```
# Upload video
video_file = client.files.upload(file=path)
```

```
import time
# Wait until the uploaded video is available
while video_file.state.name == "PROCESSING":
  print('.', end='')
  time.sleep(5)
  video_file = client.files.get(name=video_file.name)

if video_file.state.name == "FAILED":
  raise ValueError(video_file.state.name)
```

[First Entry, ..., Last Entry]

The uploaded video is ready for processing. This prompt instructs the model to provide basic information about the historical events portrayed in the video.

```
system_prompt = """
  You are historian who specializes in events caught on film.
  When you receive a video answer following questions:
  When did it happen?
  Who is the most important person in video?
  How the event is called?
"""
```

Some historic events touch on controversial topics that may get flagged by Gemini API, which blocks the response for the query.

Because of this, it might be a good idea to turn off safety settings.

```
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

```
from google.genai import types

MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
response = client.models.generate_content(
    model=f"models/{MODEL_ID}",
    contents=[
        "Analyze the video please",
        video_file
        ],
    config=types.GenerateContentConfig(
        system_instruction=system_prompt,
        safety_settings=safety_settings,
        ),
    )
print(response.text)
```

    Certainly! Here's the information about the video:
    
    Based on the video, here are the answers to your questions:
    
    - **When did it happen?** This event happened on June 12, 1987.
    - **Who is the most important person in the video?** The most important person in the video is Ronald Reagan, who was the President of the United States at the time.
    - **How is the event called?** The event is commonly referred to as President Reagan's "Tear Down This Wall" speech.

As you can see, the model correctly provided information about the dates, Ronald Reagan, who was the main subject of the video, and the name of this event.

You can delete the video to prevent unnecessary data storage.

```
# Delete video
client.files.delete(name=video_file.name)
```

## Summary

Now you know how you can prompt Gemini models with videos and use them to recognize historic events.

This notebook shows only one of many use cases. Check the [Video understanding](../quickstarts/Video_understanding.ipynb) notebook for more examples of using the Gemini API with videos.