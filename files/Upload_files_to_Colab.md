# Upload files to Google Colab

You can upload files to Google Colab to quickly experiment with your own data. For example, you can upload video or image files to use with the Files API, or a upload a text file to read in with a long context model like Gemini Flash. This example shows you how to upload files to the Colab runtime and use them in your code.

First, download the following file to your local machine:

*  [a11.txt](https://storage.googleapis.com/generativeai-downloads/data/a11.txt)

It contains a transcript of transmissions from the Apollo 11 mission, originally from https://www.nasa.gov/history/alsj/a11/a11trans.html.

Next, upload the file to Google Colab. To do so, first click **Files** on the left sidebar, then click the **Upload** button:

You're now able to use the file in Colab!


```
with open('a11.txt') as file:
  text_data = file.read()

# Print first 10 lines
for line in text_data.splitlines()[:10]:
  print(line)
```

    INTRODUCTION
    
    This is the transcription of the Technical Air-to-Ground Voice Transmission (GOSS NET 1) from the Apollo 11 mission.
    
    Communicators in the text may be identified according to the following list.
    
    Spacecraft:
    CDR	Commander	Neil A. Armstrong
    CMP	Command module pilot   	Michael Collins
    LMP	Lunar module pilot	Edwin E. ALdrin, Jr.


This makes it simple to use the file with the Gemini API.


```
%pip install -U -q "google-genai>=1.0.0"
```

    [First Entry, ..., Last Entry]

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata
from google import genai

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        'What is this transcript?',
        text_data
    ]
)
print(response.text)
```

    Based on the provided text, this is a transcript of air-to-ground voice communications from the Apollo 11 mission. It includes:
    
    *   **Introduction:** Explains the document is a transcription of GOSS NET 1 (Technical Air-to-Ground Voice Transmission) for Apollo 11. It lists abbreviations used for different speakers (Commander, Command Module Pilot, Lunar Module Pilot, various ground control and recovery personnel).
    
    *   **Abbreviations Key:** Provides a key to understand who the different communicators are (e.g., CDR = Commander, CMP = Command Module Pilot, CC = Capsule Communicator).
    
    *   **Air-to-Ground Voice Transcription:** The main body of the document, which is the transcribed dialogue between the Apollo 11 astronauts and mission control in Houston, as well as other ground stations. The text is segmented by timecode and location (e.g., MILA, GRAND BAHAMA ISLANDS, CANARY). It covers various aspects of the mission from launch to initial post-splashdown communications.
    
    In short, this is the official record of what was said between the Apollo 11 crew and ground control during the mission, providing a detailed account of procedures, observations, and conversations.