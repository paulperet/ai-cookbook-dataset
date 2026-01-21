# Using Local Files with Gemini in Google Colab

This guide demonstrates how to upload a local text file to Google Colab and use its contents with the Gemini API for analysis. We'll use a transcript from the Apollo 11 mission as an example.

## Prerequisites

Before starting, ensure you have:
1. A Google Colab notebook
2. A Google AI Studio API key
3. The sample text file [a11.txt](https://storage.googleapis.com/generativeai-downloads/data/a11.txt) downloaded to your local machine

## Step 1: Upload Your File to Colab

1. In your Colab notebook, click **Files** in the left sidebar
2. Click the **Upload** button
3. Select the `a11.txt` file from your local machine

The file is now available in your Colab runtime's file system.

## Step 2: Read and Inspect the File

Let's verify the file was uploaded correctly by reading its contents:

```python
with open('a11.txt') as file:
    text_data = file.read()

# Print first 10 lines to verify content
for line in text_data.splitlines()[:10]:
    print(line)
```

You should see output similar to:
```
INTRODUCTION

This is the transcription of the Technical Air-to-Ground Voice Transmission (GOSS NET 1) from the Apollo 11 mission.

Communicators in the text may be identified according to the following list.

Spacecraft:
CDR	Commander	Neil A. Armstrong
CMP	Command module pilot   	Michael Collins
LMP	Lunar module pilot	Edwin E. ALdrin, Jr.
```

## Step 3: Install Required Libraries

Install the Google Generative AI Python SDK:

```bash
pip install -U -q "google-genai>=1.0.0"
```

## Step 4: Configure API Authentication

To use the Gemini API, you need to store your API key as a Colab Secret:

1. Click the key icon in the left sidebar (Secrets)
2. Add a new secret named `GOOGLE_API_KEY` with your API key as the value
3. Run the following code to access the key:

```python
from google.colab import userdata
from google import genai

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 5: Analyze the File with Gemini

Now you can use Gemini to analyze the content of your uploaded file. First, select a model:

```python
MODEL_ID = "gemini-3-flash-preview"  # You can change this to other available models
```

Then, send the file content to Gemini for analysis:

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        'What is this transcript?',
        text_data
    ]
)

print(response.text)
```

Gemini will analyze the transcript and provide a summary. For the Apollo 11 transcript, you might see output like:

```
Based on the provided text, this is a transcript of air-to-ground voice communications from the Apollo 11 mission. It includes:

*   **Introduction:** Explains the document is a transcription of GOSS NET 1 (Technical Air-to-Ground Voice Transmission) for Apollo 11. It lists abbreviations used for different speakers (Commander, Command Module Pilot, Lunar Module Pilot, various ground control and recovery personnel).

*   **Abbreviations Key:** Provides a key to understand who the different communicators are (e.g., CDR = Commander, CMP = Command Module Pilot, CC = Capsule Communicator).

*   **Air-to-Ground Voice Transcription:** The main body of the document, which is the transcribed dialogue between the Apollo 11 astronauts and mission control in Houston, as well as other ground stations. The text is segmented by timecode and location (e.g., MILA, GRAND BAHAMA ISLANDS, CANARY). It covers various aspects of the mission from launch to initial post-splashdown communications.

In short, this is the official record of what was said between the Apollo 11 crew and ground control during the mission, providing a detailed account of procedures, observations, and conversations.
```

## Next Steps

Now that you've successfully uploaded and analyzed a file, you can:
- Upload different file types (PDFs, images, videos)
- Ask more specific questions about the content
- Process multiple files in sequence
- Extract structured information from unstructured text

This workflow demonstrates how easily you can bring your own data into Colab and leverage Gemini's capabilities for analysis and insight generation.