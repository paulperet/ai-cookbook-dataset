# Extracting Structured Data from Stories with the Gemini API

This guide demonstrates how to use the Gemini API to analyze a story and extract structured information—such as characters, locations, and a plot summary—into a clean JSON format. This technique is useful for content analysis, database population, or as a preprocessing step for other creative workflows.

## Prerequisites

Before you begin, ensure you have:
1. A Google AI Studio API key.
2. The `google-genai` Python library installed.

## Setup

First, install the required library and configure your API key.

```bash
pip install -U -q "google-genai>=1.0.0"
```

```python
from google.colab import userdata
from google import genai

# Retrieve your API key from Colab Secrets
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

> **Note:** If you're not using Google Colab, store your `GOOGLE_API_KEY` in a secure environment variable and load it accordingly.

## Step 1: Generate a Sample Story

To demonstrate the extraction process, you'll first use Gemini to create a fantasy story.

```python
from IPython.display import Markdown

MODEL_ID = "gemini-3-flash-preview"
prompt = "Generate a 10 paragraph fantasy story. Include at least 2 named characters and 2 named locations. Give as much detail in the story as possible."

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)
story = response.text
Markdown(story)
```

The model will generate a story similar to the following (abbreviated for clarity):

> The flickering candlelight cast dancing shadows across Elara's face as she hunched over the ancient map... She traced a delicate line with her calloused finger, following a treacherous path winding through the jagged peaks of the Dragon's Tooth mountains... Elara, a cartographer by trade but an adventurer at heart, had spent years deciphering this cryptic document, rumored to lead to the lost city of Eldoria...

## Step 2: Define the Output Schema

To structure the extracted data, define a TypedDict schema. This tells the model exactly what format you expect.

```python
from typing_extensions import TypedDict  # In Python 3.12+, use `from typing import TypedDict`

class Character(TypedDict):
    name: str
    description: str
    alignment: str

class Location(TypedDict):
    name: str
    description: str

class TextSummary(TypedDict):
    synopsis: str
    genres: list[str]
    locations: list[Location]
    characters: list[Character]
```

## Step 3: Extract Structured Data

Now, prompt the model to analyze the generated story and return a JSON object matching your schema.

```python
prompt = f"""
Generate a summary of the story. Include a list of genres, locations, and characters.

{story}
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_schema": TextSummary
    }
)
```

## Step 4: Inspect the Results

The response is automatically parsed into a Python dictionary. Use `pprint` for a clean, readable output.

```python
from pprint import pprint

pprint(response.parsed)
```

You will receive structured output like this:

```json
{
  "characters": [
    {
      "alignment": "Good",
      "description": "A cartographer and adventurer seeking the lost city of Eldoria.",
      "name": "Elara"
    },
    {
      "alignment": "Neutral",
      "description": "A creature of stone and shadow tasked with protecting the secrets of Eldoria.",
      "name": "Guardian of the Hidden Passage"
    }
  ],
  "genres": ["Fantasy", "Adventure"],
  "locations": [
    {
      "description": "A treacherous mountain range with jagged peaks, dangerous paths, and a hidden passage to Eldoria.",
      "name": "Dragon's Tooth Mountains"
    },
    {
      "description": "A legendary, vanished civilization rumored to hold unimaginable riches and powerful magic.",
      "name": "Eldoria (Lost City)"
    },
    {
      "description": "A village where tales of Eldoria are whispered in taverns.",
      "name": "Whisperwind Village"
    }
  ],
  "synopsis": "Elara, a cartographer and adventurer, seeks the lost city of Eldoria, following an ancient map through the Dragon's Tooth mountains. She faces treacherous terrain and a guardian creature, ultimately using music to connect with the guardian and realizing that Eldoria's true value lies in its wisdom, not material riches. She chooses to leave Eldoria undisturbed, gaining newfound courage and respect for legends."
}
```

## Summary

In this tutorial, you used the Gemini API to:
1. Generate a detailed fantasy story.
2. Define a structured JSON schema for extraction.
3. Prompt the model to analyze the story and return key elements—characters, locations, genres, and a synopsis—in a consistent format.

This technique transforms unstructured text into structured data, which can be fed into databases, used for content tagging, or serve as a prompt for further creative generation. The same approach works across various text formats, including articles, reports, and transcripts.

To explore more JSON-related tasks with the Gemini API, refer to the other notebooks in this directory.