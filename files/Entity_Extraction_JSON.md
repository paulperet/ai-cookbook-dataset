

# Gemini API: Entity Extraction

You will use Gemini to extract all fields that fit one of the predefined classes and label them.

```
%pip install -U -q "google-genai>=1.0.0"
```

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.

```
from google.colab import userdata
from google import genai

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Example

```
from enum import Enum
from typing_extensions import TypedDict # in python 3.12 replace typing_extensions with typing
from google.genai import types

entity_recognition_text = "John Johnson, the CEO of the Oil Inc. and Coal Inc. companies, has unveiled plans to build a new factory in Houston, Texas."
prompt = f"""
Generate list of entities in text based on the following Python class structure:

class CategoryEnum(str, Enum):
    Person = 'Person'
    Company = 'Company'
    State = 'State'
    City = 'City'

class Entity(TypedDict):
  name: str
  category: CategoryEnum

class Entities(TypedDict):
  entities: list[Entity]

{entity_recognition_text}"""
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json"
    )
  )
```

```
import json

print(json.dumps(json.loads(response.text), indent=4))
```

    {
        "entities": [
            {
                "name": "John Johnson",
                "category": "Person"
            },
            {
                "name": "Oil Inc.",
                "category": "Company"
            },
            {
                "name": "Coal Inc.",
                "category": "Company"
            },
            {
                "name": "Houston",
                "category": "City"
            },
            {
                "name": "Texas",
                "category": "State"
            }
        ]
    }

## Summary
You have used the Gemini API to extract entities of predefined categories with their labels. You extracted every person, company, state, and country. You are not limited to these categories, as this should work with any category of your choice.

Please see the other notebooks in this directory to learn more about how you can use the Gemini API for other JSON related tasks.