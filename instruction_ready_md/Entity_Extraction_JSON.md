# Entity Extraction with the Gemini API

This guide demonstrates how to use the Gemini API to perform structured entity extraction from text. You will define custom categories and extract all matching entities, receiving the results in a clean JSON format.

## Prerequisites & Setup

First, install the required Python client library and configure your API key.

### 1. Install the Library
Run the following command in your terminal or notebook cell to install the `google-genai` package.

```bash
pip install -U -q "google-genai>=1.0.0"
```

### 2. Configure Your API Key
You need a Gemini API key. Store it securely. In this example, we'll retrieve it from an environment variable named `GOOGLE_API_KEY`.

```python
import os
from google import genai

# Retrieve your API key from an environment variable
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

# Initialize the client
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 1: Define Your Entity Schema

To guide the model, you must define the structure of the entities you want to extract. We'll use Python's `Enum` for categories and `TypedDict` for the data structure.

```python
from enum import Enum
from typing import List, TypedDict

# Define the allowed categories using an Enum
class CategoryEnum(str, Enum):
    Person = 'Person'
    Company = 'Company'
    State = 'State'
    City = 'City'

# Define the structure for a single entity
class Entity(TypedDict):
    name: str
    category: CategoryEnum

# Define the structure for the final response
class Entities(TypedDict):
    entities: List[Entity]
```

## Step 2: Prepare Your Text and Prompt

Now, create the text you want to analyze and construct a prompt that includes your defined schema.

```python
# The text from which you want to extract entities
entity_recognition_text = "John Johnson, the CEO of the Oil Inc. and Coal Inc. companies, has unveiled plans to build a new factory in Houston, Texas."

# Construct the prompt, including the class definitions for context
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
```

## Step 3: Call the Gemini API

Select a model and make the API call. We'll configure it to return JSON directly.

```python
from google.genai import types

MODEL_ID = "gemini-2.0-flash-exp" # You can change this to other available models like "gemini-1.5-pro"

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
        temperature=0,           # Set to 0 for deterministic, structured outputs
        response_mime_type="application/json" # Request the response in JSON format
    )
)
```

## Step 4: Parse and Display the Results

The API returns a JSON string. Parse it and display the extracted entities.

```python
import json

# Parse the JSON response
result = json.loads(response.text)

# Pretty-print the result
print(json.dumps(result, indent=4))
```

**Expected Output:**
```json
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
```

## Summary

You have successfully used the Gemini API to perform structured entity extraction. By providing a clear schema via Python class definitions in the prompt, you instructed the model to identify and categorize entities such as people, companies, and locations.

**Key Takeaways:**
1.  **Schema Guidance:** Defining your expected output structure (using `Enum` and `TypedDict`) is crucial for getting consistent, parseable JSON.
2.  **Model Configuration:** Setting `response_mime_type="application/json"` and `temperature=0` ensures the model returns a structured, deterministic response.
3.  **Flexibility:** You are not limited to the categories used here. You can modify the `CategoryEnum` and `Entity` class to extract any type of entity relevant to your use case (e.g., `Product`, `Date`, `Event`).

This pattern is powerful for transforming unstructured text into structured data, enabling integration with databases, analytics pipelines, or other applications.