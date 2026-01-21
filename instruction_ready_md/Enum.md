# Gemini API: Constraining Output with Enums

This guide demonstrates how to use the Gemini API to constrain model outputs using Python's `Enum` class. You will learn to define a set of allowed values and ensure the model's response is one of them, which is useful for classification tasks and structured data generation.

## Prerequisites

### 1. Install the SDK
First, install the required Python client library.

```bash
pip install -q -U "google-genai>=1.0.0"
```

### 2. Configure Authentication
Set up your API key. Ensure it's stored securely. This example uses a Colab secret named `GOOGLE_API_KEY`. For other environments, you can set it as an environment variable.

```python
from google import genai

# Replace with your method of loading the API key
GOOGLE_API_KEY = "YOUR_API_KEY"  # e.g., from os.environ.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

### 3. Select a Model
Choose a Gemini model for this tutorial. The `gemini-2.5-flash-preview` is a good default for its balance of speed and capability.

```python
MODEL_ID = "gemini-2.5-flash-preview"
```

## Step 1: Basic Enum for Single-Choice Classification

In this step, you will ask the model to classify an image by selecting from a predefined list of categories.

### 1.1 Define the Enum Schema
Create an `Enum` class representing the possible instrument categories.

```python
import enum

class InstrumentClass(enum.Enum):
    PERCUSSION = "Percussion"
    STRING = "String"
    WOODWIND = "Woodwind"
    BRASS = "Brass"
    KEYBOARD = "Keyboard"
```

### 1.2 Prepare the Input Image
Download and load an image of an instrument. The model will analyze this image.

```python
import requests
from PIL import Image
from io import BytesIO

# Download an example image
image_url = "https://storage.googleapis.com/generativeai-downloads/images/instrument.jpg"
image_data = requests.get(image_url).content
instrument_image = Image.open(BytesIO(image_data))
```

### 1.3 Generate a Constrained Response
Use the `response_schema` parameter to pass your `Enum` class. Set `response_mime_type="text/x.enum"` to receive a plain text response matching one of the enum values.

```python
from google.genai import types

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[instrument_image, 'What is the category of this instrument?'],
    config=types.GenerateContentConfig(
        response_mime_type="text/x.enum",
        response_schema=InstrumentClass
    )
)

print(f"Classification: {response.text}")
```

**Expected Output:**
```
Classification: Brass
```

The model's response is constrained to one of the five `InstrumentClass` values.

### 1.4 Using JSON Output Format
You can also request the output as JSON. The result will be the same value, but formatted as a JSON string.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[instrument_image, 'What category of instrument is this?'],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=InstrumentClass
    )
)

print(f"JSON Response: {response.text}")
```

**Expected Output:**
```
JSON Response: "Brass"
```

## Step 2: Using Enums Within Complex JSON Schemas

Enums can be embedded within more complex `TypedDict` schemas to structure multi-field outputs. Here, you will generate a list of recipes, each with a name and a popularity grade from an enum.

### 2.1 Define the Nested Schema
Create a `Grade` enum and a `Recipe` `TypedDict` that uses it.

```python
import typing_extensions as typing

class Grade(enum.Enum):
    A_PLUS = 'a+'
    A = 'a'
    B = 'b'
    C = 'c'
    D = 'd'
    F = 'f'

class Recipe(typing.TypedDict):
    recipe_name: str
    grade: Grade  # The grade must be one of the defined enum values
```

### 2.2 Generate Structured Output
Request a list of `Recipe` objects. Pass `list[Recipe]` as the `response_schema`.

```python
result = client.models.generate_content(
    model=MODEL_ID,
    contents="List about 10 cookie recipes, grade them based on popularity",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[Recipe],
        # Optional: Increase timeout for longer generations
        http_options=types.HttpOptions(timeout=60000)
    ),
)
```

### 2.3 Parse and Display the Result
Load the JSON response and print it in a readable format.

```python
import json

response_data = json.loads(result.text)
print(json.dumps(response_data, indent=2))
```

**Example Output:**
```json
[
  {
    "recipe_name": "Classic Chocolate Chip",
    "grade": "a+"
  },
  {
    "recipe_name": "Oatmeal Raisin",
    "grade": "a"
  },
  {
    "recipe_name": "Peanut Butter",
    "grade": "a"
  },
  {
    "recipe_name": "Sugar Cookie",
    "grade": "a"
  },
  {
    "recipe_name": "Snickerdoodle",
    "grade": "a"
  },
  {
    "recipe_name": "Double Chocolate",
    "grade": "b"
  },
  {
    "recipe_name": "Shortbread",
    "grade": "b"
  },
  {
    "recipe_name": "Gingerbread",
    "grade": "b"
  },
  {
    "recipe_name": "White Chocolate Macadamia",
    "grade": "a"
  },
  {
    "recipe_name": "Biscotti",
    "grade": "c"
  }
]
```

Notice that every `grade` field in the output is a string that corresponds exactly to one of the values defined in the `Grade` enum.

## Summary and Next Steps

You have successfully used Python `Enum` classes to constrain Gemini API outputs. This technique is essential for:
*   **Classification tasks:** Ensuring the model picks from a valid set of labels.
*   **Structured data generation:** Enforcing consistent values for specific fields within JSON objects.

### Further Exploration
*   **Structured Output Documentation:** Dive deeper into schema definitions in the official [Structured Output](https://ai.google.dev/gemini-api/docs/structured-output) guide.
*   **API Reference:** Review the [`GenerationConfig`](https://ai.google.dev/api/generate-content#generationconfig) API reference for all configuration options.
*   **Related Patterns:**
    *   Use a full **JSON Schema** for more complex validation beyond enums (see the [JSON Mode](../quickstarts/JSON_mode.ipynb) guide).
    *   Integrate enums within **Function Calling** schemas to define precise argument options.
    *   Explore the [Text Summarization](../examples/json_capabilities/Text_Summarization.ipynb) example, which uses constrained output to format summaries consistently.
    *   See the [Object Detection](../examples/Object_detection.ipynb) example, which uses JSON output to standardize detection results.

By mastering enums and structured output, you gain precise control over the Gemini API's responses, making them more predictable and easier to integrate into your applications.