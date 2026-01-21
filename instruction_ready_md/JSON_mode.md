# Gemini API: JSON Mode Quickstart Guide

This guide demonstrates how to use the Gemini API to generate structured JSON output. You can constrain the model's response by either describing the schema in your prompt or by directly supplying a schema object.

## Prerequisites

### 1. Install the Required Library
First, install the Google Generative AI Python SDK.

```bash
pip install -U -q "google-genai>=1.0.0"
```

### 2. Configure Your API Key
Set up your API key. This example assumes you have stored your key in an environment variable or a secure secret manager. For Colab, you can use a secret named `GOOGLE_API_KEY`.

```python
from google import genai

# Replace with your method of loading the API key
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Method 1: Constrain Output via Prompt

In this method, you describe the desired JSON schema directly within your prompt.

### Step 1: Define the Prompt
Create a prompt that explicitly states the JSON structure you want the model to return.

```python
prompt = """
List a few popular cookie recipes using this JSON schema:

Recipe = {'recipe_name': str}
Return: list[Recipe]
"""
```

### Step 2: Configure the Model and Generate Content
Select a model and activate JSON mode by setting the `response_mime_type` in the generation configuration.

```python
MODEL_ID = "gemini-2.0-flash-exp"  # You can choose any available model

raw_response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config={
        'response_mime_type': 'application/json'
    },
)
```

### Step 3: Parse and Display the JSON Response
The model returns a JSON string. Parse it into a Python object and print it.

```python
import json

response = json.loads(raw_response.text)
print(json.dumps(response, indent=4))
```

**Example Output:**
```json
[
    {
        "recipe_name": "Chocolate Chip Cookies"
    },
    {
        "recipe_name": "Oatmeal Raisin Cookies"
    },
    {
        "recipe_name": "Peanut Butter Cookies"
    },
    {
        "recipe_name": "Sugar Cookies"
    },
    {
        "recipe_name": "Snickerdoodles"
    }
]
```

## Method 2: Supply a Schema Directly (Recommended)

For newer models (1.5 and beyond), you can pass a Python type or schema object directly. The model will strictly adhere to this schema, providing more robust and structured output.

### Step 1: Define a Typed Schema
Use Python's `typing` module to define a structured schema for your data.

```python
import typing

class Recipe(typing.TypedDict):
    recipe_name: str
    recipe_description: str
    recipe_ingredients: list[str]
```

### Step 2: Generate Content with the Schema
Pass the schema (`list[Recipe]` in this case) to the `response_schema` field in the configuration.

```python
result = client.models.generate_content(
    model=MODEL_ID,
    contents="List a few imaginative cookie recipes along with a one-sentence description as if you were a gourmet restaurant and their main ingredients",
    config={
        'response_mime_type': 'application/json',
        'response_schema': list[Recipe],
    },
)
```

### Step 3: Parse and Display the Result
Parse the JSON response and print it in a readable format.

```python
print(json.dumps(json.loads(result.text), indent=4))
```

**Example Output:**
```json
[
    {
        "recipe_name": "Lavender & White Chocolate Cloud Kisses",
        "recipe_description": "Delicate lavender-infused meringue cookies, airily light and meltingly tender, are kissed with swirls of creamy white chocolate.",
        "recipe_ingredients": [
            "Egg whites",
            "Sugar",
            "Dried lavender",
            "White chocolate"
        ]
    },
    {
        "recipe_name": "Smoked Paprika & Dark Chocolate Chili Sables",
        "recipe_description": "A sophisticated blend of smoky paprika, intense dark chocolate, and a whisper of chili creates a captivating sweet and savory experience in a crisp sable cookie.",
        "recipe_ingredients": [
            "All-purpose flour",
            "Unsalted butter",
            "Dark chocolate",
            "Smoked paprika",
            "Chili powder"
        ]
    }
]
```

## Summary and Next Steps

You have learned two methods to generate structured JSON output with the Gemini API:
1. **Prompt-based Schema:** Describe the format in your prompt.
2. **Direct Schema Supply:** Pass a Python type or schema object (recommended for newer models).

### Further Exploration

*   **API Documentation:** For detailed information, consult the [Structured Output](https://ai.google.dev/gemini-api/docs/structured-output) guide and the [`GenerationConfig`](https://ai.google.dev/api/generate-content#generationconfig) API reference.
*   **Related Examples:**
    *   Use constrained output for [Text Summarization](https://github.com/google-gemini/cookbook/blob/main/examples/json_capabilities/Text_Summarization.ipynb) to format summaries consistently.
    *   The [Object Detection](https://github.com/google-gemini/cookbook/blob/main/examples/Object_detection.ipynb) example uses JSON mode to standardize detection results.
*   **Other Constraint Techniques:** JSON is one way to guide model output. You can also explore:
    *   [Enum Constraints](../quickstarts/Enum.ipynb)
    *   [Function Calling](../quickstarts/Function_calling.ipynb)
    *   [Code Execution](../quickstarts/Code_Execution.ipynb)