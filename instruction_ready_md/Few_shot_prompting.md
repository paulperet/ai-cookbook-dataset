# Gemini API Cookbook: Few-Shot Prompting

This guide demonstrates how to use few-shot prompting with the Gemini API to improve model performance on tasks that require specific formats or reasoning patterns. Few-shot prompting involves providing the model with example question-answer pairs before presenting your actual query.

## Prerequisites

First, install the required library and configure your API key.

### 1. Install the Gemini SDK

```bash
pip install -U -q "google-genai>=1.0.0"
```

### 2. Import Libraries

```python
from google import genai
from google.genai import types
```

### 3. Configure Your API Key

Store your API key in a Colab Secret named `GOOGLE_API_KEY`. If you haven't set this up, refer to the [Authentication guide](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb).

```python
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 1: Define Your Model

Choose a Gemini model for the examples. We'll use the efficient `gemini-2.5-flash` model.

```python
MODEL_ID = "gemini-2.5-flash"
```

## Step 2: Implement a Sorting Task with Few-Shot Examples

In this example, we'll teach the model to sort animals by size using example pairs.

1.  **Construct the Prompt:** The prompt includes the task instruction and two example question-answer pairs. The final line presents the new question without an answer.
2.  **Generate the Response:** Send the prompt to the model.
3.  **Print the Result:** Display the model's completion.

```python
prompt = """
    Sort the animals from biggest to smallest.
    Question: Sort Tiger, Bear, Dog
    Answer: Bear > Tiger > Dog
    Question: Sort Cat, Elephant, Zebra
    Answer: Elephant > Zebra > Cat
    Question: Sort Whale, Goldfish, Monkey
    Answer:
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

print(response.text)
```

**Expected Output:**
```
Whale > Monkey > Goldfish
```

The model successfully applies the pattern from the examples to sort the new set of animals.

## Step 3: Implement a Structured Extraction Task

Few-shot prompting is especially powerful for guiding the model to produce outputs in a specific format, like JSON. Here, we'll extract city-country pairs.

1.  **Construct the Prompt:** The prompt defines the extraction task and provides examples showing the exact JSON output format.
2.  **Configure the Response:** Use `GenerateContentConfig` to request the response in `application/json` format. This helps ensure a valid JSON structure.
3.  **Generate and Print the Response.**

```python
prompt = """
    Extract cities from text, include country they are in.
    USER: I visited Mexico City and Poznan last year
    MODEL: {"Mexico City": "Mexico", "Poznan": "Poland"}
    USER: She wanted to visit Lviv, Monaco and Maputo
    MODEL: {"Lviv": "Ukraine", "Monaco": "Monaco", "Maputo": "Mozambique"}
    USER: I am currently in Austin, but I will be moving to Lisbon soon
    MODEL:
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
    ),
)

print(response.text)
```

**Expected Output:**
```json
{"Austin": "USA", "Lisbon": "Portugal"}
```

The model correctly identifies the cities and their associated countries, returning the data in the specified JSON schema.

## Next Steps

You can adapt this few-shot technique for various tasks:
*   **Classify your own data** by providing examples of different categories.
*   **Experiment with other prompting techniques**, such as zero-shot prompting (providing no examples) or chain-of-thought prompting (asking the model to reason step-by-step).
*   Explore more examples in the [Gemini Cookbook repository](https://github.com/google-gemini/cookbook).