# Text Classification with the Gemini API

This guide demonstrates how to use the Gemini API to analyze a piece of text and classify the topics it discusses, along with their relevance.

## Prerequisites

First, ensure you have the necessary Python library installed.

```bash
pip install -U -q "google-genai>=1.0.0"
```

## Setup: Configure the API Client

You will need a Gemini API key. Store it securely and initialize the client.

```python
from google.colab import userdata
from google import genai

GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 1: Generate a Sample Article

To demonstrate classification, let's first generate a sample article using Gemini. We'll ask it to write about sports and include another topic.

```python
MODEL_ID = "gemini-3-flash-preview" # You can choose other available models

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Generate a 5 paragraph article about Sports, include one other topic",
)
article = response.text
print(article[:500]) # Print the first part to verify
```

The model will generate a multi-paragraph article. For this tutorial, we'll work with an article that discusses sports, health, economics, social issues, and art.

## Step 2: Define the Classification Schema

We want the API to return a structured list of topics with their relevance. We'll define the expected data structure using Python's `enum` and `TypedDict`.

```python
import enum
from typing_extensions import TypedDict  # For Python <3.12, use `typing`

from google.genai import types

class Relevance(enum.Enum):
    WEAK = "weak"
    STRONG = "strong"

class Topic(TypedDict):
    topic: str
    relevance: Relevance
```

## Step 3: Create the System Instruction

We need to guide the model on how to perform the classification. A clear system instruction is key.

```python
sys_instruction = """
Generate topics from text. Ensure that topics are general e.g. "Health".
Strong relevance is obtained when the topic is a core tenet of the content.
Weak relevance reflects one or two mentions.
"""
```

## Step 4: Call the API for Classification

Now, we send our generated article to the model with the system instruction and request a JSON response matching our schema.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=article,
    config=types.GenerateContentConfig(
        system_instruction=sys_instruction,
        response_mime_type="application/json",
        response_schema=list[Topic],
    )
)
```

## Step 5: Parse and Display the Results

The response contains a parsed object thanks to the schema. Let's inspect the classified topics.

```python
from pprint import pprint

pprint(response.parsed)
```

**Output:**
```
[{'relevance': <Relevance.STRONG: 'strong'>, 'topic': 'Sports'},
 {'relevance': <Relevance.STRONG: 'strong'>, 'topic': 'Health'},
 {'relevance': <Relevance.WEAK: 'weak'>, 'topic': 'Economics'},
 {'relevance': <Relevance.STRONG: 'strong'>, 'topic': 'Social Issues'},
 {'relevance': <Relevance.WEAK: 'weak'>, 'topic': 'Art'}]
```

The model successfully identified the main topics. As requested, "Sports" is strongly relevant. "Health" and "Social Issues" are also core tenets, while "Economics" and "Art" are only briefly mentioned, resulting in a weak relevance score.

## Summary

You have now learned how to use the Gemini API to perform text classification into structured JSON. The key steps are:
1.  Defining a clear schema for the expected output.
2.  Crafting a precise system instruction to guide the model's analysis.
3.  Configuring the API call to request a JSON response.

You can experiment by providing your own text or by modifying the `response_schema` to include a specific, predefined list of possible topics.