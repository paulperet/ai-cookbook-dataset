# Sentiment Analysis with the Gemini API

This guide walks you through using the Gemini API to perform structured sentiment analysis on text reviews. You will define a schema for the response and classify reviews as positive, neutral, or negative with a magnitude score.

## Prerequisites

Before you begin, ensure you have a Google AI API key. You can obtain one from [Google AI Studio](https://makersuite.google.com/app/apikey).

### 1. Install the Required Library

Install the Google Generative AI Python SDK.

```bash
pip install -U "google-genai>=1.0.0"
```

### 2. Configure Your API Key

Set your API key as an environment variable. For security, avoid hardcoding it in your scripts.

```python
import os
from google import genai

# Set your API key. In a production environment, use a secure method like a secrets manager.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 1: Define the Response Schema

You will instruct the model to return a structured JSON response. Define the expected format using Python's `enum` and `TypedDict`.

```python
import enum
from typing_extensions import TypedDict

class Magnitude(enum.Enum):
    """Enum to represent the strength of a sentiment."""
    WEAK = "weak"
    STRONG = "strong"

class Sentiment(TypedDict):
    """Schema for the sentiment analysis response."""
    positive_sentiment_score: Magnitude
    negative_sentiment_score: Magnitude
    neutral_sentiment_score: Magnitude

# System instruction to guide the model's task.
system_instruct = """
Generate each sentiment score probability (positive, negative, or neutral) for the whole text.
"""
```

## Step 2: Prepare Example Reviews

Create sample reviews to test the sentiment analysis.

```python
negative_review = "This establishment is an insult to the culinary arts, with inedible food that left me questioning the chef's sanity and the health inspector's judgment."
positive_review = "This restaurant is a true gem with impeccable service and a menu that tantalizes the taste buds. Every dish is a culinary masterpiece, crafted with fresh ingredients and bursting with flavor."
neutral_review = "The restaurant offers a decent dining experience with average food and service, making it a passable choice for a casual meal."
```

## Step 3: Create a Content Generation Function

Define a helper function to call the Gemini API. This function configures the model to use your system instruction and return JSON matching the `Sentiment` schema.

```python
from google.genai import types

def generate_content(review):
    """Generates structured sentiment analysis for a given review."""
    MODEL_ID = "gemini-1.5-flash"  # You can change this to other available models.
    return client.models.generate_content(
        model=MODEL_ID,
        contents=review,
        config=types.GenerateContentConfig(
            system_instruction=system_instruct,
            response_mime_type="application/json",
            response_schema=Sentiment,
        )
    )
```

## Step 4: Analyze Sentiment for Each Review

Now, use the function to analyze the sentiment of each example review.

### Analyze the Negative Review

```python
from pprint import pprint

response = generate_content(negative_review)
pprint(response.parsed)
```

The output will show a strong negative sentiment:

```python
{
    'negative_sentiment_score': <Magnitude.STRONG: 'strong'>,
    'neutral_sentiment_score': <Magnitude.WEAK: 'weak'>,
    'positive_sentiment_score': <Magnitude.WEAK: 'weak'>
}
```

### Analyze the Positive Review

```python
response = generate_content(positive_review)
pprint(response.parsed)
```

The output will show a strong positive sentiment:

```python
{
    'negative_sentiment_score': <Magnitude.WEAK: 'weak'>,
    'neutral_sentiment_score': <Magnitude.WEAK: 'weak'>,
    'positive_sentiment_score': <Magnitude.STRONG: 'strong'>
}
```

### Analyze the Neutral Review

```python
response = generate_content(neutral_review)
pprint(response.parsed)
```

The output will show a strong neutral sentiment:

```python
{
    'negative_sentiment_score': <Magnitude.WEAK: 'weak'>,
    'neutral_sentiment_score': <Magnitude.STRONG: 'strong'>,
    'positive_sentiment_score': <Magnitude.WEAK: 'weak'>
}
```

## Summary

You have successfully used the Gemini API to perform structured sentiment analysis. By defining a clear JSON schema and providing a system instruction, you guided the model to classify text into positive, negative, and neutral categories with magnitude scores.

## Next Steps

- **Experiment with Other Texts:** Try analyzing different types of content, such as social media comments, customer support emails, or product feedback.
- **Adjust the Schema:** Modify the `Sentiment` TypedDict to include numerical scores or additional categories like "mixed."
- **Explore Other Models:** Test the analysis with different Gemini models (e.g., `gemini-1.5-pro`) to compare performance and latency.

For more advanced JSON-related tasks with the Gemini API, refer to the other tutorials in the cookbook.