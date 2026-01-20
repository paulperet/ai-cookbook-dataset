# Gemini API: Sentiment Analysis

You will use the Gemini to extract sentiment scores of reviews.

```
%pip install -U -q "google-genai>=1.0.0"
```

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.

```
from google.colab import userdata
from google import genai

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Example

Start by defining how you want your JSON to be returned and which categories you would like to classify an item by. After that, go ahead and define some examples. In this case, you are trying to classify reviews as positive, neutral, or negative.

```
import enum
from typing_extensions import TypedDict


class Magnitude(enum.Enum):
  WEAK = "weak"
  STRONG = "strong"


class Sentiment(TypedDict):
  positive_sentiment_score: Magnitude
  negative_sentiment_score: Magnitude
  neutral_sentiment_score: Magnitude


system_instruct = """
Generate each sentiment score probability (positive, negative, or neutral) for the whole text.
"""
```

```
negative_review = "This establishment is an insult to the culinary arts, with inedible food that left me questioning the chef's sanity and the health inspector's judgment."
positive_review = "This restaurant is a true gem with impeccable service and a menu that tantalizes the taste buds. Every dish is a culinary masterpiece, crafted with fresh ingredients and bursting with flavor."
neutral_review = "The restaurant offers a decent dining experience with average food and service, making it a passable choice for a casual meal."
```

Take a look at each of the probabilities returned to see how each of these reviews would be classified by the Gemini model.

Helper function to generate content from sentiment llm:

```
from google.genai import types

def generate_content(review):
    MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
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

```
from pprint import pprint

response = generate_content(negative_review)
pprint(response.parsed)
```

    {'negative_sentiment_score': <Magnitude.STRONG: 'strong'>,
     'neutral_sentiment_score': <Magnitude.WEAK: 'weak'>,
     'positive_sentiment_score': <Magnitude.WEAK: 'weak'>}

```
response = generate_content(positive_review)
pprint(response.parsed)
```

    {'negative_sentiment_score': <Magnitude.WEAK: 'weak'>,
     'neutral_sentiment_score': <Magnitude.WEAK: 'weak'>,
     'positive_sentiment_score': <Magnitude.STRONG: 'strong'>}

```
response = generate_content(neutral_review)
pprint(response.parsed)
```

    {'negative_sentiment_score': <Magnitude.WEAK: 'weak'>,
     'neutral_sentiment_score': <Magnitude.STRONG: 'strong'>,
     'positive_sentiment_score': <Magnitude.WEAK: 'weak'>}

## Summary
You have now used the Gemini API to analyze the sentiment of restaurant reviews using structured data. Try out other types of texts, such as comments under a video or emails.

Please see the other notebooks in this directory to learn more about how you can use the Gemini API for other JSON related tasks.