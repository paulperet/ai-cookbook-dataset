# Entity Extraction with the Gemini API: A Practical Guide

Use the Gemini API to automate information extraction from text. This guide demonstrates how to extract structured data like street names, phone numbers, and URLs using simple prompts.

## Prerequisites

First, install the required library and configure your API key.

### 1. Install the Client Library

```bash
pip install -U -q "google-genai>=1.0.0"
```

### 2. Configure Your API Key

Store your Gemini API key in an environment variable named `GOOGLE_API_KEY`. Then, initialize the client.

```python
from google import genai
import os

# Retrieve your API key from the environment
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

### 3. Select a Model

Choose a Gemini model for the task. For fast, cost-effective extraction, a Flash model is recommended.

```python
MODEL_ID = "gemini-2.5-flash-lite"  # Fast and efficient for extraction tasks
```

---

## Tutorial: Extracting Entities Step-by-Step

### Step 1: Extract Streets and Transport Methods

We'll start with a sample text about travel directions in Rome. Our goal is to extract all street names and forms of transportation.

First, define the text.

```python
directions = """
To reach the Colosseum from Rome's Fiumicino Airport (FCO),
your options are diverse. Take the Leonardo Express train from FCO
to Termini Station, then hop on metro line A towards Battistini and
alight at Colosseo station.
Alternatively, hop on a direct bus, like the Terravision shuttle, from
FCO to Termini, then walk a short distance to the Colosseum on
Via dei Fori Imperiali.
If you prefer a taxi, simply hail one at the airport and ask to be taken
to the Colosseum. The taxi will likely take you through Via del Corso and
Via dei Fori Imperiali.
A private transfer service offers a direct ride from FCO to the Colosseum,
bypassing the hustle of public transport.
If you're feeling adventurous, consider taking the train from
FCO to Ostiense station, then walking through the charming
Trastevere neighborhood, crossing Ponte Palatino to reach the Colosseum,
passing by the Tiber River and Via della Lungara.
Remember to validate your tickets on the metro and buses,
and be mindful of pickpockets, especially in crowded areas.
No matter which route you choose, you're sure to be awed by the
grandeur of the Colosseum.
"""
```

Now, craft a prompt that instructs the model to extract the specific entities and return them in a structured JSON format.

```python
directions_prompt = f"""
From the given text, extract the following entities and return a list of them.
Entities to extract: street name, form of transport.
Text: {directions}
Street = []
Transport = []
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=directions_prompt
)

print(response.text)
```

The model will return a structured JSON object:

```json
{
  "Street": [
    "Via dei Fori Imperiali",
    "Via del Corso",
    "Via dei Fori Imperiali",
    "Via della Lungara"
  ],
  "Transport": [
    "train",
    "metro",
    "bus",
    "shuttle",
    "taxi",
    "private transfer service",
    "public transport",
    "train"
  ]
}
```

**Note:** The output may contain duplicates (like "Via dei Fori Imperiali" and "train"). You can easily deduplicate these lists in your post-processing code.

### Step 2: Control the Output Format

You can refine the prompt to request a different output structure. For example, ask for two simple lists.

```python
directions_list_prompt = f"""
From the given text, extract the following entities and return a list of them.
Entities to extract: street name, form of transport.
Text: {directions}
Return your answer as two lists:
Street = [street names]
Transport = [forms of transport]
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=directions_list_prompt
)

print(response.text)
```

Expected output format:
```
Street = ['Via dei Fori Imperiali', 'Via del Corso', 'Via dei Fori Imperiali', 'Via della Lungara']
Transport = ['train', 'metro', 'bus', 'shuttle', 'taxi', 'transfer service']
```

### Step 3: Extract Phone Numbers

Let's apply the same technique to extract phone numbers from a customer service email.

Define the source text.

```python
customer_service_email = """
Hello,
Thank you for reaching out to our customer support team regarding your
recent purchase of our premium subscription service.
Your activation code has been sent to +87 668 098 344
Additionally, if you require immediate assistance, feel free to contact us
directly at +1 (800) 555-1234.
Our team is available Monday through Friday from 9:00 AM to 5:00 PM PST.
For after-hours support, please call our
dedicated emergency line at +87 455 555 678.
Thanks for your business and look forward to resolving any issues
you may encounter promptly.
Thank you.
"""
```

Create a prompt to extract just the phone numbers.

```python
phone_prompt = f"""
From the given text, extract the following entities and return a list of them.
Entities to extract: phone numbers.
Text: {customer_service_email}
Return your answer in a list:
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=phone_prompt
)

print(response.text)
```

The model returns a clean list:
```json
[
  "+87 668 098 344",
  "+1 (800) 555-1234",
  "+87 455 555 678"
]
```

### Step 4: Extract URLs as Markdown Links

Finally, let's extract URLs from a technical FAQ and format them as clickable Markdown links. This is useful for generating documentation.

Define the text containing URLs.

```python
url_text = """
Gemini API billing FAQs

This page provides answers to frequently asked questions about billing
for the Gemini API. For pricing information, see the pricing page
https://ai.google.dev/pricing.
For legal terms, see the terms of service
https://ai.google.dev/gemini-api/terms#paid-services.

What am I billed for?
Gemini API pricing is based on total token count, with different prices
for input tokens and output tokens. For pricing information,
see the pricing page https://ai.google.dev/pricing.

Where can I view my quota?
You can view your quota and system limits in the Google Cloud console
https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas.

Is GetTokens billed?
Requests to the GetTokens API are not billed,
and they don't count against inference quota.
"""
```

Craft a prompt that asks for unique URLs in Markdown format.

```python
url_prompt = f"""
From the given text, extract the following entities and return a list of them.
Entities to extract: URLs.
Text: {url_text}
Do not duplicate entities.
Return your answer in a markdown format:
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=url_prompt
)

print(response.text)
```

The model returns a deduplicated list formatted for Markdown:
```
- https://ai.google.dev/pricing
- https://ai.google.dev/gemini-api/terms#paid-services
- https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas
```

## Summary

You've learned how to use the Gemini API for entity extraction. The key steps are:
1.  **Define your source text.**
2.  **Craft a clear prompt** specifying the entities to extract (e.g., "street name", "phone numbers", "URLs").
3.  **Control the output format** by instructing the model to return JSON, lists, or Markdown.
4.  **Call the API** and parse the structured response.

This technique can be adapted to extract dates, product names, email addresses, or any other entity relevant to your application.