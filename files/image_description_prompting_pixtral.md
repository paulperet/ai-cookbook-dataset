# Image Description Extraction using Mistral's Pixtral API

# Image Description Extraction using Mistral's Pixtral API

In this notebook, we'll use the `Mistral` API to extract structured image descriptions in JSON format using the `Pixtral-12b-2409` model. We'll send an image URL and prompt the model to return key elements with descriptions.

## Prerequisites
Make sure you have an API key for the Mistral AI platform. We'll also show you how to load it from environment variables.

```python
# Install the Mistral Python SDK
!pip install mistralai
```

[Collecting mistralai, ..., Successfully installed eval-type-backport-0.2.2 invoke-2.2.1 mistralai-1.9.11]

## Setup
We'll load the Mistral API key from environment variables and initialize the client. Make sure your API key is saved in your environment variables as `MISTRAL_API_KEY`.

```python
%env MISTRAL_API_KEY=
```

```python
import os
from mistralai import Mistral

# Load Mistral API key from environment variables
api_key = os.environ["MISTRAL_API_KEY"]

# Model specification
model = "pixtral-12b-2409"

# Initialize the Mistral client
client = Mistral(api_key=api_key)

```

## Sending Image URL for Description
We'll prompt the model to describe the image by providing an image URL. The response will be returned in a structured JSON format with the key elements described.

```python
# Define the messages for the chat API
messages = [
    {
        "role": "system",
        "content": "Return the answer in a JSON object with the next structure: "
                   "{\"elements\": [{\"element\": \"some name of element1\", "
                   "\"description\": \"some description of element 1\"}, "
                   "{\"element\": \"some name of element2\", \"description\": "
                   "\"some description of element 2\"}]}"
    },
    {
        "role": "user",
        "content": "Describe the image"
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": "https://docs.mistral.ai/img/eiffel-tower-paris.jpg"
            }
        ]
    }
]

# Call the Mistral API to complete the chat
chat_response = client.chat.complete(
    model=model,
    messages=messages,
    response_format={
        "type": "json_object",
    }
)

# Get the content of the response
content = chat_response.choices[0].message.content

# Output the raw JSON response
print(content)

```

    {"elements": [{"element": "Eiffel Tower", "description": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower."}]}

## Parsing the JSON Response
We'll now parse the JSON response from the API and print the elements and their corresponding descriptions.

```python
import json

# Parse the JSON content
json_object = json.loads(content)
elements = json_object["elements"]

# Print each element and its description
for element in elements:
    print("Element:", element["element"])
    print("Description:", element["description"])
    print()

```

    Element: Eiffel Tower
    Description: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    

## Conclusion
In this notebook, we used the Mistral Pixtral model to describe an image by sending an image URL and receiving a structured JSON response. The descriptions provided by the model offer insights into the key elements of the image.