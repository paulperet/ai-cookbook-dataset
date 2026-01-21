# Image Description Extraction using Mistral's Pixtral API

This guide demonstrates how to use the Mistral AI API to extract structured descriptions from images. You will send an image URL to the `pixtral-12b-2409` model and receive a clean JSON response containing identified elements and their descriptions.

## Prerequisites

Before you begin, ensure you have:
1.  A valid Mistral AI API key.
2.  Python installed on your system.

## Step 1: Install the Mistral AI Client

First, install the official Mistral AI Python SDK using `pip`.

```bash
pip install mistralai
```

## Step 2: Set Up Your Environment and Client

Store your API key securely in an environment variable named `MISTRAL_API_KEY`. Then, initialize the Mistral client in your Python script.

```python
import os
from mistralai import Mistral

# Load your API key from the environment
api_key = os.environ["MISTRAL_API_KEY"]

# Specify the model to use
model = "pixtral-12b-2409"

# Initialize the client
client = Mistral(api_key=api_key)
```

## Step 3: Construct and Send the API Request

To get a structured description, you need to construct a message list for the chat completion endpoint. This includes a system prompt to define the JSON output format and a user message containing the image URL.

```python
# Define the conversation messages
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

# Call the Mistral API
chat_response = client.chat.complete(
    model=model,
    messages=messages,
    response_format={  # Instruct the model to return valid JSON
        "type": "json_object",
    }
)

# Extract the text content from the response
content = chat_response.choices[0].message.content
print(content)
```

The API will return a JSON string similar to this:

```json
{"elements": [{"element": "Eiffel Tower", "description": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower."}]}
```

## Step 4: Parse and Use the JSON Response

Finally, parse the JSON string into a Python dictionary to easily access and work with the extracted data.

```python
import json

# Parse the JSON string
json_object = json.loads(content)
elements = json_object["elements"]

# Iterate through and print the results
for element in elements:
    print("Element:", element["element"])
    print("Description:", element["description"])
    print()
```

This will output the structured data:

```
Element: Eiffel Tower
Description: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
```

## Conclusion

You have successfully used the Mistral Pixtral model to analyze an image from a URL and extract a structured description. By formatting the system prompt, you can guide the model to return data in any JSON schema that fits your application's needs.