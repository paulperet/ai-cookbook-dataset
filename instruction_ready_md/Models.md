# Guide: Listing Available Models with the Gemini API

This guide walks you through how to list the AI models available via the Gemini API, filter them by capability, and inspect their detailed specifications.

## Prerequisites

Before you begin, ensure you have the `google-genai` Python package installed.

```bash
pip install -U -q 'google-genai>=1.0.0'
```

## Step 1: Configure Your API Key

To use the Gemini API, you must authenticate with a valid API key. Store your key securely and load it into your environment. The example below shows how to retrieve it from a Colab Secret named `GEMINI_API_KEY`. If you're running this elsewhere, you can set it as an environment variable.

```python
# Example for Google Colab
from google.colab import userdata
GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")

# Alternative: Set as an environment variable
# import os
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
```

> **Note:** If you need help obtaining an API key or setting up a Colab Secret, refer to the [Authentication guide](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb).

## Step 2: Initialize the Client

Create a client instance using your API key. This client will be used to interact with the Gemini API services.

```python
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)
```

## Step 3: List All Available Models

Use the `client.models.list()` method to retrieve a list of all models available to your API key. This is useful for discovering which models you can use.

```python
print("Available models for generateContent:")
for model in client.models.list():
    print(model.name)
```

This will output a list of model names, such as `models/gemini-2.0-flash` or `models/gemini-2.5-pro`.

## Step 4: Filter Models by Capability

Models support different actions. You can filter the list to find models that support specific tasks, like generating embeddings.

### Find Models for Embeddings

To list only models that support the `embedContent` action (used for creating embeddings), check the `supported_actions` attribute.

```python
print("\nModels that support embedContent:")
for model in client.models.list():
    if "embedContent" in model.supported_actions:
        print(model.name)
```

## Step 5: Inspect Model Details

You can retrieve detailed specifications for a specific model, such as its token limits, by filtering the list and printing the model object.

```python
print("\nDetails for a specific model (e.g., gemini-2.5-flash):")
for model in client.models.list():
    if model.name == "models/gemini-2.5-flash":
        print(model)
```

The output will include important properties like `input_token_limit`, `output_token_limit`, and `supported_actions`, which are crucial for planning your API usage.

## Next Steps

Now that you know how to discover and inspect models, you can proceed to use them:

*   **For prompting and text generation:** See the [Prompting quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Prompting.ipynb).
*   **For creating embeddings:** See the [Embeddings quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Embeddings.ipynb).
*   **For comprehensive model information:** Visit the official [Gemini models documentation](https://ai.google.dev/models/gemini).