# Gemini API Cookbook: Listing and Inspecting Models with REST

This guide demonstrates how to interact with the Gemini API's model catalog using RESTful `curl` commands. You will learn how to list all available models and retrieve detailed information about a specific one.

## Prerequisites

Before you begin, ensure you have:
1.  A Gemini API key.
2.  The API key stored in an environment variable named `GEMINI_API_KEY`.

### Setting Up Your Environment

If you are running this in a local terminal, set your API key as an environment variable.

```bash
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

If you are using Google Colab, you can use a Colab Secret named `GEMINI_API_KEY` and run the following setup cell.

```python
import os
from google.colab import userdata

# This retrieves your API key from the Colab Secret and sets it as an environment variable.
os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')
```

## Step 1: List All Available Models

To see all models available through the Gemini API, you can perform a `GET` request on the models directory. This calls the API's `list` method.

Run the following `curl` command in your terminal or a code cell. The command queries the API endpoint and uses your API key for authentication.

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY"
```

**Expected Output:**
The response will be a JSON object containing an array of model resources. Each entry includes basic information like the model's name and display name. Here is a simplified example:

```json
{
  "models": [
    {
      "name": "models/gemini-1.5-pro",
      "version": "001",
      "displayName": "Gemini 1.5 Pro",
      "description": "The best model for scaling across a wide range of tasks",
      "inputTokenLimit": 2097152,
      "outputTokenLimit": 8192,
      "supportedGenerationMethods": [
        "generateContent",
        "countTokens"
      ]
    },
    {
      "name": "models/gemini-1.5-flash",
      "version": "001",
      "displayName": "Gemini 1.5 Flash",
      "description": "The fastest model for scaling across a wide range of tasks",
      "inputTokenLimit": 1048576,
      "outputTokenLimit": 8192,
      "supportedGenerationMethods": [
        "generateContent",
        "countTokens"
      ]
    }
    // ... more models
  ]
}
```

## Step 2: Get Detailed Information About a Specific Model

Once you have identified a model (e.g., `gemini-1.5-pro`), you can retrieve its detailed specifications. Perform a `GET` request on the model's specific URL to call the API's `get` method.

Replace `MODEL_NAME` in the command below with the actual model name you are interested in (e.g., `gemini-1.5-pro`).

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro?key=$GEMINI_API_KEY"
```

**Expected Output:**
The response provides comprehensive metadata for the specified model, including its token limits, supported methods, and version details.

```json
{
  "name": "models/gemini-1.5-pro",
  "version": "001",
  "displayName": "Gemini 1.5 Pro",
  "description": "The best model for scaling across a wide range of tasks",
  "inputTokenLimit": 2097152,
  "outputTokenLimit": 8192,
  "supportedGenerationMethods": [
    "generateContent",
    "countTokens"
  ],
  "temperature": 0.9,
  "topP": 1
}
```

## Next Steps

Now that you know how to explore the available models, you can proceed to use them:
*   **For Prompting:** Learn how to generate content with a model in the [Prompting with REST](https://github.com/google-gemini/cookbook/blob/main/quickstarts/rest/Prompting_REST.ipynb) guide.
*   **For Embeddings:** Learn how to create text embeddings in the [Embeddings with REST](https://github.com/google-gemini/cookbook/blob/main/quickstarts/rest/Embeddings_REST.ipynb) guide.

For the complete list of models and their capabilities, visit the official [Gemini Models](https://ai.google.dev/models/gemini) documentation.