# Guide: Using JSON Mode with the Gemini API via REST

This guide demonstrates how to use the Gemini API's JSON Mode to structure model outputs. You'll learn to send a request that instructs the model to return a JSON-formatted response based on a provided schema.

## Prerequisites

To follow this guide, you need:
1.  A **Google AI API Key**. You can [generate one here](https://aistudio.google.com/app/apikey).
2.  A terminal or an environment like Google Colab to run `curl` commands.

## Step 1: Set Your API Key

First, set your API key as an environment variable. This allows the `curl` command to authenticate your request.

**If using a terminal:**
```bash
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

**If using Google Colab:**
Run the following cell to set your key from a Colab Secret named `GOOGLE_API_KEY`.

```python
import os
from google.colab import userdata

os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

## Step 2: Choose a Model

Select a Gemini model that supports JSON mode. The example uses `gemini-3-flash-preview`, but you can choose from other available models.

```bash
# Set your chosen model ID
MODEL_ID="gemini-3-flash-preview"
```

## Step 3: Activate JSON Mode with a `curl` Request

JSON mode is activated by setting the `response_mime_type` parameter to `"application/json"` within the `generationConfig` of your API request.

The following `curl` command sends a prompt asking for cookie recipes and specifies a simple JSON schema for the response.

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
-H 'Content-Type: application/json' \
-d '{
    "contents": [{
      "parts":[
        {"text": "List a few popular cookie recipes using this JSON schema: {\"type\": \"object\", \"properties\": { \"recipe_name\": {\"type\": \"string\"}}}"}
        ]
    }],
    "generationConfig": {
        "response_mime_type": "application/json"
    }
}'
```

### Understanding the Request
*   **Endpoint:** The request is sent to the `generateContent` endpoint for your chosen model (`$MODEL_ID`).
*   **Authentication:** Your API key is passed via the URL query parameter `?key=$GOOGLE_API_KEY`.
*   **Prompt & Schema:** The `text` part contains both the instruction ("List a few popular cookie recipes") and the target JSON schema.
*   **Key Configuration:** The `"generationConfig": {"response_mime_type": "application/json"}` line activates JSON mode.

### Example Response
A successful request will return a response where the `text` field contains a valid JSON string.

```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "[{\"recipe_name\":\"Chocolate Chip Cookies\"},{\"recipe_name\":\"Peanut Butter Cookies\"},{\"recipe_name\":\"Oatmeal Raisin Cookies\"},{\"recipe_name\":\"Sugar Cookies\"},{\"recipe_name\":\"Shortbread Cookies\"}]"
          }
        ],
        "role": "model"
      }
    }
  ]
}
```

## Step 4: Deactivating JSON Mode

To return to the default, unstructured text output, you can either:
1.  Set `response_mime_type` to `"text/plain"`.
2.  Omit the `response_mime_type` parameter entirely from the `generationConfig`.

## Next Steps & Further Reading

*   **API Documentation:** For in-depth details on structured output and all configuration options, refer to the official [Structured Output](https://ai.google.dev/gemini-api/docs/structured-output) guide and the [`GenerationConfig`](https://ai.google.dev/api/generate-content#generationconfig) API reference.
*   **Related Examples:**
    *   See how JSON mode is used for [Text Summarization](../../examples/json_capabilities/Text_Summarization.ipynb) to format story summaries.
    *   Explore the [Object Detection](../../examples/Object_detection.ipynb) example, which uses JSON to normalize detection outputs.
*   **Beyond JSON:** The Gemini API offers other methods to constrain and enhance model outputs:
    *   Use an **[Enum](../../quickstarts/Enum.ipynb)** to restrict responses to a predefined list of values.
    *   Implement **[Function Calling](../../quickstarts/Function_calling.ipynb)** to connect the model to your own tools and APIs.
    *   Enable **[Code Execution](../../quickstarts/Code_Execution.ipynb)** to allow the model to write and execute code in a sandboxed environment.