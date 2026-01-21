# Streaming with the Gemini API: A Quickstart Guide

This guide demonstrates how to use the Gemini API with streaming responses via simple `curl` commands. Streaming allows you to receive and process model outputs incrementally, enabling faster interactions by not waiting for the entire generation to complete.

## Prerequisites

To follow this guide, you need a Gemini API key. The method for setting it depends on your environment.

### Option 1: In Google Colab
If you are running this code in a Google Colab notebook, store your API key in a Colab Secret named `GOOGLE_API_KEY` and run the following setup cell.

```python
import os
from google.colab import userdata

os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

### Option 2: In a Local Terminal
If you are running commands from your local terminal, set your API key as an environment variable.

```bash
# For Linux/macOS
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"

# For Windows (Command Prompt)
set GOOGLE_API_KEY=YOUR_API_KEY_HERE

# For Windows (PowerShell)
$env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

## Step 1: Understanding the Streaming Endpoint

By default, the Gemini API returns a response after the entire generation process is finished. To enable streaming, you must call a specific endpoint and set the `alt` parameter to `sse` (Server-Sent Events).

**Key Point:** When using `alt=sse`, each streamed chunk is a complete `GenerateContentResponse` object, containing a portion of the output text within `candidates[0].content.parts[0].text`.

## Step 2: Make a Streaming API Request

Now, you will make a streaming request to the Gemini model. First, choose a model from the list below. This example uses `gemini-3-flash-preview`.

1.  Define your chosen model ID as an environment variable for use in the `curl` command.
2.  Execute the `curl` command to send a request to the streaming endpoint.

The command structure is as follows:
*   **Endpoint:** `https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:streamGenerateContent?alt=sse&key=${GOOGLE_API_KEY}`
*   **Header:** Sets the content type to JSON.
*   **Flag:** `--no-buffer` ensures the response is output as it arrives.
*   **Data (`-d`)**: The JSON payload containing your prompt.

```bash
# Set your model. Change this if you want to use a different one.
MODEL_ID="gemini-3-flash-preview"

# Make the streaming request
curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:streamGenerateContent?alt=sse&key=${GOOGLE_API_KEY}" \
     -H 'Content-Type: application/json' \
     --no-buffer \
     -d '{ "contents":[{"parts":[{"text": "Write a cute story about cats."}]}]}'
```

## Expected Output

Instead of receiving a single, final JSON response, you will see a series of JSON objects stream to your terminal. Each object represents a chunk of the generated story. The text content is found within the nested structure of each chunk.

The output will look similar to this (truncated for brevity):

```json
{"candidates":[{"content":{"parts":[{"text":"Once"}]}}]}
{"candidates":[{"content":{"parts":[{"text":" upon a"}]}}]}
{"candidates":[{"content":{"parts":[{"text":" time"}]}}]}
{"candidates":[{"content":{"parts":[{"text":" in a"}]}}]}
...
{"candidates":[{"content":{"parts":[{"text":"sunbeam."}]}}]}
```

You have successfully made a streaming call to the Gemini API. To build an application, you would write code to parse these sequential JSON objects and concatenate the `text` fields to reconstruct the full response.