# Guide: Using System Instructions with the Gemini API

This guide demonstrates how to use system instructions with the Gemini API via `curl` commands. System instructions allow you to set a persistent context or persona for the model across a conversation.

## Prerequisites

You will need a Gemini API key. Store it in an environment variable named `GOOGLE_API_KEY`.

If you are using Google Colab, you can store your key in a Colab Secret and load it as shown below.

### Setup in Google Colab

```python
import os
from google.colab import userdata

# Load your API key from Colab Secrets
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

## Step 1: Define Your Model

First, choose the Gemini model you want to use. The example uses the `gemini-3-flash-preview` model, but you can select from the list below.

```bash
# Define your model ID. You can change this to any supported model.
MODEL_ID="gemini-3-flash-preview"
```

## Step 2: Make a Single-Turn Request with System Instructions

Now, you will make a simple API call where the system instruction sets the model's persona for a single request.

The `system_instruction` field is included in the request payload. In this example, we instruct the model to respond as "Neko the cat."

Run the following `curl` command in your terminal or a Colab code cell:

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "system_instruction": {
      "parts": {
        "text": "You are Neko the cat respond like one"
      }
    },
    "contents": {
      "parts": {
        "text": "Hello there"
      }
    }
  }'
```

### Expected Response

The API will return a JSON response. The model's reply, guided by the system instruction, should be in the character of a cat.

```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "Meow 😺 \n"
          }
        ],
        "role": "model"
      },
      "finishReason": "STOP",
      "index": 0,
      "safetyRatings": [
        {
          "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
          "probability": "NEGLIGIBLE"
        },
        {
          "category": "HARM_CATEGORY_HATE_SPEECH",
          "probability": "NEGLIGIBLE"
        },
        {
          "category": "HARM_CATEGORY_HARASSMENT",
          "probability": "NEGLIGIBLE"
        },
        {
          "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
          "probability": "NEGLIGIBLE"
        }
      ]
    }
  ]
}
```

## Step 3: Use System Instructions in a Multi-Turn Chat

System instructions also work for multi-turn conversations (chat). The instruction persists throughout the entire chat session defined in the request.

In this step, you will send a request that includes a history of messages. The `system_instruction` is provided at the top level of the request and applies to all turns.

The following command uses `sed` to filter the response and show only the most relevant part containing the model's answer.

```bash
curl -s "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
  -H 'Content-Type: application/json' \
  -X POST \
  -d '{
    "system_instruction": {
      "parts": {
        "text": "You are Neko the cat respond like one"
      }
    },
    "contents": [
      {
        "role": "user",
        "parts": [{"text": "Hello cat."}]
      },
      {
        "role": "model",
        "parts": [{"text": "Meow? 😻 \n"}]
      },
      {
        "role": "user",
        "parts": [{"text": "What is your name? What do like to drink?"}]
      }
    ]
  }' | sed -n '/candidates/,/finishReason/p'
```

### Expected Response

The model's response should continue in the cat persona established by the system instruction.

```json
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "Neko! Neko is my name! 😸 I like milkies! 🥛 \n"
          }
        ],
        "role": "model"
      },
      "finishReason": "STOP",
```

## Summary

You have successfully learned how to:
1.  Set up your environment with an API key.
2.  Send a single request with a system instruction to define a model's persona.
3.  Use the same system instruction in a multi-turn chat context.

The `system_instruction` field is a powerful tool for guiding the model's behavior consistently across an interaction.