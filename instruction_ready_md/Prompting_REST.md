# Gemini API Quickstart Guide: Prompting with REST

This guide walks you through using the Gemini API via RESTful `curl` commands. You'll learn how to send prompts, handle images, manage conversations, and configure model parameters.

## Prerequisites

Before you begin, ensure you have:
1.  A **Google AI API Key**. You can get one from [Google AI Studio](https://makersuite.google.com/app/apikey).
2.  Your API key stored in an environment variable named `GOOGLE_API_KEY`.

    ```bash
    # Example for setting the environment variable in your terminal
    export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```

## Step 1: Run Your First Prompt

Let's start by generating a simple text response. You'll use the `generateContent` endpoint.

1.  Choose a model. For this guide, we'll use `gemini-2.5-flash`.
2.  Construct a `curl` command that sends a JSON payload with your prompt.

```bash
MODEL_ID="gemini-2.5-flash"

curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [{
        "parts":[{"text": "Give me python code to sort a list."}]
        }]
       }'
```

**Expected Output (Abridged):**
The API will return a JSON response. The main answer is in the `candidates[0].content.parts[0].text` field.

```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "```python\n# Example list to be sorted\nmy_list = [5, 3, 1, 2, 4]\n\n# Sort the list in ascending order using the sort() method\nmy_list.sort()\n\n# Print the sorted list\nprint(my_list)\n```"
          }
        ]
      }
    }
  ]
}
```

## Step 2: Use Images in Your Prompt

The Gemini API can process images. You need to download the image, encode it in Base64, and include it in your request payload.

### 2.1 Download an Image
First, download an image to your local machine.

```bash
curl -o image.jpg "https://storage.googleapis.com/generativeai-downloads/images/jetpack.jpg"
```

### 2.2 Create the Request Payload
Now, create a JSON file (`request.json`) that contains your text prompt and the Base64-encoded image data.

**Important:** The correct `base64` command flag depends on your operating system:
*   **Linux/Google Colab:** Use `base64 -w0`
*   **macOS:** Use `base64 -i`

The following command works for Linux/Colab environments:

```bash
echo '{
  "contents":[
    {
      "parts":[
        {"text": "This image contains a sketch of a potential product along with some notes. Given the product sketch, describe the product as thoroughly as possible based on what you see in the image, making sure to note all of the product features. Return output in json format: {description: description, features: [feature1, feature2, feature3, etc]}"},
        {
          "inline_data": {
            "mime_type":"image/jpeg",
            "data": "'$(base64 -w0 image.jpg)'"
          }
        }
      ]
    }
  ]
}' > request.json
```

### 2.3 Send the Multimodal Request
Send the request using the JSON file you just created.

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=${GOOGLE_API_KEY}" \
        -H 'Content-Type: application/json' \
        -d @request.json
```

**Expected Output (Abridged):**
The model will return a structured JSON description based on the image.

```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "```json\n{\n \"description\": \"The Jetpack Backpack is a backpack that allows the user to fly...\",\n \"features\": [\n  \"retractable boosters\",\n  \"steam-powered\",\n  \"green/clean\",\n  \"15-min battery life\"\n ]\n}\n```"
          }
        ]
      }
    }
  ]
}
```

## Step 3: Have a Multi-Turn Conversation

The API supports chat by passing the entire conversation history in the `contents` array. Each turn has a `role` ("user" or "model") and `parts`.

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "contents": [
        {"role":"user",
         "parts":[{
           "text": "In one sentence, explain how a computer works to a young child."}]},
        {"role": "model",
         "parts":[{
           "text": "A computer is like a smart helper that can store information, do math problems, and follow our instructions to make things happen."}]},
        {"role": "user",
         "parts":[{
           "text": "Okay, how about a more detailed explanation to a high schooler?"}]}
      ]
    }'
```

The model's response will be a more technical explanation, continuing naturally from the provided context.

## Step 4: Configure Model Parameters

You can control the model's behavior using `generationConfig` and `safetySettings` in your request.

*   **`temperature`**: Controls randomness (0.0 = deterministic, 1.0 = creative).
*   **`maxOutputTokens`**: Limits the length of the response.
*   **`stopSequences`**: Makes the model stop generating if it encounters a specific string.
*   **`safetySettings`**: Adjusts thresholds for blocking harmful content.

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
        "contents": [{
            "parts":[
                {"text": "Give me a numbered list of cat facts."}
            ]
        }],
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_ONLY_HIGH"
            }
        ],
        "generationConfig": {
            "stopSequences": ["Title"],
            "temperature": 0.9,
            "maxOutputTokens": 2000
        }
    }'
```

**Expected Output (Abridged):**
You will receive a creative (`temperature: 0.9`) list of cat facts, stopping before any line that starts with "Title".

```json
{
  "candidates": [
    {
      "content": {
        "parts": [
          {
            "text": "1. Cats have 32 muscles in their ears, allowing them to rotate them 180 degrees.\n2. The average lifespan of a domestic cat is 12-15 years.\n..."
          }
        ]
      }
    }
  ]
}
```

## Next Steps

You've successfully made basic text, multimodal, and chat requests to the Gemini API. To dive deeper, explore the following:

*   **Advanced Safety Settings**: Learn to fine-tune content safety filters.
*   **Function Calling**: Integrate tools and APIs with your prompts.
*   **System Instructions**: Guide model behavior at a foundational level.

Check the [official Gemini API documentation](https://ai.google.dev/docs) for comprehensive details on all features and parameters.