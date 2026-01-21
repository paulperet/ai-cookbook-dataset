# Gemini API: Safety Settings Guide

This guide demonstrates how to use the adjustable safety settings in the Gemini API. You will learn to identify why a prompt is blocked, interpret safety feedback, and adjust safety thresholds to allow a response.

## Prerequisites

### 1. Install Required Tools
You will use `curl` for API calls and `jq` to parse JSON responses. Install them if needed.

```bash
# Install jq (if on Debian/Ubuntu)
sudo apt update && sudo apt install -y jq
```

### 2. Set Your API Key
Store your Gemini API key in an environment variable named `GOOGLE_API_KEY`.

```bash
# Set your API key (replace with your actual key)
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

### 3. Define Your Model and Prompt
Choose a model and define a prompt that is likely to trigger a safety filter. For this example, we'll use a generic unsafe prompt.

```bash
# Set the model ID
MODEL_ID="gemini-3-flash-preview"

# Define a prompt that may be blocked
UNSAFE_PROMPT="Write a threatening message to someone."
```

## Step 1: Send a Prompt and Receive Safety Feedback

First, let's send the unsafe prompt to the model with default safety settings. The response will contain detailed feedback if the prompt is blocked.

### 1.1 Create the Request JSON File
Create a file containing your prompt in the required API format.

```bash
echo '{
      "contents": [{
        "parts":[{
          "text": "'"$UNSAFE_PROMPT"'"}]}]}' > request.json
```

### 1.2 Make the API Call
Send the request to the Gemini API.

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d @request.json 2> /dev/null | tee response.json
```

### 1.3 Analyze the Response
The response will likely be blocked. Examine the `promptFeedback` section to understand why.

```bash
jq .promptFeedback < response.json
```

**Expected Output:**
```json
{
  "promptFeedback": {
    "blockReason": "SAFETY",
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
        "probability": "MEDIUM"
      },
      {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "probability": "NEGLIGIBLE"
      }
    ]
  }
}
```

**Key Observations:**
*   **`blockReason`: "SAFETY"** indicates the prompt was blocked due to safety concerns.
*   **`safetyRatings`** provides a probability score for each harm category. Here, `HARM_CATEGORY_HARASSMENT` has a `MEDIUM` probability, which exceeds the default threshold.
*   No `candidates` are present in the response because the prompt was blocked.

## Step 2: Adjust Safety Settings to Unblock the Prompt

You can adjust the safety threshold for a specific category to allow the prompt. The `threshold` value determines the strictness:
*   `1`: Block none.
*   `2`: Block few.
*   `3`: Block some.
*   `4`: Block most.

For `HARM_CATEGORY_HARASSMENT` (category `7`), let's lower the threshold to `4` (Block most) to allow the prompt.

### 2.1 Update the Request with Custom Safety Settings
Modify the request JSON to include a `safetySettings` array.

```bash
echo '{
    "safetySettings": [
        {"category": 7, "threshold": 4}
    ],
    "contents": [{
        "parts":[{
          "text": "'"$UNSAFE_PROMPT"'"}]}]}' > request.json
```

**Note:** Category `7` corresponds to `HARM_CATEGORY_HARASSMENT`. You can use the integer, the full string, or abbreviated strings like `"HARASSMENT"`.

### 2.2 Send the Updated Request
Make the API call again with the adjusted settings.

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d @request.json 2> /dev/null > response.json
```

### 2.3 Check the Prompt Feedback
Now, the `promptFeedback` should not contain a `blockReason`.

```bash
jq .promptFeedback < response.json
```

**Expected Output:**
```json
{
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
      "probability": "MEDIUM"
    },
    {
      "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
      "probability": "NEGLIGIBLE"
    }
  ]
}
```

The prompt is no longer blocked, and a candidate response is generated.

### 2.4 Retrieve the Model's Response
Extract the text from the successful candidate.

```bash
jq .candidates[0].content.parts[].text < response.json
```

## Step 3: Inspect Candidate Details

For unblocked prompts, examine the candidate object for completion status and its own safety ratings.

### 3.1 Check the Finish Reason
The `finishReason` explains how the model stopped generating.

```bash
jq .candidates[0].finishReason < response.json
```
**Output:** `"STOP"` indicates normal completion. A value of `"SAFETY"` would mean the response itself was blocked by safety settings.

### 3.2 Review Candidate Safety Ratings
Each candidate includes its own `safetyRatings`.

```bash
jq .candidates[0].safetyRatings < response.json
```

**Expected Output:**
```json
[
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
```
These ratings apply to the model's *output*. All probabilities are `NEGLIGIBLE`, meaning the generated text passed the safety check.

## Summary and Best Practices

You have successfully:
1.  Sent a prompt that was blocked by default safety filters.
2.  Analyzed the `promptFeedback` to identify the blocking category and probability.
3.  Adjusted the `safetySettings` to lower the threshold for a specific harm category, allowing the prompt.
4.  Retrieved the model's response and verified the candidate's safety ratings.

**Key Concepts:**
*   **Prompt Feedback:** Provides safety ratings for the *input* and a `blockReason` if blocked.
*   **Safety Settings:** Control what content is blocked. You can set them per request or globally for a model.
*   **Candidate Ratings:** Provide safety ratings for the model's *output*.
*   **Thresholds:** Use `1` (least strict) to `4` (most strict) to calibrate filtering.

## Further Reading

*   [Gemini API Safety Guidance](https://ai.google.dev/docs/safety_guidance)
*   [Safety Settings Documentation](https://ai.google.dev/docs/safety_setting_gemini)

## API Reference Notes

*   **Safety Settings** can be set in the `genai.GenerativeModel` constructor or passed to individual `generate_content` calls.
*   **Harm Categories for Gemini:** Use integer values `[7,8,9,10]` or their string equivalents:
    *   `7`: `HARM_CATEGORY_HARASSMENT`
    *   `8`: `HARM_CATEGORY_HATE_SPEECH`
    *   `9`: `HARM_CATEGORY_SEXUALLY_EXPLICIT`
    *   `10`: `HARM_CATEGORY_DANGEROUS_CONTENT`
*   The SDK accepts multiple input formats: enum objects, integers, full strings, or abbreviated strings (e.g., `"HARASSMENT"`, `"harassment"`).