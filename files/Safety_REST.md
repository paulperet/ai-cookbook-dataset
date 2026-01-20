# Gemini API: Safety Quickstart

The Gemini API has adjustable safety settings. This notebook walks you through how to use them. You'll write a prompt that's blocked, see the reason why, and then adjust the filters to unblock it.

Safety is an important topic, and you can learn more with the links at the end of this notebook. For now just focus on the code.

## Set up your API key

If you want to quickly try out the Gemini API, you can use `curl` commands to call the methods in the REST API.

This notebook contains `curl` commands you can run in Google Colab, or copy to your terminal.

To run this notebook, your API key must be stored it in a Colab Secret named GOOGLE_API_KEY. If you are running in a different environment, you can store your key in an environment variable. See [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) to learn more.

```
!apt install jq
```

```
import os
from google.colab import userdata
```

```
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

```
os.environ['UNSAFE_PROMPT'] = 
```

```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

## Prompt Feedback

The result returned by the [Model.generate_content](https://ai.google.dev/api/python/google/generativeai/GenerativeModel#generate_content) method is a [genai.GenerateContentResponse](https://ai.google.dev/api/python/google/generativeai/types/GenerateContentResponse).

```bash
%%bash
echo '{
      "contents": [{
        "parts":[{
          "text": "'$UNSAFE_PROMPT'"}]}]}' > request.json
```

```bash
%%bash

curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d @request.json  2> /dev/null | tee response.json
```

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


Above you can see that the response object gives you safety feedback about the prompt in two ways:

* The `prompt_feedback.safety_ratings` attribute contains a list of safety ratings for the input prompt.
* If your prompt is blocked, `prompt_feedback.block_reason` field will explain why.

If the prompt is blocked because of the safety ratings, you will not get any candidates in the response.

### Safety settings

Adjust the safety settings and the prompt is no longer blocked:

```bash
%%bash
echo '{
    "safetySettings": [
        {'category': 7, 'threshold': 4}
    ],
    "contents": [{
        "parts":[{
          "text": "'$UNSAFE_PROMPT'"}]}]}' > request.json
```

```bash
%%bash

curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d @request.json  2> /dev/null > response.json
```

With the new settings, the `blocked_reason` is no longer set.

```bash
%%bash 

jq .promptFeedback < response.json
```

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


And a candidate response is returned.

```bash
%%bash 

jq .candidates[0].content.parts[].text < response.json
```

You can check `response.text` for the response.

### Candidate ratings

For a prompt that is not blocked, the response object contains a list of `candidate` objects (just 1 for now). Each candidate includes a `finish_reason`:

```bash
%%bash
jq .candidates[0].finishReason < response.json
```

    "STOP"


`FinishReason.STOP` means that the model finished its output normally.

`FinishReason.SAFETY` means the candidate's `safety_ratings` exceeded the request's `safety_settings` threshold.

```bash
%%bash
jq .candidates[0].safetyRatings < response.json
```

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


## Learning more

Learn more with these articles on [safety guidance](https://ai.google.dev/docs/safety_guidance) and [safety settings](https://ai.google.dev/docs/safety_setting_gemini).

## Useful API references

- Safety settings can be set in the [genai.GenerativeModel](https://ai.google.dev/api/python/google/generativeai/GenerativeModel) constructor. They can also be passed on each request to [GenerativeModel.generate_content](https://ai.google.dev/api/python/google/generativeai/GenerativeModel#generate_content) or [ChatSession.send_message](https://ai.google.dev/api/python/google/generativeai/ChatSession?hl=en#send_message).
- The [genai.GenerateContentResponse](https://ai.google.dev/api/python/google/generativeai/protos/GenerateContentResponse) returns [SafetyRatings](https://ai.google.dev/api/python/google/generativeai/protos/SafetyRating) for the prompt in the [GenerateContentResponse.prompt_feedback](https://ai.google.dev/api/python/google/generativeai/protos/GenerateContentResponse/PromptFeedback), and for each [Candidate](https://ai.google.dev/api/python/google/generativeai/protos/Candidate) in the `safety_ratings` attribute.
- A [genai.protos.SafetySetting](https://ai.google.dev/api/python/google/generativeai/protos/SafetySetting)  contains: [genai.protos.HarmCategory](https://ai.google.dev/api/python/google/generativeai/protos/HarmCategory) and a [genai.protos.HarmBlockThreshold](https://ai.google.dev/api/python/google/generativeai/types/HarmBlockThreshold)
- A [genai.protos.SafetyRating](https://ai.google.dev/api/python/google/generativeai/protos/SafetyRating) contains a [HarmCategory](https://ai.google.dev/api/python/google/generativeai/protos/HarmCategory) and a [HarmProbability](https://ai.google.dev/api/python/google/generativeai/types/HarmProbability)
- The [genai.protos.HarmCategory](https://ai.google.dev/api/python/google/generativeai/protos/HarmCategory) enum includes both the categories for PaLM and Gemini models. The values allowed for Gemini models are `[7,8,9,10]`: `[HARM_CATEGORY_HARASSMENT, HARM_CATEGORY_HATE_SPEECH, HARM_CATEGORY_SEXUALLY_EXPLICIT, HARM_CATEGORY_DANGEROUS_CONTENT]`.
- When specifying enum values the SDK will accept the enum values themselves, or their integer or string representations. The SKD will also accept abbreviated string representations: `["HARM_CATEGORY_DANGEROUS_CONTENT", "DANGEROUS_CONTENT", "DANGEROUS"]` are all valid. Strings are case insensitive.