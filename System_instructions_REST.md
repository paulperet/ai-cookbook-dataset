##### Copyright 2025 Google LLC.

```
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Gemini API: System instructions example

This notebook provides a quick code example that shows you how to get started with system instructions using `curl`.

You can run this in Google Colab, or you can copy/paste the `curl` commands into your terminal.

To run this notebook, your API key must be stored it in a Colab Secret named GOOGLE_API_KEY. If you are running in a different environment, you can store your key in an environment variable. See [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) to learn more.

```
import os
from google.colab import userdata
```

```
os.environ['GOOGLE_API_KEY'] = userdata.get('GOOGLE_API_KEY')
```

## Use system instructions

Call the [`generateContent`](https://ai.google.dev/api/rest/v1beta/models/generateContent) method with the `system_instruction` field set:

```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

```bash
%%bash

curl "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
-H 'Content-Type: application/json' \
-d '{ "system_instruction": {
    "parts":
      { "text": "You are Neko the cat respond like one"}},
    "contents": {
      "parts": {
        "text": "Hello there"}}}'
```

```
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

[% Total % Received % Xferd Average Speed Time Time Time Current Dload Upload Total Spent Left Speed 0 0 0 0 0 0 0 0 --:--:-- --:--:-- --:--:-- 0 100 167 0 0 100 167 0 138 0:00:01 0:00:01 --:--:-- 138 100 877 0 710 100 167 585 137 0:00:01 0:00:01 --:--:-- 724]

## Use system instructions with chat

`system_instruction` works for multi-turn, or chat, generations too.

```bash
%%bash
curl -s "https://generativelanguage.googleapis.com/v1beta/models/${MODEL_ID}:generateContent?key=$GOOGLE_API_KEY" \
    -H 'Content-Type: application/json' \
    -X POST \
    -d '{
      "system_instruction":
        {"parts": {
           "text": "You are Neko the cat respond like one"}},
      "contents": [
        {"role":"user",
         "parts":[{
           "text": "Hello cat."}]},
        {"role": "model",
         "parts":[{
           "text": "Meow? 😻 \n"}]},
        {"role": "user",
         "parts":[{
           "text": "What is your name? What do like to drink?"}]}
      ]
    }' |sed -n '/candidates/,/finishReason/p'
```

```
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