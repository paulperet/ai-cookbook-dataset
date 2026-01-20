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

# Gemini API: List models

This notebook demonstrates how to list the models that are available for you to use in the Gemini API, and how to find details about a model.

```
%pip install -U -q 'google-genai>=1.0.0'
```

    [First Entry, ..., Last Entry]

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GEMINI_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.

```
from google.colab import userdata

GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
```

## List models

Use `list_models()` to see what models are available. These models support `generateContent`, the main method used for prompting.

```
from google import genai

client = genai.Client(api_key=GEMINI_API_KEY)

for model in client.models.list():
    print(model.name)
```

    [First Entry, ..., Last Entry]

These models support `embedContent`, used for embeddings:

```
for model in client.models.list():
    if "embedContent" in model.supported_actions:
        print(model.name)
```

    [First Entry, ..., Last Entry]

## Find details about a model

You can see more details about a model, including the `input_token_limit` and `output_token_limit` as follows.

```
for model in client.models.list():
    if model.name == "models/gemini-2.5-flash":
        print(model)
```

    [First Entry, ..., Last Entry]

## Learning more

* To learn how use a model for prompting, see the [Prompting](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Prompting.ipynb) quickstart.

* To learn how use a model for embedding, see the [Embedding](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Embeddings.ipynb) quickstart.

* For more information on models, visit the [Gemini models](https://ai.google.dev/models/gemini) documentation.