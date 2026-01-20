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

# Gemini API: Self-ask prompting

Self ask prompting is similar to chain of thought, but instead of going step by step as one answer, it asks itself questions that will help answer the query. Like the chain of thought, it helps the model to think analytically.

```
%pip install -U -q "google-genai>=1.0.0"
```

Additionally, select the model you want to use from the available options below:

```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.

```
from google.colab import userdata
from google import genai

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Example

```
from IPython.display import Markdown

prompt = """
  Question: Who was the president of the united states when Mozart died?
  Are follow up questions needed?: yes.
  Follow up: When did Mozart died?
  Intermediate answer: 1791.
  Follow up: Who was the president of the united states in 1791?
  Intermediate answer: George Washington.
  Final answer: When Mozart died George Washington was the president of the USA.

  Question: Where did the Emperor of Japan, who ruled the year Maria
  Skłodowska was born, die?
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt
)

Markdown(response.text)
```

Are follow up questions needed?: Yes.
Follow up: When was Maria Skłodowska born?
Intermediate answer: 1867.
Follow up: Who was the Emperor of Japan in 1867?
Intermediate answer: Emperor Kōmei.
Follow up: When did Emperor Kōmei die?
Intermediate answer: 1867.
Final answer: Emperor Kōmei died in Kyoto.

## Additional note
Self-ask prompting works well with function calling. Follow-up questions can be used as input to a function, which e.g. searches the internet. The question and answer from the function can be added back to the prompt. During the next query to the model, it can either create another function call or return the final answer.

For a related example, please see the [Search re-ranking using Gemini embeddings](https://github.com/google-gemini/cookbook/blob/22ba52659005defc53ce2d6717fb9fedf1d661f1/examples/Search_reranking_using_embeddings.ipynb) example in the Gemini Cookbook.