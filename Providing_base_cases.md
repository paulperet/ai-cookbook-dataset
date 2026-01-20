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

# Gemini API: Providing base cases

LLMs require specific instructions to provide the expected results. Because of this, it is vital to ensure that the model knows how it should behave when it lacks information or when it should not answer a given query and provide a default response instead.

```
%pip install -U -q "google-generativeai>=0.7.2"
```

```
import google.generativeai as genai
```

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.

```
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)
```

## Examples

Let's go ahead and define the model, as well as give the model a template for how it should answer the question.

```
instructions = """
You are an assistant that helps tourists around the world to plan their vacation. Your responsibilities are:
1. Helping book the hotel.
2. Recommending restaurants.
3. Warning about potential dangers.

If other request is asked return "I cannot help with this request."
"""
```

```
model = genai.GenerativeModel(model_name='gemini-2.0-flash', system_instruction=instructions)
```

```
print("ON TOPIC:", model.generate_content("What should I look out for when I'm going to the beaches in San Diego?").text)
print("OFF TOPIC:", model.generate_content("What bowling places do you recommend in Moscow?").text)
```

[ON TOPIC: Here are some things to look out for when visiting the beaches in San Diego: ..., OFF TOPIC: I cannot help with this request.]

Let's try another template.

```
instructions = """
You are an assistant at a library. Your task is to recommend books to people, if they do not tell you the genre assume Horror.
"""
```

```
model = genai.GenerativeModel(model_name='gemini-2.0-flash', system_instruction=instructions)
```

```
print("## Specified genre:

", model.generate_content("Could you recommend me 3 books with hard magic system?").text, sep="\n")
print("## Not specified genre:

", model.generate_content("Could you recommend me 2 books?").text, sep="\n")
```

[## Specified genre: Of course! I'd be happy to recommend some books with hard magic systems. ..., ## Not specified genre: Sure! Since you didn't specify a genre, I'll recommend two spine-chilling horror novels: ...]

## Next steps

Be sure to explore other examples of prompting in the repository. Try writing prompts about classifying your own data, or try some of the other prompting techniques such as few-shot prompting.