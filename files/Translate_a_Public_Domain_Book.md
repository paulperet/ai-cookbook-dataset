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
limitations under the License.
```

# Gemini API: Translate a public domain book

In this notebook, you will explore Gemini model as a translation tool, demonstrating how to prepare data, create effective prompts, and save results into a `.txt` file.

This approach significantly improves knowledge accessibility. Various sources provide open-source books. In this notebook, you will use [Project Gutenberg](https://www.gutenberg.org/) as a resource.

This platform provides access to a wide range of books available for free download in PDF, eBook, TXT formats and more.

```
%pip install -U -q "google-genai>=1.0.0" tqdm
```

```
from tqdm import tqdm

from IPython.display import Markdown
```

## Configure your API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for an example.

```
from google import genai
from google.colab import userdata
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Data preparation

You will translate a well-known book written by Arthur Conan Doyle from Polish (trans. Eugenia Żmijewska) to English about the detective Sherlock Holmes. Here are the titles in Polish and English:

* Polish Title: *Tajemnica Baskerville'ów: dziwne przygody Sherlocka Holmes*

* English Title: *The Hound of the Baskervilles*

```
!curl https://www.gutenberg.org/cache/epub/34079/pg34079.txt > Sherlock.txt
```

```
with open("/content/Sherlock.txt") as f:
      book = f.read()
```

Books contain all sorts of fictional or historical descriptions, some of them rather literal and might cause the model to stop from performing translation query. To prevent some of those exceptions users are able to change `safety_setting` from default to more open approach.

```
from google.genai import types
safety_settings =[
            types.SafetySetting(
              category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
              threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
              category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
              threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
              category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
              threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
              category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
              threshold=types.HarmBlockThreshold.BLOCK_NONE,
            )
        ]
```

Go ahead and initialize the Gemini Flash model for this task.

```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}


def generate_output(prompt):
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
          safety_settings=safety_settings
        )

    )

    try:
        return response.text
    except Exception as ex:
        raise ex
```

Begin by counting how many tokens are in the book to know if you need to proceed with the next step (split text into smaller chunks).

## Token Information

LLMs are constrained by token limits. In order to stay within the limit of Gemini Flash, it's important to ensure the following:

* Input token limit should not exceed: 1,048,576 tokens
* Output token limit should not exceed: 8,192 tokens

Since input chunks of post-translation text will serve as outputs, it's advisable to split them into smaller segments.

As you will see in the example, there are over 1000 small chunks. Calling API for the token count of each one would take a long time. Because of this estimate 1 token is equal to 4 characters will be used.

To account for the possible miscalculation of tokens and output potentially being longer than input each chunk will use up to 4*5000 characters. Remember to adjust the maximum to your needs.

For more details on tokens in Gemini models, refer to the documentation [here](https://ai.google.dev/gemini-api/docs/models/gemini
).

### Split text into chunks
Adjust the number of whitespace characters used for splitting based on the book you are working with.

Then, ensure to consider token limitations to receive complete responses from the model.

```
chunks = book[:book.find("END OF THE PROJECT GUTENBERG EBOOK TAJEMNICA BASKERVILLE'ÓW: DZIWNE PRZYGODY SHERLOCKA HOLMES")].split("\n\n")

num_chunks = len(chunks)
print(f"Number of chunks: {num_chunks}")
```

```
estimated_token_counts = []

for chunk in chunks:
    if not chunk:  # Check if chunk is empty or None
        continue  # Skip this iteration if chunk is empty

    # Process non-empty chunk
    estimated_token_count = len(chunk)/4
    estimated_token_counts.append(estimated_token_count)

# You can print number of estimated tokens in each non empty chunk to see what are you working with
print(estimated_token_counts)
```

[134.5, 16.25, 6.5, 7.25, 25.0, 4.0, 58.25, 26.0, 50.25, 148.0, 10.25, 11.0, 7.5, 8.5, 10.0, 9.5, 9.25, 7.75, 9.25, 9.5, 7.25, 11.25, 0.75, 5.0, 136.75, 22.25, 37.0, 12.5, 73.25, 45.0, 11.5, 24.25, 8.5, 45.5, 11.75, 31.75, 50.0, 48.0, 45.25, 22.0, 38.25, 52.25, 6.75, 6.25, 3.75, 54.25, 19.5, 15.75, 24.25, 45.25, 4.5, 152.75, 22.0, 72.0, 135.0, 91.5, 9.75, 53.5, 38.0, 5.25, 120.0, 128.0, 37.5, 9.25, 3.5, 10.25, 15.75, 11.25, 15.5, 27.75, 31.25, 16.5, 13.75, 10.25, 73.25, 44.5, 66.0, 13.5, 28.0, 11.25, 43.0, 33.5, 1.0, 7.5, 14.5, 17.5, 6.25, 5.5, 4.0, 78.5, 16.5, 9.0, 111.0, 17.75, 35.75, 34.5, 15.5, 21.0, 30.5, 57.5, 33.75, 21.75, 158.75, 60.5, 39.0, 110.0, 125.75, 124.5, 66.0, 55.75, 40.25, 63.0, 54.5, 84.75, 45.0, 33.5, 90.5, 138.25, 37.75, 10.0, 38.75, 5.5, 10.25, 12.25, 10.0, 66.0, 19.5, 125.75, 62.5, 50.25, 59.25, 44.5, 131.25, 65.75, 20.25, 9.0, 65.5, 32.0, 50.75, 52.75, 68.5, 60.25, 48.25, 54.25, 10.0, 15.5, 8.5, 92.75, 1.75, 17.0, 24.25, 150.75, 29.75, 106.75, 103.75, 76.25, 59.75, 44.75, 76.25, 126.25, 82.75, 5.0, 1.75, 5.75, 21.25, 14.25, 1.25, 2.0, 26.0, 9.75, 14.75, 6.25, 2.25, 12.0, 31.5, 7.0, 18.75, 8.0, 1.75, 5.5, 7.25, 6.0, 1.75, 5.25, 28.0, 10.75, 14.75, 12.0, 8.0, 5.75, 2.25, 20.0, 14.75, 9.75, 12.75, 39.5, 9.75, 12.5, 5.25, 20.0, 9.0, 4.75, 5.5, 9.0, 1.75, 10.75, 10.0, 9.75, 5.75, 10.0, 22.25, 4.75, 12.25, 23.5, 15.0, 9.5, 83.25, 42.25, 6.5, 23.75, 18.25, 7.75, 9.25, 22.75, 3.5, 95.75, 16.75, 77.75, 88.75, 5.25, 10.25, 42.25, 7.0, 62.0, 10.25, 149.25, 15.25, 154.5, 9.0, 36.0, 10.5, 45.5, 70.75, 32.5, 2.75, 18.5, 10.75, 49.25, 6.5, 20.75, 30.75, 5.25, 9.25, 3.0, 7.25, 14.75, 8.0, 8.25, 78.5, 77.75, 81.5, 7.25, 8.0, 14.5, 8.25, 21.5, 12.25, 7.25, 2.25, 71.0, 9.0, 1.75, 8.25, 14.5, 3.75, 118.25, 6.0, 11.0, 15.75, 136.5, 11.5, 18.0, 20.25, 47.0, 6.0, 22.5, 3.5, 94.75, 9.0, 51.0, 10.0, 88.0, 0.75, 5.75, 51.0, 75.25, 15.25, 37.0, 24.75, 13.0, 62.0, 28.75, 20.5, 11.0, 35.5, 10.75, 31.25, 12.5, 20.75, 33.75, 25.5, 26.75, 33.0, 62.5, 3.5, 19.25, 7.75, 21.75, 64.25, 34.5, 22.25, 27.5, 23.0, 27.0, 40.25, 11.25, 40.0, 66.25, 20.25, 13.5, 61.75, 16.5, 23.5, 15.5, 6.5, 19.0, 198.25, 16.75, 34.75, 4.75, 140.75, 24.25, 24.5, 3.25, 10.5, 15.5, 10.25, 12.5, 9.75, 33.25, 19.5, 45.5, 15.75, 18.25, 17.25, 85.75, 20.0, 32.5, 36.25, 25.5, 22.75, 127.75, 26.75, 30.25, 17.0, 44.25, 5.25, 12.5, 26.75, 7.0, 39.5, 36.5, 70.25, 11.0, 5.5, 13.5, 16.75, 50.75, 12.0, 32.5, 124.25, 11.0, 53.25, 43.0, 21.5, 6.75, 4.75, 4.5, 88.75, 4.75, 113.25, 25.75, 119.25, 25.25, 44.5, 7.0, 19.25, 17.0, 28.5, 14.5, 29.5, 6.75, 22.75, 32.0, 31.5, 3.5, 8.75, 4.25, 19.5, 4.25, 39.75, 4.75, 38.75, 2.5, 151.25, 0.75, 4.25, 64.0, 15.25, 19.0, 15.25, 3.0, 47.25, 27.25, 23.5, 9.25, 16.5, 37.5, 26.25, 17.5, 11.0, 48.25, 28.5, 31.0, 6.0, 9.0, 9.25, 52.25, 11.75, 22.0, 27.25, 10.75, 28.0, 8.0, 16.5, 83.75, 35.25, 10.5, 2.25, 5.0, 101.25, 7.0, 5.75, 37.5, 22.25, 9.0, 4.5, 12.75, 4.25, 48.75, 2.75, 66.0, 25.5, 43.75, 31.75, 3.25, 19.25, 11.0, 10.75, 20.0, 7.5, 42.25, 6.75, 28.25, 3.75, 7.5, 8.75, 14.25, 38.75, 64.0, 56.5, 21.5, 54.25, 21.5, 16.75, 12.25, 89.0, 44.75, 8.0, 61.75, 12.5, 52.5, 11.0, 8.75, 27.0, 29.5, 34.25, 29.75, 9.5, 45.0, 10.0, 22.0, 25.5, 7.5, 22.0, 26.5, 10.25, 5.75, 15.75, 25.5, 88.25, 57.75, 18.75, 11.25, 12.25, 9.75, 21.0, 12.0, 31.25, 14.25, 25.75, 16.25, 71.75, 43.75, 20.0, 15.0, 24.5, 8.25, 42.5, 10.75, 37.0, 48.25, 5.0, 7.0, 6.0, 7.75, 1.75, 10.0, 14.75, 4.75, 28.75, 25.25, 1.75, 20.0, 97.