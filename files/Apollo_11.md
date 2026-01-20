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

# Prompting with an Apollo 11 transcript

This notebook provides a quick example of how to prompt Gemini using a text file. In this case, you'll use a 400 page transcript from Apollo 11.

```
%pip install -U -q "google-genai>=1.0.0"
```

### Setup your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see Authentication for an example.

```
from google.colab import userdata
from google import genai

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

Download the transcript.

```
!wget https://storage.googleapis.com/generativeai-downloads/data/a11.txt
```

[First Entry, ..., Last Entry]

Upload the file using the File API so its easier to pass it to the model later on.

```
text_file_name = "a11.txt"
print(f"Uploading file...")
text_file = client.files.upload(file=text_file_name)
print(f"Completed upload: {text_file.uri}")
```

## Generate Content

After the file has been uploaded, you can make `client.models.generate_content` requests that reference the File API URI. Then you will ask the model to find a few lighthearted moments.

```
prompt = "Find four lighthearted moments in this text file."

MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}

response = client.models.generate_content(
  model=f"models/{MODEL_ID}",
  contents=[
   prompt,
   text_file
  ],
  config={
   "httpOptions": {"timeout": 600}
  }
)

print(response.text)
```

## Delete File

Files are automatically deleted after 2 days or you can manually delete them using `files.delete()`.

```
client.files.delete(name=text_file.name)
```

## Learning more

The File API accepts files under 2GB in size and can store up to 20GB of files per project. Learn more about the File API here.