```
# @title Licensed under the Apache License, Version 2.0 (the "License");
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Google I/O 2025 - Live coding experiences with the Gemini API

Welcome to the official Colab notebook from the **Google I/O 2025 live coding session** on the Gemini API!

This notebook serves as a comprehensive, hands-on guide to exploring the cutting-edge capabilities of the Gemini models, as demonstrated live during the presentation. You will dive deep into how developers can leverage the Gemini API to build powerful, innovative, and highly intelligent AI-powered applications.

Throughout this interactive session, you'll find practical demonstrations covering the **latest advancements** in Gemini, including:

*   **Generative Media (GenMedia models):** Learn to create stunning images with **Imagen3** and the experimental **Gemini 2.0 Flash image generation**, and generate dynamic videos with the powerful **Veo2** model.
*   **Advanced Multimodality:** Understand and generate content across various modalities, seamlessly combining text, images, and videos in your prompts and responses.
*   **Text-to-Speech (TTS):** Transform written text into natural-sounding audio, exploring customizable voices, language options, and even multi-speaker dialogues.
*   **Intelligent Tool Use:** Empower Gemini with built-in tools like **Code Execution** (for solving complex problems in a sandbox), real-time **grounding via Google Search**, and **URL context** to interact with external systems and fetch factual information directly from the web.
*   **Adaptive Thinking & Agentic Solutions:** Discover how Gemini models can perform internal reasoning and problem-solving with their **thinking capability**, and how to build complex, multi-step AI agents using the **Google Agent Development Kit (ADK)** for advanced use cases.

This notebook is designed to be fully runnable, allowing you to execute the code, experiment with different prompts, and directly experience the versatility and power of the Gemini API. Get ready to unlock new possibilities and code the future!

## Setup

Before diving into the exciting world of Gemini API, we need to set up our environment. This involves installing the necessary SDK and configuring your API key.

### Install SDK

The `google-genai` Python SDK is essential for interacting with the Gemini API. This SDK provides a streamlined way to access different Gemini models and their functionalities.

Install the SDK from [PyPI](https://github.com/googleapis/python-genai).


```
%pip install -U -q 'google-genai>=1.16'
```

[First Entry, ..., Last Entry]

### Setup your API key

To authenticate your requests with the Gemini API, you need an API key. This key allows you to access Google's powerful generative AI models. It's recommended to store your API key securely, for instance, as a Colab Secret named `GOOGLE_API_KEY`.

If you don't already have an API key or you aren't sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

### Initialize SDK client

With the `google-genai` SDK, initializing the client is straightforward. You pass your API key to `genai.Client`, and the client handles communication with the Gemini API. Individual model settings are then applied in each API call.


```
import time
from google import genai
from google.genai import types
from IPython.display import Video, HTML, Markdown, Image

client = genai.Client(api_key=GOOGLE_API_KEY)
```

# Working with GenMedia models

The Gemini API offers access to various GenMedia models, enabling advanced capabilities like generating images and videos from text prompts. These models push the boundaries of what's possible in creative content generation.

## Generating images with Imagen3

[Imagen 3](https://ai.google.dev/gemini-api/docs/imagen) is Google's most advanced text-to-image model, capable of producing highly detailed images with rich lighting and fewer artifacts than previous versions. It excels in scenarios where image quality and specific artistic styles are paramount.

Warning Badge

⚠️

Imagen is a paid-only feature and won't work if you are on the free tier.

### Select the Imagen3 model to be used

The `imagen-3.0-generate-002` model is specifically designed for high-quality image generation from textual prompts.


```
MODEL_ID = "imagen-3.0-generate-002" # @param {isTemplate: true}
```

### Imagen3 image generation prompt

When generating images with Imagen 3, you provide a descriptive prompt to guide the output. You can also specify parameters like the number of images, person_generation (to allow or disallow generating images of people), and aspect_ratio for different output dimensions.


```
%%time

prompt = """
  Dynamic anime illustration: A happy Brazilian man with short grey hair and a
  grey beard, mid-presentation at a tech conference. He's wearing a fun blue
  short-sleeve shirt covered in mini avocado prints. Capture a funny, energetic
  moment where he's clearly enjoying himself, perhaps with an exaggerated joyful
  expression or a humorous gesture, stage background visible.
"""

number_of_images = 1 # @param {type:"slider", min:1, max:4, step:1}
person_generation = "ALLOW_ADULT" # @param ['DONT_ALLOW', 'ALLOW_ADULT']
aspect_ratio = "1:1" # @param ["1:1", "3:4", "4:3", "16:9", "9:16"]

result = client.models.generate_images(
    model=MODEL_ID,
    prompt=prompt,
    config=dict(
        number_of_images=number_of_images,
        output_mime_type="image/jpeg",
        person_generation=person_generation,
        aspect_ratio=aspect_ratio
    )
)
```

    CPU times: user 82.6 ms, sys: 13 ms, total: 95.6 ms
    Wall time: 4.59 s


After generation, the `result.generated_images` object contains the generated images, which can then be displayed.


```
for generated_image in result.generated_images:
  imagen_image = generated_image.image.show()
```

## Generating images with Gemini 2.0 Flash image out model (experimental)


The `gemini-2.0-flash-preview-image-generation model` extends Gemini's multimodal capabilities to include conversational image generation and editing. This model can generate images along with text responses, making it highly versatile for mixed-media content creation.

### Select the Gemini 2.0 image out model

This model is specifically designed for generating and editing images conversationally.


```
MODEL_ID = "gemini-2.0-flash-preview-image-generation"
```

### Gemini 2.0 Flash image generation prompt (with interleaved text)

This example demonstrates how Gemini 2.0 Flash can generate both text and images in an interleaved fashion, providing a rich, conversational output that combines instructions with visual aids. You must explicitly set response_modalities to `['Text', 'Image']` to enable this feature.


```
%%time

contents = """
  Show me how to cook a Brazilian cuscuz with coconut milk and grated coconut.
  Include detailed step by step guidance with images.
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=contents,
    config=types.GenerateContentConfig(
        response_modalities=['Text', 'Image']
    )
)
```

    CPU times: user 459 ms, sys: 53.7 ms, total: 513 ms
    Wall time: 19.6 s


The output `response.candidates.content.parts` can contain both text and inline image data, which are then displayed accordingly.


```
for part in response.candidates[0].content.parts:
    if part.text is not None:
      display(Markdown(part.text))
    elif part.inline_data is not None:
      mime = part.inline_data.mime_type
      print(mime)
      data = part.inline_data.data
      display(Image(data=data, width=512, height=512))
```

### Gemini 2.0 Flash image generation prompt

This example focuses on generating an image based on a descriptive text prompt, similar to Imagen 3, but utilizing the Gemini 2.0 Flash model's capabilities for image generation within a text-based generation flow.


```
%%time

prompt = """
  Dynamic anime illustration: A happy Brazilian man with short grey hair and a
  grey beard, mid-presentation at a tech conference. He's wearing a fun blue
  short-sleeve shirt covered in mini avocado prints. Capture a funny, energetic
  moment where he's clearly enjoying himself, perhaps with an exaggerated joyful
  expression or a humorous gesture, stage background visible.
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
        response_modalities=['Text', 'Image']
    )
)
```

    CPU times: user 112 ms, sys: 6.52 ms, total: 118 ms
    Wall time: 3.5 s


The generated content is then processed to display the image parts.


```
for part in response.candidates[0].content.parts:
  if part.inline_data is not None and part.inline_data.mime_type is not None:
    mime = part.inline_data.mime_type
    print(mime)
    data = part.inline_data.data
    display(Image(data=data))
```

    image/png

#### Saving the generated image

The generated image data can be extracted from the response and saved locally, typically as a PNG file.


```
import pathlib

for part in response.candidates[0].content.parts:
    if part.text is not None:
      continue
    elif part.inline_data is not None:
      mime = part.inline_data.mime_type
      data = part.inline_data.data
      pathlib.Path("gemini_imgout.png").write_bytes(data)
```

### Editing images with Gemini 2.0 Flash image out

Gemini 2.0 Flash also supports image editing. You can provide an image as input along with a text prompt describing the desired modifications. This allows for conversational image manipulation.


```
%%time

import PIL

prompt = """
  make the image background in full white and add a wireless presentation
  clicker on the hand of the person
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        prompt,
        PIL.Image.open('gemini_imgout.png')
    ],
    config=types.GenerateContentConfig(
        response_modalities=['Text', 'Image']
    )
)
```

    CPU times: user 678 ms, sys: 15.6 ms, total: 694 ms
    Wall time: 5.79 s


The edited image and any accompanying text are then displayed.


```
for part in response.candidates[0].content.parts:
    if part.text is not None:
      display(Markdown(part.text))
    elif part.inline_data is not None:
      mime = part.inline_data.mime_type
      print(mime)
      data = part.inline_data.data
      display(Image(data=data))
```

    image/png

The edited image is saved, overwriting the previous `gemini_imgout.png` file (for future usage in this notebook).


```
for part in response.candidates[0].content.parts:
    if part.text is not None:
      continue
    elif part.inline_data is not None:
      mime = part.inline_data.mime_type
      data = part.inline_data.data
      pathlib.Path("gemini_imgout.png").write_bytes(data)
```

## Generating videos with the Veo2 model

Veo 2 is Google's advanced text-to-video and image-to-video model, capable of generating high-quality videos with detailed cinematic and visual styles. It can capture prompt nuances and maintain consistency across frames.

Princing warning Badge

⚠️

Veo is a paid-only feature and won't work if you are on the free tier.

### Select the Veo2 model to be used

The `veo-2.0-generate-001` model is used for video generation tasks.


```
VEO_MODEL_ID = "veo-2.0-generate-001"
```

### Run a text-to-video generation prompt

This section demonstrates how to generate videos directly from a text prompt. You can specify various configurations like `person_generation`, `aspect_ratio`, `number_of_videos`, `duration`, and a `negative_prompt` to guide the video generation process. Video generation is an asynchronous operation, so the code includes a loop to wait for the operation to complete.


```
%%time

import time
from google.genai import types
from IPython.display import Video, HTML

prompt = """
  Dynamic anime scene: A happy Brazilian man with short grey hair and a
  grey beard, mid-presentation at a tech conference. He's wearing a fun blue
  short-sleeve shirt covered in mini avocado prints. Capture a funny, energetic
  moment where he's clearly enjoying himself, perhaps with an exaggerated joyful
  expression or a humorous gesture, stage background visible.
"""

# Optional parameters
negative_prompt = "" # @param {type: "string"}
person_generation = "allow_adult"  # @param ["dont_allow", "allow_adult"]
aspect_ratio = "16:9" # @param ["16:9", "9:16"]
number_of_videos = 1 # @param {type:"slider", min:1, max:4, step:1}
duration = 8 # @param {type:"slider", min:5, max:8, step:1}

operation = client.models.generate_videos(
    model=VEO_MODEL_ID,
    prompt=prompt,
    config=types.GenerateVideosConfig(
      # At the moment the config must not be empty
      person_generation=person_generation,
      aspect_ratio=aspect_ratio,  # 16:9 or 9:16
      number_of_videos=number_of_videos, # supported value is 1-4
      negative_prompt=negative_prompt,
      duration_seconds=duration, # supported value is 5-8
    ),
)

# Waiting for the video(s) to be generated
while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)
    print(operation)

print(operation.result.generated_videos)
```

    [First Entry, ..., Last Entry]
    CPU times: user 188 ms, sys: 37.1 ms, total: 225 ms
    Wall time: 40.7 s


### See the video generation results

Once the video generation operation is complete, the generated video(s) can be downloaded and displayed within the notebook. Generated videos are stored for 2 days on the server, so it's important to save a local copy if needed.


```
for n, generated_video in enumerate(operation.result.generated_videos):
  client.files.download(file=generated_video.video)
  generated_video.video.save(f'video{n}.mp4') # Saves the video(s)
  display(generated_video.video.show()) # Displays the video(s) in a notebook
```

### Run a image-to-video generation prompt

Veo 2 can also generate videos from an input image, using the image as the starting frame. This allows you to bring static images to life by adding motion and narrative based on a text prompt.


```
%%time

import io
from PIL import Image

prompt = """
  Dynamic anime scene: A happy Brazilian man with short grey hair and a
  grey beard, mid-presentation at a tech conference. He's wearing a fun blue
  short-sleeve shirt covered in mini avocado prints. Capture a funny, energetic
  moment where he's clearly enjoying himself, perhaps with an exaggerated joyful
  expression or a humorous gesture, stage background visible.
"""

image_name = "gemini_imgout.png"

# Optional parameters
negative_prompt = "ugly, low quality" # @param {type: "string"}
aspect_ratio = "16:9" # @param ["16:9", "9:16"]
number_of_videos = 1 # @param {type:"slider", min:1, max:4, step:1}
duration = 8 # @param {type:"slider", min:5, max:8, step:1}

# Loading the image
im = Image.open(image_name)

# converting the image to bytes
image_bytes_io = io.BytesIO()
im.save(image_bytes_io, format=im.format)
image_bytes = image_bytes_io.getvalue()

operation = client.models.generate_videos(
    model=VEO_MODEL_ID,
    prompt=prompt,
    image=types.Image(image_bytes=image_bytes, mime_type=im.format),
    config=types.GenerateVideosConfig(
      # At the moment the config must not be empty
      aspect_ratio = aspect_ratio,  # 16:9 or 9:16
      number_of_videos = number_of_videos, # supported value is 1-4
      negative_prompt = negative_prompt,
      duration_seconds = duration, # supported value is 5-8
    ),
)

# Waiting for the video(s) to be generated
while not operation.done:
    time.sleep(20)
    operation = client.operations.get(operation)
    print(operation)

print(operation.result.generated_videos)
```

    [First Entry, ..., Last Entry]
    CPU times: user 670 ms, sys: 27.9 ms, total: 698 ms
    Wall time: 41.4 s


The generated videos are then saved and displayed.


```
for n, generated_video in enumerate(operation.result.generated_videos):
  client.files.download(file=generated_video.video)
  generated_video.video.save(f'video{n}.mp4') # Saves the video(s)
  display(generated_video.video.show()) # Displays the video(s) in a notebook
```

## Generating text-to-speech (TTS) with Gemini models

The Gemini API offers native text-to-speech (TTS) capabilities, allowing you to transform text into natural-sounding audio. This feature provides fine-grained control over various aspects of speech, including style, accent, pace, and tone.

### Select the TTS model to be used

The `