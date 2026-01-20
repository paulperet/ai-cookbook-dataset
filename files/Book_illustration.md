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

# Illustrating a book using Gemini 2.5 Image

In this guide, you are going to use multiple Gemini features (long context, multimodality, structured output, file API, chat mode...) in conjunction with the Gemini 2.5 image model (aka. nano banana) to illustrate a book.

Each concept will be explained along the way, but if you need a simpler introduction to Gemini Image generation model, check the [getting started](../quickstarts/Get_Started_Nano_Banana.ipynb) notebook, or the [Image generation documentation](https://ai.google.dev/gemini-api/docs/image-generation).

Note: for the sake of the notebook's size (and your billing if you run it), the number of images has been limited to 3 characters and 3 chapters each time, but feel free to remove the limitation if you want more with your own experimentations.

Also note that this notebook used to use [Imagen](https://ai.google.dev/gemini-api/docs/imagen) models instead of Gemini 2.5 Image. If you are interested in the Imagen version, checked-out this [old version](../../c604f672f621186f609b1d977a918250eaca19f2/examples/Book_illustration.ipynb).

## 0/ Setup

This section install the SDK, set it up using your [API key](../quickstarts/Authentication.ipynb), imports the relevant libs, downloads the sample videos and upload them to Gemini.

Just collapse (click on the little arrow on the left of the title) and run this section if you want to jump straight to the examples (just don't forget to run it otherwise nothing will work).

### Install SDK

```
%pip install -U -q "google-genai"
```

### Setup your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.

```
from google.colab import userdata

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
```

### Initialize SDK client

With the new SDK you now only need to initialize a client with you API key (or OAuth if using [Vertex AI](https://link_to_vertex_AI)). The model is now set in each call.

```
from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)
```

### Imports

Some imports to display markdown text and images in Colab.

```
import json
from PIL import Image
from IPython.display import display, Markdown
```

### Select models

```
IMAGE_MODEL_ID = "gemini-2.5-flash-image"  # @param ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"] {"allow-input":true, isTemplate: true}
GEMINI_MODEL_ID = "gemini-2.5-flash" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

For the sake of the notebook's size (and your billing if you run it), the number of images has been limited to 3 characters and 3 chapters each time, but feel free to remove the limitation if you want more with your own experimentations.

```
max_character_images = 3 # @param {type:"integer",isTemplate: true, min:1}
max_chapter_images = 3 # @param {type:"integer",isTemplate: true, min:1}
```

# Illustrate a book: The Wind in the Willows

## 1/ Get a book and upload using the File API

Start by downloading a book from the open-source [Project Gutenberg](www.gutenberg.org) library. For example, it can be [The Wind in the Willows](https://en.wikipedia.org/wiki/The_Wind_in_the_Willows) from Kenneth Grahame.

`client.files.upload` is used to upload the file so that Gemini can easily access it.

```
import requests

url = "https://www.gutenberg.org/cache/epub/289/pg289.txt"  # @param {type:"string"}

response = requests.get(url)
with open("book.txt", "wb") as file:
    file.write(response.content)

book = client.files.upload(file="book.txt")

```

## 2/ Start the chat

You are going to use [chat mode](https://ai.google.dev/gemini-api/docs/text-generation?lang=python#chat) here so that Gemini will keep the history of what you asked it, and also so that you don't have to send it the book every time. More details on chat mode in the [Get Started](https://colab.sandbox.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Get_started.ipynb#scrollTo=b6sB7W-jdGxJ) notebook.

You should also define the format of the output you want using [structured output](https://ai.google.dev/gemini-api/docs/structured-output?lang=python#generate-json). You will mainly use Gemini to generate prompts so let's define a Pydantic model with two fields, a name and a prompt:

```
from pydantic import BaseModel

class Prompts(BaseModel):
    name: str
    prompt: str

```

`client.chats.create` starts the chat and defines its main parameters (model and the output you want).

```
# Re-run this cell if you want to start anew.
chat = client.chats.create(
    model=GEMINI_MODEL_ID,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[Prompts],
    ),
)

chat.send_message(
    [
        "Here's a book, to illustrate using Gemini 2.5 Image. Don't say anything for now, instructions will follow.",
        book
    ]
);
```

The first message sent to the model is just to give it a bit of context ("*to illustrate using Gemini 2.5 Image*"), and more importantly give it the book.

It could have been done during the next step, especially since you're not interested in what the model has to say this time, but splitting the two steps makes it clearer.

## 3/ Define a style

If you want to test a specific style, just write it down and Gemini will use it. Still, tell Gemini about it so it will adapt the prompts it will generate accordingly.

If you prefer to let Gemini choose the best style for the book, leave the style empty and ask Gemini to define a style fitting to the book.

```
style = "" # @param {type:"string", "placeholder":"Write your own style or leave empty to let Gemini generate one"}

if style=="":
  response = chat.send_message("""
    Can you define a art style that would fit the story?
    Just give us the prompt for the art syle that will added to the furture prompts.
    """)
  style = json.loads(response.text)[0]["prompt"]
else:
  chat.send_message(f"""
    The art style will be:"{style}".
    Keep that in mind when generating future prompts.
    Keep quiet for now, instructions will follow.
  """)

display(Markdown(f"### Style:"))
print(style)

style = f'Follow this style: "{style}" '
```

### Style:

    Classic storybook illustration, gentle whimsical realism, soft watercolor and pen-and-ink style, with warm, inviting lighting and rich detail, depicting anthropomorphic animals in the English countryside.

Let's also define some more instructions which will act as "system instructions" or a negative prompt to tell the model what you do not want to see (text on the images).

```
system_instructions = """
  There must be no text on the image, it should not look like a cover page.
  It should be an full illustration with no borders, titles, nor description.
  Stay family-friendly with uplifting colors.
  Each produced should be a simple image, no panels.
"""
```

## 4/ Generate portraits of the main characters

You are now ready to start generating images, starting with the main characters.

Ask Gemini to describe each of the main characters (excluding children as Gemini 2.5 Image can't generate images of them) and check that the output follows the format requested.

```
response = chat.send_message("""
  Can you describe the main characters (only the adults) and
  prepare a prompt describing them with as much details as possible (use the descriptions from the book)
  so Gemini 2.5 Image can generate images of them? Each prompt should be at least 50 words.
""")

characters = json.loads(response.text)

print(json.dumps(characters, indent=4))
```

    [
        {
            "name": "Mole",
            "prompt": "A small, good-hearted anthropomorphic mole with black fur, showing faint splashes of whitewash from spring-cleaning. He has a keen, curious expression, often appearing wide-eyed and easily excited by new discoveries like the river. When wet, he is a 'squashy, pulpy lump of misery' or has a 'bedraggled appearance.' He is loyal, eager, and appreciative of home comforts, sometimes timid but capable of bravery when pushed, beaming with delight at familiar objects."
        },
        {
            "name": "Water Rat",
            "prompt": "A sociable, good-natured anthropomorphic Water Rat with small, neat ears and thick, silky brown hair. He embodies a love for the river, often seen sculling or preparing for outings. His eyes are clear, dark, and brown, though they can sometimes appear glazed and shifting grey when under a powerful spell. He possesses a practical and observant demeanor, yet also has a dreamy, poetic side, capable of being stern but always a loyal and responsible friend to Mole."
        },
        {
            "name": "Mr. Toad",
            "prompt": "A short, stout anthropomorphic toad, prone to dramatic shifts in emotion and appearance. He is often depicted in eccentric motor-car attire including goggles, cap, gaiters, and an enormous overcoat, or in a comical washerwoman's disguise with a squat figure, cotton print gown, and rusty black bonnet. His expression ranges from boastful, conceited, and self-important to terrified, self-pitying, and filled with a furious, swollen pride. Despite his flaws, he is ultimately good-hearted."
        },
        {
            "name": "Mr. Badger",
            "prompt": "A reclusive but kindly anthropomorphic badger, appearing rough and touzled, often in a long dressing-gown with down-at-heel slippers, holding a flat candlestick or a stout cudgel. His whiskers are bristling, and his expression can be gruff and suspicious, but also paternal, thoughtful, and placid. He commands respect with his solid qualities, wise counsel, and firm, no-nonsense demeanor, embodying a grounded, enduring spirit rooted deep in the Wild Wood."
        },
        {
            "name": "Otter",
            "prompt": "A cheerful and social anthropomorphic otter with a broad, glistening muzzle and strong, white teeth, often seen shaking water from his coat. He is boisterous and friendly, quick to laugh, but also demonstrates a deep, caring nature as a father. His keen eyes reflect a playful spirit, and he moves with a confident, knowing air, familiar with all river paths, always ready for a chat and a meal."
        },
        {
            "name": "Sea Rat",
            "prompt": "A lean, keen-featured anthropomorphic rat with thin, long paws and small gold earrings in his neatly-set, well-shaped ears. He wears a faded blue knitted jersey and patched, stained breeches. His eyes are much wrinkled at the corners, shining with a distant, experienced light, sometimes reflecting the 'foam-streaked grey-green of leaping Northern seas,' conveying a captivating, wanderlust-filled persona that is both independent and an engaging storyteller."
        }
    ]

Now that you have the prompts, you just need to loop on all the characters and have Gemini 2.5 Image generate an image for them. This model uses the same API as the text generation models.

Like before, for the sake of consistency, we are going to use chat mode, but within a different instance.

For an extensive explanation on the Gemini 2.5 Image model and its options, check the [getting started with Gemini 2.5 Image](../quickstarts/Get_Started_Nano_Banana.ipynb) notebook. But here's a quick overview of what being used here:
* `prompt` is the prompt passed down to Gemini 2.5 Image. You're not just sending what Gemini has generate to describe the chacaters but also our style and our system instructions.
* `response_modalities=['Image']` because we only want images
* `aspect_ratio="9:16"` because we want portraits images

Note that we could have used system instructions but the model currently ignores them so we decided to pass them as message.

```
image_chat = client.chats.create(
    model=IMAGE_MODEL_ID,
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="9:16"
        )
    )
)

image_chat.send_message(f"""
  You are going to generate portrait images to illustrate The Wind in the Willows from Kenneth Grahame.
  The style we want you to follow is: {style}
  Also follow those rules: {system_instructions}
""")

for character in characters[:max_character_images]:
  display(Markdown(f"### {character['name']}"))
  display(Markdown(character['prompt']))

  response = image_chat.send_message(f"Create an illustration for {character['name']} following this description: {character['prompt']}")

  for part in response.parts:
    if part.inline_data:
      generated_image = part.as_image()
      generated_image.show()
      break

# Be careful; long output (see below)
```

### Mole

A small, good-hearted anthropomorphic mole with black fur, showing faint splashes of whitewash from spring-cleaning. He has a keen, curious expression, often appearing wide-eyed and easily excited by new discoveries like the river. When wet, he is a 'squashy, pulpy lump of misery' or has a 'bedraggled appearance.' He is loyal, eager, and appreciative of home comforts, sometimes timid but capable of bravery when pushed, beaming with delight at familiar objects.

### Water Rat

A sociable, good-natured anthropomorphic Water Rat with small, neat ears and thick, silky brown hair. He embodies a love for the river, often seen sculling or preparing for outings. His eyes are clear, dark, and brown, though they can sometimes appear glazed and shifting grey when under a powerful spell. He possesses a practical and observant demeanor, yet also has a dreamy, poetic side, capable of being stern but always a loyal and responsible friend to Mole.

### Mr. Toad

A short, stout anthropomorphic toad, prone to dramatic shifts in emotion and appearance. He is often depicted in eccentric motor-car attire including goggles, cap, gaiters, and an enormous overcoat, or in a comical washerwoman's disguise with a squat figure, cotton print gown, and rusty black bonnet. His expression ranges from boastful, conceited, and self-important to terrified, self-pitying, and filled with a furious, swollen pride. Despite his flaws, he is ultimately good-hearted.

## 5/ Illustrate the chapters of the book

After the characters, it's now time to create illustrations for the content of the book. You are going to ask Gemini to generate prompts for each chapter and then ask Gemini 2.5 Image to generate images based on those prompts.

```
response = chat.send_message("Now, for each chapters of the book, give me a prompt to illustrate what happens in it. Be very descriptive, especially of the characters. Be very descriptive and remember to tell their name and to reuse the character prompts if they appear in the images. Each character should at least be described with 30 words.")

chapters = json.loads(response.text)[:max_chapter_images]

print(json.dumps(chapters, indent=4))
```

    [
        {
            "name": "Chapter I. The River Bank",
            "prompt": "Classic storybook illustration, gentle whimsical realism, soft watercolor and pen-and-ink style, with warm, inviting lighting and rich detail, depicting anthropomorphic animals in the English countryside. A small, good-hearted anthropomorphic mole with black fur, Mole (showing faint splashes of whitewash from spring-cleaning, his eyes wide with wonder, eager and appreciative of new experiences), sits on a sun-drenched riverbank, trailing a paw in the water. Beside him, a sociable, good-natured anthropomorphic Water Rat (with small, neat ears and thick, silky brown hair, his dark and clear eyes gazing dreamily), sculls a small blue and white boat across the sparkling river. In the distance, a broad, glistening muzzle of Otter (a cheerful and social anthropomorphic otter with strong, white teeth, familiar with all river paths), emerges from the river, approaching the bank."
        },
        {
            "name": "Chapter II. The Open Road",
            "prompt": "Classic storybook illustration, gentle whimsical realism, soft watercolor and pen-and-ink style, with warm, inviting lighting and rich detail, depicting anthropomorphic animals in the English countryside. Mr. Toad (a short, stout anthropomorphic toad, prone to dramatic shifts in emotion and appearance, currently filled with boastful, conceited self-importance), enthusiastically gestures towards his new canary-yellow gypsy caravan with green trim and red wheels. The caravan lies wrecked in a ditch. Mr. Toad sits in the dusty road, eyes fixed on a vanishing, magnificent, immense red motor-car (a dramatic speck in the distance), murmuring 'Poop-poop!' with a look of blissful obsession.