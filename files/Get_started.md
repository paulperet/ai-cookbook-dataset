# Gemini API: Getting started with Gemini models

---
> **Gemini 3 Pro/Flash**: If you are only interested in the new [Gemini 3 models](https://ai.google.dev/gemini-api/docs/gemini-3) new capabilities ([thinking levels](#thinking_level), [media resolution](#media_resolution) and [thoughts signatures](#thoughts_signature), jump directly to the [dedicated section](#gemini3) at the end of this notebook.

---


The **[Google Gen AI SDK](https://github.com/googleapis/python-genai)** provides a unified interface to [Gemini models](https://ai.google.dev/gemini-api/docs/models) through both the [Gemini Developer API](https://ai.google.dev/gemini-api/docs) and the Gemini API on [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/overview). With a few exceptions, code that runs on one platform will run on both. This notebook uses the Developer API.

This notebook will walk you through:

* [Installing and setting-up](#setup) the Google GenAI SDK
* [Text](#text_prompt) and [multimodal](#multimodal_prompt) prompting
* Setting [system instructions](#system_instructions)
* Control the [thinking](#thinking) process
* Counting [tokens](#count_tokens)
* Configuring [safety filters](#safety_filters)
* Initiating a [multi-turn chat](#chat)
* Generating a [content stream](#stream) and sending [asynchronous](#async) requests
* [Controlling generated output](#json)
* Using [function calling](#function_calling)
* Grounding your requests using [file uploads](#file_api), [Google Search](#search_grounding), [Google Maps](#maps), [Youtube](#youtube_link) or by add [URLs](#url_context) to you prompt
* Using [context caching](#caching)
* Generating [text embeddings](#embeddings)

More details about the SDK on the [documentation](https://ai.google.dev/gemini-api/docs/sdks).

Feature-specific models have their own dedicated guides:
* Podcast and speech generation using [Gemini TTS](./Get_started_TTS.ipynb),
* Live interaction with [Gemini Live](./Get_started_LiveAPI.ipynb),
* Image generation using [Imagen](./Get_started_imagen.ipynb),
* Video generation using [Veo](./Get_started_Veo.ipynb),
* Music generation using [Lyria RealTime](./Get_started_LyriaRealTime.ipynb).

<a name="setup"></a>
## Setup

### Install SDK

Install the SDK from [PyPI](https://github.com/googleapis/python-genai). It's recommended to always use the latest version.


```
%pip install -U -q 'google-genai>=1.51.0' # 1.51 is needed for Gemini 3 pro thinking levels support
```

[First Entry, ..., Last Entry]

### Setup your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GEMINI_API_KEY`. If you don't already have an API key or you aren't sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata

GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
```

### Initialize SDK client

With the new SDK, now you only need to initialize a client with you API key (or OAuth if using [Vertex AI](https://cloud.google.com/vertex-ai)). The model is now set in each call.


```
from google import genai
from google.genai import types

client = genai.Client(api_key=GEMINI_API_KEY)
```

### Choose a model

Select the model you want to use in this guide. You can either select one from the list or enter a model name manually. Keep in mind that some models, such as the 2.5 ones are thinking models and thus take slightly more time to respond. For more details, you can see [thinking notebook](./Get_started_thinking.ipynb) to learn how to control the thinking.

Feel free to select [Gemini 3 Pro](https://ai.google.dev/gemini-api/docs/models#gemini-3-pro) if you want to try our newest model, but keep in mind that it has no free tier.

For a full overview of all Gemini models, check the [documentation](https://ai.google.dev/gemini-api/docs/models/gemini).


```
MODEL_ID = "gemini-2.5-flash" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-3-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

<a name="text_prompt"></a>
## Send text prompts

Use the `generate_content` method to generate responses to your prompts. You can pass text directly to `generate_content` and use the `.text` property to get the text content of the response. Note that the `.text` field will work when there's only one part in the output.


```
from IPython.display import Markdown

response = client.models.generate_content(
    model=MODEL_ID,
    contents="What's the largest planet in our solar system?"
)

display(Markdown(response.text))
```


The largest planet in our solar system is **Jupiter**.

It's a gas giant and is more than twice as massive as all the other planets combined!


<a name="system_instructions"></a>
## Add system instructions

You can also add system instructions to give the model direction on how to respond and which persona it should use. This is especially useful for mixture-of-experts models like the the pro models.


```
system_instruction = "You are a pirate and are explaining things to 5 years old kids."

response = client.models.generate_content(
    model=MODEL_ID,
    contents="What's the largest planet in our solar system?",
    config=types.GenerateContentConfig(
        system_instruction=system_instruction,
    )
)

display(Markdown(response.text))

```


Ahoy there, me hearties! Gather 'round, gather 'round, and let old Cap'n tell ye a grand secret of the skies!

Ye wanna know which planet is the biggest, eh? The most enormous, the most ginormous, the proper behemoth of them all?

Well, batten down the hatches and listen close! It be the mighty **JUPITER**!

Shiver me timbers, that planet is a true giant! Imagine yer very own Earth, the one we're standin' on right now... well, ye could fit *over a thousand* of those little Earths right inside Jupiter! It's that big!

It's like the biggest, swirliest marble ye ever did see, made of gas and storms, with a giant red eye always lookin' out into space! A real king of the planets, it is!

What do ye think of that, eh? A mighty, mighty world, Jupiter is!


<a name="count_tokens"></a>
## Count tokens

Tokens are the basic inputs to the Gemini models. You can use the `count_tokens` method to calculate the number of input tokens before sending a request to the Gemini API.


```
response = client.models.count_tokens(
    model=MODEL_ID,
    contents="What's the highest mountain in Africa?",
)

print(f"This prompt was worth {response.total_tokens} tokens.")
```

    This prompt was worth 10 tokens.


<a name="parameters"></a>
## Configure model parameters

You can include parameter values in each call that you send to a model to control how the model generates a response.

Learn more about [experimenting with parameter values](https://ai.google.dev/gemini-api/docs/prompting-strategies#model-parameters) in the documentation.


```
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Tell me how the internet works, but pretend I'm a puppy who only understands squeaky toys.",
    config=types.GenerateContentConfig(
        temperature=0.4, # Temperature of 1 is strongly recommended for Gemini 3 Pro
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        seed=5,
        stop_sequences=["STOP!"],
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
)

display(Markdown(response.text))
```




*Woof!* Sit! Stay! Who’s a good boy? You are! Listen close. Ears up!

Okay, imagine the whole world is one giant backyard.

**1. The Magic Window (Your Computer)**
You are sitting in front of the Magic Window. You see a picture of a ball. You want the ball. You tap the Magic Window with your paw. That tap is a **SQUEAK**.

**2. The Invisible Leash (Wi-Fi)**
When you Squeak, the sound doesn't stay here. It runs very fast! It runs along the Invisible Leash. *Zoom!* Like when you get the zoomies!

**3. The Router (The Fetch Master)**
The Squeak runs to a little blinking box in the corner. That box is the Fetch Master. The Fetch Master catches your Squeak and throws it *really far* out of the house. *Go long!*

**4. The Wires (The Tunnels)**
Your Squeak runs through tunnels under the ground. It runs past the mailman (grrr!), past the squirrels, all the way to a giant building far away.

**5. The Server (The Giant Toy Box)**
The giant building is the Giant Toy Box. It has *all* the squeaky toys in the world.
Your Squeak arrives and barks, "I WANT THE RED BALL!"
The Giant Toy Box hears you. It finds the Red Ball.

**6. Packets (Chewing the Toy)**
But wait! The Red Ball is too big to fit through the tunnels!
So, the Giant Toy Box chews the ball into tiny, tiny pieces. *Chomp chomp chomp.*
Don't worry! It’s okay!
It throws all the tiny chewed pieces back into the tunnels. *Fetch!*

**7. Reassembly (The Miracle)**
The tiny pieces run back past the squirrels, past the mailman, through the Fetch Master box, and onto your Magic Window.
Your Magic Window catches all the tiny pieces and—*SQUEAK!*—glues them back together instantly!

Now the Red Ball is on your screen.

**SQUEAK SQUEAK!** Good internet! Good boy!



<a name="thinking"></a>
## Control the thinking process

All models since the 2.5 generation are thinking models, which means that they are first analysing your request, strategizing about how to answer and only afterwards starting to answer you. This is very useful for complex requests but at the cost of some latency.

Check the [dedicated guide](./Get_started_thinking.ipynb) for more details.

### Check the thought process

By adding the `include_thoughts=True` option in the config, you can check the though proces of the model.


```
prompt = "A man moves his car to an hotel and tells the owner he’s bankrupt. Why?"

response = client.models.generate_content(
  model=MODEL_ID,
  contents=prompt,
  config=types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(
      include_thoughts=True
    )
  )
)

for part in response.parts:
  if not part.text:
    continue
  if part.thought:
    display(Markdown("### Thought summary:"))
    display(Markdown(part.text))
    print()
  else:
    display(Markdown("### Answer:"))
    display(Markdown(part.text))
    print()

print(f"We used {response.usage_metadata.thoughts_token_count} tokens for the thinking phase and {response.usage_metadata.prompt_token_count} for the output.")
```


### Thought summary:



**Unraveling the Riddle: A Monopoly-Focused Thought Process**

Alright, let's break this down. A man moves a car to a hotel and then declares bankruptcy. My initial instinct is to look beyond the literal. Is this some insurance scam? No, that's not it. I need to think outside the box, maybe a game scenario? The car and hotel seem oddly specific… Wait a minute! "Car," "Hotel," "Bankrupt"... *Monopoly*! That has to be it.

In *Monopoly*, you move your car token around the board and end up paying rent if you land on someone else's property, especially if it's got a hotel. This man is obviously playing as the car token, lands on a hotel property owned by someone else, and the rent is too steep, bam, bankruptcy!

The riddle is playing on our literal interpretation. The key is recognizing the symbolism. It's a classic lateral thinking puzzle. Moving the car represents landing on a hotel property, the owner is the opponent, and bankruptcy is the consequence of not being able to pay the rent. Simple enough. It is **Monopoly**. That's the answer.





    



### Answer:



He is playing **Monopoly**.

The man is playing as the racecar token. He landed on a property owned by the other player (the "owner") that had a hotel on it, and the rent was high enough to bankrupt him.


    
    We used 575 tokens for the thinking phase and 20 for the output.


### Disable thinking

On flash and flash-lite models, you can disable the thinking by setting its `thinking_budget` to 0.


```
if "-pro" not in MODEL_ID:
  response = client.models.generate_content(
    model=MODEL_ID,
    contents="Quicky tell me a joke about unicorns.",
    config=types.GenerateContentConfig(
      thinking_config=types.ThinkingConfig(
        thinking_budget=0
      )
    )
  )

  display(Markdown(response.text))
```




Why did the unicorn run across the road?

To get to the other rainbow!



Inversely, you can also use `thinking_budget` to set it even higher (up to 24576 tokens).

For Gemini 3, please check the [dedicated section](#thinking_level) at the end of this guide.

<a name="multimodal_prompt"></a>
## Send multimodal prompts

Use Gemini model, a multimodal model that supports multimodal prompts. You can include text, [PDF documents](../quickstarts/PDF_Files.ipynb), images, [audio](../quickstarts/Audio.ipynb) and [videos](../quickstarts/Video.ipynb) in your prompt requests and get text or code responses. Check the [File API](#file_api) section below for more examples.

In this first example, you'll download an image from a specified URL, save it as a byte stream and then write those bytes to a local file named `jetpack.png`.


```
import requests
import pathlib
from PIL import Image

IMG = "https://storage.googleapis.com/generativeai-downloads/data/jetpack.png" # @param {type: "string"}

img_bytes = requests.get(IMG).content

img_path = pathlib.Path('jetpack.png')
img_path.write_bytes(img_bytes)
```




    1567837



Now send the image, and ask Gemini to generate a short blog post based on it.


```
from IPython.display import display, Markdown
image = Image.open(img_path)
image.thumbnail([512,512])

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        image,
        "Write a short and engaging blog post based on this picture."
    ]
)

display(image)
Markdown(response.text)
```





**Title: Beat the Traffic in Style: Introducing the Jetpack Backpack Concept**

We’ve all been there. You’re five minutes late, the train is delayed, or the highway is a parking lot, and you find yourself wishing you could just *lift off* and fly over the chaos.

Well, it looks like the solution might be sitting on a notepad right now.

We recently stumbled upon a concept sketch for what might be the greatest invention for the modern commuter: **The Jetpack Backpack.** It’s part James Bond, part Silicon Valley, and 100% awesome.

Here is why this sketch needs to become a reality immediately:

**1. Stealth Mode Commuting**
The biggest problem with traditional jetpacks? They are bulky and awkward. This design changes the game. It is lightweight and explicitly designed to **"look like a normal backpack."** You can land outside your office building, retract the boosters, and walk in looking professional.

**2. Practicality Meets Sci-Fi**
It isn't just for flying; it's for working. The design features **padded strap support** for comfort and is spacious enough to **fit an 18" laptop.** You can carry your mobile office *while* hovering 50 feet above the sidewalk.

**3. Eco-Friendly Flight**
Forget burning jet fuel. This concept is **steam-powered**, making it "green and clean." Plus, in a nod to modern convenience, it features **USB-C charging**. Just plug it in next to your phone at night, and you’re ready for takeoff in the morning.

**The Catch?**
It has a **15-minute battery life**. It’s not going to get you across the country, but it is the perfect amount of time to skip that one terrible intersection or make a very dramatic entrance at a rooftop party.

So, who is ready to invest? We aren't saying this is available in stores yet, but if it were, our commute would look a whole lot cooler.

**Would you wear the Jetpack Backpack? Let us know in the comments!**



<a name="images"></a>
## Generate Images

Gemini can output images directly as part of a conversation using the [Image generation](./Image_out.ipynb) models (aka "Nano-banana).


```
from IPython.display import Image, Markdown

response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents='Hi, can you create a 3