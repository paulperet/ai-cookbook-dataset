

# Gemini Native Image generation (aka 🍌Nano-Banana models)

<a target="_blank" href="https://colab.research.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Get_Started_Nano_Banana.ipynb"></a>

---
> **🍌Nano Banana Pro**: If you are only interested in the new [Gemini 3 Pro Image](https://ai.google.dev/gemini-api/docs/models#gemini-3-pro-image) model, jump directly to the [dedicated section](#nano-banana-pro).

---


This notebook will show you how to use the native Image-output feature of Gemini, using the model multimodal capabilities to output both images and texts, and iterate on an image through a discussion.

There are now 2 models you can use:
* `gemini-2.5-flash-image` aka. "nano-banana": Cheap and fast yet powerful. This should be your default choice.
* `gemini-3-pro-image-preview` aka "nano-banana-pro": More powerful thanks to its **thinking** capabilities and its access to real-wold data using **Google Search**. It really shines at creating diagrams and grounded images. And cherry on top, it can create 2K and 4K images!

These models are really good at:
* **Maintaining character consistency**: Preserve a subject’s appearance across multiple generated images and scenes
* **Performing intelligent editing**: Enable precise, prompt-based edits like inpainting (adding/changing objects), outpainting, and targeted transformations within an image
* **Compose and merge images**: Intelligently combine elements from multiple images into a single, photorealistic composite (maximum 3 with flash, 14 with pro)
* **Leverage multimodal reasoning**: Build features that understand visual context, such as following complex instructions on a hand-drawn diagram

Following this guide, you'll learn how to do all those things and even more.

<!-- Princing warning Badge -->
<table>
  <tr>
    <!-- Emoji -->
    <td bgcolor="#ffe680">
      <font size=30>⚠️</font>
    </td>
    <!-- Text Content Cell -->
    <td bgcolor="#ffe680">
      <h3><font color=black><font color='#217bfe'><a href="https://ai.google.dev/gemini-api/docs/billing#enable-cloud-billing">Enable billing</font></a> to use Image Generation. This is a pay-as-you-go feature (cf. <a href="https://ai.google.dev/pricing#gemini-2.5-flash-image-preview"><font color='#217bfe'>pricing</font></a>).</font></h3>
    </td>
  </tr>
</table>

Note that [Imagen](./Get_started_imagen.ipynb) models also offer image generation but in a slightly different way as the Image-out feature has been developed to work iteratively so if you want to make sure certain details are clearly followed, and you are ready to iterate on the image until it's exactly what you envision, Image-out is for you.

Check the [documentation](https://ai.google.dev/gemini-api/docs/image-generation#choose-a-model) for more details on both features and some more advice on when to use each one.

## Setup

### Install SDK


```
%pip install -U -q "google-genai>=1.40.0" # minimum version needed for the aspect ratio
```

### Setup your API key

To run the following cell, your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see [Authentication](../quickstarts/Authentication.ipynb) for an example.


```
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

### Initialize SDK client

With the new SDK you now only need to initialize a client with your API key (or OAuth if using Vertex AI). The model is now set in each call.


```
from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)
```

### Select a model

You can choose between two models:
* `gemini-2.5-flash-image` aka. "nano-banana": Cheap and fast yet powerful. This should be your default choice.
* `gemini-3-pro-image-preview` aka "nano-banana-pro": Has thinking and google search grounding, and can even output 2K and 4K images (cf. [dedicated section](#nano-banana-pro))


```
MODEL_ID = "gemini-2.5-flash-image" # @param ["gemini-2.5-flash-image", "gemini-3-pro-image-preview"] {"allow-input":true, isTemplate: true}
```

### Utils

These two functions will help you manage the outputs of the model.

Compared to when you simply generate text, this time the output will contain multiple parts, some one them being text while others will be images. You'll also have to take into account that there could be multiple images so you cannot stop at the first one.



```
from IPython.display import display, Markdown
import pathlib

# Loop over all parts and display them either as text or images
def display_response(response):
  for part in response.parts:
    if part.thought: # We don't want to see the thoughts
      continue
    if part.text:
      display(Markdown(part.text))
    elif image:= part.as_image():
      image.show()

# Save the image
# If there are multiple ones, only the last one will be saved
def save_image(response, path):
  for part in response.parts:
    if image:= part.as_image():
      image.save(path)
```

## Generate images

Using the Gemini Image generation model is the same as using any Gemini model: you simply call `generate_content`.

You can set the `response_modalities` to indicate to the model that you are expecting text and images in the output but it's optional as this is expected with this model.

If you just want an image and don't need text, you can set `response_modalities=['Image']`.


```
prompt = 'Create a photorealistic image of a siamese cat with a green left eye and a blue right one and red patches on his face and a black and pink nose' # @param {type:"string"}

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
        response_modalities=['Text', 'Image'] # response_modalities=['Image'] if you only want the images
    )
)

display_response(response)
save_image(response, 'cat.png')
```


Here is your requested image: 



    


## Edit images

You can also do image editing, simply pass the original image as part of the prompt. Don't limit yourself to simple edit, Gemini is able to keep the character consistency and reprensent you character in different behaviors or places.


```
import PIL

text_prompt = "Create a side view picture of that cat, in a tropical forest, eating a nano-banana, under the stars" # @param {type:"string"}

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        text_prompt,
        PIL.Image.open('cat.png')
    ]
)

display_response(response)
save_image(response, 'cat_tropical.png')
```


    


As you can see, you can clearly recognize the same cat with its peculiar nose and eyes.

## Control aspect ratio

You can control the aspect ratio of the output image. The model's primary behavior is to match the size of your input images; otherwise, it defaults to generating square (1:1) images.

To do so, add an `aspect_ratio` value to the `image_config` as you can see in the cell below. The different ratios available and the size of the image generated are listed in this table:

| Aspect ratio | Resolution | Tokens |
| --- | --- | --- |
| 1:1 | 1024x1024 | 1290 |
| 2:3 | 832x1248 | 1290 |
| 3:2 | 1248x832 | 1290 |
| 3:4 | 864x1184 | 1290 |
| 4:3 | 1184x864 | 1290 |
| 4:5 | 896x1152 | 1290 |
| 5:4 | 1152x896 | 1290 |
| 9:16 | 768x1344 | 1290 |
| 16:9 | 1344x768 | 1290 |
| 21:9 | 1536x672 | 1290 |

Note that the number of tokens stays the same for all aspect ratio.


```
import PIL

text_prompt = "Now the cat should keep the same attitude, but be well dressed in fancy restaurant and eat a fancy nano banana." # @param {type:"string"}
aspect_ratio = "16:9" # @param ["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"]

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        text_prompt,
        PIL.Image.open('cat_tropical.png')
    ],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
        )
    )

)

display_response(response)
save_image(response, 'cat_resaurant.png')
```


    


## Get multiple images (ex: tell stories)

So far you've only generated one image per call, but you can request way more than that! Let's try a baking receipe or telling a story.


```
prompt = "Show me how to bake macarons with images" # @param ["Show me how to bake macarons with images","Create a beautifully entertaining 8 part story with 8 images with two blue characters and their adventures in the 1960s music scene. The story is thrilling throughout with emotional highs and lows and ending on a great twist and high note. Do not include any words or text on the images but tell the story purely through the imagery itself. "] {"allow-input":true}

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
)

display_response(response)

# Be careful; long output (see below)
```

The output of the previous code cell could not be saved in the notebook without making it too big to be managed by Github, but here are some examples of what it should look like when you run it when asking for a story, or for a baking receipe:

----------
**Prompt**: *Create a beautifully entertaining 8 part story with 8 images with two blue characters and their adventures in the 1960s music scene. The story is thrilling throughout with emotional highs and lows and ending on a great twist and high note. Do not include any words or text on the images but tell the story purely through the imagery itself.*
(Images have been stitched together)

----------
**Prompt**: *Show me how to bake macarons with images*


That sounds delicious! Here's a simplified guide on how to bake macarons. While it can be a bit tricky, practice makes perfect!

**Ingredients you'll need:**

*   **For the Macaron Shells:**
    *   100g almond flour
    *   100g powdered sugar
    *   75g granulated sugar
    *   2 egg whites (aged for a day or two at room temp, if possible, for better stability)
    *   Pinch of salt (optional)
    *   Food coloring (gel or powder, not liquid)

*   **For the Filling:** (Buttercream, ganache, or jam are popular choices)

---

**Step 1: Prepare your dry ingredients.**
Sift together the almond flour and powdered sugar into a bowl. This step is crucial for achieving smooth macaron shells, as it removes any lumps.


**Step 2: Make the meringue.**
In a separate, clean bowl, beat the egg whites with a pinch of salt (if using) until foamy. Gradually add the granulated sugar, continuing to beat until you achieve stiff, glossy peaks. If you're using food coloring, add it now. The meringue should be firm enough that you can turn the bowl upside down without it falling out.

**Step 3: Combine dry ingredients with meringue (Macaronage).**
Gently fold the sifted almond flour and powdered sugar into the meringue in two or three additions. This is called "macaronage" and is the most critical step. You want to mix until the batter flows like "lava" or a slowly ribboning consistency when you lift your spatula. Be careful not to overmix, or your macarons will be flat; under-mixing will result in lumpy shells.

**Step 4: Pipe the macarons.**
Transfer the batter to a piping bag fitted with a round tip. Pipe uniform circles onto baking sheets lined with parchment paper or silicone mats. Leave some space between each macaron.

**Step 5: Tap and Rest.**
Firmly tap the baking sheets on your counter several times to release any air bubbles. Use a toothpick to pop any remaining bubbles. This helps create smooth tops and the characteristic "feet." Let the piped macarons rest at room temperature for 30-60 minutes, or until a skin forms on top. When you gently touch a shell, it shouldn't feel sticky. This "drying" step is essential for the feet to develop properly.

**Step 6: Bake the macarons.**
Preheat your oven to 300°F (150°C). Bake one tray at a time for 12-15 minutes. The exact time can vary by oven. They are done when they have developed "feet" and don't wobble when gently touched.

**Step 7: Cool and Fill.**
Once baked, let the macaron shells cool completely on the baking sheet before carefully peeling them off. This prevents them from breaking.  Then, match them up by size and pipe or spread your chosen filling onto one shell before sandwiching it with another.

Finally, let them mature in the refrigerator for at least 24 hours. This allows the flavors to meld and the shells to soften to the perfect chewy consistency.

Enjoy your homemade macarons!

-----

## Chat mode (recommended method)

So far you've used unary calls, but Image-out is actually made to work better with chat mode as it's easier to iterate on an image turn after turn.


```
chat = client.chats.create(
    model=MODEL_ID,
)
```


```
message = "create a image of a plastic toy fox figurine in a kid's bedroom, it can have accessories but no weapon" # @param {type:"string"}

response = chat.send_message(message)
display_response(response)
save_image(response, "figurine.png")
```


Here is an image of a plastic toy fox figurine in a kid's bedroom, with accessories: 



    



```
message = "Add a blue planet on the figuring's helmet or hat (add one if needed)" # @param {type:"string"}
response = chat.send_message(message)
display_response(response)
```


    



```
message = 'Move that figurine on a beach' # @param {type:"string"}
response = chat.send_message(message)
display_response(response)
```


    



```
message = 'Now it should be base-jumping from a spaceship with a wingsuit' # @param {type:"string"}
response = chat.send_message(message)
display_response(response)
```


    



```
message = 'Cooking a barbecue with an apron' # @param {type:"string"}
response = chat.send_message(message)
display_response(response)
```


    



```
message = 'What about chilling in a spa?' # @param {type:"string"}
response = chat.send_message(message)
display_response(response)
```


    


You can also control the aspect ratio of the output image in chat mode.

To do so, add an `aspect_ratio` value to the `image_config` as you can see in the cell below.


```
message = "Bring it back to the bedroom" # @param {type:"string"}
response = chat.send_message(
    message,
    config=types.GenerateContentConfig(
        image_config=types.ImageConfig(aspect_ratio="16:9"),
    ),
)
display_response(response)
```


    


## Mix multiple pictures

You can also mix multiple images (up to 3 with nano-banana, 14 with nano-banana-pro, 6 with high fidelity), either because there are multiple characters in your image, or because you want to hightlight a certain product, or set the background.


```
import PIL

text_prompt = "Create a picture of that figurine riding that cat in a fantasy world." # @param {type:"string"}

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        text_prompt,
        PIL.Image.open('cat.png'),
        PIL.Image.open('figurine.png')
    ],
)

display_response(response)
```


What a fun idea! Here’s that brave figurine riding the cat through a fantastical world. 



    


<a name="nano-banana-pro"></a>
## Nano-Banana Pro

Compared to the flash model that you love, the pro version is able to go further in understanding your requests since it's a [**thinking**](#thinking). Sent it your most complex requests and it will be able to fullfill your desires.

It's able to use [**search grounding**](#grounding) to even bertter understand the subjects your are talking about and access to up-to-date informations.

You'll be able to control the [output resolution](#image_size) and generate up to 4K images.

Lastly, it now supports way more languages!

Note that the pro model is more expensive than the flash one, especially when generating 4K images (cf. [pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-2.5-flash-image)).


```
# @title Run this cell to set everything up (especially if you jumped directly to this section)

from google.colab import userdata
from google import genai
from