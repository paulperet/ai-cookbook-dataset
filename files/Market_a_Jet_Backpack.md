

# Create a marketing campaign from a product sketch of a Jet Backpack

This notebook contains a code example of using the Gemini API to analyze a a product sketch (in this case, a drawing of a Jet Backpack), create a marketing campaign for it, and output taglines in JSON format.

## Setup


```
%pip install -U -q "google-genai>=1.0.0"
```


```
import PIL.Image
from IPython.display import display, Image, HTML
import ipywidgets as widgets
```

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see the [Authentication](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) quickstart for an example.


```
from google import genai
from google.colab import userdata

GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

Additionally, select the model you want to use from the available options below:


```
MODEL_ID = "gemini-3-flash-preview" # @param ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-preview", "gemini-3-pro-preview"] {"allow-input":true, isTemplate: true}
```

## Marketing Campaign
- Product Name
- Description
- Feature List / Descriptions
- H1
- H2


## Analyze Product Sketch

First you will download a sample image to be used:


```
productSketchUrl = "https://storage.googleapis.com/generativeai-downloads/images/jetpack.jpg"
!curl -o jetpack.jpg {productSketchUrl}
```

[First Entry, ..., Last Entry]

You can view the sample image to understand the prompts you are going to work with:


```
img = PIL.Image.open('jetpack.jpg')
display(Image('jetpack.jpg', width=300))
```

Now define a prompt to analyze the sample image:


```
analyzePrompt = """
    This image contains a sketch of a potential product along with some notes.
    Given the product sketch, describe the product as thoroughly
    as possible based on what you see in the image, making sure to note
    all of the product features.

    Return output in json format.
"""
```

- Set the model to return JSON by setting `response_mime_type="application/json"`.
- Describe the schema for the response using a `TypedDict`.


```
from typing_extensions import TypedDict

class Response(TypedDict):
  description: str
  features: list[str]
```


```
from google.genai import types

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[analyzePrompt, img],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Response)
)
```


```
import json

productInfo = json.loads(response.text)

print(json.dumps(productInfo, indent=4))
```

    {
        "description": "The image is a sketch of a product called the Jetpack Backpack. It appears to be a normal looking, lightweight backpack with padded strap supports that fits an 18\" laptop. The backpack has retractable boosters and has USB-C charging. It has a 15 minute battery life and is steam-powered, making it a green/clean energy source.",
        "features": [
            "Fits 18\" laptop",
            "Lightweight",
            "Padded strap support",
            "Retractable boosters",
            "USB-C charging",
            "15-min battery life",
            "Steam-powered"
        ]
    }


> Note: Here the model is just following text instructions for how the output json should be formatted. The API also supports a **strict JSON mode** where you specify a schema, and the API uses "Controlled Generation" (aka "Constrained Decoding") to ensure the model follows the schema, see the [JSON mode quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/JSON_mode.ipynb) for details.

## Generate marketing ideas

Now using the image you can use Gemini API to generate marketing names ideas:


```
namePrompt = """
    You are a marketing whiz and writer trying to come up
    with a name for the product shown in the image.
    Come up with ten varied, interesting possible names.
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[namePrompt, img],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[str])
)

names = json.loads(response.text)
# Create a Dropdown widget to choose a name from the
# returned possible names
dropdown = widgets.Dropdown(
    options=names,
    value=names[0],  # default value
    description='Name:',
    disabled=False,
)
display(dropdown)
```

Finally you can work on generating a page for your product campaign:


```
name = dropdown.value
```


```
websiteCopyPrompt = f"""
  You're a marketing whiz and expert copywriter. You're writing
  website copy for a product named {name}. Your first job is to come
  up with H1 H2 copy. These are brief, pithy sentences or phrases that
  are the first and second things the customer sees when they land on the
  splash page. Here are some examples:
  [{{
    "h1": "A feeling is canned",
    "h2": "drinks and powders to help you feel calm cool and collected\
    despite the stressful world around you"
  }},
  {{
    "h1": "Design. Publish. Done.",
    "h2": "Stop rebuilding your designs from scratch. In Framer, everything\
    you put on the canvas is ready to be published to the web."
  }}]

  Create the same json output for a product named "{name}" with description\
  "{productInfo['description']}".
  Output ten different options as json in an array.
"""
```


```
class Headings(TypedDict):
  h1:str
  h2:str
```


```
copyResponse = client.models.generate_content(
    model=MODEL_ID,
    contents=[websiteCopyPrompt, img],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[Headings])
)
```


```
copy = json.loads(copyResponse.text)
```


```
print(json.dumps(copy, indent=4))
```

    [
        {
            "h1": "SteamPack: Eco-Flight",
            "h2": "The steam-powered backpack that gets you there sustainably."
        },
        {
            "h1": "Your Green Commute, Elevated",
            "h2": "SteamPack: The eco-friendly jetpack backpack."
        },
        {
            "h1": "Fly Green, Travel Light",
            "h2": "Introducing SteamPack: The lightweight, steam-powered jetpack backpack."
        },
        {
            "h1": "SteamPack: Eco-Friendly Flight",
            "h2": "Soar above the traffic with this sustainable jetpack backpack."
        },
        {
            "h1": "The Future of Green Travel is Here",
            "h2": "SteamPack: Your lightweight, steam-powered solution for short hops."
        },
        {
            "h1": "Upgrade Your Commute",
            "h2": "SteamPack: The lightweight backpack that turns into a steam-powered jetpack."
        },
        {
            "h1": "Green Power, Personal Flight",
            "h2": "Introducing SteamPack: the backpack that gives you a boost, sustainably."
        },
        {
            "h1": "Beyond Backpacks. Beyond Expectations.",
            "h2": "SteamPack: The steam-powered commute solution."
        },
        {
            "h1": "The Eco-Friendly Way To Fly.",
            "h2": "SteamPack: Your lightweight, steam-powered backpack companion."
        },
        {
            "h1": "SteamPack: Lightweight and Clean",
            "h2": "A sustainable alternative for short distance travel."
        }
    ]



```
h1 = copy[2]['h1']
h2 = copy[2]['h2']
```


```
htmlPrompt = f"""
    Generate HTML and CSS for a splash page for a new product called {name}.
    Output only HTML and CSS and do not link to any external resources.
    Include the top level title: "{h1}" with the subtitle: "{h2}".

    Return the HTML directly, do not wrap it in triple-back-ticks (```).
"""
```


```
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[htmlPrompt])
print(response.text)
```

    ```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SteamPack - Fly Green, Travel Light</title>
        <style>
            body {
                font-family: sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f0f0f0;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                background: linear-gradient(to bottom, #a8edea, #fed6e3);
            }
    
            .splash-container {
                text-align: center;
                padding: 40px;
                background-color: rgba(255, 255, 255, 0.8);
                border-radius: 10px;
                box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.1);
                max-width: 800px;
                width: 90%;
            }
    
            h1 {
                font-size: 3em;
                margin-bottom: 10px;
                color: #333;
            }
    
            h2 {
                font-size: 1.5em;
                color: #666;
                font-weight: normal;
            }
    
            .cta-button {
                display: inline-block;
                padding: 15px 30px;
                background-color: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin-top: 20px;
                font-size: 1.2em;
            }
    
            .cta-button:hover {
                background-color: #3e8e41;
            }
        </style>
    </head>
    <body>
        <div class="splash-container">
            <h1>Fly Green, Travel Light</h1>
            <h2>Introducing SteamPack: The lightweight, steam-powered jetpack backpack.</h2>
            <a href="#" class="cta-button">Learn More</a>
        </div>
    </body>
    </html>
    ```



```
HTML(response.text)
```