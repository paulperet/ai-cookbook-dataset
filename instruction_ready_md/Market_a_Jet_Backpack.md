# Guide: Creating a Marketing Campaign from a Product Sketch with Gemini

This guide demonstrates how to use the Gemini API to analyze a product sketch, extract key features, and generate a complete marketing campaign, including product names, taglines, and a splash page.

## Prerequisites

Before you begin, ensure you have the following:

1.  **Python Environment:** A Python environment (like Google Colab or a local Jupyter notebook).
2.  **Google AI API Key:** A valid API key for Google's Gemini models. Store it securely.
3.  **Required Libraries:** Install the necessary Python packages.

### Step 1: Install and Import Libraries

First, install the `google-genai` client library and import the required modules.

```bash
pip install -U -q "google-genai>=1.0.0"
```

```python
import PIL.Image
import json
from IPython.display import display, Image, HTML
import ipywidgets as widgets
from typing_extensions import TypedDict
from google import genai
from google.genai import types
```

### Step 2: Configure the Gemini Client

Initialize the Gemini client using your API key. This example assumes the key is stored in a Colab Secret named `GOOGLE_API_KEY`. Adjust the method for accessing your key if you are in a different environment.

```python
# For Google Colab:
from google.colab import userdata
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')

# For other environments, you might set it directly:
# GOOGLE_API_KEY = "YOUR_API_KEY_HERE"

client = genai.Client(api_key=GOOGLE_API_KEY)
```

### Step 3: Select a Model

Choose which Gemini model you want to use for this task.

```python
MODEL_ID = "gemini-1.5-flash"  # A good, fast model for this workflow
# Other options: "gemini-1.5-pro", "gemini-2.0-flash-exp"
```

## Part 1: Analyze the Product Sketch

You will start by downloading a sample product sketch (a jetpack backpack) and using Gemini to analyze its features.

### Step 1.1: Download and View the Sketch

Download the sample image from a public URL and display it to understand what you're working with.

```python
# Download the image
productSketchUrl = "https://storage.googleapis.com/generativeai-downloads/images/jetpack.jpg"
!curl -o jetpack.jpg {productSketchUrl}

# Display the image
img = PIL.Image.open('jetpack.jpg')
display(Image('jetpack.jpg', width=300))
```

### Step 1.2: Define the Analysis Prompt

Create a prompt that instructs the model to describe the product and list its features based on the sketch.

```python
analyzePrompt = """
    This image contains a sketch of a potential product along with some notes.
    Given the product sketch, describe the product as thoroughly
    as possible based on what you see in the image, making sure to note
    all of the product features.

    Return output in json format.
"""
```

### Step 1.3: Define the Response Schema

To ensure structured output, define the expected JSON schema using a `TypedDict`. This tells the model the exact format you want.

```python
class Response(TypedDict):
  description: str
  features: list[str]
```

### Step 1.4: Generate the Product Analysis

Call the Gemini API with the prompt, image, and your schema configuration. The `response_mime_type="application/json"` parameter instructs the model to return valid JSON.

```python
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[analyzePrompt, img],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Response)
)

# Parse and print the result
productInfo = json.loads(response.text)
print(json.dumps(productInfo, indent=4))
```

**Expected Output:**
The model will return a JSON object containing a detailed description and a list of features extracted from the sketch.

```json
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
```

> **Note:** This example uses standard prompting for JSON formatting. For production use, consider the **strict JSON mode** which uses constrained decoding to guarantee schema compliance. See the [JSON mode quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/JSON_mode.ipynb) for details.

## Part 2: Generate Marketing Ideas

With the product analysis complete, you can now generate creative marketing assets.

### Step 2.1: Generate Potential Product Names

Create a prompt to generate a list of catchy product names.

```python
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
print("Potential Names:", names)
```

To make it interactive, you can create a dropdown widget to select your favorite name.

```python
dropdown = widgets.Dropdown(
    options=names,
    value=names[0],  # default value
    description='Name:',
    disabled=False,
)
display(dropdown)
```

### Step 2.2: Select a Name

Once you've chosen a name from the dropdown, assign it to a variable for the next steps.

```python
selected_name = dropdown.value
print(f"Selected Name: {selected_name}")
```

## Part 3: Create Website Copy and a Splash Page

Now, use the selected name and product information to generate compelling website headlines (H1 and H2 tags) and a full HTML splash page.

### Step 3.1: Generate Headline Options

Craft a prompt that asks the model to generate multiple headline pairs (H1 and H2) for the product's splash page, providing examples for context.

```python
websiteCopyPrompt = f"""
  You're a marketing whiz and expert copywriter. You're writing
  website copy for a product named {selected_name}. Your first job is to come
  up with H1 H2 copy. These are brief, pithy sentences or phrases that
  are the first and second things the customer sees when they land on the
  splash page. Here are some examples:
  [{{
    "h1": "A feeling is canned",
    "h2": "drinks and powders to help you feel calm cool and collected despite the stressful world around you"
  }},
  {{
    "h1": "Design. Publish. Done.",
    "h2": "Stop rebuilding your designs from scratch. In Framer, everything you put on the canvas is ready to be published to the web."
  }}]

  Create the same json output for a product named "{selected_name}" with description "{productInfo['description']}".
  Output ten different options as json in an array.
"""

class Headings(TypedDict):
  h1: str
  h2: str

copyResponse = client.models.generate_content(
    model=MODEL_ID,
    contents=[websiteCopyPrompt, img],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[Headings])
)

headline_options = json.loads(copyResponse.text)
print(json.dumps(headline_options, indent=4))
```

**Expected Output:**
You will receive a list of 10 headline pairs. For example:

```json
[
    {
        "h1": "SteamPack: Eco-Flight",
        "h2": "The steam-powered backpack that gets you there sustainably."
    },
    {
        "h1": "Your Green Commute, Elevated",
        "h2": "SteamPack: The eco-friendly jetpack backpack."
    }
    // ... more options
]
```

### Step 3.2: Select Headlines and Generate HTML

Choose one of the generated headline pairs (for example, the third one) and use it to prompt the model to create a complete HTML/CSS splash page.

```python
# Select a headline pair (e.g., the third option)
chosen_h1 = headline_options[2]['h1']
chosen_h2 = headline_options[2]['h2']

htmlPrompt = f"""
    Generate HTML and CSS for a splash page for a new product called {selected_name}.
    Output only HTML and CSS and do not link to any external resources.
    Include the top level title: "{chosen_h1}" with the subtitle: "{chosen_h2}".

    Return the HTML directly, do not wrap it in triple-back-ticks (```).
"""

response = client.models.generate_content(
    model=MODEL_ID,
    contents=[htmlPrompt])

# The response contains the raw HTML code
html_code = response.text
print(html_code)
```

### Step 3.3: Render the HTML (Optional)

If you are in an environment that supports it (like a Jupyter notebook), you can render the generated HTML directly to see a preview of your marketing splash page.

```python
# Display the generated HTML page
HTML(html_code)
```

## Summary

You have successfully built a pipeline that:
1.  **Analyzes** a product sketch to extract a description and feature list.
2.  **Generates** creative marketing names and allows for interactive selection.
3.  **Creates** professional headline copy (H1/H2) for a website.
4.  **Builds** a complete, styled HTML splash page for the product campaign.

This workflow demonstrates the power of multimodal AI (processing both images and text) for accelerating creative and marketing tasks. You can adapt this template for any product sketch or concept by simply changing the input image and tweaking the prompts.