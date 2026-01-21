# Working with Charts, Graphs, and Slide Decks Using the Gemini API

Gemini models are powerful multimodal LLMs capable of processing both text and image inputs. This guide demonstrates how to use the Gemini Flash model to extract and interpret data from various visual formats, including charts, graphs, and entire slide decks.

## Prerequisites

Before you begin, ensure you have the following:

1.  A Google AI API key. If you don't have one, follow the [Authentication guide](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb).
2.  This tutorial assumes you are using Google Colab. Your API key should be stored in a Colab Secret named `GOOGLE_API_KEY`.

## Step 1: Install and Import Required Libraries

First, install the necessary Python client and import the required modules.

```bash
pip install -U -q "google-genai>=1.0.0"
```

```python
import os
import time
from glob import glob

from PIL import Image
from IPython.display import Markdown, display
```

## Step 2: Configure the Gemini Client

Initialize the Gemini client using your API key.

```python
from google import genai
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## Step 3: Prepare the Image Dataset

You will use images from [Priyanka Vergadia's GCPSketchnote](https://github.com/priyankavergadia/GCPSketchnote) repository, which contains detailed diagrams of Google Cloud Platform services. These images are licensed under the Creative Commons Attribution 4.0 International Public License.

Clone the repository and load the image file paths.

```bash
git clone https://github.com/priyankavergadia/GCPSketchnote.git
```

```python
images_with_duplicates = glob("/content/GCPSketchnote/images/*")

# Remove duplicate images with different extensions
images = []
encountered = set()
for path in images_with_duplicates:
    path_without_extension, extension = os.path.splitext(path)
    if path_without_extension not in encountered and extension != ".pdf":
        images.append(path)
        encountered.add(path_without_extension)

print(f"Loaded {len(images)} unique images.")
```

## Step 4: Define Helper Functions

Create utility functions to resize images (to reduce data usage) and to query the Gemini model.

```python
from google.genai import types

# Make images fit better on screen and decrease data used for requests
def shrink_image(image: Image, ratio=2):
    width, height = image.size
    return image.convert('RGB').resize((width // ratio, height // ratio))

MODEL_ID = "gemini-3-flash-preview"  # You can change this model as needed

def generate_content_from_image(prompt, image_paths):
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[prompt] + [shrink_image(Image.open(image_path)) for image_path in image_paths],
    )
    return response.text
```

## Step 5: Interpret a Single Chart

Let's test the model's ability to extract specific data from a chart. First, download an example chart.

```bash
chart_path_gif = "chart.gif"
!curl https://storage.googleapis.com/gweb-uniblog-publish-prod/original_images/gemini_final_text_table_amendment_13_12_23.gif > $chart_path_gif
```

The Gemini API currently does not support `.gif` files, so you need to convert the image to `.jpg`.

```python
chart_path_jpg = "chart.jpg"
image = Image.open(chart_path_gif)
image = image.convert('RGB')
image.save(chart_path_jpg)
```

Now, query the model to interpret the chart. You will ask it to compare model performance in a specific benchmark category.

```python
prompt = """You are a tool that interprets tables. Which model (Gemini Ultra or GPT-4) is better in the 'Math' category in MATH benchmark?"""
response = generate_content_from_image(prompt, [chart_path_jpg])
display(Markdown(response))
```

**Expected Output:**
> In the MATH benchmark, Gemini Ultra scores 53.2% and GPT-4 scores 52.9%. Therefore, Gemini Ultra is better in the 'Math' category in the MATH benchmark.

## Step 6: Extract Information from a Single Slide

Next, let's analyze a single slide describing a Google Cloud service. You'll use an image from the cloned repository.

```python
image_path = "/content/GCPSketchnote/images/pubsub.jpg"
```

Start with a simple request to describe the image.

```python
prompt = "Describe the image in 5 sentences."
response = generate_content_from_image(prompt, [image_path])
display(Markdown(response))
```

Now, ask a more targeted question to extract specific information about patterns shown in the image.

```python
prompt = "Explain the different pub/sub patterns using the image. Ignore the rest."
response = generate_content_from_image(prompt, [image_path])
display(Markdown(response))
```

**Expected Output Snippet:**
> Okay, let's break down the Pub/Sub patterns presented in the image...
> 1.  **MANY-TO-ONE (FAN-IN):** Multiple publishers send messages to separate topics...
> 2.  **MANY-TO-MANY:** Multiple publishers send messages to a single topic...
> 3.  **ONE-TO-MANY (FAN-OUT):** A single publisher sends messages to a single topic...

## Step 7: Process an Entire Slide Deck

A key strength of the Gemini Flash model is its ability to process a large number of images in a single request (up to 3,600). This allows you to analyze entire slide decks without splitting them.

You will use the first four images from the dataset and ask the model to generate quiz questions based on their content.

```python
prompt = """
Your job is to create a set of questions to check knowledge of various
GCP products. Write for each image the topic and an example question.
"""

response = generate_content_from_image(prompt, images[:4])
display(Markdown(response))
```

**Expected Output Snippet:**
> Okay, here's a set of questions based on the provided images...
> **Image 1: Compute Options (Compute Engine, Kubernetes Engine, Cloud Run, Cloud Functions)**
> *   **Topic:** Choosing the right compute option
> *   **Example Question:** "Your team needs to migrate an existing application that relies on specific kernel modules..."

## Summary

The Gemini API's ability to process images like charts, graphs, and slide decks showcases the power of multimodal LLMs. By understanding visual elements, you can unlock new insights, automate analysis, and save significant time.

Potential applications are vast, from creating accessible descriptions of visual content for the disabled community to automating the extraction of key points from business presentations. This tutorial provides the foundation—now it's your turn to build upon it and explore further.