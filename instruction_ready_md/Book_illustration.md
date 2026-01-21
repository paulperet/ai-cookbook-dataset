# Illustrating a Book with Gemini 2.5 Image

In this guide, you will use multiple Gemini features—including long context, multimodality, structured output, the File API, and chat mode—in conjunction with the Gemini 2.5 Image model to illustrate a book. You'll generate character portraits and chapter scenes, all while maintaining a consistent artistic style.

> **Note:** To manage notebook size and potential billing costs, this tutorial limits the number of generated images. You can adjust these limits later for your own experiments.

## Prerequisites & Setup

Before you begin, ensure you have the necessary libraries installed and your API key configured.

### 1. Install the SDK

Run the following command to install the Google Generative AI SDK.

```bash
pip install -U -q "google-genai"
```

### 2. Set Up Your API Key

Your API key must be stored in a Colab Secret named `GOOGLE_API_KEY`. If you need to create one, refer to the [Authentication guide](../quickstarts/Authentication.ipynb).

```python
from google.colab import userdata

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
```

### 3. Initialize the Client

Initialize the Gemini client with your API key.

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=GOOGLE_API_KEY)
```

### 4. Import Helper Libraries

Import libraries for displaying markdown and images.

```python
import json
from PIL import Image
from IPython.display import display, Markdown
```

### 5. Select Models and Set Limits

Choose the models you'll use and set limits for the number of images to generate.

```python
# Select the image and text generation models
IMAGE_MODEL_ID = "gemini-2.5-flash-image"
GEMINI_MODEL_ID = "gemini-2.5-flash"

# Limit the number of images for this tutorial
max_character_images = 3
max_chapter_images = 3
```

---

## Step 1: Obtain and Upload a Book

You'll start by downloading a book from Project Gutenberg and uploading it to Gemini using the File API, which allows the model to easily reference the text.

```python
import requests

# Download "The Wind in the Willows" from Project Gutenberg
url = "https://www.gutenberg.org/cache/epub/289/pg289.txt"
response = requests.get(url)

with open("book.txt", "wb") as file:
    file.write(response.content)

# Upload the file to Gemini
book = client.files.upload(file="book.txt")
```

## Step 2: Start a Chat Session

You will use Gemini's chat mode to maintain conversation history and avoid re-sending the book with each request. You'll also define a structured output format using a Pydantic model to get consistent responses.

```python
from pydantic import BaseModel

# Define the structure for the prompts Gemini will generate
class Prompts(BaseModel):
    name: str
    prompt: str

# Create a new chat session
chat = client.chats.create(
    model=GEMINI_MODEL_ID,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[Prompts],
    ),
)

# Provide the book to the model as context
chat.send_message(
    [
        "Here's a book, to illustrate using Gemini 2.5 Image. Don't say anything for now, instructions will follow.",
        book
    ]
)
```

## Step 3: Define an Artistic Style

You can either specify a style or let Gemini propose one that fits the book's tone.

```python
# Optionally, define your own style. Leave empty to let Gemini choose.
style = ""

if style == "":
    # Ask Gemini to define a fitting art style
    response = chat.send_message("""
        Can you define an art style that would fit the story?
        Just give us the prompt for the art style that will be added to future prompts.
        """)
    style = json.loads(response.text)[0]["prompt"]
else:
    # Inform Gemini of your chosen style
    chat.send_message(f"""
        The art style will be: "{style}".
        Keep that in mind when generating future prompts.
        Keep quiet for now, instructions will follow.
    """)

# Display the chosen style
display(Markdown(f"### Style:"))
print(style)

# Format the style for use in prompts
style = f'Follow this style: "{style}" '
```

**Example Output:**
```
Classic storybook illustration, gentle whimsical realism, soft watercolor and pen-and-ink style, with warm, inviting lighting and rich detail, depicting anthropomorphic animals in the English countryside.
```

Now, define some system instructions to guide the image generation, acting as a "negative prompt."

```python
system_instructions = """
    There must be no text on the image, it should not look like a cover page.
    It should be a full illustration with no borders, titles, nor description.
    Stay family-friendly with uplifting colors.
    Each produced should be a simple image, no panels.
"""
```

## Step 4: Generate Character Portraits

First, ask Gemini to generate detailed descriptions of the main characters, which will serve as prompts for the image model.

```python
# Request character descriptions from Gemini
response = chat.send_message("""
    Can you describe the main characters (only the adults) and
    prepare a prompt describing them with as much detail as possible (use the descriptions from the book)
    so Gemini 2.5 Image can generate images of them? Each prompt should be at least 50 words.
""")

characters = json.loads(response.text)
print(json.dumps(characters, indent=4))
```

**Example Output (truncated):**
```json
[
    {
        "name": "Mole",
        "prompt": "A small, good-hearted anthropomorphic mole with black fur..."
    },
    {
        "name": "Water Rat",
        "prompt": "A sociable, good-natured anthropomorphic Water Rat..."
    },
    ...
]
```

Next, create a new chat session with the Gemini 2.5 Image model to generate portrait images.

```python
# Create a chat session for image generation
image_chat = client.chats.create(
    model=IMAGE_MODEL_ID,
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="9:16"  # Portrait orientation
        )
    )
)

# Provide the style and system instructions to the image model
image_chat.send_message(f"""
    You are going to generate portrait images to illustrate The Wind in the Willows from Kenneth Grahame.
    The style we want you to follow is: {style}
    Also follow those rules: {system_instructions}
""")

# Loop through characters and generate images
for character in characters[:max_character_images]:
    display(Markdown(f"### {character['name']}"))
    display(Markdown(character['prompt']))

    # Request an image for the character
    response = image_chat.send_message(
        f"Create an illustration for {character['name']} following this description: {character['prompt']}"
    )

    # Display the generated image
    for part in response.parts:
        if part.inline_data:
            generated_image = part.as_image()
            generated_image.show()
            break
```

## Step 5: Illustrate the Book's Chapters

Finally, ask Gemini to create prompts for key scenes in each chapter, then generate the corresponding images.

```python
# Request chapter illustration prompts from Gemini
response = chat.send_message("""
    Now, for each chapter of the book, give me a prompt to illustrate what happens in it.
    Be very descriptive, especially of the characters. Be very descriptive and remember to tell their name and to reuse the character prompts if they appear in the images.
    Each character should at least be described with 30 words.
""")

# Limit the number of chapters for this tutorial
chapters = json.loads(response.text)[:max_chapter_images]
print(json.dumps(chapters, indent=4))
```

**Example Output (truncated):**
```json
[
    {
        "name": "Chapter I. The River Bank",
        "prompt": "Classic storybook illustration... Mole sits on a sun-drenched riverbank..."
    },
    ...
]
```

You can now use these chapter prompts with the `image_chat` session (as done for characters) to generate full scene illustrations, following the same style and rules.

## Summary

You have successfully:
1.  Uploaded a book to Gemini using the File API.
2.  Established a chat session for structured, context-aware prompting.
3.  Defined (or generated) a consistent artistic style.
4.  Created detailed character descriptions and generated their portraits.
5.  Generated descriptive prompts for key chapter scenes.

This workflow demonstrates how to combine Gemini's text and image generation capabilities to create a cohesive set of illustrations for a narrative. You can extend this by generating more images, experimenting with different styles, or illustrating other books.