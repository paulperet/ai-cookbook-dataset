# Asynchronous Requests with the Gemini API

This guide demonstrates how to make asynchronous and parallel requests using the Gemini API's Python SDK and Python's built-in `asyncio` library. You'll learn to efficiently handle multiple API calls, particularly when working with external resources like images.

## Prerequisites

Ensure you have the required packages installed.

```bash
pip install -U 'google-genai>=1.0.0' aiohttp
```

## Setup: API Key and Model Selection

First, configure your API key and select a model. This example assumes your key is stored in an environment variable named `GOOGLE_API_KEY`.

```python
import os
from google import genai

# Set your API key from an environment variable
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

# Select your model
MODEL_ID = "gemini-3-flash-preview"  # Example model; choose one that fits your needs
```

## Step 1: Prepare Your Prompt and Image URLs

Define a simple prompt and a list of image filenames hosted online.

```python
prompt = "Describe this image in just 3 words."

img_filenames = ["firefighter.jpg", "elephants.jpeg", "jetpack.jpg"]
img_dir = "https://storage.googleapis.com/generativeai-downloads/images/"
```

## Step 2: Process Local Images Asynchronously

If you have local files, you can process them one at a time using the async API. This allows the event loop to yield to other tasks while waiting for each API response.

```python
import PIL.Image

async def describe_local_images():
    for img_filename in img_filenames:
        # Open the local image file
        img = PIL.Image.open(img_filename)
        # Make an asynchronous API call
        response = await client.aio.models.generate_content(
            model=MODEL_ID,
            contents=[prompt, img]
        )
        print(response.text)

# To run this function, you would use: asyncio.run(describe_local_images())
```

**Output:**
```
Boy, cat, tree.
Forest elephant family
Jetpack Backpack Concept
```

## Step 3: Download and Process Images in Parallel

A more realistic scenario involves downloading images from URLs and processing them concurrently. This example uses `aiohttp` for asynchronous HTTP requests and chains tasks for maximum efficiency.

### 3.1 Define Helper Functions

Create coroutines to download an image and to process it using the Gemini API.

```python
import io
import aiohttp
import asyncio

async def download_image(session: aiohttp.ClientSession, img_url: str) -> PIL.Image.Image:
    """Downloads an image from a URL and returns a PIL Image object."""
    async with session.get(img_url) as img_resp:
        buffer = io.BytesIO()
        buffer.write(await img_resp.read())
        return PIL.Image.open(buffer)

async def process_image(img_future: asyncio.Future) -> str:
    """
    Summarizes an image using the Gemini API.
    Accepts a Future representing the image download task.
    """
    # Await the future to get the actual image, then call the API
    image = await img_future
    response = await client.aio.models.generate_content(
        model=MODEL_ID,
        contents=[prompt, image]
    )
    return response.text
```

### 3.2 Orchestrate Parallel Execution

The main function creates download tasks and immediately chains API processing tasks to them, then gathers results as they complete.

```python
async def download_and_describe():
    async with aiohttp.ClientSession() as session:
        response_futures = []

        for img_filename in img_filenames:
            # Create a future for downloading the image
            img_url = img_dir + img_filename
            img_future = download_image(session, img_url)

            # Immediately create a task to process the image once it's downloaded
            text_future = asyncio.ensure_future(process_image(img_future))
            response_futures.append(text_future)

        print(f"Download and content generation queued for {len(response_futures)} images.")
        print()

        # Process and print responses in the order they complete
        for future in asyncio.as_completed(response_futures):
            result = await future
            print(result)

# To run: asyncio.run(download_and_describe())
```

**Output:**
```
Download and content generation queued for 3 images.

Wild elephant family.
Jetpack backpack concept
Cat, person, tree.
```

## How It Works

1.  **Task Chaining:** Each image download is represented by a `Future`. The `process_image` coroutine awaits this future, meaning the API call only starts once the image download is complete. This creates an efficient pipeline.
2.  **Concurrent Execution:** By using `asyncio.ensure_future`, we schedule the `process_image` task immediately. All download and processing tasks are managed concurrently by the event loop.
3.  **As-Completed Processing:** `asyncio.as_completed` yields futures as they finish, allowing you to handle results out of their original order, which can improve perceived performance.

## Next Steps

*   Explore the full [`AsyncClient`](https://googleapis.github.io/python-genai/genai.html#genai.client.AsyncClient) class in the Python SDK reference for more advanced asynchronous methods.
*   Deepen your understanding of concurrency by reading the official Python documentation for the [`asyncio`](https://docs.python.org/3/library/asyncio.html) library.