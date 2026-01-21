# Multi-Modal LLM Guide: Image Reasoning with Mistral Pixtral-12B

This guide demonstrates how to use the MistralAI MultiModal LLM abstraction for image understanding and reasoning with the `pixtral-12b-2409` model. You will learn to perform synchronous and asynchronous completions, stream responses, and analyze both remote and local images.

## Prerequisites

First, install the required packages and set up your environment.

```bash
pip install llama-index-multi-modal-llms-mistralai matplotlib
```

```python
import os
from IPython.display import Markdown, display

# Replace with your actual MistralAI API key
os.environ["MISTRAL_API_KEY"] = "<YOUR API KEY>"
```

## Step 1: Initialize the Multi-Modal LLM

Import and instantiate the `MistralAIMultiModal` client, specifying the model and maximum new tokens.

```python
from llama_index.multi_modal_llms.mistralai import MistralAIMultiModal

mistralai_mm_llm = MistralAIMultiModal(
    model="pixtral-12b-2409",
    max_new_tokens=300
)
```

## Step 2: Load and Inspect Images from URLs

We'll start by loading two images from public URLs and displaying them for context.

```python
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls

image_urls = [
    "https://tripfixers.com/wp-content/uploads/2019/11/eiffel-tower-with-snow.jpeg",
    "https://cdn.statcdn.com/Infographic/images/normal/30322.jpeg",
]

image_documents = load_image_urls(image_urls)
```

To verify the images, you can inspect them using `matplotlib`.

```python
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

# Display the first image
img_response = requests.get(image_urls[0], headers=headers)
img = Image.open(BytesIO(img_response.content))
plt.imshow(img)
plt.show()

# Display the second image
img_response = requests.get(image_urls[1], headers=headers)
img = Image.open(BytesIO(img_response.content))
plt.imshow(img)
plt.show()
```

## Step 3: Perform a Synchronous Completion

Now, use the `complete` method to get a description of the images.

```python
complete_response = mistralai_mm_llm.complete(
    prompt="Describe the images as an alternative text in a few words",
    image_documents=image_documents,
)

display(Markdown(f"{complete_response}"))
```

**Example Output:**
> The image consists of two distinct parts. The first part is a photograph of the Eiffel Tower in Paris, France, covered in snow... The second part is an infographic titled "France's Social Divide."...

## Step 4: Stream a Completion

For real-time token generation, use the `stream_complete` method.

```python
stream_complete_response = mistralai_mm_llm.stream_complete(
    prompt="give me more context for this images in a few words",
    image_documents=image_documents,
)

for r in stream_complete_response:
    print(r.delta, end="")
```

## Step 5: Perform Asynchronous Completions

The library also supports asynchronous operations for non-blocking calls.

### Async Complete

```python
response_acomplete = await mistralai_mm_llm.acomplete(
    prompt="Describe the images as an alternative text in a few words",
    image_documents=image_documents,
)

display(Markdown(f"{response_acomplete}"))
```

### Async Stream Complete

```python
response_astream_complete = await mistralai_mm_llm.astream_complete(
    prompt="Describe the images as an alternative text in a few words",
    image_documents=image_documents,
)

async for delta in response_astream_complete:
    print(delta.delta, end="")
```

## Step 6: Compare Two Images

Load two different images and ask the model to compare them.

```python
image_urls_compare = [
    "https://tripfixers.com/wp-content/uploads/2019/11/eiffel-tower-with-snow.jpeg",
    "https://assets.visitorscoverage.com/production/wp-content/uploads/2024/04/AdobeStock_626542468-min-1024x683.jpeg",
]

image_documents_compare = load_image_urls(image_urls_compare)

response_multi = mistralai_mm_llm.complete(
    prompt="What are the differences between two images?",
    image_documents=image_documents_compare,
)

display(Markdown(f"{response_multi}"))
```

**Example Output:**
> The first image shows the Eiffel Tower in Paris, France, covered in snow... while the second image shows a tennis court with a large crowd of people watching a tennis match...

## Step 7: Process a Local Image File

You can also analyze images stored locally on your machine.

First, download an example image (a receipt).

```bash
wget 'https://www.boredpanda.com/blog/wp-content/uploads/2022/11/interesting-receipts-102-6364c8d181c6a__700.jpg' -O 'receipt.jpg'
```

Load the local image using `SimpleDirectoryReader`.

```python
from llama_index.core import SimpleDirectoryReader

image_documents_local = SimpleDirectoryReader(
    input_files=["./receipt.jpg"]
).load_data()

response_local = mistralai_mm_llm.complete(
    prompt="Transcribe the text in the image",
    image_documents=image_documents_local,
)

display(Markdown(f"{response_local}"))
```

**Example Output:**
> Dine-in  
> Cashier: Raul  
> 02-Apr-2022 5:01:56P  
> 1 EMPANADA - BEEF $3.00  
> ...

## Summary

You have successfully used the MistralAI MultiModal LLM to:
1. Initialize the client with the Pixtral-12B model.
2. Load and inspect images from URLs.
3. Perform synchronous and asynchronous completions.
4. Stream responses token-by-token.
5. Compare multiple images.
6. Process and transcribe text from a local image file.

This workflow enables powerful multi-modal reasoning, combining visual understanding with natural language generation for diverse applications.