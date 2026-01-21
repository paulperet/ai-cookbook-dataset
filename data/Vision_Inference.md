# Local Inference with Phi-3-Vision: A Step-by-Step Guide

Phi-3-vision-128k-instruct is a powerful multimodal model that combines language understanding with visual perception. This guide will walk you through setting up the model and using it for three practical tasks: image analysis, Optical Character Recognition (OCR), and multi-image comparison.

## Prerequisites

Before you begin, ensure you have **Python 3.10 or higher** installed. You will also need to install the required libraries.

### 1. Install Core Dependencies
Open your terminal and run the following commands to install the necessary packages.

```bash
pip install transformers -U
pip install datasets -U
pip install torch -U
```

### 2. Install Flash Attention (Optional but Recommended)
For optimal performance, especially with CUDA 11.6+, install the flash-attn library.

```bash
pip install flash-attn --no-build-isolation
```

## Step 1: Initial Setup and Model Loading

First, you need to import the required modules and load the Phi-3-Vision model and its processor.

```python
from PIL import Image
import requests
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# Define the model ID
model_id = "microsoft/Phi-3-vision-128k-instruct"

# Initialize the processor and model
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto").cuda()

# Define the special tokens for the chat prompt format
user_prompt = '<|user|>\n'
assistant_prompt = '<|assistant|>\n'
prompt_suffix = "<|end|>\n"
```

**Explanation:**
*   We import libraries for image handling (`PIL`), web requests (`requests`), and the core PyTorch and Transformers components.
*   The `AutoProcessor` handles tokenization and image preprocessing specific to the Phi-3-Vision model.
*   The model is loaded with `torch_dtype="auto"` for efficient memory usage and moved to the GPU with `.cuda()`.
*   The `user_prompt`, `assistant_prompt`, and `prompt_suffix` variables define the special tokens required to format instructions correctly for this model.

## Step 2: Analyze an Image

Let's start by having the model analyze an image and describe its content. We'll use a chart showing NVIDIA's data center revenue.

```python
# 1. Construct the prompt with the image placeholder and your question.
prompt = f"{user_prompt}<|image_1|>\nCould you please introduce this stock to me?{prompt_suffix}{assistant_prompt}"

# 2. Load an image from a URL.
url = "https://g.foolcdn.com/editorial/images/767633/nvidiadatacenterrevenuefy2017tofy2024.png"
image = Image.open(requests.get(url, stream=True).raw)

# 3. Process the inputs (text + image) for the model.
inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

# 4. Generate a response from the model.
generate_ids = model.generate(**inputs,
                              max_new_tokens=1000,
                              eos_token_id=processor.tokenizer.eos_token_id,
                              )

# 5. Extract only the newly generated tokens (excluding the input prompt).
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

# 6. Decode the generated token IDs back into readable text.
response = processor.batch_decode(generate_ids,
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]
print(response)
```

**Expected Output:**
The model will analyze the revenue chart and provide a description of NVIDIA as a company.

```
Certainly! Nvidia Corporation is a global leader in advanced computing and artificial intelligence (AI). The company designs and develops graphics processing units (GPUs), which are specialized hardware accelerators used to process and render images and video. Nvidia's GPUs are widely used in professional visualization, data centers, and gaming. The company also provides software and services to enhance the capabilities of its GPUs. Nvidia's innovative technologies have applications in various industries, including automotive, healthcare, and entertainment. The company's stock is publicly traded and can be found on major stock exchanges.
```

## Step 3: Perform OCR (Text Extraction)

Next, we'll use the model's vision capabilities to read text directly from an image, a task known as Optical Character Recognition (OCR).

```python
# 1. Construct a prompt asking for specific text information.
prompt = f"{user_prompt}<|image_1|>\nHelp me get the title and author information of this book?{prompt_suffix}{assistant_prompt}"

# 2. Load an image of a book cover.
url = "https://marketplace.canva.com/EAFPHUaBrFc/1/0/1003w/canva-black-and-white-modern-alone-story-book-cover-QHBKwQnsgzs.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 3. Process and generate a response (same steps as before).
inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
generate_ids = model.generate(**inputs,
                              max_new_tokens=1000,
                              eos_token_id=processor.tokenizer.eos_token_id,
                              )
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids,
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]
print(response)
```

**Expected Output:**
The model will successfully extract the text from the book cover image.

```
The title of the book is "ALONE" and the author is Morgan Maxwell.
```

## Step 4: Compare Multiple Images

Phi-3-Vision can process and compare multiple images in a single prompt. Let's use it to find differences between two similar images.

```python
# 1. Construct a prompt with two image placeholders.
prompt = f"{user_prompt}<|image_1|>\n<|image_2|>\n What is difference in this two images?{prompt_suffix}{assistant_prompt}"

# 2. Load two different images.
url1 = "https://hinhnen.ibongda.net/upload/wallpaper/doi-bong/2012/11/22/arsenal-wallpaper-free.jpg"
image_1 = Image.open(requests.get(url1, stream=True).raw)

url2 = "https://assets-webp.khelnow.com/d7293de2fa93b29528da214253f1d8d0/news/uploads/2021/07/Arsenal-1024x576.jpg.webp"
image_2 = Image.open(requests.get(url2, stream=True).raw)

# 3. Pass the list of images to the processor.
images = [image_1, image_2]
inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

# 4. Generate and decode the response.
generate_ids = model.generate(**inputs,
                              max_new_tokens=1000,
                              eos_token_id=processor.tokenizer.eos_token_id,
                              )
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(response)
```

**Expected Output:**
The model will provide a comparative analysis of the two scenes.

```
The first image shows a group of soccer players from the Arsenal Football Club posing for a team photo with their trophies, while the second image shows a group of soccer players from the Arsenal Football Club celebrating a victory with a large crowd of fans in the background. The difference between the two images is the context in which the photos were taken, with the first image focusing on the team and their trophies, and the second image capturing a moment of celebration and victory.
```

## Summary

You have successfully set up the Phi-3-vision-128k-instruct model and used it for three distinct multimodal tasks:
1.  **Image Analysis:** Describing the content and context of a financial chart.
2.  **OCR:** Extracting specific text (title and author) from a book cover.
3.  **Multi-Image Comparison:** Identifying and explaining the differences between two related images.

The key pattern for using this model is to structure your prompt with the `<|user|>`, `<|image_n|>`, and `<|assistant|>` tokens correctly, then pass both the text prompt and the image(s) to the processor. You can adapt this pattern for a wide variety of visual question-answering and reasoning tasks.