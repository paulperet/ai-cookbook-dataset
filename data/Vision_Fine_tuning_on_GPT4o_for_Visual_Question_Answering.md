# Vision Fine-tuning on GPT-4o for Visual Question Answering

## Introduction

We're excited to announce the launch of [Vision Fine-Tuning on GPT-4o](https://openai.com/index/introducing-vision-to-the-fine-tuning-api/), a cutting-edge multimodal fine-tuning capability that empowers developers to fine-tune GPT-4o using both **images** and **text**. With this new feature, you can customize models to have stronger image understanding capabilities, unlocking possibilities across various industries and applications.

From **advanced visual search** to **improved object detection** for autonomous vehicles or smart cities, vision fine-tuning enables you to craft solutions tailored to your specific needs. By combining text and image inputs, this product is uniquely positioned for tasks like **visual question answering**, where detailed, context-aware answers are derived from analyzing images.

In this guide, we’ll walk you through the steps to fine-tune GPT-4o with multimodal inputs. Specifically, we’ll demonstrate how to train a model for answering questions related to **images of books**, but the potential applications span countless domains—from **web design** and **education** to **healthcare** and **research**.

For more information, check out the full [Documentation](https://platform.openai.com/docs/guides/fine-tuning/vision).

## Prerequisites

First, ensure you have the necessary Python libraries installed. You'll need `openai`, `datasets`, `pandas`, `Pillow`, and `tqdm`.

```bash
pip install openai datasets pandas Pillow tqdm
```

Now, import the required modules and initialize the OpenAI client.

```python
from openai import OpenAI
import json
import os
from datasets import load_dataset
import pandas as pd
from io import BytesIO
from PIL import Image
import base64
from tqdm import tqdm

client = OpenAI()
```

## Step 1: Load and Prepare the Dataset

We will work with a dataset of question-answer pairs on images of books from the [OCR-VQA dataset](https://ocr-vqa.github.io/), accessible through HuggingFace. This dataset contains images of books with associated questions about their title, author, edition, year, and genre.

### 1.1 Load the Dataset

Load the dataset from HuggingFace.

```python
# load dataset
ds = load_dataset("howard-hou/OCR-VQA")
```

### 1.2 Sample and Preprocess the Data

We'll sample a small subset for training, validation, and testing. We'll also convert byte-string images to PIL Image objects and explode the `questions` and `answers` columns to create individual QA pairs.

```python
# sample 150 training examples, 50 validation examples and 100 test examples
ds_train = ds['train'].shuffle(seed=42).select(range(150))
ds_val = ds['validation'].shuffle(seed=42).select(range(50))
ds_test = ds['test'].shuffle(seed=42).select(range(100))

# convert to pandas dataframe
ds_train = ds_train.to_pandas()
ds_val = ds_val.to_pandas()
ds_test = ds_test.to_pandas()

# convert byte strings to images
ds_train['image'] = ds_train['image'].apply(lambda x: Image.open(BytesIO(x['bytes'])))
ds_val['image'] = ds_val['image'].apply(lambda x: Image.open(BytesIO(x['bytes'])))
ds_test['image'] = ds_test['image'].apply(lambda x: Image.open(BytesIO(x['bytes'])))

# explode the 'questions' and 'answers' columns
ds_train = ds_train.explode(['questions', 'answers'])
ds_val = ds_val.explode(['questions', 'answers'])
ds_test = ds_test.explode(['questions', 'answers'])

# rename columns
ds_train = ds_train.rename(columns={'questions': 'question', 'answers': 'answer'})
ds_val = ds_val.rename(columns={'questions': 'question', 'answers': 'answer'})
ds_test = ds_test.rename(columns={'questions': 'question', 'answers': 'answer'})

# create unique ids for each example
ds_train = ds_train.reset_index(drop=True)
ds_val = ds_val.reset_index(drop=True)
ds_test = ds_test.reset_index(drop=True)

# select columns
ds_train = ds_train[['question', 'answer', 'image']]
ds_val = ds_val[['question', 'answer', 'image']]
ds_test = ds_test[['question', 'answer', 'image']]
```

### 1.3 Inspect a Sample

Let's inspect a random sample from the training set to understand the data.

```python
from IPython.display import display

# display a random training example
print('QUESTION:', ds_train.iloc[198]['question'])
display(ds_train.iloc[198]['image'])
print('ANSWER:', ds_train.iloc[198]['answer'])
```

**Output:**
```
QUESTION: What is the title of this book?
ANSWER: Patty's Patterns - Advanced Series Vol. 1 & 2: 100 Full-Page Patterns Value Bundle
```

This example shows a complex question where the model must identify the correct title from multiple text elements in the image.

## Step 2: Define the System Prompt

Clear system instructions guide the model on how to process the training data. Define a detailed prompt that outlines the reasoning steps and output format.

```python
SYSTEM_PROMPT = """
Generate an answer to the question based on the image of the book provided.
Questions will include both open-ended questions and binary "yes/no" questions.
The questions will inquire about the title, author, edition, year and genre of the book in the image.

You will read the question and examine the corresponding image to provide an accurate answer.

# Steps

1. **Read the Question:** Carefully analyze the question to understand what information is being asked.
2. **Examine the Image:**
   - **Identify Relevant Bounding Boxes (if applicable):** For questions requiring specific details like the title or author, focus on the relevant areas or bounding boxes within the image to extract the necessary text. There may be multiple relevant bounding boxes in the image, so be sure to consider all relevant areas.
   - **Analyze the Whole Image:** For questions that need general reasoning (e.g., "Is this book related to Children's Books?"), consider the entire image, including title, graphics, colors, and overall design elements.
3. **Formulate a Reasoned Answer:**
   - For binary questions (yes/no), use evidence from the image to support your answer.
   - For open-ended questions, provide the exact text from the image or a concise phrase that best describes the requested information.

# Output Format

- Provide your answer in a concise and clear manner. Always return the final conclusion only, no additional text or reasoning.
- If the question is binary, answer with "Yes" or "No."
- For open-ended questions requesting specific details (e.g., title, author), return the exact text from the image.
- For questions about general attributes like "genre," return a single word or phrase that best describes it.

# Notes

- Always prioritize accuracy and clarity in your responses.
- If multiple authors are listed, return the first author listed.
- If the information is not present in the image, try to reason about the question using the information you can gather from the image e.g. if the author is not listed, use the title and genre to find the author.
- Ensure reasoning steps logically lead to the conclusions before stating your final answer.

# Examples
You will be provided with examples of questions and corresponding images of book covers, along with the reasoning and conclusion for each example. Use these examples to guide your reasoning process."""
```

## Step 3: Encode Images for Fine-tuning

Images must be encoded in base64 format and be in RGB or RGBA mode. Define a helper function to handle this conversion.

```python
def encode_image(image, quality=100):
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Convert to RGB
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
```

The `quality` parameter (1-100) can be adjusted to control file size, which is useful if you approach the 1GB limit for a fine-tuning job.

## Step 4: Create Few-Shot Examples

Few-shot examples provide the model with a pattern for reasoning. We'll create a list of example messages using samples from the training set.

```python
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "**Example 1:**\n\n**Question:** Who wrote this book?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(ds_train.iloc[286]['image'], quality=50)}"}}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "**Reasoning:** The cover clearly displays two authors' names, 'Evelyn M. Thomson' and 'Orlen N. Johnson,' at the bottom of the cover, with Evelyn M. Thomson listed first. Typically, the first-listed author is considered the primary author or main contributor.\n\n**Conclusion:** Evelyn Thomson"}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "**Example 2:**\n\n**Question:** What is the title of this book?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(ds_train.iloc[22]['image'], quality=50)}"}}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "**Answer:**\n\n**Reasoning:** The cover prominently displays the title across the top and center of the image. The full title reads, 'Computer Systems: An Integrated Approach to Architecture and Operating Systems,' with each component of the title clearly separated and formatted to stand out.\n\n**Conclusion:** Computer Systems: An Integrated Approach to Architecture and Operating Systems"}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "**Example 3:**\n\n**Question:** Is this book related to Children's Books?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(ds_train.iloc[492]['image'], quality=50)}"}}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "**Answer:**\n\n**Reasoning:** The cover illustration features a whimsical mermaid holding a red shoe, with gentle, child-friendly artwork that suggests it is targeted toward a young audience. Additionally, the style and imagery are typical of children's literature.\n\n**Conclusion:** Yes"}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "**Example 4:**\n\n**Question:** Is this book related to History?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(ds_train.iloc[68]['image'], quality=50)}"}}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "**Answer:**\n\n**Reasoning:** The title 'Oliver Wendell Holmes, Jr.: Civil War Soldier, Supreme Court Justice' clearly indicates that this book focuses on the life of Oliver Wendell Holmes, Jr., providing a biographical account rather than a general historical analysis. Although it references historical elements (Civil War, Supreme Court), the primary focus is on the individual rather than historical events as a whole.\n\n**Conclusion:** No"}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "**Example 5:**\n\n**Question:** What is the genre of this book?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(ds_train.iloc[42]['image'], quality=50)}"}}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "**Answer:**\n\n**Reasoning:** The cover prominently features an image of a train station and the title 'Railway Depots, Stations & Terminals,' which directly suggests a focus on railway infrastructure. This points to the book being related to topics within Engineering & Transportation.\n\n**Conclusion:** Engineering & Transportation"}
        ]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "**Example 6:**\n\n**Question:** What type of book is this?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(ds_train.iloc[334]['image'], quality=50)}"}}
        ]
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "**Answer:**\n\n**Reasoning:** The title 'Principles and Practice of Modern Chromatographic Methods' suggests a focus on chromatography, a scientific technique used in chemistry and biology. This aligns with the academic and technical nature typical of books in the 'Science & Math' category.\n\n**Conclusion:** Science & Math"}
        ]
    }
]
```

## Step 5: Construct the Training Dataset

Now, iterate through the training set to construct the final message format required for fine-tuning. Each training example must be a conversation in the Chat Completions API format, including the system prompt, few-shot examples, the user's question with the image, and the assistant's answer.

We recommend at least 10 examples for fine-tuning, with noticeable improvements often seen with 50-100 examples. Here, we'll use our entire training sample of 721 QA pairs.

```python
# constructing the training set
json_data = []

for idx, example in tqdm(ds_train.iterrows(), total=len(ds_train)):
    system_message = {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}]
    }

    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Question [{idx}]: {example['question']}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(example['image'], quality=50)}"}}
        ]
    }

    assistant_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": example["answer"]}]
    }

    all_messages = [system_message] + FEW_SHOT_EXAMPLES + [user_message, assistant_message]

    json_data.append({"messages": all_messages})
```

## Step 6: Save the Training Data

Save the final training set in a `.jsonl` file, where each line represents a single training example.

```python
# Save to a .jsonl file
with open('vision_fine_tuning_data.jsonl', 'w') as f:
    for entry in json_data:
        f.write(json.dumps(entry) + '\n')
```

You have now prepared a multimodal dataset ready for vision fine-tuning with GPT-4o. The next steps would involve uploading this file via the OpenAI API and initiating the fine-tuning job, as detailed in the official [fine-tuning documentation](https://platform.openai.com/docs/guides/fine-tuning).