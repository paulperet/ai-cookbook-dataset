# Vision Fine-tuning on GPT-4o for Visual Question Answering

We're excited to announce the launch of [Vision Fine-Tuning on GPT-4o](https://openai.com/index/introducing-vision-to-the-fine-tuning-api/), a cutting-edge multimodal fine-tuning capability that empowers developers to fine-tune GPT-4o using both **images** and **text**. With this new feature, you can customize models to have stronger image understanding capabilities, unlocking possibilities across various industries and applications.

From **advanced visual search** to **improved object detection** for autonomous vehicles or smart cities, vision fine-tuning enables you to craft solutions tailored to your specific needs. By combining text and image inputs, this product is uniquely positioned for tasks like **visual question answering**, where detailed, context-aware answers are derived from analyzing images. In general, this seems to be most effective when the model is presented with questions and images that resemble the training data as we are able to teach the model how to search and identify relevant parts of the image to answer the question correctly. Similarly to fine-tuning on text inputs, vision fine-tuning is not as useful for teaching the model new information.

In this guide, we’ll walk you through the steps to fine-tune GPT-4o with multimodal inputs. Specifically, we’ll demonstrate how to train a model for answering questions related to **images of books**, but the potential applications span countless domains—from **web design** and **education** to **healthcare** and **research**.

Whether you're looking to build smarter defect detection models for manufacturing, enhance complex document processing and diagram understanding, or develop applications with better visual comprehension for a variety of other use cases, this guide will show you just how fast and easy it is to get started.

For more information, check out the full [Documentation](https://platform.openai.com/docs/guides/fine-tuning/vision).


```python
from openai import OpenAI, ChatCompletion
import json
import os

client = OpenAI()
```

### Load Dataset

We will work with a dataset of question-answer pairs on images of books from the [OCR-VQA dataset](https://ocr-vqa.github.io/), accessible through HuggingFace. This dataset contains 207,572 images of books with associated question-answer pairs inquiring about title, author, edition, year and genre of the book. In total, the dataset contains ~1M QA pairs. For the purposes of this guide, we will only use a small subset of the dataset to train, validate and test our model.

We believe that this dataset will be well suited for fine-tuning on multimodal inputs as it requires the model to not only accurately identify relevant bounding boxes to extract key information, but also reason about the content of the image to answer the question correctly.


```python
from datasets import load_dataset

# load dataset
ds = load_dataset("howard-hou/OCR-VQA")
```

We'll begin by sampling 150 training examples, 50 validation examples and 100 test examples. We will also explode the `questions` and `answers` columns to create a single QA pair for each row. Additionally, since our images are stored as byte strings, we'll convert them to images for processing.


```python
import pandas as pd
from io import BytesIO
from PIL import Image

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

Let's inspect a random sample from the training set.

In this example, the question prompts the model to determine the title of the book. In this case, the answer is quite ambiguous as there is the main title "Patty's Patterns - Advanced Series Vol. 1 & 2" as well as the subtitle "100 Full-Page Patterns Value Bundle" which are found in different parts of the image. Also, the name of the author here is not an individual, but a group called "Penny Farthing Graphics" which could be mistaken as part of the title.

This type of task is typical in visual question answering, where the model must interpret complex images and provide accurate, context-specific responses. By training on these kinds of questions, we can enhance the model's ability to perform detailed image analysis across a variety of domains.


```python
from IPython.display import display

# display a random training example
print('QUESTION:', ds_train.iloc[198]['question'])
display(ds_train.iloc[198]['image'])
print('ANSWER:', ds_train.iloc[198]['answer'])

```

    QUESTION: What is the title of this book?



    
    ANSWER: Patty's Patterns - Advanced Series Vol. 1 & 2: 100 Full-Page Patterns Value Bundle


### Data Preparation

To ensure successful fine-tuning of our model, it’s crucial to properly structure the training data. Correctly formatting the data helps avoid validation errors during training and ensures the model can effectively learn from both text and image inputs. The good news is, this process is quite straightforward.

Each example in the training dataset should be a conversation in the same format as the **Chat Completions API**. Specifically, this means structuring the data as a series of **messages**, where each message includes a `role` (such as "user" or "assistant") and the `content` of the message.

Since we are working with both text and images for vision fine-tuning, we’ll construct these messages to include both content types. For each training sample, the question about the image is presented as a user message, and the corresponding answer is provided as an assistant message.

Images can be included in one of two ways:
* As **HTTP URLs**, referencing the location of the image.
* As **data URLs** containing the image encoded in **base64**.

Here’s an example of how the message format should look:


```python
{
    "messages": 
    [
        {
            "role": "system",
            "content": "Use the image to answer the question."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the title of this book?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<encoded_image>"}}
            ]
        }
    ]
}
```

Let's start by defining the **system instructions** for our model. These instructions provide the model with important context, guiding how it should behave when processing the training data. Clear and concise system instructions are particularly useful to make sure the model reasons well on both text and images.


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

To ensure our images are properly formatted for vision fine-tuning, they must be in **base64 format** and either **RGB or RGBA**. This ensures the model can accurately process the images during training. Below is a function that handles the encoding of images, while also converting them to the correct format if necessary.

This function allows us to control the quality of the image encoding, which can be useful if we want to reduce the size of the file. 100 is the highest quality, and 1 is the lowest. The maximum file size for a fine-tuning job is 1GB, but we are unlikely to see improvements with a very large amount of training data. Nevertheless, we can use the `quality` parameter to reduce the size of the file if needed to accomodate file size limits.


```python
import base64

def encode_image(image, quality=100):
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Convert to RGB
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality) 
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
```

We will also include **Few-Shot examples** from the training set as user and assistant messages to help guide the model's reasoning process.



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

Now that we have our system instructions, few-shot examples, and the image encoding function in place, the next step is to iterate through the training set and construct the messages required for fine-tuning. As a reminder, each training example must be formatted as a conversation and must include both the image (in base64 format) and the corresponding question and answer.

To fine-tune GPT-4o, we recommend providing at least **10 examples**, but you’ll typically see noticeable improvements with **50 to 100** training examples. In this case, we'll go all-in and fine-tune the model using our larger training sample of **150 images, and 721 QA pairs**.


```python
from tqdm import tqdm

# constructing the training set
json_data = []

for idx, example in tqdm(ds_train.iterrows()):
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

    [721it, ..., 721it/s]


We save our final training set in a `.jsonl` file where each line in the file represents a single example in the training dataset.


```