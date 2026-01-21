# Multimodal RAG with CLIP Embeddings and GPT-4 Vision: A Step-by-Step Guide

This guide walks you through building a Multimodal Retrieval-Augmented Generation (RAG) system. Unlike traditional text-based RAG, this system integrates visual data by directly embedding images using the CLIP model. This approach bypasses the lossy process of generating text captions, leading to more accurate retrieval. We'll demonstrate this by searching a knowledge base of technology images to answer user queries about an uploaded image.

## Prerequisites & Setup

Before we begin, ensure you have the necessary Python packages installed.

```bash
pip install clip torch pillow faiss-cpu numpy openai tqdm
pip install git+https://github.com/openai/CLIP.git
```

Now, let's import the required libraries.

```python
# Model and Core Libraries
import faiss
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from openai import OpenAI

# Helper Libraries
from tqdm import tqdm
import os
import numpy as np
import pickle
from typing import List, Union, Tuple

# Image Processing and Visualization
from PIL import Image
import matplotlib.pyplot as plt
import base64

# Initialize the OpenAI client
client = OpenAI()
```

## Step 1: Load the CLIP Model

We start by loading the pre-trained CLIP model and its associated image preprocessor. This model will be used to generate embeddings for our images.

```python
# Load the CLIP model. Use "cuda" if you have a GPU available.
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
```

## Step 2: Create the Image Embedding Database

Our knowledge base will be a collection of image embeddings. We'll create these from a directory of JPEG images.

### 2.1 Gather Image Paths

First, we define a helper function to collect all `.jpeg` file paths from a specified directory.

```python
def get_image_paths(directory: str, number: int = None) -> List[str]:
    """
    Retrieves paths to all .jpeg files in a directory.
    Args:
        directory: Path to the folder containing images.
        number: If specified, returns only the first 'number' of images.
    Returns:
        A list of full image file paths.
    """
    image_paths = []
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.jpeg'):
            full_path = os.path.join(directory, filename)
            image_paths.append(full_path)
            # Early return if a specific number is requested
            if number is not None and count == number:
                return [image_paths[-1]]
            count += 1
    return image_paths

# Define the path to your image database
image_directory = 'image_database/'
image_paths = get_image_paths(image_directory)
```

### 2.2 Generate Image Embeddings

Next, we create a function that uses the CLIP model to generate an embedding vector for each image. The `preprocess` function handles resizing, normalization, and other transformations required by the model.

```python
def get_features_from_image_path(image_paths: List[str]) -> torch.Tensor:
    """
    Generates CLIP embeddings for a list of image file paths.
    Args:
        image_paths: List of paths to image files.
    Returns:
        A tensor containing the embedding vectors for all images.
    """
    # Preprocess each image
    images = [preprocess(Image.open(path).convert("RGB")) for path in image_paths]
    # Stack images into a single batch tensor
    image_input = torch.tensor(np.stack(images))
    # Generate embeddings (no gradient calculation needed)
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
    return image_features

# Generate embeddings for all images in our database
image_features = get_features_from_image_path(image_paths)
```

### 2.3 Build the Vector Index

We use FAISS, a library for efficient similarity search, to create an index of our image embeddings. We'll use the inner product (`IndexFlatIP`) as our similarity metric.

```python
# Initialize a FAISS index for inner product similarity
index = faiss.IndexFlatIP(image_features.shape[1])
# Add the embeddings to the index
index.add(image_features)
print(f"Index built with {index.ntotal} vectors.")
```

### 2.4 Load Image Descriptions

For the RAG context, we need textual descriptions for each image. We assume these are stored in a `description.json` file, where each line is a JSON object with `image_path` and `description` keys.

```python
# Load the description data
data = []
with open('description.json', 'r') as file:
    for line in file:
        data.append(json.loads(line))

def find_entry(data: List[dict], key: str, value: str) -> Union[dict, None]:
    """
    Searches a list of dictionaries for an entry with a specific key-value pair.
    """
    for entry in data:
        if entry.get(key) == value:
            return entry
    return None
```

## Step 3: Prepare a User Query Image

For this tutorial, we'll use a sample image as our user's query. This image (`train1.jpeg`) is of a "DELTA Pro Ultra Whole House Battery Generator," a piece of technology unveiled at CES 2024.

```python
user_image_path = 'train1.jpeg'
# (Optional) Display the query image
# im = Image.open(user_image_path)
# plt.imshow(im)
# plt.show()
```

## Step 4: Query GPT-4 Vision (Baseline)

Before using our RAG system, let's see how GPT-4 Vision performs on the query image without any additional context. This highlights the model's limitations when faced with unfamiliar objects.

First, we create helper functions to encode the image and query the vision model.

```python
def encode_image(image_path: str) -> str:
    """Encodes an image file to a base64 string."""
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode('utf-8')

def image_query(query: str, image_path: str) -> str:
    """
    Sends a query and an image to the GPT-4 Vision model.
    Args:
        query: The text question or instruction.
        image_path: Path to the image file.
    Returns:
        The model's text response.
    """
    response = client.chat.completions.create(
        model='gpt-4-vision-preview',
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                        },
                    }
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

# Ask GPT-4V to label the image
baseline_label = image_query('Write a short label of what is shown in this image?', user_image_path)
print(f"GPT-4V Baseline Label: {baseline_label}")
```

**Expected Output:**
```
GPT-4V Baseline Label: Autonomous Delivery Robot
```

The model misidentifies the battery generator as a robot because this specific product was not in its training data. This demonstrates the need for grounding the model with relevant context from our knowledge base.

## Step 5: Perform Semantic Image Search

Now, we'll use our CLIP-powered vector database to find the most similar images to the user's query.

### 5.1 Get Embedding for the Query Image

```python
# Generate an embedding for the user's image
query_embedding = get_features_from_image_path([user_image_path])
```

### 5.2 Search the FAISS Index

We search for the top 2 most similar images in our database. FAISS returns the indices of these images and their similarity scores (distances).

```python
# Search for the 2 nearest neighbors
k = 2
distances, indices = index.search(query_embedding.reshape(1, -1), k)

# Process results: pair indices with distances and sort by similarity (higher distance = more similar for inner product)
indices = indices[0]
distances = distances[0]
results = list(zip(indices, distances))
results.sort(key=lambda x: x[1], reverse=True)  # Sort descending by similarity score

print("Search Results (Index, Similarity Score):")
for idx, score in results:
    print(f"  - Index: {idx}, Score: {score:.4f}")
```

### 5.3 Retrieve the Most Similar Image

We use the top result's index to get the file path and its associated description from our JSON data.

```python
# Get the path of the most similar image
most_similar_idx = results[0][0]
most_similar_path = get_image_paths(image_directory, most_similar_idx)[0]

# Find the description for this image
description_entry = find_entry(data, 'image_path', most_similar_path)

if description_entry:
    print(f"Retrieved description for: {most_similar_path}")
else:
    print("Warning: No description found for the retrieved image.")
```

## Step 6: Answer a User Query with RAG

Finally, we combine the retrieved image and its description to answer a specific user question. We construct a prompt that includes both the user's query and the relevant context.

```python
# Define the user's question
user_question = 'What is the capacity of this item?'

# Construct the RAG prompt
prompt = f"""
Below is a user query. Please answer the query using the provided description and image as context.

User Query:
{user_question}

Contextual Description:
{description_entry['description']}
"""

# Query GPT-4 Vision with the prompt and the retrieved image
final_answer = image_query(prompt, most_similar_path)
print("\n--- Final Answer ---")
print(final_answer)
```

**Expected Output:**
```
--- Final Answer ---
The portable home battery DELTA Pro has a base capacity of 3.6kWh. This capacity can be expanded up to 25kWh with additional batteries. The image showcases the DELTA Pro, which has an impressive 3600W power capacity for AC output as well.
```

Success! By first finding the most visually similar image in our knowledge base and then providing its description as context, GPT-4 Vision can accurately answer a detailed question about an unfamiliar product.

## Conclusion

In this guide, you've built a functional Multimodal RAG pipeline:

1.  **Created a Visual Knowledge Base:** You used the CLIP model to generate embeddings for a directory of images and stored them in a FAISS index for efficient similarity search.
2.  **Performed Contextual Retrieval:** For a user-uploaded image, you found the most visually similar image in your database and retrieved its associated text description.
3.  **Generated Grounded Answers:** You provided both the retrieved image and description to GPT-4 Vision, enabling it to answer specific questions accurately, even about objects not in its original training data.

**Potential Enhancements:**
*   **Fine-tuning CLIP:** Improve embedding quality by fine-tuning CLIP on your specific domain of images.
*   **Advanced Retrieval:** Implement more sophisticated search strategies, such as hybrid search combining text and image embeddings or re-ranking results.
*   **Prompt Engineering:** Optimize the prompt template to better integrate the visual and textual context for more precise answers.

This pattern is powerful and can be adapted to various domains like e-commerce, medical imaging, or industrial inspection, wherever visual data is key to understanding.