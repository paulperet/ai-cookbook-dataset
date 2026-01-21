# Multimodal Embeddings for Similarity Search with Hugging Face and FAISS

_Authored by: [Merve Noyan](https://huggingface.co/merve)_

## Overview

Embeddings are dense, semantically meaningful vector representations of data. They enable powerful applications like similarity search, zero-shot classification, and model training. In this tutorial, you'll learn how to create and index multimodal embeddings—for both text and images—using the Hugging Face ecosystem and FAISS. We'll use the CLIP model, which jointly understands text and images, to build a search system that can retrieve similar items using either a text prompt or an image.

## Prerequisites

Ensure you have the necessary libraries installed.

```bash
pip install -q datasets faiss-gpu transformers sentencepiece
```

## Step 1: Import Libraries and Load the Model

We'll use the CLIP model from Hugging Face, which provides separate encoders for text and images.

```python
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
import faiss
import numpy as np

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Load the CLIP model and its processors
model = AutoModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
```

## Step 2: Load the Dataset

We'll use a small captioning dataset, `jmhessel/newyorker_caption_contest`, which contains images and their descriptions.

```python
from datasets import load_dataset

ds = load_dataset("jmhessel/newyorker_caption_contest", "explanation")
dataset = ds["train"]  # We'll work with the training split
```

Let's inspect a sample to understand the data structure.

```python
# Display the first image
print("Image description:", dataset[0]["image_description"])
# The image itself can be viewed in a notebook with: dataset[0]["image"]
```

## Step 3: Generate Text Embeddings

The `datasets` library simplifies embedding generation. We'll map a function over the dataset to create a new column containing text embeddings.

```python
def embed_text(example):
    # Tokenize the description and move to GPU
    inputs = tokenizer([example["image_description"]], truncation=True, return_tensors="pt").to("cuda")
    # Get text features from CLIP
    embeddings = model.get_text_features(**inputs)[0].detach().cpu().numpy()
    return {'embeddings': embeddings}

# Apply the function to create embeddings
ds_with_embeddings = dataset.map(embed_text)
```

## Step 4: Generate Image Embeddings

Similarly, we'll create embeddings for the images.

```python
def embed_image(example):
    # Process the image and move to GPU
    inputs = processor([example["image"]], return_tensors="pt").to("cuda")
    # Get image features from CLIP
    image_embeddings = model.get_image_features(**inputs)[0].detach().cpu().numpy()
    return {'image_embeddings': image_embeddings}

# Apply the function to add image embeddings
ds_with_embeddings = ds_with_embeddings.map(embed_image)
```

## Step 5: Build FAISS Indexes

FAISS enables efficient similarity search. We'll create separate indexes for text and image embeddings.

```python
# Create a FAISS index for text embeddings
ds_with_embeddings.add_faiss_index(column='embeddings')

# Create a FAISS index for image embeddings
ds_with_embeddings.add_faiss_index(column='image_embeddings')
```

## Step 6: Query with Text

Now you can search the dataset using a text prompt. The system will return the most similar items based on text embeddings.

```python
# Define a search prompt
prompt = "a snowy day"

# Generate embedding for the prompt
prompt_embedding = model.get_text_features(
    **tokenizer([prompt], return_tensors="pt", truncation=True).to("cuda")
)[0].detach().cpu().numpy()

# Perform the search
scores, retrieved_examples = ds_with_embeddings.get_nearest_examples(
    'embeddings', prompt_embedding, k=1
)

# Display the result
print("Closest description:", retrieved_examples["image_description"][0])
# The closest image can be displayed with: retrieved_examples["image"][0]
```

## Step 7: Query with an Image

You can also use an image as a query. Let's search for images similar to a picture of a beaver.

```python
import requests

# Load an example image from a URL
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/beaver.png"
image = Image.open(requests.get(url, stream=True).raw)

# Generate embedding for the image
img_embedding = model.get_image_features(
    **processor([image], return_tensors="pt").to("cuda")
)[0].detach().cpu().numpy()

# Search for similar images in the dataset
scores, retrieved_examples = ds_with_embeddings.get_nearest_examples(
    'image_embeddings', img_embedding, k=1
)

# Display the result
print("Closest description:", retrieved_examples["image_description"][0])
# The closest image can be displayed with: retrieved_examples["image"][0]
```

## Step 8: Save and Load the FAISS Indexes

To reuse the indexes later, save them locally.

```python
# Save text embeddings index
ds_with_embeddings.save_faiss_index('embeddings', 'embeddings/embeddings.faiss')

# Save image embeddings index
ds_with_embeddings.save_faiss_index('image_embeddings', 'embeddings/image_embeddings.faiss')
```

### Push to the Hugging Face Hub (Optional)

For collaboration and persistence, you can store the indexes in a dataset repository on the Hugging Face Hub.

```python
from huggingface_hub import HfApi, notebook_login

# Log in to the Hub
notebook_login()

# Create a dataset repository
api = HfApi()
api.create_repo("your-username/faiss_embeddings", repo_type="dataset")

# Upload the saved indexes
api.upload_folder(
    folder_path="./embeddings",
    repo_id="your-username/faiss_embeddings",
    repo_type="dataset",
)
```

### Load Indexes from the Hub

You can download and load the indexes into a new dataset.

```python
from huggingface_hub import snapshot_download

# Download the indexes
snapshot_download(
    repo_id="your-username/faiss_embeddings",
    repo_type="dataset",
    local_dir="downloaded_embeddings"
)

# Load the text embeddings index into a dataset
ds = ds["train"]
ds.load_faiss_index('embeddings', './downloaded_embeddings/embeddings.faiss')

# Perform a search with the loaded index
prompt = "people under the rain"
prompt_embedding = model.get_text_features(
    **tokenizer([prompt], return_tensors="pt", truncation=True).to("cuda")
)[0].detach().cpu().numpy()

scores, retrieved_examples = ds.get_nearest_examples('embeddings', prompt_embedding, k=1)

# Display the result
print("Closest description:", retrieved_examples["image_description"][0])
# The closest image can be displayed with: retrieved_examples["image"][0]
```

## Conclusion

You've successfully built a multimodal similarity search system using CLIP, Hugging Face `datasets`, and FAISS. This system can retrieve relevant images using either text or image queries. The indexes can be saved, shared via the Hugging Face Hub, and reloaded for future use. This foundation can be extended to larger datasets and integrated into applications like e-commerce search or content recommendation systems.