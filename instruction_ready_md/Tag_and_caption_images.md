# Guide: Tagging and Searching Clothing Images with the Gemini API

This guide demonstrates how to use the Gemini API's vision and embedding capabilities to create a searchable database of clothing images. You will generate descriptive tags and captions for images, then use text or other images to find similar items.

## Prerequisites

Ensure you have the following installed and configured:

### 1. Install Required Libraries
```bash
pip install -U -q "google-genai>=1.0.0"
```

### 2. Import Modules
```python
from google import genai
from google.genai import types
from PIL import Image as PILImage
import time
import numpy as np
import pandas as pd
from glob import glob
import ast
```

### 3. Configure Your API Key
Store your Gemini API key in an environment variable named `GOOGLE_API_KEY`. Then, initialize the client:

```python
import os

api_key = os.environ.get('GOOGLE_API_KEY')
client = genai.Client(api_key=api_key)
```

## Step 1: Download the Sample Dataset

You'll use a small dataset of clothing images for this tutorial.

```bash
# Download the dataset
!wget https://storage.googleapis.com/generativeai-downloads/data/clothes-dataset.zip

# Extract the files
!unzip -o clothes-dataset.zip
```

Load the image paths into a list:

```python
images = glob("/content/clothes-dataset/*")
images.sort(reverse=True)
```

## Step 2: Create a Helper Function for Vision Tasks

To manage API quotas and ensure consistent calls, create a helper function that processes images with a configurable delay.

```python
MODEL_ID = 'gemini-2.0-flash'  # You can change this to other Gemini models

def generate_text_using_image(prompt, image_path, sleep_time=4):
    """Generate text from an image using Gemini with rate limiting."""
    start = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[PILImage.open(image_path)],
        config=types.GenerateContentConfig(
            system_instruction=prompt
        ),
    )
    end = time.perf_counter()
    duration = end - start
    # Enforce a minimum delay between calls
    time.sleep(sleep_time - duration if duration < sleep_time else 0)
    return response.text
```

## Step 3: Define a Controlled Vocabulary for Tags

Create a predefined list of keywords to standardize tags across all images.

```python
keywords = np.concatenate((
    ["flannel", "shorts", "pants", "dress", "T-shirt", "shirt", "suit"],
    ["women", "men", "boys", "girls"],
    ["casual", "sport", "elegant"],
    ["fall", "winter", "spring", "summer"],
    ["red", "violet", "blue", "green", "yellow", "orange", "black", "white"],
    ["polyester", "cotton", "denim", "silk", "leather", "wool", "fur"]
))
```

## Step 4: Generate Keywords for Each Image

Craft a prompt that instructs the model to extract only relevant tags from the predefined list.

```python
keyword_prompt = f"""
You are an expert in clothing that specializes in tagging images of clothes,
shoes, and accessories.
Your job is to extract all relevant keywords from a photo that will help describe an item.
You are going to see an image, extract only the keywords for the clothing, and try to provide as many keywords as possible.

Allowed keywords: {list(keywords)}

Extract tags only when it is obvious that it describes the main item in the image.
Return the keywords as a list of strings:

example1: ["blue", "shoes", "denim"]
example2: ["sport", "skirt", "cotton", "blue", "red"]
"""

def generate_keywords(image_path):
    return generate_text_using_image(keyword_prompt, image_path)
```

Test the keyword generation on a few images:

```python
from IPython.display import Image, display

for image_path in images[:4]:
    response_text = generate_keywords(image_path)
    display(Image(image_path))
    print(response_text)
```

**Example Output:**
```
["shorts", "denim", "blue"]
["suit", "men", "blue", "elegant"]
["suit", "blue", "black", "men", "elegant"]
["T-shirt", "cotton", "casual", "women", "spring", "summer", "red"]
```

## Step 5: Correct and Deduplicate Keywords

The model might return synonyms or invalid terms. Use embeddings to map generated keywords to the nearest valid term in your predefined list.

### 5.1 Create Embeddings for the Predefined Keywords
```python
EMBEDDINGS_MODEL_ID = "embedding-001"

def embed(text):
    """Generate an embedding vector for the given text."""
    embedding = client.models.embed_content(
        model=EMBEDDINGS_MODEL_ID,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="semantic_similarity"
        )
    )
    return np.array(embedding.embeddings[0].values)

# Create a DataFrame with keywords and their embeddings
keywords_df = pd.DataFrame({'Keywords': keywords})
keywords_df["Embeddings"] = keywords_df['Keywords'].apply(embed)
```

### 5.2 Define a Similarity Function
```python
def cosine_similarity(array_1, array_2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(array_1, array_2) / (np.linalg.norm(array_1) * np.linalg.norm(array_2))
```

### 5.3 Map Generated Keywords to Predefined Ones
```python
def replace_word_with_most_similar(keyword, keywords_df, threshold=0.7):
    """Map a keyword to the most similar predefined term, if above threshold."""
    if keyword in keywords_df["Keywords"].values:
        return keyword
    embedding = embed(keyword)
    similarities = keywords_df['Embeddings'].apply(
        lambda row_embedding: cosine_similarity(embedding, row_embedding)
    )
    most_similar_keyword_index = similarities.idxmax()
    if similarities[most_similar_keyword_index] < threshold:
        return None  # No suitable match found
    return keywords_df.loc[most_similar_keyword_index, "Keywords"]

# Test the mapping
for word in ["purple", "tank top", "everyday"]:
    print(word, "->", replace_word_with_most_similar(word, keywords_df))
```

**Output:**
```
purple -> violet
tank top -> T-shirt
everyday -> casual
```

### 5.4 Apply Mapping to Generated Keyword Lists
```python
def map_generated_keywords_to_predefined(generated_keywords, keywords_df=keywords_df):
    """Convert a list of generated keywords to a set of mapped, valid keywords."""
    output_keywords = set()
    for keyword in generated_keywords:
        mapped_keyword = replace_word_with_most_similar(keyword, keywords_df)
        if mapped_keyword:
            output_keywords.add(mapped_keyword)
    return output_keywords

# Example usage
print(map_generated_keywords_to_predefined(["white", "business", "sport", "women", "polyester"]))
print(map_generated_keywords_to_predefined(["blue", "jeans", "women", "denim", "casual"]))
```

**Output:**
```
{'polyester', 'women', 'white', 'sport'}
{'women', 'blue', 'casual', 'denim'}
```

## Step 6: Generate Descriptive Captions

Create a prompt for generating concise, one-sentence descriptions of each clothing item.

```python
caption_prompt = """
You are an expert in clothing that specializes in describing images of clothes, shoes and accessories.
Your job is to extract information from a photo that will help describe an item.
You are going to see an image, focus only on the piece of clothing, ignore surroundings.
Be specific, but stay concise, the description should only be one sentence long.
Most important aspects are color, type of clothing, material, style and who it is meant for.
If you are not sure about a part of the image, ignore it.
"""

def generate_caption(image_path):
    return generate_text_using_image(caption_prompt, image_path)

# Test caption generation
for image_path in images[8:10]:
    response_text = generate_caption(image_path)
    display(Image(image_path))
    print(response_text)
```

**Example Output:**
```
This is a red, short-sleeved, knee-length women's dress with a colorful floral pattern.
This is a khaki button-up shirt with two chest pockets and long sleeves, designed for men.
```

## Step 7: Build a Searchable Dataset

Now, combine keywords and captions for a set of images and compute their embeddings to enable search.

### 7.1 Generate Metadata for Each Image
```python
def generate_keyword_and_caption(image_path):
    """Generate and process keywords and a caption for a single image."""
    raw_keywords = generate_keywords(image_path)
    try:
        # Convert string representation of list to actual list
        keyword_list = ast.literal_eval(raw_keywords)
        mapped_keywords = map_generated_keywords_to_predefined(keyword_list)
    except SyntaxError:
        # If parsing fails, use an empty set
        mapped_keywords = set()
    caption = generate_caption(image_path)
    return {
        "image_path": image_path,
        "keywords": mapped_keywords,
        "caption": caption
    }

# Process the first 8 images as our search database
described_df = pd.DataFrame([generate_keyword_and_caption(image_path) for image_path in images[:8]])
```

### 7.2 Create Combined Embeddings
Create a single embedding vector for each image by combining its keywords and caption.

```python
def embed_row(row):
    """Create an embedding from the combined keywords and caption of a row."""
    text = ", ".join(row["keywords"]) + ".\n" + row["caption"]
    return embed(text)

described_df["embeddings"] = described_df.apply(embed_row, axis=1)
```

## Step 8: Search Using Natural Language Queries

Create a function to find the most similar image based on a text query.

```python
def find_image_from_text(text):
    """Return the path of the image most similar to the text query."""
    text_embedding = embed(text)
    similarities = described_df['embeddings'].apply(
        lambda row_embedding: cosine_similarity(text_embedding, row_embedding)
    )
    most_fitting_image_index = similarities.idxmax()
    return described_df.loc[most_fitting_image_index, "image_path"]

# Example searches
display(Image(find_image_from_text("A suit for a wedding.")))
display(Image(find_image_from_text("A colorful dress.")))
```

## Step 9: Search Using Example Images

You can also find similar clothing items by providing another image as a query.

```python
def find_image_from_image(query_image_path):
    """Return the path of the image most similar to the query image."""
    # Generate metadata for the query image
    query_metadata = generate_keyword_and_caption(query_image_path)
    # Create an embedding for the query
    query_embedding = embed_row(query_metadata)
    # Compare to the database
    similarities = described_df['embeddings'].apply(
        lambda row_embedding: cosine_similarity(query_embedding, row_embedding)
    )
    most_fitting_image_index = similarities.idxmax()
    return described_df.loc[most_fitting_image_index, "image_path"]

# Test with images from the later part of the dataset (not in the search DB)
query_image = images[8]
display(Image(query_image))
display(Image(find_image_from_image(query_image)))
```

## Summary

You have successfully built a system that:
1. **Generates standardized tags and captions** for clothing images using Gemini's vision capabilities.
2. **Corrects and maps keywords** to a controlled vocabulary using semantic similarity with embeddings.
3. **Creates a searchable database** by computing embeddings for each image's combined metadata.
4. **Enables search** using either natural language queries or example images.

This workflow can be extended to larger catalogs, integrated into e-commerce platforms, or adapted for other visual search applications.