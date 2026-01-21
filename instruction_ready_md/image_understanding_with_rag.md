# Image Understanding with RAG using OpenAI's Vision & Responses APIs

Welcome! This guide demonstrates how to build a Retrieval-Augmented Generation (RAG) system using OpenAI’s Vision and Responses APIs. It focuses on multimodal data, combining image and text inputs to analyze customer experiences. The system leverages GPT-5 and integrates image understanding with file search to provide context-aware responses.

Multimodal datasets are increasingly common, particularly in domains like healthcare, where records often contain both visual data (e.g., radiology scans) and accompanying text (e.g., clinical notes). Real-world datasets also tend to be noisy, with incomplete or missing information, making it critical to analyze multiple modalities in tandem.

This tutorial focuses on a customer service use case: evaluating customer feedback that may include photos and written reviews. You’ll learn how to synthetically generate both image and text inputs, use file search for context retrieval, and apply the Evals API to assess how incorporating image understanding impacts overall performance.

## Table of Contents

1.  [Setup & Dependencies](#setup-and-dependencies)
2.  [Example Generations](#example-generations)
3.  [Data Processing](#data-processing)
4.  [Populating Vector Store](#populating-vector-store)
5.  [Retrieval and Filtering](#retrieval-and-filtering)
6.  [Evaluation and Analysis](#evaluation-and-analysis)

## Setup and Dependencies

First, install the required libraries.

```bash
pip install openai evals pandas numpy matplotlib tqdm ipython --upgrade --quiet
```

Now, import the necessary modules and set up your OpenAI client and cache directory.

```python
import base64
from io import BytesIO
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI
from IPython.display import display, Image
from tqdm.notebook import tqdm

cache_dir = Path('.local_cache')
cache_dir.mkdir(parents=True, exist_ok=True)

client = OpenAI()
```

## Example Generations

Generating high-quality training and evaluation data for machine learning tasks can be costly and time-consuming. Synthetic data offers a practical and scalable alternative. In this section, you'll use the OpenAI Image API to generate synthetic images and the Responses API to create synthetic text, enabling efficient prototyping.

### 1. Generate a Synthetic Image

Let's start by generating a synthetic image of a gourmet pasta dish.

```python
prompt = ("Gourmet pasta neatly plated with garnish and sides on a white ceramic plate, "
          "photographed from above on a restaurant table. Soft shadows and vibrant colors.")
cache_path = f".local_cache/{hash(prompt)}.png"

if not os.path.exists(cache_path):
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024"
    )
    
    with open(cache_path, "wb") as f:
        f.write(base64.b64decode(response.data[0].b64_json))
    print(f"Generated and cached: {cache_path}")
else:
    print(f"Loading from cache: {cache_path}")
```

### 2. Generate a Synthetic Text Review

Next, create a function to generate synthetic food delivery reviews with a specified sentiment.

```python
def generate_food_delivery_review(sentiment: str = 'positive') -> str:
    """
    Generate a synthetic food delivery review with the specified sentiment.
    
    Args:
        sentiment: An adjective such as 'positive' or 'negative'.
    
    Returns:
        Generated review text
    """
    prompt = "Write a very concise, realistic customer review for a recent food delivery."
    prompt += f" The review should reflect a {sentiment} experience."
    
    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "minimal"},
        input=[{"role": "user", "content": prompt}]
    )
    return response.output_text

review = generate_food_delivery_review()
print(review)
```

**Example Output:**
```
Order arrived 10 minutes early, food was hot and packaged securely. Tacos were fresh, well-seasoned, and the salsa tasted homemade. Driver was friendly, followed instructions, and left it at the door. Will definitely order again.
```

## Data Processing

In this example, you'll work with a pre-generated synthetic dataset of customer feedback that includes short text snippets, images from customer reviews, and occasionally combined multimodal entries. You can also generate your own synthetic dataset using the examples provided above to tailor the data to your specific use case.

### 1. Download the Dataset

Download the synthetic dataset and associated images.

```bash
mkdir -p .local_cache/images
wget https://raw.githubusercontent.com/robtinn/image_understanding_rag_dataset/main/data/df.csv -O .local_cache/df.csv

wget https://raw.githubusercontent.com/robtinn/image_understanding_rag_dataset/main/data/images/1.png -O .local_cache/images/1.png
wget https://raw.githubusercontent.com/robtinn/image_understanding_rag_dataset/main/data/images/2.png -O .local_cache/images/2.png
wget https://raw.githubusercontent.com/robtinn/image_understanding_rag_dataset/main/data/images/3.png -O .local_cache/images/3.png
wget https://raw.githubusercontent.com/robtinn/image_understanding_rag_dataset/main/data/images/4.png -O .local_cache/images/4.png
wget https://raw.githubusercontent.com/robtinn/image_understanding_rag_dataset/main/data/images/5.png -O .local_cache/images/5.png
wget https://raw.githubusercontent.com/robtinn/image_understanding_rag_dataset/main/data/images/6.png -O .local_cache/images/6.png
wget https://raw.githubusercontent.com/robtinn/image_understanding_rag_dataset/main/data/images/7.png -O .local_cache/images/7.png
```

### 2. Analyze Images for Sentiment

Create helper functions to encode images and analyze their sentiment using the Vision API.

```python
def encode_image(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze_image_sentiment(image_path: str) -> str:
    """Analyze food delivery image and return sentiment analysis."""
    base64_image = encode_image(image_path)
    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "minimal"},
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Analyze this food delivery image. Respond with a brief description and sentiment (positive/negative) in one line."
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        }],
    )
    return response.output_text.strip()
```

### 3. Process the Dataset

Load the dataset, analyze images to extract sentiment, and create a combined `full_sentiment` column.

```python
df = pd.read_csv(".local_cache/df.csv")
cache_dir = Path(".local_cache")

for idx, row in df[~df['image_path'].isna()].iterrows():
    image_path = cache_dir / 'images' / row['image_path']
    sentiment = analyze_image_sentiment(str(image_path))
    df.at[idx, 'full_sentiment'] = f"{row['text']} {sentiment}" if pd.notna(row['text']) else sentiment
    print(f"Processed {row['image_path']}")

df['full_sentiment'] = df['full_sentiment'].fillna(df['text'])

output_path = cache_dir / "df_full_sentiment.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved results to {output_path}")
```

### 4. View the Processed Data

Display the first few rows of the processed DataFrame to verify the results.

```python
pd.set_option('display.max_colwidth', 100)  # Increase from default (50) to view full sentiment
display(df.head())
```

## Populating Vector Store

This example uses OpenAI's built-in vector store and file search capabilities to build a RAG system that can analyze customer experiences from their feedback, which can be both visual and text-based. You will create two vector stores for comparisons: one with image understanding and one without.

### 1. Create Vector Stores

Create two separate vector stores to hold your context data.

```python
text_vector_store = client.vector_stores.create(
    name="food_delivery_reviews_text",
    metadata={
        "purpose": "text_understanding",
        "created_by": "notebook",
        "version": "1.0"
    }
)
text_vector_store_id = text_vector_store.id

text_image_vector_store = client.vector_stores.create(
    name="food_delivery_reviews_text_image",
    metadata={
        "purpose": "text_image_understanding",
        "created_by": "notebook",
        "version": "1.0"
    }
)
text_image_vector_store_id = text_image_vector_store.id

print("Vector Store IDs:")
print(f"  Text:       {text_vector_store_id}")
print(f"  Text+Image: {text_image_vector_store_id}")
```

### 2. Upload Files to Vector Stores

Define a function to upload your processed data to the vector stores and set metadata attributes (like `month`) for filtering.

```python
def upload_files_to_vector_store(vector_store_id, df, column_name="full_sentiment"):
    file_ids = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Uploading context files"):
        if pd.isna(row[column_name]):
            file_stream = BytesIO('No information available.'.encode('utf-8'))
        else:
            file_stream = BytesIO(row[column_name].encode('utf-8'))
        file_stream.name = f"context_{row.get('id', i)}_{row.get('month', '')}.txt"
        
        file = client.vector_stores.files.upload(
            vector_store_id=vector_store_id,
            file=file_stream
        )
        file_ids.append(file.id)

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Updating file attributes"):
        client.vector_stores.files.update(
            vector_store_id=vector_store_id,
            file_id=file_ids[i],
            attributes={"month": row["month"]}
        )
```

Now, upload the data to both vector stores.

```python
upload_files_to_vector_store(text_image_vector_store_id, df)
upload_files_to_vector_store(text_vector_store_id, df, column_name="text")
```

## Retrieval and Filtering

You can now analyze your dataset with natural language queries using File Search. For the text-only dataset, you'll see that information is missing that could inform your analysis. The only positive review for spaghetti in July has visual feedback. The RAG system with only text-based context is uncertain about positive details, but with image context provided, the second RAG system can give a more accurate response.

### 1. Query the Text-Only Vector Store

First, query the text-only vector store for comments about 'spaghetti' in July.

```python
query = "Where there any comments about the 'spaghetti'?"
print(f"🔍 Query: {query}\n")

response = client.responses.create(
    model="gpt-5",
    input=query,
    tools=[{
        "type": "file_search",
        "vector_store_ids": [text_vector_store_id],
        "filters": {
            "type": "eq",
            "key": "month",
            "value": "july"
        }
    }]
)

print("📝 Response:")
print("-" * 40)
print(response.output_text)
```

**Example Output:**
```
🔍 Query: Where there any comments about the 'spaghetti'?

📝 Response:
----------------------------------------
I couldn’t find any comments that explicitly mention “spaghetti.” The closest related note says “Pasta was overcooked” in context_9_july.txt . If you have a specific date or file in mind, I can check that directly.
```

### 2. Query the Text+Image Vector Store

Now, perform the same query on the vector store that includes image context.

```python
query = "Where there any comments about the 'spaghetti'?"
print(f"🔍 Query: {query}\n")

response = client.responses.create(
    model="gpt-5",
    input=query,
    tools=[{
        "type": "file_search",
        "vector_store_ids": [text_image_vector_store_id],
        "filters": {
            "type": "eq",
            "key": "month",
            "value": "july"
        }
    }]
)

print("📝 Response:")
print("-" * 40)
print(response.output_text)
```

**Example Output:**
```
🔍 Query: Where there any comments about the 'spaghetti'?

📝 Response:
----------------------------------------
Yes. There’s a positive note describing “a neatly plated spaghetti in tomato sauce with parsley, served alongside arugula, garlic bread, and grated cheese.”
```

### 3. Display Retrieved Images (Optional)

To verify the results, you can display the images retrieved by the search. First, create a mapping from context filenames to image paths.

```python
IMAGE_ID_MAPPING = {
    f"context_{row['id']}_{row['month']}.txt": row["image_path"]
    for _, row in df[~df['image_path'].isna()].iterrows()
}

def display_retrieved_images(response, cache_dir: str = ".local_cache"):
    """
    Display images from the retrieved search results.
    
    Args:
        response: The response object from the search query
        cache_dir: Directory where images are stored
        
    Returns:
        Dict mapping filenames to image paths for the displayed images
    """
    # Get the annotations from the response
    try:
        annotations = response.output[3].content[0].annotations
        retrieved_files = {result.filename for result in annotations}
    except (AttributeError, IndexError):
        print("No search results found in the response.")
        return {}

    # Display matching images
    displayed_images = {}
    for file in retrieved_files:
        if file in IMAGE_ID_MAPPING and IMAGE_ID_MAPPING[file]:
            image_path = Path(cache_dir) / 'images' / IMAGE_ID_MAPPING[file]
            print(f"Displaying image for {file}:")
            displayed_images[file] = str(image_path)
    
    return displayed_images

displayed = display_retrieved_images(response)
print(f"Displayed {len(displayed)} images")
```

### 4. Query for Negative Pizza Reviews

Let's test another query: look for negative reviews about pizza in June and check if the pizza was burnt.

```python
query = "Were there any negative reviews for pizza, and if so, was the pizza burnt?"
print(f"🔍 Query: {query}\n")

response = client.responses.create(
    model="gpt-5",
    input=query,
    tools=[{
        "type": "file_search",
        "vector_store_ids": [text_image_vector_store_id],
        "filters": {
            "type": "eq",
            "key": "month",
            "value": "june"
        }
    }]
)

print("📝 Response:")
print("-" * 40)
print(response.output_text)
```

**Example Output:**
```
🔍 Query: Were there any negative reviews for pizza, and if so, was the pizza burnt?

📝 Response:
----------------------------------------
Yes. One review explicitly describes a “burnt pepperoni pizza with charred crust and grease stains in the box” and is marked as negative sentiment .
```

You can again use the `display_retrieved_images` function to see the corresponding image.

## Evaluation and Analysis

As your dataset evolves and you want to evaluate new models, you can use the OpenAI Evaluation API to assess the performance of your system for sentiment analysis. In this simple example, using the `string_check` criteria, you'll check if the model's output matches one of the three possible ground truth values: `positive`, `negative`, or `unclear`.

### 1. Prepare Evaluation Data

Create a function to format your DataFrame into the structure required for the Evals API.

```python
def prepare_evaluation_data(
    df: pd.DataFrame,
    text_col: str = "full_sentiment",
    label_col: str = "label"
) -> list:
    """
    Prepare evaluation data items from a DataFrame.
    
    Args:
        df: Input pandas DataFrame.
        text_col: Column containing the input text.
        label_col: Column containing the ground truth label.
    
    Returns:
        List of dicts formatted for evaluation.
    """
    return [
        {"item": {"input": str(row[text_col]), "ground_truth": row[label_col]}}
        for _, row in df.iterrows()
    ]
```

### 2. Create an Evaluation Object

First, create an evaluation object that defines the task and the success criteria.

```python
eval_obj = client.evals.create(
    name="food-categorization-eval",
    data_source_config={
        "type": "custom",
        "item_schema": {
            "type": "object",
            "properties": {
                "input": {"type": "string"},
                "ground_truth": {"type": "string"}
            },
            "required": ["input", "ground_truth"]
        },
        "include_sample_schema": True
    },
    testing_criteria=[
        {
            "type": "string_check",
            "name": "Match output to human label",
            "input": "{{sample.output_text}}",
            "reference": "{{item.ground_truth}}",
            "operation": "eq"
        }
    ]
)
eval_id = eval_obj.id
print(f"Created evaluation with ID: {eval_id}")
```

### 3. Create and Launch Evaluation Runs

Define a function to create and launch an evaluation run, then run it for both your text-only and text+image datasets.

```python
def create_eval_run(evaluation_data: list, eval_id: str) -> str:
    """
    Create and launch an evaluation run.
    
    Args:
        evaluation_data: List of evaluation items.
        eval_id: The evaluation object ID.
    
    Returns:
        The run ID as a string.
    """
    eval_config = {
        "type": "completions",
        "model": "gpt-5",
        "input_messages": {
            "type": "template",
            "template": [
                {
                    "type": "message",
                    "role": "user",
                    "content": {
                        "type": "input_text",
                        "text":