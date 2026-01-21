# Clothing Matchmaker App: Combining GPT-4o Mini with RAG

This guide demonstrates how to build a clothing recommendation system using OpenAI's GPT-4o mini model for image analysis and a custom Retrieval-Augmented Generation (RAG) pipeline for matching. You will create an application that analyzes an image of a clothing item, extracts its key features, and finds complementary items from a knowledge base.

## Prerequisites & Setup

Ensure you have Python installed. Then, install the required libraries.

```bash
pip install openai tenacity tqdm numpy tiktoken pandas
```

Now, import the necessary modules and set up your OpenAI client.

```python
import pandas as pd
import numpy as np
import json
import ast
import tiktoken
import base64
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm
from tenacity import retry, wait_random_exponential, stop_after_attempt
from IPython.display import HTML, display

# Initialize the OpenAI client
client = OpenAI()

# Model constants
GPT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_COST_PER_1K_TOKENS = 0.00013
```

## Step 1: Prepare the Knowledge Base

You will use a sample clothing dataset. For this tutorial, we'll work with a subset, but the code is designed to scale.

Load the dataset from a CSV file.

```python
styles_filepath = "data/sample_clothes/sample_styles.csv"
styles_df = pd.read_csv(styles_filepath, on_bad_lines='skip')
print(f"Dataset loaded successfully. It contains {len(styles_df)} items.")
print(styles_df.head())
```

## Step 2: Generate Embeddings for the Dataset

To enable semantic search, you need to create vector embeddings for each item's description. We'll implement a batched, parallelized embedding function for efficiency.

First, define the core embedding function with retry logic for reliability.

```python
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(10))
def get_embeddings(input_texts: list):
    """Fetch embeddings from OpenAI's API for a list of text inputs."""
    response = client.embeddings.create(
        input=input_texts,
        model=EMBEDDING_MODEL
    ).data
    return [data.embedding for data in response]
```

Next, create helper functions to batch the data and manage parallel execution.

```python
def batchify(iterable, batch_size=1):
    """Split an iterable into batches of a specified size."""
    length = len(iterable)
    for start_idx in range(0, length, batch_size):
        yield iterable[start_idx: min(start_idx + batch_size, length)]

def embed_corpus(
    corpus: list,
    batch_size=64,
    num_workers=8,
    max_context_len=8191,
):
    """
    Generate embeddings for a corpus of text in parallel batches.
    Calculates and displays token usage and estimated cost.
    """
    # Tokenize and truncate the corpus
    encoding = tiktoken.get_encoding("cl100k_base")
    encoded_corpus = [
        encoded[:max_context_len] for encoded in encoding.encode_batch(corpus)
    ]

    # Calculate statistics
    num_tokens = sum(len(article) for article in encoded_corpus)
    cost_to_embed_tokens = num_tokens / 1000 * EMBEDDING_COST_PER_1K_TOKENS
    print(
        f"Number of articles: {len(encoded_corpus)}\n"
        f"Total tokens: {num_tokens}\n"
        f"Estimated embedding cost: ${cost_to_embed_tokens:.4f} USD"
    )

    # Process embeddings in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(get_embeddings, text_batch)
            for text_batch in batchify(encoded_corpus, batch_size)
        ]

        # Progress bar
        with tqdm(total=len(encoded_corpus)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(batch_size)

        # Collect results
        embeddings = []
        for future in futures:
            embeddings.extend(future.result())

    return embeddings

def generate_embeddings(df, column_name):
    """Generate embeddings for a specified DataFrame column and add them as a new column."""
    descriptions = df[column_name].astype(str).tolist()
    embeddings = embed_corpus(descriptions)
    df['embeddings'] = embeddings
    print("Embeddings created and added to the DataFrame.")
    return df
```

Now, generate embeddings for the `productDisplayName` column in your dataset.

```python
# Generate embeddings (costs ~$0.001 for 1K items)
styles_df = generate_embeddings(styles_df, 'productDisplayName')

# Save the DataFrame with embeddings for future use
styles_df.to_csv('data/sample_clothes/sample_styles_with_embeddings.csv', index=False)
print("Embeddings saved to 'sample_styles_with_embeddings.csv'.")
```

> **Note:** If you prefer to skip the embedding generation, you can load a pre-computed version. Uncomment and run the following code instead.
> ```python
> # styles_df = pd.read_csv('data/sample_clothes/sample_styles_with_embeddings.csv', on_bad_lines='skip')
> # styles_df['embeddings'] = styles_df['embeddings'].apply(ast.literal_eval)
> ```

## Step 3: Build the Matching Algorithm

You will implement a cosine similarity search to find items in the knowledge base that are most similar to a given query embedding.

First, define a manual cosine similarity function.

```python
def cosine_similarity_manual(vec1, vec2):
    """Calculate the cosine similarity between two vectors."""
    vec1 = np.array(vec1, dtype=float)
    vec2 = np.array(vec2, dtype=float)

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)
```

Next, create the function that finds the top-k most similar items above a similarity threshold.

```python
def find_similar_items(input_embedding, embeddings, threshold=0.5, top_k=2):
    """
    Find the most similar items based on cosine similarity.
    
    Args:
        input_embedding: The query vector.
        embeddings: List of vectors to search through.
        threshold: Minimum similarity score for a match.
        top_k: Number of top matches to return.
    
    Returns:
        List of tuples (index, similarity_score) for the top matches.
    """
    # Calculate similarities
    similarities = [
        (idx, cosine_similarity_manual(input_embedding, vec))
        for idx, vec in enumerate(embeddings)
    ]
    
    # Filter by threshold
    filtered = [(idx, sim) for idx, sim in similarities if sim >= threshold]
    
    # Sort and return top-k
    sorted_indices = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_indices
```

Finally, create a wrapper function that takes text descriptions, generates their embeddings, and finds matching items from the DataFrame.

```python
def find_matching_items_with_rag(df_items, item_descriptions):
    """
    For each input description, find the most similar items in the DataFrame.
    
    Args:
        df_items: DataFrame containing an 'embeddings' column.
        item_descriptions: List of text descriptions to match.
    
    Returns:
        List of matching DataFrame rows.
    """
    embeddings = df_items['embeddings'].tolist()
    similar_items = []

    for desc in item_descriptions:
        # Generate embedding for the query description
        input_embedding = get_embeddings([desc])
        # Find similar items
        similar_indices = find_similar_items(input_embedding[0], embeddings, threshold=0.6)
        # Collect the matching rows
        similar_items.extend([df_items.iloc[i[0]] for i in similar_indices])

    return similar_items
```

## Step 4: Create the Image Analysis Module

The core of the application uses GPT-4o mini to analyze an image of a clothing item and extract structured data. The model will suggest complementary items, a category, and a gender label.

First, define a utility function to encode images.

```python
def encode_image_to_base64(image_path):
    """Encode an image file to a base64 string."""
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode('utf-8')
```

Now, create the function that calls the GPT-4o mini model with a structured prompt.

```python
def analyze_image(image_base64, subcategories):
    """
    Analyze a clothing image using GPT-4o mini.
    
    Args:
        image_base64: Base64-encoded string of the image.
        subcategories: List of possible article types (for the category field).
    
    Returns:
        A JSON string containing 'items', 'category', and 'gender'.
    """
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
Given an image of an item of clothing, analyze the item and generate a JSON output with the following fields: "items", "category", and "gender".
Use your understanding of fashion trends, styles, and gender preferences to provide accurate and relevant suggestions for how to complete the outfit.

The 'items' field should be a list of items that would go well with the item in the picture. Each item should be a concise title containing style, color, and gender (e.g., "Fitted White Women's T-shirt").

The 'category' must be chosen from this list: {subcategories}.

The 'gender' must be chosen from this list: [Men, Women, Boys, Girls, Unisex].

Do not include the description of the item in the picture. Do not include the ```json ``` tag in the output.

Example Input: An image representing a black leather jacket.
Example Output: {{"items": ["Fitted White Women's T-shirt", "White Canvas Sneakers", "Women's Black Skinny Jeans"], "category": "Jackets", "gender": "Women"}}
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    }
                ],
            }
        ]
    )
    return response.choices[0].message.content
```

## Step 5: Run the Complete Pipeline

Now, you will combine all components to process an image and get clothing recommendations.

### 5.1 Analyze a Test Image

Select a test image from your sample dataset and analyze it.

```python
# Path to your sample images
image_path = "data/sample_clothes/sample_images/"
test_image = "2133.jpg"  # Example: Men's shirt

# Encode the image
full_path = image_path + test_image
encoded_image = encode_image_to_base64(full_path)

# Get unique subcategories from the dataset for the prompt
unique_subcategories = styles_df['articleType'].unique()

# Analyze the image
analysis_result = analyze_image(encoded_image, unique_subcategories)
image_analysis = json.loads(analysis_result)

print("Image Analysis Result:")
print(json.dumps(image_analysis, indent=2))
```

### 5.2 Find Matching Items

Use the analysis results to filter the dataset and find complementary items.

```python
# Extract features from the analysis
item_descriptions = image_analysis['items']
item_category = image_analysis['category']
item_gender = image_analysis['gender']

# Filter the dataset: same gender (or unisex) and different category
filtered_items = styles_df.loc[styles_df['gender'].isin([item_gender, 'Unisex'])]
filtered_items = filtered_items[filtered_items['articleType'] != item_category]
print(f"Filtered dataset contains {len(filtered_items)} items for matching.")

# Find matching items using RAG
matching_items = find_matching_items_with_rag(filtered_items, item_descriptions)
print(f"Found {len(matching_items)} matching items.")
```

### 5.3 Display the Results

Create a simple HTML display to show the matching items.

```python
def display_matching_items(matching_items_df, image_base_path="data/sample_clothes/sample_images/"):
    """Display images of matching items in the notebook."""
    html = "<div style='display: flex; flex-wrap: wrap;'>"
    for _, row in matching_items_df.iterrows():
        img_filename = f"{row['id']}.jpg"
        img_path = image_base_path + img_filename
        html += f"<img src='{img_path}' style='width: 200px; margin: 10px;'/>"
    html += "</div>"
    display(HTML(html))

# Convert matching items list to a DataFrame for display
if matching_items:
    matching_df = pd.DataFrame(matching_items)
    display_matching_items(matching_df)
else:
    print("No matching items found.")
```

## Summary

You have successfully built a Clothing Matchmaker application that:

1. **Generates Embeddings**: Creates vector representations for a clothing dataset.
2. **Analyzes Images**: Uses GPT-4o mini to extract structured features (complementary items, category, gender) from an image.
3. **Performs Semantic Search**: Implements a RAG pipeline with cosine similarity to find matching items from the knowledge base.
4. **Filters Intelligently**: Applies business logic (gender, category) to refine recommendations.

This pipeline is modular and scalable. You can replace the local matching algorithm with a production-grade vector database, expand the dataset, or refine the analysis prompt for more complex styling rules.