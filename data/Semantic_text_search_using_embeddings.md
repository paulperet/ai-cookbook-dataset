# Semantic Text Search with Embeddings: A Practical Guide

This guide demonstrates how to perform semantic text search using embeddings. You'll learn to efficiently search through a dataset of product reviews by embedding your query and finding the most similar entries, all at a low computational cost.

## Prerequisites

Ensure you have the necessary libraries installed and the dataset ready.

```bash
pip install pandas numpy openai
```

## Step 1: Load the Dataset with Embeddings

First, load the dataset containing pre-computed embeddings for product reviews.

```python
import pandas as pd
import numpy as np
from ast import literal_eval

# Path to the dataset (ensure this file exists in your environment)
datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"

# Load the CSV file
df = pd.read_csv(datafile_path)

# Convert the string representation of embeddings into NumPy arrays
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

print(f"Dataset loaded with {len(df)} reviews.")
```

## Step 2: Define the Semantic Search Function

To perform semantic search, you'll need a function that:
1. Embeds the search query.
2. Computes the cosine similarity between the query embedding and each review embedding.
3. Returns the top `n` most similar reviews.

Create a utility module `utils/embeddings_utils.py` with the following functions, or define them inline.

```python
# If creating utils/embeddings_utils.py, include:
# from openai import OpenAI
# import numpy as np

# client = OpenAI(api_key="your-api-key")

# def get_embedding(text, model="text-embedding-3-small"):
#     text = text.replace("\n", " ")
#     return client.embeddings.create(input=[text], model=model).data[0].embedding

# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# For simplicity, we'll assume these functions are available.
from utils.embeddings_utils import get_embedding, cosine_similarity

def search_reviews(df, product_description, n=3, pprint=True):
    """
    Search reviews by semantic similarity to a product description.

    Args:
        df (pd.DataFrame): DataFrame containing an 'embedding' column.
        product_description (str): The search query.
        n (int): Number of top results to return.
        pprint (bool): Whether to print formatted results.

    Returns:
        pd.Series: Top n most similar reviews.
    """
    # Generate embedding for the search query
    product_embedding = get_embedding(
        product_description,
        model="text-embedding-3-small"
    )

    # Compute cosine similarity for each review
    df["similarity"] = df.embedding.apply(
        lambda x: cosine_similarity(x, product_embedding)
    )

    # Sort by similarity and select top n results
    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .combined.str.replace("Title: ", "")
        .str.replace("; Content:", ": ")
    )

    # Print results in a readable format
    if pprint:
        for r in results:
            print(r[:200])  # Print first 200 characters
            print()

    return results
```

## Step 3: Execute Semantic Searches

Now, use the `search_reviews` function to perform various semantic searches.

### Search for "delicious beans"

```python
results = search_reviews(df, "delicious beans", n=3)
```

**Output:**
```
Delicious!:  I enjoy this white beans seasoning, it gives a rich flavor to the beans I just love it, my mother in law didn't know about this Zatarain's brand and now she is traying different seasoning

Fantastic Instant Refried beans:  Fantastic Instant Refried Beans have been a staple for my family now for nearly 20 years.  All 7 of us love it and my grown kids are passing on the tradition.

Delicious:  While there may be better coffee beans available, this is my first purchase and my first time grinding my own beans.  I read several reviews before purchasing this brand, and am extremely 
```

### Search for "whole wheat pasta"

```python
results = search_reviews(df, "whole wheat pasta", n=3)
```

**Output:**
```
Tasty and Quick Pasta:  Barilla Whole Grain Fusilli with Vegetable Marinara is tasty and has an excellent chunky vegetable marinara.  I just wish there was more of it.  If you aren't starving or on a 

sooo good:  tastes so good. Worth the money. My boyfriend hates wheat pasta and LOVES this. cooks fast tastes great.I love this brand and started buying more of their pastas. Bulk is best.

Bland and vaguely gamy tasting, skip this one:  As far as prepared dinner kits go, "Barilla Whole Grain Mezze Penne with Tomato and Basil Sauce" just did not do it for me...and this is coming from a p
```

### Search for "bad delivery"

This query efficiently surfaces reviews about delivery issues.

```python
results = search_reviews(df, "bad delivery", n=1)
```

**Output:**
```
great product, poor delivery:  The coffee is excellent and I am a repeat buyer.  Problem this time was with the UPS delivery.  They left the box in front of my garage door in the middle of the drivewa
```

### Search for "spoilt"

Find reviews mentioning product spoilage or damage.

```python
results = search_reviews(df, "spoilt", n=1)
```

**Output:**
```
Disappointed:  The metal cover has severely disformed. And most of the cookies inside have been crushed into small pieces. Shopping experience is awful. I'll never buy it online again.
```

### Search for "pet food"

```python
results = search_reviews(df, "pet food", n=2)
```

**Output:**
```
Great food!:  I wanted a food for a a dog with skin problems. His skin greatly improved with the switch, though he still itches some.  He loves the food. No recalls, American made with American ingred

Great food!:  I wanted a food for a a dog with skin problems. His skin greatly improved with the switch, though he still itches some.  He loves the food. No recalls, American made with American ingred
```

## Summary

You've successfully implemented a semantic search system using embeddings. This approach allows you to:

- **Understand Context:** Go beyond keyword matching to find conceptually similar reviews.
- **Operate Efficiently:** Perform searches quickly and at low cost once embeddings are pre-computed.
- **Extract Insights:** Rapidly identify patterns like delivery failures or product defects.

To scale this system, consider using optimized similarity search algorithms (e.g., HNSW, FAISS) for larger datasets. The core principle remains the same: embed your query, compute similarity, and retrieve the most relevant results.