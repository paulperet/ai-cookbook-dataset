# Generating Embeddings from a Dataset: A Step-by-Step Guide

This guide walks you through the process of generating text embeddings for a dataset using OpenAI's embedding models. You will load a sample dataset, preprocess the text, and create vector embeddings suitable for downstream AI tasks like search, clustering, or retrieval.

## Prerequisites

Before you begin, ensure you have the necessary Python packages installed. You can install them using pip:

```bash
pip install pandas openai tiktoken
```

You will also need an OpenAI API key. Set it as an environment variable:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Step 1: Import Libraries and Configure the Model

First, import the required libraries and configure the embedding model you'll use.

```python
import pandas as pd
import tiktoken
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI()

# Configuration for the embedding model
embedding_model = "text-embedding-3-small"
embedding_encoding = "cl100k_base"  # Encoding used by the model
max_tokens = 8000  # Maximum context length for the model is 8191
```

We also need a helper function to call the OpenAI API and retrieve embeddings.

```python
def get_embedding(text, model=embedding_model):
    """Helper function to get an embedding from the OpenAI API."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding
```

## Step 2: Load and Inspect the Dataset

We'll use a sample dataset of Amazon fine food reviews. For this tutorial, we provide a filtered subset containing 1,000 reviews.

```python
# Load the dataset from a CSV file
input_datapath = "data/fine_food_reviews_1k.csv"
df = pd.read_csv(input_datapath, index_col=0)

# Select and rename relevant columns
df = df[["Time", "ProductId", "UserId", "Score", "Summary", "Text"]]
df = df.dropna()  # Remove any rows with missing values

# Combine the review summary and text into a single field
df["combined"] = (
    "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
)

# Display the first few rows to understand the structure
print(df.head(2))
```

## Step 3: Preprocess and Filter the Data

To ensure efficient processing, we will filter the dataset to the 1,000 most recent reviews and remove any that are too long for our model's context window.

```python
# Subsample to the most recent 1000 reviews
top_n = 1000
df = df.sort_values("Time").tail(top_n * 2)  # Take a larger sample anticipating some will be filtered out
df.drop("Time", axis=1, inplace=True)  # Remove the timestamp column as we no longer need it

# Initialize the tokenizer to count tokens
encoding = tiktoken.get_encoding(embedding_encoding)

# Calculate the number of tokens for each combined review
df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))

# Filter out reviews that exceed the model's maximum token limit
df = df[df.n_tokens <= max_tokens].tail(top_n)

print(f"Final dataset size: {len(df)} reviews")
```

## Step 4: Generate and Save Embeddings

Now, we will generate embeddings for each review in our processed dataset. This step involves API calls and may take a few minutes to complete.

```python
# Apply the embedding function to each 'combined' text entry
print("Generating embeddings. This may take a few minutes...")
df["embedding"] = df.combined.apply(lambda x: get_embedding(x, model=embedding_model))

# Save the DataFrame with embeddings to a new CSV file for future use
output_path = "data/fine_food_reviews_with_embeddings_1k.csv"
df.to_csv(output_path)
print(f"Embeddings saved to {output_path}")
```

## Step 5: Verify the Embeddings (Optional)

To confirm everything is working, you can quickly test the embedding function on a simple string and inspect the result.

```python
# Generate an embedding for a test string
test_embedding = get_embedding("hi", model=embedding_model)
print(f"Embedding vector for 'hi' (first 5 dimensions): {test_embedding[:5]}")
print(f"Embedding vector length: {len(test_embedding)}")
```

## Summary

You have successfully loaded a text dataset, preprocessed it by combining fields and filtering by length, and generated vector embeddings using OpenAI's `text-embedding-3-small` model. The final dataset, saved as a CSV file, now includes a numerical embedding for each review, ready to be used in applications like semantic search, recommendation systems, or clustering analysis.

**Next Steps:** You can load the saved CSV file (`fine_food_reviews_with_embeddings_1k.csv`) and use the embeddings to build a simple search index or visualize the data in a lower-dimensional space using techniques like PCA or UMAP.