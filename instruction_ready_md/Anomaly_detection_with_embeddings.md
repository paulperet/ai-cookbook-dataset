# Anomaly Detection with Embeddings: A Step-by-Step Guide

## Overview

This tutorial demonstrates how to use embeddings from the Gemini API to detect potential outliers in your dataset. You'll visualize a subset of the 20 Newsgroup dataset using t-SNE and identify outliers that fall outside a defined radius from the central point of each categorical cluster.

## Prerequisites

Before you begin, ensure you have the necessary libraries installed and your API key configured.

### Step 1: Install Required Packages

```bash
pip install -U -q "google-genai>=1.0.0"
```

### Step 2: Configure API Authentication

To run the following code, you need to store your API key in an environment variable named `GOOGLE_API_KEY`. If you don't have an API key, see the [Authentication quickstart](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Authentication.ipynb) for setup instructions.

```python
from google import genai

# Configure your API key
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
client = genai.Client(api_key=API_KEY)
```

## Prepare Your Dataset

### Step 3: Load the 20 Newsgroups Dataset

We'll use the training subset of the 20 Newsgroups Text Dataset from SciKit-Learn, which contains 18,000 newsgroup posts across 20 topics.

```python
from sklearn.datasets import fetch_20newsgroups

# Load the training dataset
newsgroups_train = fetch_20newsgroups(subset="train")

# View the available categories
print("Available categories:")
for i, name in enumerate(newsgroups_train.target_names):
    print(f"{i}: {name}")
```

### Step 4: Clean and Preprocess the Text Data

The raw text contains email addresses, names, and other metadata that we should clean before generating embeddings.

```python
import re
import pandas as pd

# Clean the text data
def clean_text(text):
    # Remove email addresses
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "", text)
    # Remove names in parentheses
    text = re.sub(r"\([^()]*\)", "", text)
    # Remove "From: " prefix
    text = text.replace("From: ", "")
    # Remove "\nSubject: " prefix
    text = text.replace("\nSubject: ", "")
    # Truncate to 5,000 characters if longer
    return text[:5000] if len(text) > 5000 else text

# Apply cleaning to all documents
cleaned_data = [clean_text(d) for d in newsgroups_train.data]

# Create a DataFrame
df_train = pd.DataFrame({
    "Text": cleaned_data,
    "Label": newsgroups_train.target
})

# Map label indices to category names
df_train["Class Name"] = df_train["Label"].map(
    lambda x: newsgroups_train.target_names[x]
)

print(f"Dataset shape: {df_train.shape}")
print(df_train.head())
```

### Step 5: Sample and Filter the Data

To make the tutorial manageable, we'll sample 150 documents from each category and focus on science-related topics.

```python
# Sample 150 documents from each category
SAMPLE_SIZE = 150
df_sampled = df_train.groupby("Label", group_keys=False).apply(
    lambda x: x.sample(min(len(x), SAMPLE_SIZE), random_state=42)
)

# Filter to only science categories
science_categories = ["sci.crypt", "sci.electronics", "sci.med", "sci.space"]
df_train = df_sampled[df_sampled["Class Name"].isin(science_categories)].reset_index(drop=True)

print(f"Filtered dataset shape: {df_train.shape}")
print("\nCategory distribution:")
print(df_train["Class Name"].value_counts())
```

## Create Embeddings

### Step 6: Understand Embedding Task Types

The Gemini embeddings API supports different task types. For clustering and anomaly detection, we'll use the `CLUSTERING` task type.

Available task types include:
- `RETRIEVAL_QUERY`: For search queries
- `RETRIEVAL_DOCUMENT`: For search documents
- `SEMANTIC_SIMILARITY`: For semantic similarity tasks
- `CLASSIFICATION`: For classification tasks
- `CLUSTERING`: For clustering tasks (our use case)

### Step 7: Implement the Embedding Function

```python
from tqdm.auto import tqdm
from google.genai import types
from google.api_core import retry
import numpy as np

def make_embed_text_fn(model_name):
    """
    Create a function to embed text using the Gemini API.
    
    Args:
        model_name: The embedding model to use
        
    Returns:
        Function that takes a list of texts and returns embeddings
    """
    @retry.Retry(timeout=300.0)
    def embed_fn(texts: list[str]) -> list[list[float]]:
        # Embed the batch of texts with CLUSTERING task type
        embeddings = client.models.embed_content(
            model=model_name,
            contents=texts,
            config=types.EmbedContentConfig(task_type="CLUSTERING"),
        ).embeddings
        return np.array([embedding.values for embedding in embeddings])
    
    return embed_fn

def create_embeddings(df, batch_size=100):
    """
    Create embeddings for all texts in the DataFrame.
    
    Args:
        df: DataFrame containing a "Text" column
        batch_size: Number of texts to process in each batch
        
    Returns:
        DataFrame with added "Embeddings" column
    """
    MODEL_ID = "text-embedding-004"
    model = f"models/{MODEL_ID}"
    embed_fn = make_embed_text_fn(model)
    
    all_embeddings = []
    
    # Process texts in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Creating embeddings"):
        batch = df["Text"].iloc[i:i + batch_size].tolist()
        embeddings = embed_fn(batch)
        all_embeddings.extend(embeddings)
    
    # Add embeddings to DataFrame
    df = df.copy()
    df["Embeddings"] = all_embeddings
    return df

# Generate embeddings for our dataset
df_train = create_embeddings(df_train)
print(f"Embedding dimension: {len(df_train['Embeddings'][0])}")
```

## Dimensionality Reduction

### Step 8: Prepare Embeddings for Visualization

The embeddings are 768-dimensional vectors. To visualize them, we need to reduce their dimensionality to 2D or 3D.

```python
# Convert embeddings to numpy array
X = np.array(df_train["Embeddings"].to_list(), dtype=np.float32)
print(f"Embeddings shape: {X.shape}")
```

### Step 9: Apply t-SNE for Dimensionality Reduction

t-SNE (t-Distributed Stochastic Neighbor Embedding) is particularly good at preserving local structure, making it ideal for visualizing clusters.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Apply t-SNE to reduce to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X)

# Add 2D coordinates to DataFrame
df_train["x"] = X_2d[:, 0]
df_train["y"] = X_2d[:, 1]

# Visualize the clusters
plt.figure(figsize=(10, 8))
colors = {"sci.crypt": "red", "sci.electronics": "blue", 
          "sci.med": "green", "sci.space": "orange"}

for category, color in colors.items():
    mask = df_train["Class Name"] == category
    plt.scatter(df_train.loc[mask, "x"], df_train.loc[mask, "y"], 
                c=color, label=category, alpha=0.6)

plt.title("t-SNE Visualization of Science Newsgroup Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.show()
```

## Detect Anomalies

### Step 10: Calculate Cluster Centers and Distances

Now we'll identify potential outliers by calculating the distance of each point from its cluster center.

```python
from scipy.spatial.distance import cdist

def detect_anomalies(df, threshold_std=2.0):
    """
    Detect anomalies based on distance from cluster centers.
    
    Args:
        df: DataFrame with embeddings and cluster labels
        threshold_std: Number of standard deviations for anomaly threshold
        
    Returns:
        DataFrame with anomaly scores and labels
    """
    df_result = df.copy()
    df_result["anomaly_score"] = 0.0
    df_result["is_anomaly"] = False
    
    # Calculate for each category
    for category in df["Class Name"].unique():
        # Get embeddings for this category
        mask = df["Class Name"] == category
        category_embeddings = np.array(df.loc[mask, "Embeddings"].to_list())
        
        # Calculate cluster center (mean of embeddings)
        cluster_center = np.mean(category_embeddings, axis=0)
        
        # Calculate distances from center
        distances = cdist(category_embeddings, [cluster_center], metric='euclidean').flatten()
        
        # Calculate anomaly threshold (mean + n*std)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + threshold_std * std_dist
        
        # Mark anomalies
        df_result.loc[mask, "anomaly_score"] = distances
        df_result.loc[mask, "is_anomaly"] = distances > threshold
    
    return df_result

# Detect anomalies
df_with_anomalies = detect_anomalies(df_train, threshold_std=2.0)

# Count anomalies per category
print("Anomalies detected per category:")
print(df_with_anomalies.groupby("Class Name")["is_anomaly"].sum())
```

### Step 11: Visualize Anomalies

Let's visualize the anomalies on our t-SNE plot.

```python
plt.figure(figsize=(12, 10))

# Plot all points
for category, color in colors.items():
    mask = (df_with_anomalies["Class Name"] == category) & (~df_with_anomalies["is_anomaly"])
    plt.scatter(df_with_anomalies.loc[mask, "x"], 
                df_with_anomalies.loc[mask, "y"], 
                c=color, label=category, alpha=0.5, s=50)

# Highlight anomalies
anomaly_mask = df_with_anomalies["is_anomaly"]
plt.scatter(df_with_anomalies.loc[anomaly_mask, "x"], 
            df_with_anomalies.loc[anomaly_mask, "y"], 
            c="black", marker="x", s=200, linewidths=2, 
            label="Anomalies", alpha=0.8)

plt.title("Anomaly Detection in Science Newsgroup Embeddings")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.show()
```

### Step 12: Examine Detected Anomalies

Let's look at some of the detected anomalies to understand why they might be outliers.

```python
# Display top anomalies with highest anomaly scores
top_anomalies = df_with_anomalies[df_with_anomalies["is_anomaly"]].sort_values(
    "anomaly_score", ascending=False
).head(5)

print("Top 5 anomalies detected:")
print("=" * 80)

for idx, row in top_anomalies.iterrows():
    print(f"\nCategory: {row['Class Name']}")
    print(f"Anomaly Score: {row['anomaly_score']:.4f}")
    print(f"Text preview: {row['Text'][:200]}...")
    print("-" * 80)
```

## Summary

In this tutorial, you've learned how to:

1. **Prepare text data** by cleaning and sampling from the 20 Newsgroups dataset
2. **Generate embeddings** using the Gemini API with the appropriate task type for clustering
3. **Visualize embeddings** using t-SNE dimensionality reduction
4. **Detect anomalies** by calculating distances from cluster centers
5. **Identify and examine outliers** that deviate significantly from their category clusters

This approach can be adapted to various domains where you need to identify unusual documents, products, or data points within categorized datasets. The key insight is that embeddings capture semantic meaning, allowing us to find documents that are semantically distant from others in their category.

## Next Steps

- Experiment with different threshold values for anomaly detection
- Try other dimensionality reduction techniques like UMAP
- Apply this method to your own datasets
- Explore using different embedding models or task types
- Implement more sophisticated anomaly detection algorithms on the embeddings