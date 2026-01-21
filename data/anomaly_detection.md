# Anomaly Detection with Embeddings: A Step-by-Step Guide

## Overview

This tutorial demonstrates how to use embeddings from the Gemini API to detect potential outliers in a dataset. You will visualize a subset of the 20 Newsgroup dataset using t-SNE and identify outliers that fall outside a defined radius from the central point of each categorical cluster.

For more information on getting started with Gemini API embeddings, check out the [Get Started guide](../quickstarts/Get_started.ipynb).

## Prerequisites

To follow this guide, you need:
- Python 3.11+
- A Gemini API key (available from [Google AI Studio](https://aistudio.google.com/))

## Setup

### 1. Install Required Libraries
First, install the Gemini API Python library and other dependencies.

```bash
pip install -U google-genai numpy pandas matplotlib seaborn scikit-learn tqdm
```

### 2. Import Libraries
Import the necessary modules for data processing, visualization, and API interaction.

```python
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google import genai
from google.genai import types

from sklearn.datasets import fetch_20newsgroups
from sklearn.manifold import TSNE
```

### 3. Configure the Gemini API Client
Before using the Gemini API, you must obtain and configure your API key.

```python
# Replace with your actual API key or set it as an environment variable
GEMINI_API_KEY = "YOUR_API_KEY_HERE"

client = genai.Client(api_key=GEMINI_API_KEY)
```

**Key Point:** Choose an embedding model and stick with it for consistency. Different models produce incompatible embeddings.

```python
# List available embedding models
for m in client.models.list():
    if 'embedContent' in m.supported_actions:
        print(m.name)
```

Select the model you want to use:

```python
MODEL_ID = "gemini-embedding-001"
```

## Prepare the Dataset

### 4. Load the 20 Newsgroups Dataset
The [20 Newsgroups Text Dataset](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) contains 18,000 newsgroup posts across 20 topics. We'll use the training subset.

```python
newsgroups_train = fetch_20newsgroups(subset='train')

# View the available categories
print(newsgroups_train.target_names)
```

### 5. Clean and Preprocess the Text Data
Clean the text by removing emails, names, and other extraneous information, then truncate long entries.

```python
# Apply cleaning functions
newsgroups_train.data = [re.sub(r'[\w\.-]+@[\w\.-]+', '', d) for d in newsgroups_train.data]  # Remove email
newsgroups_train.data = [re.sub(r"\([^()]*\)", "", d) for d in newsgroups_train.data]  # Remove names
newsgroups_train.data = [d.replace("From: ", "") for d in newsgroups_train.data]  # Remove "From: "
newsgroups_train.data = [d.replace("\nSubject: ", "") for d in newsgroups_train.data]  # Remove "\nSubject: "

# Truncate each entry to 5,000 characters
newsgroups_train.data = [d[0:5000] if len(d) > 5000 else d for d in newsgroups_train.data]
```

### 6. Create a DataFrame
Organize the training data into a pandas DataFrame for easier manipulation.

```python
df_train = pd.DataFrame(newsgroups_train.data, columns=['Text'])
df_train['Label'] = newsgroups_train.target
df_train['Class Name'] = df_train['Label'].map(newsgroups_train.target_names.__getitem__)

print(df_train.head())
print(f"DataFrame shape: {df_train.shape}")
```

### 7. Sample and Filter the Data
To make the analysis manageable, sample 150 points per category and filter to only science-related topics.

```python
# Take a sample of each label category
SAMPLE_SIZE = 150
df_train = (df_train.groupby('Label', as_index=False)
                    .apply(lambda x: x.sample(SAMPLE_SIZE))
                    .reset_index(drop=True))

# Filter to science categories
df_train = df_train[df_train['Class Name'].str.contains('sci')]

# Reset index
df_train = df_train.reset_index()

print(df_train['Class Name'].value_counts())
print(f"Filtered DataFrame shape: {df_train.shape}")
```

## Generate Embeddings

### 8. Create an Embedding Function
The Gemini embedding model supports several task types. For clustering analysis, use the `CLUSTERING` task type.

```python
from tqdm.auto import tqdm
tqdm.pandas()

def make_embed_text_fn(model):
    def embed_fn(text: str) -> list[float]:
        # Set task_type to CLUSTERING for this analysis
        result = client.models.embed_content(
            model=model,
            contents=text,
            config=types.EmbedContentConfig(task_type="CLUSTERING")
        )
        return np.array(result.embeddings[0].values)
    return embed_fn

def create_embeddings(df):
    df['Embeddings'] = df['Text'].progress_apply(make_embed_text_fn(MODEL_ID))
    return df
```

### 9. Generate Embeddings for the Dataset
Apply the embedding function to your text data.

```python
df_train = create_embeddings(df_train)
df_train.drop('index', axis=1, inplace=True)

print("Embedding generation complete.")
print(f"Sample embedding dimension: {len(df_train['Embeddings'][0])}")
```

## Dimensionality Reduction

### 10. Prepare Embeddings for Visualization
The embedding vectors have 3072 dimensions. To visualize them, we need to reduce dimensionality to 2D or 3D.

```python
# Convert embeddings to a numpy array
X = np.array(df_train['Embeddings'].to_list(), dtype=np.float32)
print(f"Embedding array shape: {X.shape}")
```

### 11. Apply t-SNE for Dimensionality Reduction
t-Distributed Stochastic Neighbor Embedding (t-SNE) preserves local structure, making it ideal for visualizing clusters.

```python
tsne = TSNE(random_state=0)
tsne_results = tsne.fit_transform(X)

# Create a DataFrame for the t-SNE results
df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
df_tsne['Class Name'] = df_train['Class Name']

print(df_tsne.head())
```

## Next Steps

You now have:
1. A cleaned and sampled dataset of science-related newsgroup posts
2. High-dimensional embeddings generated using the Gemini API
3. 2D representations of those embeddings via t-SNE

In the next part of this tutorial, you'll learn how to:
- Visualize the t-SNE results to identify clusters
- Calculate cluster centroids
- Define anomaly thresholds based on distance from centroids
- Detect and analyze outliers in your dataset

The foundation is set for performing meaningful anomaly detection on text data using semantic embeddings.