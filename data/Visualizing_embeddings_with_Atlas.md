# Visualizing OpenAI Embeddings with Atlas: A Step-by-Step Guide

In this tutorial, you'll learn how to visualize OpenAI text embeddings using [Atlas](https://atlas.nomic.ai), a powerful tool for exploring high-dimensional data in your web browser. We'll walk through uploading a dataset of food review embeddings and creating an interactive visualization.

## Prerequisites

Before you begin, ensure you have the following installed:

```bash
pip install pandas numpy nomic
```

You will also need:
- A CSV file containing your text embeddings (we'll use a sample dataset of food reviews).
- An Atlas API key for authentication (a demo key is provided in this example).

## Step 1: Load and Prepare Your Embedding Data

First, import the necessary libraries and load your dataset. We'll use a sample file containing 1,000 food reviews with pre-computed embeddings.

```python
import pandas as pd
import numpy as np
from ast import literal_eval

# Load the embeddings from a CSV file
datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"
df = pd.read_csv(datafile_path)

# Convert the string representation of embeddings to a NumPy array
embeddings = np.array(df.embedding.apply(literal_eval).to_list())

# Clean up the DataFrame: remove the embedding column and rename the index column
df = df.drop('embedding', axis=1)
df = df.rename(columns={'Unnamed: 0': 'id'})
```

**What's happening here?**
- We load a CSV file containing review text and their corresponding embedding vectors.
- The embeddings are stored as string representations of lists, so we use `literal_eval` to convert them to actual Python lists.
- We then convert the list of lists into a NumPy array for efficient processing.
- Finally, we clean the DataFrame by removing the original embedding column and renaming the index column to `'id'`.

## Step 2: Authenticate with Atlas

Now, authenticate with the Atlas service using your API key. This example uses a demo key, but you should replace it with your own for production use.

```python
import nomic
from nomic import atlas

# Authenticate with your Atlas API key
nomic.login('7xDPkYXSYDc1_ErdTPIcoAR9RNd8YDlkS3nVNXcVoIMZ6')  # Demo key
```

## Step 3: Create an Atlas Project and Map

With your data prepared and authentication set, you can now create an Atlas project and generate a visualization map.

```python
# Convert the DataFrame to a list of dictionaries for Atlas
data = df.to_dict('records')

# Create the Atlas project and map
project = atlas.map_embeddings(
    embeddings=embeddings,
    data=data,
    id_field='id',
    colorable_fields=['Score']
)

# Access the first map in the project
map = project.maps[0]
```

**Understanding the parameters:**
- `embeddings`: The NumPy array containing your embedding vectors.
- `data`: The metadata associated with each embedding (e.g., review text, score, etc.).
- `id_field`: The unique identifier field in your metadata.
- `colorable_fields`: The metadata fields you can use to color-code points in the visualization (here, the review score).

## Step 4: Explore Your Visualization

After creating the map, you can interact with it directly in your Jupyter notebook.

```python
# Display the map in your notebook
map
```

When you run this cell, you'll see output similar to:

```
Project: meek-laborer
Projection ID: 463f4614-7689-47e4-b55b-1da0cc679559
Explore on atlas.nomic.ai
```

This provides:
- Your project name (automatically generated)
- A unique projection ID
- A link to explore the visualization in the Atlas web interface

## Next Steps

Your embeddings are now visualized in Atlas! You can:

1. **Explore in Jupyter:** Interact with the embedded visualization to zoom, pan, and select points.
2. **Open in Browser:** Click the provided link to access the full Atlas web interface with additional analysis tools.
3. **Color by Score:** Use the `Score` field to color-code points and identify patterns in review sentiment.

Atlas enables you to discover clusters, outliers, and relationships in your embedding space that might not be apparent from the raw data alone. This visualization can help validate embedding quality, identify data issues, or generate hypotheses for further analysis.

## Summary

In this tutorial, you've learned how to:
1. Load and preprocess embedding data from a CSV file
2. Authenticate with the Atlas API
3. Create an interactive visualization of your embeddings
4. Explore the results both in Jupyter and through the web interface

This workflow provides a powerful way to understand and communicate insights from your AI models' embedding spaces.