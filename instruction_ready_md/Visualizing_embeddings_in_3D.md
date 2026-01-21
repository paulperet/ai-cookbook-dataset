# Visualizing Embeddings in 3D: A Step-by-Step Guide

This tutorial demonstrates how to visualize high-dimensional text embeddings in a 3D space. We'll use Principal Component Analysis (PCA) to reduce 1536-dimensional embeddings down to 3 dimensions, then create an interactive 3D plot to explore the data. The dataset is a curated sample from DBpedia.

## Prerequisites & Setup

First, ensure you have the necessary libraries installed. If you haven't already, run the following installation commands in your environment.

```bash
pip install pandas scikit-learn matplotlib
```

Now, let's import the required modules.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
```

## Step 1: Load the Dataset and Generate Embeddings

We'll start by loading the sample data and generating embeddings for each text entry.

### 1.1 Load the DBpedia Samples

The dataset `dbpedia_samples.jsonl` contains 200 randomly sampled entries from the DBpedia validation dataset.

```python
# Load the dataset
samples = pd.read_json("data/dbpedia_samples.jsonl", lines=True)

# Display the unique categories and their counts
categories = sorted(samples["category"].unique())
print("Categories of DBpedia samples:")
print(samples["category"].value_counts())

# Show the first few rows of the dataframe
print("\nFirst few samples:")
print(samples.head())
```

**Output:**
```
Categories of DBpedia samples:
Artist                    21
Film                      19
Plant                     19
OfficeHolder              18
Company                   17
NaturalPlace              16
Athlete                   16
Village                   12
WrittenWork               11
Building                  11
Album                     11
Animal                    11
EducationalInstitution    10
MeanOfTransportation       8
Name: category, dtype: int64

First few samples:
# ... (DataFrame preview)
```

### 1.2 Generate Text Embeddings

Next, we'll convert the text data into vector embeddings using a pre-trained model. This example assumes you have a utility function `get_embeddings` that queries an embedding API (like OpenAI's). The function sends a batch request for all 200 samples.

```python
# Import the helper function (ensure utils.embeddings_utils is in your path)
from utils.embeddings_utils import get_embeddings

# Generate embeddings for all text entries
# This will create a matrix of shape (200, 1536) for the 'text-embedding-3-small' model
matrix = get_embeddings(samples["text"].to_list(), model="text-embedding-3-small")
```

The variable `matrix` now contains a 2D NumPy array where each row is a 1536-dimensional vector representing a text sample.

## Step 2: Reduce Dimensionality with PCA

To visualize these high-dimensional vectors, we need to project them into a 3D space. We'll use PCA for this dimensionality reduction.

```python
# Initialize PCA to reduce to 3 principal components
pca = PCA(n_components=3)

# Fit PCA on the embedding matrix and transform the data
vis_dims = pca.fit_transform(matrix)

# Store the 3D coordinates as a new column in our dataframe
samples["embed_vis"] = vis_dims.tolist()
```

The `vis_dims` array now has the shape `(200, 3)`. Each sample is represented by just three numbers (x, y, z) suitable for 3D plotting.

## Step 3: Create a 3D Visualization

Now we'll plot the reduced embeddings in an interactive 3D scatter plot, color-coding points by their category.

```python
# Set up the matplotlib backend for interactive plotting (e.g., in Jupyter)
%matplotlib widget

# Create a 3D figure
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(projection='3d')

# Use a colormap with enough distinct colors for our categories
cmap = plt.get_cmap("tab20")

# Plot each category separately to assign labels
for i, cat in enumerate(categories):
    # Extract the 3D coordinates for all samples of this category
    sub_matrix = np.array(samples[samples["category"] == cat]["embed_vis"].to_list())
    x = sub_matrix[:, 0]
    y = sub_matrix[:, 1]
    z = sub_matrix[:, 2]
    
    # Assign a consistent color for this category
    colors = [cmap(i / len(categories))] * len(sub_matrix)
    
    # Add the points to the plot
    ax.scatter(x, y, zs=z, zdir='z', c=colors, label=cat)

# Add axis labels and a legend
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend(bbox_to_anchor=(1.1, 1))

# Display the plot
plt.show()
```

## Understanding the Results

After running the code, you'll see an interactive 3D plot. Here's what to look for:

*   **Clustering:** Points of the same color (category) that are close together indicate the embeddings successfully captured semantic similarity. For example, all "Artist" entries might form a distinct cluster.
*   **Separation:** Clear gaps between different colored clusters show the model distinguishes well between categories.
*   **PCA Axes:** The X, Y, and Z axes represent the three most significant directions of variance in the original 1536-dimensional space, as determined by PCA.

This visualization helps validate the quality of your embeddings and provides an intuitive way to explore the relationships within your text data.

## Next Steps

*   **Experiment:** Try different embedding models (e.g., `text-embedding-3-large`) and compare the resulting plots.
*   **Alternative Reduction:** Use t-SNE or UMAP instead of PCA for potentially better preservation of local structures.
*   **Larger Datasets:** Apply the same technique to your own datasets to explore their semantic landscape.

By following this guide, you've successfully transformed high-dimensional text embeddings into an interpretable 3D visualization, a valuable technique for debugging and understanding your NLP pipelines.