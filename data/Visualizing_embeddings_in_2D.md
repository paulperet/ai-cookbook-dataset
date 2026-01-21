# Visualizing Embeddings in 2D with t-SNE

This guide walks you through reducing high-dimensional text embeddings to 2D using t-SNE (t-Distributed Stochastic Neighbor Embedding) and visualizing them with a scatter plot. This technique helps you understand how semantic relationships between text documents (in this case, product reviews) are represented in the embedding space.

## Prerequisites

Ensure you have the required libraries installed and the dataset ready.

```bash
pip install pandas scikit-learn matplotlib numpy
```

The dataset `fine_food_reviews_with_embeddings_1k.csv` should be located in a `data/` directory. This file contains 1,000 Amazon food reviews with pre-computed text embeddings (e.g., from OpenAI's `text-embedding-ada-002` model, which produces 1536-dimensional vectors).

## Step 1: Import Libraries and Load Data

First, import the necessary libraries and load the dataset containing the embeddings.

```python
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from ast import literal_eval

# Load the dataset
datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"
df = pd.read_csv(datafile_path)
```

## Step 2: Prepare the Embedding Matrix

The embeddings are stored in the DataFrame as string representations of lists. We need to convert them into a NumPy array for processing.

```python
# Convert the 'embedding' column from string to list, then to a NumPy array
matrix = np.array(df['embedding'].apply(literal_eval).to_list())
print(f"Embedding matrix shape: {matrix.shape}")
```

You should see an output like `(1000, 1536)`, confirming you have 1000 reviews, each represented by a 1536-dimensional vector.

## Step 3: Reduce Dimensionality with t-SNE

We'll use t-SNE to project the 1536-dimensional embeddings into a 2D space, making them plottable. t-SNE is excellent for visualizing high-dimensional data while preserving local structures.

```python
# Initialize and fit the t-SNE model
tsne = TSNE(n_components=2, perplexity=15, random_state=42,
            init='random', learning_rate=200)

vis_dims = tsne.fit_transform(matrix)
print(f"Reduced dimensions shape: {vis_dims.shape}")
```

The `vis_dims` variable now contains the 2D coordinates for each review. The parameters `perplexity`, `init`, and `learning_rate` are tuned for this specific dataset to produce a clear visualization.

## Step 4: Visualize the 2D Embeddings

We'll create a scatter plot where each point represents a review. Points are colored by their star rating (1 to 5 stars), and we'll mark the average location for each rating with an "X".

```python
# Define a color map from red (1-star) to green (5-star)
colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]

# Extract x and y coordinates from the t-SNE output
x_coords = [x for x, y in vis_dims]
y_coords = [y for x, y in vis_dims]

# Create color indices based on star ratings (1-5 -> 0-4)
color_indices = df['Score'].values - 1

# Create the scatter plot
colormap = matplotlib.colors.ListedColormap(colors)
plt.figure(figsize=(10, 8))
plt.scatter(x_coords, y_coords, c=color_indices, cmap=colormap, alpha=0.3)

# Calculate and plot the average location for each star rating
for score in range(5):
    avg_x = np.array(x_coords)[df['Score'] - 1 == score].mean()
    avg_y = np.array(y_coords)[df['Score'] - 1 == score].mean()
    color = colors[score]
    plt.scatter(avg_x, avg_y, marker='x', color=color, s=100)

plt.title("Amazon Food Review Ratings Visualized with t-SNE", fontsize=14)
plt.show()
```

## Understanding the Visualization

*   **Color Gradient:** Reviews are colored by their star rating:
    *   **Red:** 1-star (negative)
    *   **Dark Orange:** 2-star
    *   **Gold:** 3-star (neutral)
    *   **Turquoise:** 4-star
    *   **Dark Green:** 5-star (positive)
*   **Cluster "X" Marks:** The large "X" markers indicate the average position for all reviews of a given rating.
*   **Interpretation:** You will likely observe a gradient or separation in the 2D space, with positive reviews (greens) clustering in one region and negative reviews (reds) in another. This demonstrates that the original embeddings encode semantic information about sentiment, which t-SNE preserves in a lower dimension.

## Next Steps

This visualization confirms that the embeddings meaningfully represent the content and sentiment of the reviews. You can extend this analysis by:
*   Experimenting with t-SNE parameters (`perplexity`, `learning_rate`) to improve separation.
*   Using other dimensionality reduction techniques like UMAP for comparison.
*   Applying clustering algorithms (e.g., K-Means) directly on the 2D projections to identify review groups automatically.