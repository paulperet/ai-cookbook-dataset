# Guide: Clustering Customer Reviews with K-Means and OpenAI

This guide demonstrates how to discover hidden groupings within a dataset of customer reviews using K-Means clustering and OpenAI's GPT-4. You'll learn to cluster review embeddings, visualize the results, and use an LLM to interpret and label each cluster's theme.

## Prerequisites

Ensure you have the following Python libraries installed. You can install them using `pip`.

```bash
pip install numpy pandas scikit-learn matplotlib openai
```

You will also need:
*   A CSV file containing review text and pre-computed embeddings. This guide uses `fine_food_reviews_with_embeddings_1k.csv`.
*   An OpenAI API key set in your environment variables as `OPENAI_API_KEY`.

## Step 1: Load and Prepare the Data

First, import the necessary libraries and load your dataset. The embeddings are stored as strings in the CSV, so we need to convert them into NumPy arrays for processing.

```python
import numpy as np
import pandas as pd
from ast import literal_eval

# Load the data
datafile_path = "./data/fine_food_reviews_with_embeddings_1k.csv"
df = pd.read_csv(datafile_path)

# Convert the string representation of embeddings into a NumPy array
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

# Create a matrix of all embeddings for clustering
matrix = np.vstack(df.embedding.values)
print(f"Embedding matrix shape: {matrix.shape}")
```

## Step 2: Apply K-Means Clustering

We'll use the K-Means algorithm from scikit-learn to group the reviews into clusters. The number of clusters (`n_clusters`) is a key parameter you can adjust based on your needs—more clusters will reveal finer-grained patterns.

```python
from sklearn.cluster import KMeans

# Define the number of clusters
n_clusters = 4

# Initialize and fit the K-Means model
kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
kmeans.fit(matrix)

# Assign the cluster labels back to the DataFrame
labels = kmeans.labels_
df["Cluster"] = labels

# Examine the average review score per cluster
print("Average Score by Cluster:")
print(df.groupby("Cluster").Score.mean().sort_values())
```

## Step 3: Visualize the Clusters in 2D

To get an intuitive sense of the clusters, we can project the high-dimensional embeddings into a 2D space using t-SNE and plot them.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Reduce dimensionality for visualization
tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
vis_dims2 = tsne.fit_transform(matrix)

x = [x for x, y in vis_dims2]
y = [y for x, y in vis_dims2]

# Define colors for each cluster and plot
colors = ["purple", "green", "red", "blue"]
for cluster_id, color in enumerate(colors):
    # Select points belonging to the current cluster
    xs = np.array(x)[df.Cluster == cluster_id]
    ys = np.array(y)[df.Cluster == cluster_id]
    plt.scatter(xs, ys, color=color, alpha=0.3, label=f'Cluster {cluster_id}')

    # Mark the cluster center
    avg_x = xs.mean()
    avg_y = ys.mean()
    plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)

plt.title("Clusters Visualized in 2D using t-SNE")
plt.legend()
plt.show()
```

The visualization helps identify how distinct the clusters are. For example, in one run, the green cluster (Cluster 1) appeared quite separate from the others.

## Step 4: Interpret and Label Clusters with GPT-4

Now, let's understand what each cluster represents. We will sample a few reviews from each cluster and use GPT-4 to generate a descriptive theme.

```python
from openai import OpenAI
import os

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Number of reviews to sample per cluster for analysis
rev_per_cluster = 5

for i in range(n_clusters):
    print(f"\n--- Cluster {i} ---")

    # Sample reviews and format them
    reviews = "\n".join(
        df[df.Cluster == i]
        .combined.str.replace("Title: ", "")
        .str.replace("\n\nContent: ", ":  ")
        .sample(rev_per_cluster, random_state=42)
        .values
    )

    # Create a prompt for GPT-4 to find the common theme
    messages = [
        {"role": "user", "content": f'What do the following customer reviews have in common?\n\nCustomer reviews:\n"""\n{reviews}\n"""\n\nTheme:'}
    ]

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
        max_tokens=64
    )

    theme = response.choices[0].message.content.replace("\n", "")
    print(f"Theme: {theme}")

    # Display the sampled reviews for manual inspection
    sample_cluster_rows = df[df.Cluster == i].sample(rev_per_cluster, random_state=42)
    for j in range(rev_per_cluster):
        score = sample_cluster_rows.Score.values[j]
        summary = sample_cluster_rows.Summary.values[j]
        text_preview = sample_cluster_rows.Text.str[:70].values[j]
        print(f"  Score {score}, '{summary}': {text_preview}...")
    print("-" * 100)
```

### Example Output Themes:
*   **Cluster 0:** Food products purchased on Amazon.
*   **Cluster 1:** Pet food reviews.
*   **Cluster 2:** Reviews about different types of coffee.
*   **Cluster 3:** Food and drink products.

## Key Considerations

*   **Cluster Count:** The number of clusters you choose (`n_clusters`) directly impacts the results. A smaller number highlights the broadest patterns in your data, while a larger number reveals more specific, niche groupings.
*   **Cluster Interpretation:** The generated themes are based on a small sample. For production use, you might want to sample more reviews or refine the prompt to ensure the labels are accurate and useful for your specific application.
*   **Use Case Alignment:** The clusters identified by K-Means are based on embedding similarity, which may not always align perfectly with the categories you have in mind. Experimentation is key.

This workflow provides a powerful method to explore and categorize large volumes of text data, revealing insights that might not be immediately apparent.