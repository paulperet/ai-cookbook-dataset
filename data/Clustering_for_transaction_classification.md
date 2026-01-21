# Clustering for Transaction Classification: A Step-by-Step Guide

This guide demonstrates how to use clustering on unlabeled transaction data and leverage a Large Language Model (LLM) to generate human-readable descriptions for each cluster. This technique allows you to categorize and label previously unclassified datasets meaningfully.

## Prerequisites & Setup

Before starting, ensure you have the necessary libraries installed and your environment configured.

### 1. Install Required Packages
Run the following command to install the required Python packages:

```bash
pip install openai pandas numpy scikit-learn matplotlib python-dotenv
```

### 2. Import Libraries and Configure API
Create a new Python script or notebook and import the necessary modules. Configure your OpenAI API key securely.

```python
# Import libraries
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from ast import literal_eval

# Optional: Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))
COMPLETIONS_MODEL = "gpt-3.5-turbo"

# Define the path to your dataset with precomputed embeddings
embedding_path = "data/library_transactions_with_embeddings_359.csv"
```

**Note:** This guide assumes you have a CSV file containing transaction data with precomputed embeddings. The method for generating these embeddings is covered in a separate tutorial on multiclass classification.

## Step 1: Load and Inspect the Data

Load the dataset containing your transactions and their corresponding embeddings.

```python
# Load the main DataFrame
df = pd.read_csv(embedding_path)
print(df.head())
```

Next, prepare the embedding matrix for clustering. The embeddings are stored as string representations of lists in the CSV, so we need to convert them back to NumPy arrays.

```python
# Load embeddings and convert the string representation to a NumPy array
embedding_df = pd.read_csv(embedding_path)
embedding_df["embedding"] = embedding_df.embedding.apply(literal_eval).apply(np.array)

# Stack all embedding vectors into a single matrix
matrix = np.vstack(embedding_df.embedding.values)
print(f"Embedding matrix shape: {matrix.shape}")
```

This should output `(359, 1536)`, confirming you have 359 transactions, each represented by a 1536-dimensional embedding vector.

## Step 2: Apply K-Means Clustering

Now, you will cluster the transactions based on their embedding vectors using the K-Means algorithm.

```python
# Define the number of clusters
n_clusters = 5

# Initialize and fit the K-Means model
kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
kmeans.fit(matrix)

# Assign cluster labels to your DataFrame
labels = kmeans.labels_
embedding_df["Cluster"] = labels
```

## Step 3: Visualize the Clusters (Optional)

To better understand the separation between clusters, you can project the high-dimensional embeddings into a 2D space using t-SNE and create a scatter plot.

```python
# Reduce dimensionality for visualization
tsne = TSNE(
    n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200
)
vis_dims2 = tsne.fit_transform(matrix)

x = [x for x, y in vis_dims2]
y = [y for x, y in vis_dims2]

# Define colors for each cluster
colors = ["purple", "green", "red", "blue", "yellow"]

# Plot each cluster
for category, color in enumerate(colors):
    xs = np.array(x)[embedding_df.Cluster == category]
    ys = np.array(y)[embedding_df.Cluster == category]
    plt.scatter(xs, ys, color=color, alpha=0.3, label=f'Cluster {category}')

    # Mark the cluster center
    avg_x = xs.mean()
    avg_y = ys.mean()
    plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)

plt.title("Clusters Visualized in 2D Using t-SNE")
plt.legend()
plt.show()
```

This visualization helps you see how well the transactions are grouped and if any clusters overlap significantly.

## Step 4: Generate Human-Readable Cluster Descriptions

This is the core of the tutorial. You will use GPT-3.5-turbo to analyze a sample of transactions from each cluster and generate a concise, thematic description.

```python
# Define how many sample transactions to show the model per cluster
transactions_per_cluster = 10

for i in range(n_clusters):
    print(f"\n--- Cluster {i} ---")
    
    # Prepare a sample of transactions for this cluster
    # Clean the text by removing field labels for clarity
    transactions = "\n".join(
        embedding_df[embedding_df.Cluster == i]
        .combined.str.replace("Supplier: ", "")
        .str.replace("Description: ", ":  ")
        .str.replace("Value: ", ":  ")
        .sample(transactions_per_cluster, random_state=42)
        .values
    )
    
    # Create a prompt for the LLM
    prompt = f'''We want to group these transactions into meaningful clusters so we can target the areas we are spending the most money. 
What do the following transactions have in common?\n\nTransactions:\n"""\n{transactions}\n"""\n\nTheme:'''
    
    # Call the OpenAI API
    response = client.chat.completions.create(
        model=COMPLETIONS_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,        # Low temperature for consistent, focused outputs
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    
    # Extract and print the model's description
    theme = response.choices[0].message.content.replace("\n", "")
    print(f"Theme: {theme}\n")
    
    # Print the sample transactions used for this description
    print("Sample Transactions:")
    sample_cluster_rows = embedding_df[embedding_df.Cluster == i].sample(transactions_per_cluster, random_state=42)
    for j in range(transactions_per_cluster):
        print(f"  - {sample_cluster_rows.Supplier.values[j]}: {sample_cluster_rows.Description.values[j]}")
    
    print("-" * 80)
```

### Expected Output Summary

The model will generate a theme for each cluster. For example:
*   **Cluster 0:** Expenses like electricity, rates, and IT equipment.
*   **Cluster 1:** Payments for goods and services like student bursaries, archival papers, and subscriptions.
*   **Cluster 2:** All spending related to a specific location, "Kelvin Hall".
*   **Cluster 3:** Facility management fees and services.
*   **Cluster 4:** Construction or refurbishment work.

## Conclusion and Next Steps

You have successfully transformed an unlabeled dataset into meaningful categories. The LLM has provided insightful descriptions, even making nuanced connections (e.g., linking legal deposits to literary archives) without explicit guidance.

While the visualization shows some cluster overlap—indicating potential areas for tuning—you now have a foundational set of labeled clusters. You can use these labels to train a multiclass classifier, enabling you to categorize new, unseen transaction data automatically.

**Potential Improvements:**
*   Experiment with different values for `n_clusters`.
*   Try other clustering algorithms like DBSCAN or Agglomerative Clustering.
*   Refine the LLM prompt to generate more specific or formatted descriptions.
*   Use these cluster labels as training data for a supervised learning model to scale classification to larger datasets.