# Visualizing Embeddings with Weights & Biases

This guide walks you through uploading a dataset of text embeddings to **Weights & Biases (W&B)** and visualizing them using interactive 2D projections. You'll learn how to log your data as a W&B Table and explore it with dimension reduction techniques like PCA, UMAP, and t-SNE directly in the W&B interface.

## Prerequisites

Ensure you have the necessary Python packages installed:

```bash
pip install pandas scikit-learn numpy wandb
```

## Step 1: Load Your Embeddings Dataset

First, load the dataset containing your text embeddings. This example uses a CSV file where one column holds the embedding vectors.

```python
import pandas as pd
import numpy as np
from ast import literal_eval

# Path to your embeddings file
datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"

# Load the CSV
df = pd.read_csv(datafile_path)

# Convert the string representation of embeddings to a NumPy array
matrix = np.array(df.embedding.apply(literal_eval).to_list())
```

## Step 2: Prepare Data for Weights & Biases

We'll structure the data for logging to W&B. Each row will represent a review, containing the original data columns and each dimension of its embedding vector.

```python
import wandb

# Identify the original data columns (excluding the first ID column and the embedding column)
original_cols = df.columns[1:-1].tolist()

# Create column names for each embedding dimension (e.g., 'emb_0', 'emb_1', ...)
embedding_cols = ['emb_' + str(idx) for idx in range(len(matrix[0]))]

# Combine all column names
table_cols = original_cols + embedding_cols
```

## Step 3: Log the Data as a W&B Table

Now, initialize a W&B run and create a table to log your data. This table will be available in the W&B dashboard for visualization.

```python
# Start a W&B run in your project
with wandb.init(project='openai_embeddings'):
    # Create a W&B Table with our combined columns
    table = wandb.Table(columns=table_cols)

    # Iterate through each row in the DataFrame
    for i, row in enumerate(df.to_dict(orient="records")):
        # Extract the original data values
        original_data = [row[col_name] for col_name in original_cols]
        # Extract the embedding vector for this row
        embedding_data = matrix[i].tolist()
        # Add the combined data as a new row in the table
        table.add_data(*(original_data + embedding_data))

    # Log the completed table to W&B
    wandb.log({'openai_embedding_table': table})
```

After executing this code, your data is uploaded. A link to the W&B run will appear in your console—click it to open the dashboard.

## Step 4: Visualize with 2D Projection

In the W&B dashboard:
1. Navigate to the run you just created.
2. Find the `openai_embedding_table` in the run's tables.
3. Click the **gear icon (⚙️)** in the top-right corner of the table panel.
4. Change the **"Render As:"** setting to **"Combined 2D Projection"**.

W&B will automatically generate an interactive visualization using dimension reduction algorithms. You can switch between **PCA, UMAP, and t-SNE** to explore different views of your high-dimensional embeddings in 2D space.

## Next Steps

Use this visualization to:
- Identify clusters of similar reviews.
- Spot outliers or anomalies in your data.
- Validate the quality and semantic structure captured by your embeddings.

For a live example, visit: [http://wandb.me/openai_embeddings](http://wandb.me/openai_embeddings)

This workflow provides a powerful, shareable way to analyze and present your embedding data, making it easier to derive insights and collaborate with your team.