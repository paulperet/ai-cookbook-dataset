# Visualizing Embeddings with Kangas: A Step-by-Step Guide

In this tutorial, you will learn how to use Kangas to create an interactive DataGrid for visualizing text embeddings. We'll work with a dataset of product reviews, convert their embeddings into a visualizable format, and explore the results through 2D projections.

## What is Kangas?

[Kangas](https://github.com/comet-ml/kangas/) is an open-source, mixed-media tool designed for data scientists. It provides a dataframe-like interface that supports various data types, including embeddings, images, and text. Developed by [Comet](https://comet.com/), Kangas helps streamline the process of analyzing and visualizing complex data, making it easier to move models into production.

## Prerequisites

Before you begin, ensure you have the Kangas library installed.

```bash
pip install kangas
```

## Step 1: Import Kangas and Load Your Data

First, import the Kangas library and load your dataset. In this example, we'll use a publicly available CSV file containing food reviews and their corresponding embeddings.

```python
import kangas as kg

# Load the dataset from a remote URL
data = kg.read_csv("https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/fine_food_reviews_with_embeddings_1k.csv")
```

## Step 2: Inspect the Loaded Data

After loading the data, it's helpful to understand its structure. Use the `info()` method to view the columns and data types.

```python
# Display column information
data.info()
```

To see a preview of the data, simply call the DataGrid object.

```python
# Show the first and last rows
data
```

## Step 3: Create a New DataGrid with Embeddings

The original CSV stores embeddings as strings. We'll convert these into Kangas Embedding objects, which enable advanced visualization features.

```python
import ast  # For converting string representations of lists into actual lists

# Initialize a new DataGrid
dg = kg.DataGrid(
    name="openai_embeddings",
    columns=data.get_columns(),
    converters={"Score": str},  # Convert Score to string for categorical grouping
)

# Process each row to create Embedding objects
for row in data:
    # The embedding is in the 9th column (index 8)
    embedding_list = ast.literal_eval(row[8])
    
    # Create an Embedding object
    row[8] = kg.Embedding(
        embedding_list,
        name=str(row[3]),  # Use the product ID as the embedding name
        text="%s - %.10s" % (row[3], row[4]),  # Create a descriptive label
        projection="umap",  # Specify UMAP for 2D projection
    )
    dg.append(row)

# Verify the new DataGrid structure
dg.info()
```

## Step 4: Save the DataGrid

Once the DataGrid is prepared, save it for future use or sharing.

```python
dg.save()
```

## Step 5: Visualize the Embeddings

Kangas allows you to render the DataGrid directly in your environment. The embeddings are automatically projected into 2D space using UMAP.

```python
# Display the DataGrid with all columns
dg.show()
```

Scroll to the right in the output to see the 2D projection for each row. Points are colored according to the review's "Score" value.

## Step 6: Group and Filter the Visualization

To better understand how different scores cluster, group the data by the "Score" column.

```python
# Show rows grouped by Score, sorted, and limited to 5 rows per group
dg.show(group="Score", sort="Score", rows=5, select="Score,embedding")
```

This view isolates each score group, making it easier to compare embedding distributions across different review ratings.

## Next Steps

You have successfully created a Kangas DataGrid with visualized embeddings. To explore a live example of this datagrid, visit:  
[https://kangas.comet.com/?datagrid=/data/openai_embeddings.datagrid](https://kangas.comet.com/?datagrid=/data/openai_embeddings.datagrid)

Kangas provides many additional features for data analysis and visualization. Consider experimenting with different projections, adding more metadata columns, or integrating Kangas into your own data pipelines for enhanced insight into your embedding data.