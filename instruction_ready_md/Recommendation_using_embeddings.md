# Building a Content Recommendation System with Embeddings

This guide demonstrates how to build a content recommendation system using text embeddings and nearest neighbor search. You'll learn to find similar news articles from a dataset, a technique applicable to product recommendations, content discovery, and search enhancement.

## Prerequisites

Ensure you have the required libraries installed:

```bash
pip install pandas openai scikit-learn plotly
```

## 1. Setup and Imports

First, import the necessary libraries and define the embedding model.

```python
import pandas as pd
import pickle
from utils.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)

EMBEDDING_MODEL = "text-embedding-3-small"
```

## 2. Load the Dataset

We'll use a sample from the AG News Corpus. Load the dataset and examine its structure.

```python
# Load data
dataset_path = "data/AG_news_samples.csv"
df = pd.read_csv(dataset_path)

# Display the first few rows
n_examples = 5
print(df.head(n_examples))
```

Let's view the full content of these examples:

```python
# Print title, description, and label for each example
for idx, row in df.head(n_examples).iterrows():
    print("")
    print(f"Title: {row['title']}")
    print(f"Description: {row['description']}")
    print(f"Label: {row['label']}")
```

## 3. Create an Embedding Cache

To avoid recomputing embeddings repeatedly, implement a caching system that stores embeddings locally.

```python
# Set path to embedding cache
embedding_cache_path = "data/recommendations_embeddings_cache.pkl"

# Load existing cache or create a new one
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}

# Save cache to disk
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)

def embedding_from_string(
    string: str,
    model: str = EMBEDDING_MODEL,
    embedding_cache=embedding_cache
) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]
```

Test the caching function:

```python
# Get an example embedding
example_string = df["description"].values[0]
print(f"\nExample string: {example_string}")

example_embedding = embedding_from_string(example_string)
print(f"\nExample embedding (first 10 dimensions): {example_embedding[:10]}...")
```

## 4. Build the Recommendation Function

Create a function that finds and displays the k-nearest neighbors for a given article.

```python
def print_recommendations_from_strings(
    strings: list[str],
    index_of_source_string: int,
    k_nearest_neighbors: int = 1,
    model=EMBEDDING_MODEL,
) -> list[int]:
    """Print out the k nearest neighbors of a given string."""
    # Get embeddings for all strings
    embeddings = [embedding_from_string(string, model=model) for string in strings]

    # Get the embedding of the source string
    query_embedding = embeddings[index_of_source_string]

    # Calculate distances between source embedding and others
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")
    
    # Get indices of nearest neighbors
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)

    # Print source string
    query_string = strings[index_of_source_string]
    print(f"Source string: {query_string}")
    
    # Print k nearest neighbors
    k_counter = 0
    for i in indices_of_nearest_neighbors:
        # Skip identical matches
        if query_string == strings[i]:
            continue
        # Stop after k recommendations
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1

        # Print similar strings and their distances
        print(
            f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
        String: {strings[i]}
        Distance: {distances[i]:0.3f}"""
        )

    return indices_of_nearest_neighbors
```

## 5. Generate Recommendations

### Example 1: Tony Blair Article

Find articles similar to the first article about Tony Blair.

```python
article_descriptions = df["description"].tolist()

tony_blair_articles = print_recommendations_from_strings(
    strings=article_descriptions,
    index_of_source_string=0,
    k_nearest_neighbors=5,
)
```

The recommendations should include articles mentioning Tony Blair or related topics like London climate change discussions.

### Example 2: NVIDIA Chipset Security Article

Find articles similar to the second article about NVIDIA's secure chipset.

```python
chipset_security_articles = print_recommendations_from_strings(
    strings=article_descriptions,
    index_of_source_string=1,
    k_nearest_neighbors=5,
)
```

Notice the top recommendation has a significantly smaller distance (e.g., 0.11 vs 0.14+), indicating a very similar article about computer security from PC World.

## 6. Visualize Embeddings with t-SNE

To understand how embeddings cluster similar content, visualize them using t-SNE dimensionality reduction.

```python
# Get embeddings for all article descriptions
embeddings = [embedding_from_string(string) for string in article_descriptions]

# Compress embeddings to 2D using t-SNE
tsne_components = tsne_components_from_embeddings(embeddings)

# Get article labels for coloring
labels = df["label"].tolist()

# Create visualization
chart_from_components(
    components=tsne_components,
    labels=labels,
    strings=article_descriptions,
    width=600,
    height=500,
    title="t-SNE components of article descriptions",
)
```

The visualization shows clear clustering by article category, demonstrating that embeddings capture semantic similarity without explicit label information.

## 7. Visualize Nearest Neighbors

Create specialized visualizations highlighting source articles and their recommendations.

```python
def nearest_neighbor_labels(
    list_of_indices: list[int],
    k_nearest_neighbors: int = 5
) -> list[str]:
    """Return a list of labels to color the k nearest neighbors."""
    labels = ["Other" for _ in list_of_indices]
    source_index = list_of_indices[0]
    labels[source_index] = "Source"
    for i in range(k_nearest_neighbors):
        nearest_neighbor_index = list_of_indices[i + 1]
        labels[nearest_neighbor_index] = f"Nearest neighbor (top {k_nearest_neighbors})"
    return labels

# Create labels for our examples
tony_blair_labels = nearest_neighbor_labels(tony_blair_articles, k_nearest_neighbors=5)
chipset_security_labels = nearest_neighbor_labels(chipset_security_articles, k_nearest_neighbors=5)

# Visualize Tony Blair article neighbors
chart_from_components(
    components=tsne_components,
    labels=tony_blair_labels,
    strings=article_descriptions,
    width=600,
    height=500,
    title="Nearest neighbors of the Tony Blair article",
    category_orders={"label": ["Other", "Nearest neighbor (top 5)", "Source"]},
)

# Visualize chipset security article neighbors
chart_from_components(
    components=tsne_components,
    labels=chipset_security_labels,
    strings=article_descriptions,
    width=600,
    height=500,
    title="Nearest neighbors of the chipset security article",
    category_orders={"label": ["Other", "Nearest neighbor (top 5)", "Source"]},
)
```

Note that nearest neighbors in the full 2048-dimensional embedding space may not appear closest in the compressed 2D visualization, as dimensionality reduction discards some information.

## Key Takeaways

1. **Embeddings effectively capture semantic similarity** between text documents
2. **Caching embeddings** significantly improves performance for repeated queries
3. **Cosine distance** works well for measuring similarity between embedding vectors
4. **Visualization helps validate** that embeddings cluster similar content meaningfully
5. **This approach scales** to various recommendation use cases beyond news articles

## Advanced Applications

For production systems, consider:
- Incorporating embeddings as features in machine learning models
- Combining with user behavior data for personalized recommendations
- Using embeddings for cold-start recommendations (new items with no user data)
- Implementing approximate nearest neighbor search for large-scale datasets

The foundation you've built here can be extended to power recommendation engines across e-commerce, content platforms, and knowledge bases.