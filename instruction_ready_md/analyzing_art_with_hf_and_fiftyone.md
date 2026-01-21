# Analyzing Artistic Styles with Multimodal Embeddings: A Step-by-Step Guide

In this tutorial, you will learn how to analyze artistic styles in images using multimodal embeddings and computed attributes. We'll use the WikiArt dataset from Hugging Face, load it into FiftyOne for data management, and perform several analyses: similarity search, clustering, uniqueness scoring, and basic image quality assessment.

## Prerequisites

First, install the required libraries. We recommend using a virtual environment.

```bash
pip install -U transformers huggingface_hub fiftyone umap-learn
```

For faster downloads from Hugging Face Hub, install `hf-transfer`:

```bash
pip install hf-transfer
```

Then, enable it by setting an environment variable in your Python script or notebook:

```python
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
```

**Note:** This guide was tested with `transformers==4.40.0`, `huggingface_hub==0.22.2`, and `fiftyone==0.23.8`.

Now, import the necessary modules.

```python
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
from fiftyone import ViewField as F
import fiftyone.utils.huggingface as fouh
```

## Step 1: Load the WikiArt Dataset

We'll load the first 1,000 samples from the WikiArt dataset on Hugging Face Hub directly into a FiftyOne dataset using its Parquet format.

```python
dataset = fouh.load_from_hub(
    "huggan/wikiart",          # Repository ID
    format="parquet",          # Data format
    classification_fields=["artist", "style", "genre"], # Fields to treat as classifications
    max_samples=1000,          # Limit samples for speed
    name="wikiart",            # Dataset name in FiftyOne
)
```

Print a summary to inspect the dataset structure.

```python
print(dataset)
```

You can visualize the dataset in the FiftyOne App.

```python
session = fo.launch_app(dataset)
```

Let's list the unique artists in our subset.

```python
artists = dataset.distinct("artist.label")
print(artists)
```

## Step 2: Perform Similarity Search with Multimodal Embeddings

To find visually similar artwork, we'll generate embeddings using a pre-trained CLIP model. These embeddings enable both image-to-image and text-to-image (semantic) search.

Compute embeddings and build a similarity index using FiftyOne Brain.

```python
fob.compute_similarity(
    dataset,
    model="zero-shot-classification-transformer-torch", # Model type from zoo
    name_or_path="openai/clip-vit-base-patch32",        # Hugging Face model ID
    embeddings="clip_embeddings",                       # Field name for embeddings
    brain_key="clip_sim",                               # Key for the index
    batch_size=32,
)
```

**Alternative:** You can load the model directly from 🤗 Transformers and pass the object.

```python
from transformers import CLIPModel
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
fob.compute_similarity(
    dataset,
    model=model,
    embeddings="clip_embeddings",
    brain_key="clip_sim"
)
```

**How to Use:**
1.  Refresh the FiftyOne App (`session`).
2.  Select an image in the grid.
3.  Click the "photo" icon to see its most similar images based on the CLIP embeddings.

**Semantic Search:** Click the search icon in the app and enter a text query (e.g., "pastel trees"). The CLIP model will embed your text and find the most similar images in the dataset.

## Step 3: Visualize and Cluster Artistic Styles

Let's explore underlying patterns in the data using dimensionality reduction and clustering.

### 3.1 Dimensionality Reduction with UMAP

Reduce the 512-dimensional CLIP embeddings to 2D for visualization.

```python
fob.compute_visualization(
    dataset,
    embeddings="clip_embeddings",
    method="umap",
    brain_key="clip_vis"
)
```

In the FiftyOne App, open the embeddings panel. You can color the 2D points by fields like `artist` or `genre` to see how well these attributes are captured by the embeddings.

### 3.2 Clustering with K-Means

To group images algorithmically, we'll use the FiftyOne Clustering Plugin. First, download it.

```python
!fiftyone plugins download https://github.com/jacobmarks/clustering-plugin
```

**In the App:**
1.  Refresh the app.
2.  Press the backtick `` ` `` key to open the operator list.
3.  Type "cluster" and select the clustering operator.
4.  In the interactive panel, choose `clip_embeddings` as the field and select K-Means with, for example, 10 clusters.
5.  Run the operator. You can then visualize and analyze the resulting clusters in the app.

## Step 4: Quantify Image Uniqueness

We can assign a uniqueness score to each image based on how similar its embedding is to its neighbors.

Compute uniqueness scores using the existing embeddings.

```python
fob.compute_uniqueness(dataset, embeddings="clip_embeddings")
```

Now, `uniqueness` is a field on your samples. You can sort the dataset to find the most and least unique images.

```python
# View the most unique images
most_unique_view = dataset.sort_by("uniqueness", reverse=True)
session.view = most_unique_view

# View the least unique images
least_unique_view = dataset.sort_by("uniqueness", reverse=False)
session.view = least_unique_view
```

Let's find which artist, on average, creates the most unique work in our dataset.

```python
artist_unique_scores = {
    artist: dataset.match(F("artist.label") == artist).mean("uniqueness")
    for artist in artists
}

# Sort artists by average uniqueness
sorted_artists = sorted(artist_unique_scores, key=artist_unique_scores.get, reverse=True)

for artist in sorted_artists:
    print(f"{artist}: {artist_unique_scores[artist]}")
```

You can then view a specific artist's work.

```python
kustodiev_view = dataset.match(F("artist.label") == "boris-kustodiev")
session.view = kustodiev_view
```

## Step 5: Analyze Basic Image Qualities

Beyond deep learning features, simple image metrics can provide valuable insights. We'll use the FiftyOne Image Quality Plugin.

Download the plugin.

```python
!fiftyone plugins download https://github.com/jacobmarks/image-quality-issues/
```

**In the App:**
1.  Refresh and open the operator list (`` ` ``).
2.  Type `compute` and select an operator like `compute_brightness`.
3.  Run the operator. A new field (e.g., `brightness`) will be added to your samples.
4.  You can now color your UMAP visualization or filter your dataset by this new metric. Repeat this process for `contrast` and `saturation`.

This allows you to explore correlations between technical image properties and artistic labels like style or genre.

## Next Steps and Resources

You've learned how to use multimodal embeddings for search and clustering, compute uniqueness scores, and analyze basic image qualities. These techniques are broadly applicable to any visual dataset.

**To go further:**
*   **Try a New Dataset:** Load a different dataset from the [Hugging Face Hub](https://docs.voxel51.com/integrations/huggingface.html#loading-datasets-from-the-hub).
*   **Zero-Shot Classification:** Use a vision-language model to categorize images without training. See the [FiftyOne tutorial](https://docs.voxel51.com/tutorials/zero_shot_classification.html).
*   **Image Captioning:** Generate text descriptions for images and analyze those. Explore the [FiftyOne Image Captioning Plugin](https://github.com/jacobmarks/fiftyone-image-captioning-plugin).

### 📚 Key Resources
*   [FiftyOne & Hugging Face Hub Integration](https://docs.voxel51.com/integrations/huggingface.html#huggingface-hub)
*   [FiftyOne & 🤗 Transformers Integration](https://docs.voxel51.com/integrations/huggingface.html#transformers-library)
*   [FiftyOne Vector Search](https://voxel51.com/vector-search/)
*   [Dimensionality Reduction Tutorial](https://docs.voxel51.com/tutorials/dimension_reduction.html)
*   [Clustering Tutorial](https://docs.voxel51.com/tutorials/clustering.html)
*   [Uniqueness Tutorial](https://docs.voxel51.com/tutorials/uniqueness.html)

FiftyOne is an open-source toolkit for building better computer vision datasets and models. Learn more and contribute at the [FiftyOne GitHub repository](https://github.com/voxel51/fiftyone/).