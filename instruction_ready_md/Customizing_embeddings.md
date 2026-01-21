# Guide: Customizing Embeddings for Task-Specific Similarity

This guide demonstrates how to customize OpenAI embeddings to improve performance on a specific task—in this case, identifying logically entailed sentence pairs. By learning a linear transformation (a matrix) from your data, you can create "custom embeddings" that emphasize features relevant to your use case. This method has been shown to reduce error rates by up to 50% in binary classification tasks.

## Prerequisites

Ensure you have the required libraries installed. You can install them via pip if needed.

```bash
pip install numpy pandas plotly scikit-learn torch openai
```

## Step 1: Imports and Setup

Begin by importing the necessary modules and defining key parameters.

```python
from typing import List, Tuple
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import random
from sklearn.model_selection import train_test_split
import torch

from utils.embeddings_utils import get_embedding, cosine_similarity

# Configuration
embedding_cache_path = "data/snli_embedding_cache.pkl"
default_embedding_engine = "text-embedding-3-small"
num_pairs_to_embed = 1000
local_dataset_path = "data/snli_1.0_train_2k.csv"
```

## Step 2: Define Data Processing Function

You'll need a function to load and preprocess your dataset. The output must be a DataFrame with three columns: `text_1`, `text_2`, and `label` (where `1` indicates similar pairs and `-1` indicates dissimilar pairs).

```python
def process_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the SNLI dataset to extract entailment pairs.
    Adapt this function for your own dataset.
    """
    df["label"] = df["gold_label"]
    df = df[df["label"].isin(["entailment"])]
    df["label"] = df["label"].apply(
        lambda x: {"entailment": 1, "contradiction": -1}[x]
    )
    df = df.rename(columns={"sentence1": "text_1", "sentence2": "text_2"})
    df = df[["text_1", "text_2", "label"]]
    df = df.head(num_pairs_to_embed)
    return df
```

## Step 3: Load and Prepare the Dataset

Load your dataset and apply the preprocessing function.

```python
df = pd.read_csv(local_dataset_path)
df = process_input_data(df)
print(df.head())
```

**Expected Output:**
```
                                              text_1  ... label
2  A person on a horse jumps over a broken down...  ...     1
4                Children smiling and waving at camera  ...     1
7  A boy is jumping on skateboard in the middle...  ...     1
14         Two blond women are hugging one another.  ...     1
17  A few people in a restaurant setting, one o...  ...     1
```

## Step 4: Split Data into Training and Test Sets

Split the data before generating synthetic examples to avoid data leakage.

```python
test_fraction = 0.5
random_seed = 123
train_df, test_df = train_test_split(
    df, test_size=test_fraction, stratify=df["label"], random_state=random_seed
)
train_df.loc[:, "dataset"] = "train"
test_df.loc[:, "dataset"] = "test"
```

## Step 5: Generate Synthetic Negative Pairs

If your dataset contains only positive examples (like entailment pairs), you can generate synthetic negatives by combining sentences from different pairs.

```python
def dataframe_of_negatives(dataframe_of_positives: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame of negative pairs from positive pairs."""
    texts = set(dataframe_of_positives["text_1"].values) | set(
        dataframe_of_positives["text_2"].values
    )
    all_pairs = {(t1, t2) for t1 in texts for t2 in texts if t1 < t2}
    positive_pairs = set(
        tuple(text_pair)
        for text_pair in dataframe_of_positives[["text_1", "text_2"]].values
    )
    negative_pairs = all_pairs - positive_pairs
    df_of_negatives = pd.DataFrame(list(negative_pairs), columns=["text_1", "text_2"])
    df_of_negatives["label"] = -1
    return df_of_negatives

# Generate negatives for both training and test sets
negatives_per_positive = 1
train_df_negatives = dataframe_of_negatives(train_df)
train_df_negatives["dataset"] = "train"
test_df_negatives = dataframe_of_negatives(test_df)
test_df_negatives["dataset"] = "test"

# Combine positives and sampled negatives
train_df = pd.concat([
    train_df,
    train_df_negatives.sample(
        n=len(train_df) * negatives_per_positive, random_state=random_seed
    )
])
test_df = pd.concat([
    test_df,
    test_df_negatives.sample(
        n=len(test_df) * negatives_per_positive, random_state=random_seed
    )
])

df = pd.concat([train_df, test_df])
```

## Step 6: Compute and Cache Embeddings

To avoid recomputing embeddings, use a local cache. This function retrieves embeddings from the cache or calls the OpenAI API if needed.

```python
# Load existing cache or download a precomputed one
try:
    with open(embedding_cache_path, "rb") as f:
        embedding_cache = pickle.load(f)
except FileNotFoundError:
    precomputed_embedding_cache_path = "https://cdn.openai.com/API/examples/data/snli_embedding_cache.pkl"
    embedding_cache = pd.read_pickle(precomputed_embedding_cache_path)

def get_embedding_with_cache(
    text: str,
    engine: str = default_embedding_engine,
    embedding_cache: dict = embedding_cache,
    embedding_cache_path: str = embedding_cache_path,
) -> list:
    """Retrieve embedding from cache or compute and store it."""
    if (text, engine) not in embedding_cache.keys():
        embedding_cache[(text, engine)] = get_embedding(text, engine)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(text, engine)]

# Compute embeddings for both text columns
for column in ["text_1", "text_2"]:
    df[f"{column}_embedding"] = df[column].apply(get_embedding_with_cache)

# Compute cosine similarity between each pair of embeddings
df["cosine_similarity"] = df.apply(
    lambda row: cosine_similarity(row["text_1_embedding"], row["text_2_embedding"]),
    axis=1,
)
```

## Step 7: Evaluate Baseline Similarity

Before customization, evaluate the baseline accuracy of using raw cosine similarity as a predictor.

```python
def accuracy_and_se(cosine_similarity: float, labeled_similarity: int) -> Tuple[float]:
    """Compute maximum accuracy and standard error over all possible thresholds."""
    accuracies = []
    for threshold_thousandths in range(-1000, 1000, 1):
        threshold = threshold_thousandths / 1000
        total = 0
        correct = 0
        for cs, ls in zip(cosine_similarity, labeled_similarity):
            total += 1
            if cs > threshold:
                prediction = 1
            else:
                prediction = -1
            if prediction == ls:
                correct += 1
        accuracy = correct / total
        accuracies.append(accuracy)
    a = max(accuracies)
    n = len(cosine_similarity)
    standard_error = (a * (1 - a) / n) ** 0.5
    return a, standard_error

# Display similarity distributions and baseline accuracy
for dataset in ["train", "test"]:
    data = df[df["dataset"] == dataset]
    a, se = accuracy_and_se(data["cosine_similarity"], data["label"])
    print(f"{dataset} accuracy: {a:0.1%} ± {1.96 * se:0.1%}")
```

**Expected Output:**
```
train accuracy: 89.1% ± 2.4%
test accuracy: 88.8% ± 2.4%
```

## Step 8: Define Helper Functions for Custom Embeddings

Create functions to apply a learned matrix to embeddings and compute new similarities.

```python
def embedding_multiplied_by_matrix(
    embedding: List[float], matrix: torch.tensor
) -> np.array:
    """Apply a linear transformation to an embedding vector."""
    embedding_tensor = torch.tensor(embedding).float()
    modified_embedding = embedding_tensor @ matrix
    return modified_embedding.detach().numpy()

def apply_matrix_to_embeddings_dataframe(matrix: torch.tensor, df: pd.DataFrame):
    """Add custom embeddings and their cosine similarities to the DataFrame."""
    for column in ["text_1_embedding", "text_2_embedding"]:
        df[f"{column}_custom"] = df[column].apply(
            lambda x: embedding_multiplied_by_matrix(x, matrix)
        )
    df["cosine_similarity_custom"] = df.apply(
        lambda row: cosine_similarity(
            row["text_1_embedding_custom"], row["text_2_embedding_custom"]
        ),
        axis=1,
    )
```

## Step 9: Optimize the Transformation Matrix

Train a matrix to minimize the mean squared error between predicted and actual similarity labels.

```python
def optimize_matrix(
    modified_embedding_length: int = 2048,
    batch_size: int = 100,
    max_epochs: int = 100,
    learning_rate: float = 100.0,
    dropout_fraction: float = 0.0,
    df: pd.DataFrame = df,
    print_progress: bool = True,
) -> torch.tensor:
    """
    Learn a matrix that improves similarity prediction on the training set.
    Returns the optimized matrix.
    """
    run_id = random.randint(0, 2 ** 31 - 1)

    # Convert DataFrame columns to PyTorch tensors
    def tensors_from_dataframe(
        df: pd.DataFrame,
        embedding_column_1: str,
        embedding_column_2: str,
        similarity_label_column: str,
    ) -> Tuple[torch.tensor]:
        e1 = np.stack(np.array(df[embedding_column_1].values))
        e2 = np.stack(np.array(df[embedding_column_2].values))
        s = np.stack(np.array(df[similarity_label_column].astype("float").values))

        e1 = torch.from_numpy(e1).float()
        e2 = torch.from_numpy(e2).float()
        s = torch.from_numpy(s).float()
        return e1, e2, s

    # Prepare training and test tensors
    e1_train, e2_train, s_train = tensors_from_dataframe(
        df[df["dataset"] == "train"], "text_1_embedding", "text_2_embedding", "label"
    )
    e1_test, e2_test, s_test = tensors_from_dataframe(
        df[df["dataset"] == "test"], "text_1_embedding", "text_2_embedding", "label"
    )

    # Create DataLoader for batching
    dataset = torch.utils.data.TensorDataset(e1_train, e2_train, s_train)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Define model: cosine similarity of transformed embeddings
    def model(embedding_1, embedding_2, matrix, dropout_fraction=dropout_fraction):
        e1 = torch.nn.functional.dropout(embedding_1, p=dropout_fraction)
        e2 = torch.nn.functional.dropout(embedding_2, p=dropout_fraction)
        modified_embedding_1 = e1 @ matrix
        modified_embedding_2 = e2 @ matrix
        similarity = torch.nn.functional.cosine_similarity(
            modified_embedding_1, modified_embedding_2
        )
        return similarity

    # Loss function: Mean Squared Error
    def mse_loss(predictions, targets):
        difference = predictions - targets
        return torch.sum(difference * difference) / difference.numel()

    # Initialize a random matrix with gradients enabled
    embedding_length = len(df["text_1_embedding"].values[0])
    matrix = torch.randn(
        embedding_length, modified_embedding_length, requires_grad=True
    )

    # Training loop
    for epoch in range(1, 1 + max_epochs):
        for a, b, actual_similarity in train_loader:
            predicted_similarity = model(a, b, matrix)
            loss = mse_loss(predicted_similarity, actual_similarity)
            loss.backward()
            with torch.no_grad():
                matrix -= matrix.grad * learning_rate
                matrix.grad.zero_()

        # Evaluate on test set
        test_predictions = model(e1_test, e2_test, matrix)
        test_loss = mse_loss(test_predictions, s_test)

        # Apply current matrix to compute custom similarities
        apply_matrix_to_embeddings_dataframe(matrix, df)

        # Calculate and display accuracy
        if print_progress:
            for dataset_name in ["train", "test"]:
                data = df[df["dataset"] == dataset_name]
                a, se = accuracy_and_se(data["cosine_similarity_custom"], data["label"])
                print(
                    f"Epoch {epoch}/{max_epochs}: {dataset_name} accuracy: {a:0.1%} ± {1.96 * se:0.1%}"
                )

    return matrix

# Run optimization
optimized_matrix = optimize_matrix()
```

During training, you will see output like:
```
Epoch 1/100: train accuracy: 92.3% ± 2.1%
Epoch 1/100: test accuracy: 91.5% ± 2.3%
...
```

## Step 10: Apply the Optimized Matrix

Finally, use the learned matrix to transform new embeddings for inference.

```python
# Example: compute custom similarity for a new pair
new_text_1 = "A child is playing with a dog."
new_text_2 = "An animal is outdoors."

new_embedding_1 = get_embedding_with_cache(new_text_1)
new_embedding_2 = get_embedding_with_cache(new_text_2)

custom_embedding_1 = embedding_multiplied_by_matrix(new_embedding_1, optimized_matrix)
custom_embedding_2 = embedding_multiplied_by_matrix(new_embedding_2, optimized_matrix)

custom_similarity = cosine_similarity(custom_embedding_1, custom_embedding_2)
print(f"Custom cosine similarity: {custom_similarity:.4f}")
```

## Summary

You have successfully learned a linear transformation to tailor OpenAI embeddings to your specific similarity task. By following these steps—processing your data, generating synthetic examples, caching embeddings, and optimizing a projection matrix—you can significantly improve the accuracy of similarity-based predictions. This method is flexible and can be adapted to various binary or multiclass classification and clustering problems.