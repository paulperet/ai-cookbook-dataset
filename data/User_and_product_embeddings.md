# Guide: Building and Evaluating User & Product Embeddings for Recommendation

This tutorial walks you through creating user and product embeddings from review text and evaluating their predictive power for review scores. We'll use a dataset of fine food reviews with precomputed text embeddings.

## Prerequisites

Ensure you have the required libraries installed and the dataset file available.

```bash
pip install pandas numpy scikit-learn matplotlib statsmodels
```

The dataset `fine_food_reviews_with_embeddings_1k.csv` should be in a `data/` directory. This file contains review text and its corresponding embeddings, which you can generate by following the [Get_embeddings_from_dataset Notebook](Get_embeddings_from_dataset.ipynb).

## Step 1: Load and Prepare the Data

First, import the necessary libraries and load the dataset.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ast import literal_eval

# Load the dataset
df = pd.read_csv('data/fine_food_reviews_with_embeddings_1k.csv', index_col=0)

# Preview the data
print(df.head(2))
```

The dataset includes a column named `embedding` containing string representations of vector embeddings. We'll convert these to NumPy arrays for processing.

```python
# Convert the embedding strings to NumPy arrays
df['babbage_similarity'] = df["embedding"].apply(literal_eval).apply(np.array)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df, df.Score, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
```

## Step 2: Calculate User and Product Embeddings

We create embeddings for each user and product by averaging all their associated review embeddings from the training set.

```python
# Calculate average embeddings per user and per product
user_embeddings = X_train.groupby('UserId').babbage_similarity.apply(np.mean)
prod_embeddings = X_train.groupby('ProductId').babbage_similarity.apply(np.mean)

print(f"Number of unique users: {len(user_embeddings)}")
print(f"Number of unique products: {len(prod_embeddings)}")
```

You'll notice that most users and products appear only once in this sample dataset. In a larger dataset, these embeddings would capture more nuanced patterns.

## Step 3: Evaluate Embeddings on the Test Set

Now, we'll assess how well these embeddings predict review scores in the unseen test set. We compute the cosine similarity between each user-product pair.

Define a helper function to calculate cosine similarity. Ensure you have the `cosine_similarity` function available (e.g., from `utils.embeddings_utils`).

```python
from utils.embeddings_utils import cosine_similarity

def evaluate_single_match(row):
    """
    For a given review (row), fetch the user and product embeddings
    and compute their cosine similarity.
    """
    user_id = row.UserId
    product_id = row.ProductId
    try:
        user_embedding = user_embeddings[user_id]
        product_embedding = prod_embeddings[product_id]
        similarity = cosine_similarity(user_embedding, product_embedding)
        return similarity
    except KeyError:
        # If user or product isn't in the training set, return NaN
        return np.nan

# Apply the function to each row in the test set
X_test['cosine_similarity'] = X_test.apply(evaluate_single_match, axis=1)

# Normalize similarities to percentiles for easier interpretation
X_test['percentile_cosine_similarity'] = X_test.cosine_similarity.rank(pct=True)

# Preview the results
print(X_test[['UserId', 'ProductId', 'Score', 'cosine_similarity', 'percentile_cosine_similarity']].head())
```

## Step 4: Analyze the Relationship Between Similarity and Review Score

Let's quantify how well the cosine similarity predicts the actual review score (1 to 5 stars).

### 4.1 Calculate Correlation

Compute the correlation between the percentile similarity and the review score.

```python
correlation = X_test[['percentile_cosine_similarity', 'Score']].corr().values[0,1]
print(f'Correlation between user-product similarity percentile and review score: {100*correlation:.2f}%')
```

### 4.2 Visualize the Distribution

Create a boxplot to see how similarity percentiles distribute across different review scores.

```python
import matplotlib.pyplot as plt

# Create a boxplot grouped by review score
X_test.boxplot(column='percentile_cosine_similarity', by='Score')
plt.title('Cosine Similarity Percentile by Review Score')
plt.suptitle('')  # Remove default subtitle
plt.xlabel('Review Score (Stars)')
plt.ylabel('Cosine Similarity Percentile')
plt.show()
```

## Interpretation

You should observe a weak positive trend: higher cosine similarity between user and product embeddings tends to correlate with higher review scores. This suggests that embeddings derived from review text can partially predict user satisfaction—even before the product is received.

While the signal is not strong, it provides a complementary feature to traditional collaborative filtering methods. Incorporating these text-based embeddings could enhance existing recommendation systems.

## Next Steps

- Experiment with larger datasets to see if the signal strengthens.
- Combine these embeddings with other features (e.g., user demographics, product categories) in a machine learning model.
- Explore different aggregation methods for creating user/product embeddings (e.g., weighted averages, attention mechanisms).

By following this guide, you've built a foundational system for leveraging text embeddings in recommendation tasks.