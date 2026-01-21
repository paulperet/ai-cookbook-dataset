# Regression with Embeddings: Predicting Review Scores

This guide demonstrates how to use text embeddings to perform regression—predicting a numerical score—on a dataset of Amazon food reviews. We'll predict a review's star rating (1-5) based solely on the embedding of its text.

## Prerequisites

Ensure you have the required libraries installed. You can install them via pip if needed.

```bash
pip install pandas numpy scikit-learn
```

## Step 1: Import Libraries and Load Data

We begin by importing the necessary libraries and loading our dataset, which contains pre-computed embeddings for 1,000 reviews.

```python
import pandas as pd
import numpy as np
from ast import literal_eval

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset containing embeddings
datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"
df = pd.read_csv(datafile_path)

# Convert the string representation of embeddings back to numpy arrays
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)
```

## Step 2: Prepare Training and Testing Sets

To evaluate our model realistically, we split the data into training and testing subsets. The embeddings serve as our features (`X`), and the `Score` column is our target (`y`).

```python
# Split data into features (embeddings) and target (score)
X_train, X_test, y_train, y_test = train_test_split(
    list(df.embedding.values),
    df.Score,
    test_size=0.2,
    random_state=42
)
```

## Step 3: Train a Random Forest Regressor

We'll use a Random Forest Regressor, a robust ensemble method, to learn the relationship between the embeddings and the review scores.

```python
# Initialize and train the model
rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(X_train, y_train)

# Generate predictions on the test set
preds = rfr.predict(X_test)
```

## Step 4: Evaluate Model Performance

Let's calculate standard regression metrics—Mean Squared Error (MSE) and Mean Absolute Error (MAE)—to quantify how well our model performs.

```python
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)

print(f"text-embedding-3-small performance on 1k Amazon reviews: mse={mse:.2f}, mae={mae:.2f}")
```

**Output:**
```
text-embedding-3-small performance on 1k Amazon reviews: mse=0.65, mae=0.52
```

## Step 5: Establish a Baseline

To contextualize our model's performance, we compare it against a simple baseline: always predicting the mean score of the training set.

```python
# Calculate baseline performance (predicting the mean for every sample)
bmse = mean_squared_error(y_test, np.repeat(y_test.mean(), len(y_test)))
bmae = mean_absolute_error(y_test, np.repeat(y_test.mean(), len(y_test)))

print(f"Dummy mean prediction performance on Amazon reviews: mse={bmse:.2f}, mae={bmae:.2f}")
```

**Output:**
```
Dummy mean prediction performance on Amazon reviews: mse=1.73, mae=1.03
```

## Analysis

Our model achieves an MAE of **0.52**, meaning its predictions are, on average, about half a star off from the actual score. This is a significant improvement over the baseline MAE of **1.03**. In practical terms, this performance is roughly equivalent to predicting half of the reviews perfectly and the other half off by just one star.

## Next Steps

The embeddings have proven effective for regression. You can extend this approach by:
- Training a **classifier** to predict discrete categories (e.g., "positive"/"negative").
- Integrating embeddings as features into an existing machine learning pipeline to encode free-text inputs.

This technique enables you to leverage rich semantic information from text for a variety of predictive tasks.