# Zero-Shot Sentiment Classification with Embeddings

## Overview
This guide demonstrates how to perform zero-shot sentiment classification using text embeddings. You'll classify Amazon food reviews as positive or negative without any labeled training data by comparing review embeddings to descriptive class embeddings.

## Prerequisites

Ensure you have the required libraries installed:

```bash
pip install pandas numpy scikit-learn openai
```

## Step 1: Setup and Data Preparation

First, import the necessary libraries and load the dataset containing pre-computed embeddings.

```python
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.metrics import classification_report, PrecisionRecallDisplay

# Define the embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# Load the dataset
datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"
df = pd.read_csv(datafile_path)

# Convert the stored string embeddings back to numpy arrays
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)

# Filter out neutral (3-star) reviews and create binary sentiment labels
df = df[df.Score != 3]
df["sentiment"] = df.Score.replace({1: "negative", 2: "negative", 4: "positive", 5: "positive"})
```

**What we did:**
- Loaded a dataset of 1,000 food reviews with pre-computed embeddings.
- Converted the string-formatted embeddings into NumPy arrays for processing.
- Removed 3-star (neutral) reviews and mapped 1-2 star reviews as `negative` and 4-5 star reviews as `positive`.

## Step 2: Define the Zero-Shot Classification Function

We'll create a reusable function that classifies reviews by comparing their embeddings to embeddings of class descriptions.

```python
from utils.embeddings_utils import cosine_similarity, get_embedding

def evaluate_embeddings_approach(labels=['negative', 'positive'], model=EMBEDDING_MODEL):
    """
    Perform zero-shot classification using label embeddings.

    Args:
        labels: List of label descriptions (e.g., ['negative', 'positive'])
        model: The embedding model to use
    """
    # Generate embeddings for each label description
    label_embeddings = [get_embedding(label, model=model) for label in labels]

    def label_score(review_embedding, label_embeddings):
        """
        Calculate the classification score for a review.
        Positive score indicates positive sentiment.
        """
        return (cosine_similarity(review_embedding, label_embeddings[1]) - 
                cosine_similarity(review_embedding, label_embeddings[0]))

    # Calculate scores and predictions for all reviews
    probas = df["embedding"].apply(lambda x: label_score(x, label_embeddings))
    preds = probas.apply(lambda x: 'positive' if x > 0 else 'negative')

    # Print classification report
    report = classification_report(df.sentiment, preds)
    print(report)

    # Plot precision-recall curve
    display = PrecisionRecallDisplay.from_predictions(df.sentiment, probas, pos_label='positive')
    _ = display.ax_.set_title("2-class Precision-Recall curve")
```

**How it works:**
1. **Embed the labels:** Convert the text descriptions of each class into embedding vectors.
2. **Score each review:** For each review embedding, calculate the difference in cosine similarity to the positive label embedding versus the negative label embedding.
3. **Make predictions:** If the score is positive, classify as positive sentiment; otherwise, classify as negative.
4. **Evaluate:** Generate a classification report and precision-recall curve to assess performance.

## Step 3: Initial Classification with Simple Labels

Let's start with the simplest approach using single-word labels.

```python
# Evaluate with simple label names
evaluate_embeddings_approach(labels=['negative', 'positive'], model=EMBEDDING_MODEL)
```

**Output:**
```
              precision    recall  f1-score   support

    negative       0.54      0.92      0.68       136
    positive       0.98      0.87      0.92       789

    accuracy                           0.87       925
   macro avg       0.76      0.89      0.80       925
weighted avg       0.92      0.87      0.89       925
```

**Analysis:**
The classifier already performs well with 87% accuracy, showing strong performance on positive reviews (98% precision) but weaker performance on negative reviews (54% precision). The high recall for negative reviews (92%) indicates it's catching most negative cases but with many false positives.

## Step 4: Improve Performance with Descriptive Labels

Now let's use more descriptive label names that provide better context for the embedding model.

```python
# Evaluate with descriptive label names
evaluate_embeddings_approach(
    labels=['An Amazon review with a negative sentiment.', 
            'An Amazon review with a positive sentiment.']
)
```

**Output:**
```
              precision    recall  f1-score   support

    negative       0.76      0.96      0.85       136
    positive       0.99      0.95      0.97       789

    accuracy                           0.95       925
   macro avg       0.88      0.96      0.91       925
weighted avg       0.96      0.95      0.95       925
```

**Key Improvement:**
- **Accuracy increased from 87% to 95%**
- **Negative review precision improved from 54% to 76%**
- **F1-score for negative reviews increased from 0.68 to 0.85**

The more descriptive labels provide better context for the embedding model, resulting in significantly improved classification performance across all metrics.

## Step 5: Understanding the Precision-Recall Curve

The precision-recall curve generated by our function shows the trade-off between precision and recall at different classification thresholds. You can adjust the threshold to favor either precision or recall based on your application needs:

- **High-precision threshold:** Use when false positives are costly (e.g., filtering inappropriate content)
- **High-recall threshold:** Use when you need to catch all instances of a class (e.g., safety-critical applications)

## Conclusion

Zero-shot classification with embeddings provides a powerful approach to sentiment analysis without requiring labeled training data. Key takeaways:

1. **Descriptive labels matter:** Using context-rich label descriptions ("An Amazon review with a positive sentiment") significantly outperforms simple labels ("positive").

2. **Embedding similarity works:** The cosine similarity between review embeddings and label embeddings effectively captures semantic relationships.

3. **Adjustable thresholds:** The continuous prediction scores allow you to tune the precision-recall trade-off for your specific use case.

This approach can be extended to other classification tasks by simply changing the label descriptions, making it a versatile tool for various text classification problems.