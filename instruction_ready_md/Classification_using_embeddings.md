# Text Classification with Embeddings: A Practical Guide

This guide demonstrates how to perform text classification using embeddings. While fine-tuned models often outperform embedding-based approaches for classification tasks, this method provides a solid baseline and is particularly useful when you have limited labeled data. Here, we'll predict a food review's star rating (1-5) based on the text embedding.

## Prerequisites

First, ensure you have the necessary libraries installed and data prepared.

```bash
pip install pandas numpy scikit-learn
```

## Step 1: Load and Prepare the Data

We'll load a dataset of food reviews that already includes pre-computed embeddings. The dataset should be split into training and testing sets to evaluate performance on unseen data.

```python
import pandas as pd
import numpy as np
from ast import literal_eval

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset containing review text, scores, and embeddings
datafile_path = "data/fine_food_reviews_with_embeddings_1k.csv"
df = pd.read_csv(datafile_path)

# Convert the embedding column from string format to a NumPy array
df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)
```

## Step 2: Split Data into Training and Testing Sets

Separate the embeddings (features) and the star ratings (labels), then split them into training (80%) and testing (20%) subsets.

```python
# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    list(df.embedding.values),  # Features: embedding vectors
    df.Score,                    # Labels: star ratings (1-5)
    test_size=0.2,
    random_state=42
)
```

## Step 3: Train a Random Forest Classifier

We'll use a Random Forest classifier, a robust ensemble method suitable for this multi-class classification task.

```python
# Initialize and train the classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Generate predictions and probability estimates on the test set
preds = clf.predict(X_test)
probas = clf.predict_proba(X_test)
```

## Step 4: Evaluate Model Performance

Examine the classification report to understand precision, recall, and F1-score for each class.

```python
# Print a detailed classification report
report = classification_report(y_test, preds)
print(report)
```

**Output:**
```
              precision    recall  f1-score   support

           1       0.90      0.45      0.60        20
           2       1.00      0.38      0.55         8
           3       1.00      0.18      0.31        11
           4       0.88      0.26      0.40        27
           5       0.76      1.00      0.86       134

    accuracy                           0.78       200
   macro avg       0.91      0.45      0.54       200
weighted avg       0.81      0.78      0.73       200
```

**Interpretation:**
The model achieves 78% overall accuracy. Performance is strongest for 5-star reviews (high recall and F1-score), which is expected as they are the most frequent class in the dataset. Mid-range ratings (2-4 stars) show lower recall, indicating the model struggles to distinguish these more nuanced categories. This could be due to dataset imbalance or inherent subjectivity in how users assign intermediate scores.

## Step 5: Visualize Precision-Recall Trade-offs (Optional)

For a deeper evaluation, you can plot precision-recall curves for each class. This requires a custom utility function.

```python
# If you have the utility module available
from utils.embeddings_utils import plot_multiclass_precision_recall

plot_multiclass_precision_recall(probas, y_test, [1, 2, 3, 4, 5], clf)
```

**Expected Insight:**
The visualization will likely confirm that 5-star and 1-star reviews are easier for the model to predict accurately, while distinguishing between 2-4 stars remains challenging. Increasing the volume of training data, especially for the under-represented mid-range classes, could improve performance.

## Summary

You've successfully built a text classifier using embeddings and a Random Forest model. This approach provides a strong baseline, though for production tasks with sufficient data, fine-tuning a dedicated language model may yield better results. Key takeaways:

1.  **Embeddings as Features:** Text embeddings effectively capture semantic meaning for use in traditional ML models.
2.  **Class Imbalance Matters:** Performance is best on frequent classes (5-star reviews); collecting more balanced data can improve results.
3.  **Subjectivity is Challenging:** Nuanced distinctions (e.g., between 3 and 4 stars) are harder to model, reflecting real-world rating ambiguity.

To extend this work, consider experimenting with other classifiers (like SVM or gradient boosting), applying techniques to handle class imbalance, or incorporating additional metadata features alongside embeddings.