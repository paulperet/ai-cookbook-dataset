# Detecting Issues in a Text Dataset with Cleanlab

In this tutorial, you will use Cleanlab to automatically detect various data quality issues in a text classification dataset. We'll work with a subset of the Banking77-OOS dataset, which contains customer service requests from an online bank, but you can apply the same workflow to your own text data.

Cleanlab helps identify problematic examples such as:
- **Mislabeled data** (label errors)
- **Out-of-scope examples** (outliers)
- **Near-duplicate entries**

By filtering or correcting these issues before model training, you can build more reliable and performant machine learning systems.

## Prerequisites

First, install the required libraries. If you're running this in a Colab notebook, consider enabling a GPU (Runtime > Change runtime type > Hardware accelerator > GPU) for faster embedding computation.

```bash
pip install -U scikit-learn sentence-transformers datasets
pip install -U "cleanlab[datalab]"
```

Now, import the necessary modules.

```python
import re
import string
import random
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from cleanlab import Datalab

# Set seeds for reproducibility
SEED = 123456
np.random.seed(SEED)
random.seed(SEED)
pd.set_option("display.max_colwidth", None)
```

## Step 1: Load and Explore the Dataset

We'll load the first 1,000 examples from the Banking77 dataset.

```python
dataset = load_dataset("PolyAI/banking77", split="train")
data = pd.DataFrame(dataset[:1000])
data.head()
```

The dataset contains two columns:
- `text`: The customer service request utterance.
- `label`: The intent category (label) for that example.

Let's extract the raw texts and labels into separate arrays and check the number of unique classes.

```python
raw_texts, labels = data["text"].values, data["label"].values
num_classes = len(set(labels))

print(f"This dataset has {num_classes} classes.")
print(f"Classes: {set(labels)}")
```

**Output:**
```
This dataset has 7 classes.
Classes: {32, 34, 36, 11, 13, 46, 17}
```

You can inspect any example by its index:

```python
i = 1  # Change this to view other examples
print(f"Example Label: {labels[i]}")
print(f"Example Text: {raw_texts[i]}")
```

**Output:**
```
Example Label: 11
Example Text: What can I do if my card still hasn't arrived after 2 weeks?
```

> **Bringing Your Own Data?**  
> You can replace `raw_texts` and `labels` with your own text data and labels at this point. The rest of the tutorial will work the same way.

## Step 2: Generate Text Embeddings

To use the text data with machine learning models, we first convert each utterance into a dense vector (embedding). We'll use a pretrained Transformer model from the Sentence Transformers library.

```python
transformer = SentenceTransformer('google/electra-small-discriminator')
text_embeddings = transformer.encode(raw_texts)
```

The `text_embeddings` array now contains a numeric vector for each text example. Our subsequent classifier will operate on these embeddings.

## Step 3: Train a Classifier and Obtain Out-of-Sample Predictions

Cleanlab requires **out-of-sample predicted probabilities** to reliably detect label issues. This means we need predictions for each datapoint from a model that was *not* trained on that datapoint.

We'll use a simple Logistic Regression model and generate out-of-sample predictions via cross-validation.

```python
model = LogisticRegression(max_iter=400)
pred_probs = cross_val_predict(model, text_embeddings, labels, method="predict_proba")
```

The `pred_probs` array contains the predicted class probabilities for each example, with columns ordered lexicographically by class name (as required by Cleanlab).

## Step 4: Audit the Dataset with Cleanlab

Now we'll use Cleanlab's `Datalab` to audit our data. First, package the data into a dictionary.

```python
data_dict = {"texts": raw_texts, "labels": labels}
```

Create a `Datalab` instance and run the audit by providing the predicted probabilities and feature embeddings. The more information you provide, the more types of issues Cleanlab can detect.

```python
lab = Datalab(data_dict, label_name="labels")
lab.find_issues(pred_probs=pred_probs, features=text_embeddings)
```

The audit process will output a summary of the issues found. For example:

```
Finding null issues ...
Finding label issues ...
Finding outlier issues ...
Fitting OOD estimator based on provided features ...
Finding near_duplicate issues ...
Finding non_iid issues ...
Finding class_imbalance issues ...
Finding underperforming_group issues ...

Audit complete. 62 issues found in the dataset.
```

## Step 5: Review the Audit Report

Generate a detailed report to see what issues were discovered.

```python
lab.report()
```

The report will summarize the different types of issues, their counts, and show the most severe examples for each issue type. Here’s an example summary:

```
Here is a summary of the different kinds of issues found in the data:

    issue_type       num_issues
       outlier               37
near_duplicate               14
         label               10
       non_iid                1

Dataset Information: num_examples: 1000, num_classes: 7
```

The report includes detailed sections for each issue type (outlier, near_duplicate, label, non_iid), explaining what they mean and showing the most affected examples.

## Step 6: Examine Label Issues in Detail

Let's focus on the label issues (potential mislabelings). Get the detailed DataFrame for label issues.

```python
label_issues = lab.get_issues("label")
label_issues.head()
```

This DataFrame includes:
- `is_label_issue`: Boolean flag indicating if the example is likely mislabeled.
- `label_score`: Quality score between 0 and 1 (lower = more likely mislabeled).
- `given_label`: The original label.
- `predicted_label`: The label suggested by Cleanlab's analysis.

Filter to see only the examples flagged as label issues and identify the top 5 most likely errors.

```python
identified_label_issues = label_issues[label_issues["is_label_issue"] == True]
lowest_quality_labels = label_issues["label_score"].argsort()[:5].to_numpy()

print(
    f"Cleanlab found {len(identified_label_issues)} potential label errors in the dataset.\n"
    f"Here are indices of the top 5 most likely errors: \n {lowest_quality_labels}"
)
```

**Output:**
```
Cleanlab found 10 potential label errors in the dataset.
Here are indices of the top 5 most likely errors: 
 [379 100 300 485 159]
```

## Step 7: Inspect the Most Likely Label Errors

Create a DataFrame to compare the original text, given label, and Cleanlab's suggested label for the top candidates.

```python
data_with_suggested_labels = pd.DataFrame(
    {"text": raw_texts, "given_label": labels, "suggested_label": label_issues["predicted_label"]}
)
data_with_suggested_labels.iloc[lowest_quality_labels]
```

This will display the top 5 examples that Cleanlab suspects are mislabeled, allowing you to manually review and decide whether to correct them.

## Summary

In this tutorial, you learned how to:
1. Load a text classification dataset.
2. Generate text embeddings using a pretrained Transformer model.
3. Train a classifier and obtain out-of-sample predicted probabilities.
4. Use Cleanlab's `Datalab` to audit the dataset for various issues.
5. Identify and review the most likely label errors.

You can now apply this same workflow to your own text datasets to improve data quality before model training. For more advanced usage, refer to the [Cleanlab documentation](https://docs.cleanlab.ai/).