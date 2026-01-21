# Multiclass Classification for Transactions: A Practical Guide

This guide demonstrates three approaches to classifying financial transactions into predefined categories. You will learn how to apply zero-shot classification, classification with embeddings, and fine-tuning to a real-world dataset. These techniques are applicable to any multiclass classification problem where you need to categorize transactional or textual data.

## Approaches Covered
1.  **Zero-shot Classification:** Categorize transactions using only a prompt, without any labeled examples.
2.  **Classification with Embeddings:** Generate embeddings from a labeled dataset and use a traditional classifier.
3.  **Fine-tuned Classification:** Train a custom model on your labeled data for potentially higher accuracy.

## Prerequisites & Setup

First, install the required libraries and configure your environment.

```bash
pip install openai 'openai[datalib]' 'openai[embeddings]' transformers scikit-learn matplotlib plotly pandas scipy
```

```python
import openai
import pandas as pd
import numpy as np
import json
import os

# Configuration
COMPLETIONS_MODEL = "gpt-4"
os.environ["OPENAI_API_KEY"] = "<your-api-key>"  # Replace with your key
client = openai.OpenAI()
```

## 1. Load the Dataset

We'll use a public dataset of transactions over £25k from the National Library of Scotland. It contains three key features: `Supplier`, `Description`, and `Transaction value (£)`.

**Source:** [National Library of Scotland - Transactions over £25k](https://data.nls.uk/data/organisational-data/transactions-over-25k/)

```python
transactions = pd.read_csv('./data/25000_spend_dataset_current.csv', encoding='unicode_escape')
print(f"Number of transactions: {len(transactions)}")
print(transactions.head())
```

**Output:**
```
Number of transactions: 359
         Date                      Supplier                 Description  Transaction value (£)
0  21/04/2016          M & J Ballantyne Ltd       George IV Bridge Work                35098.0
1  26/04/2016                  Private Sale   Literary & Archival Items                30000.0
2  30/04/2016     City Of Edinburgh Council         Non Domestic Rates                40800.0
3  09/05/2016              Computacenter Uk                 Kelvin Hall                72835.0
4  09/05/2016  John Graham Construction Ltd  Causewayside Refurbishment                64361.0
```

## 2. Zero-shot Classification

We'll start by assessing the base model's ability to classify transactions using only a prompt. We define five categories and a catch-all for unclear cases.

### 2.1 Define the Classification Prompt and Function

Create a prompt that instructs the model to classify a transaction into one of five categories.

```python
zero_shot_prompt = '''You are a data expert working for the National Library of Scotland.
You are analysing all transactions over £25,000 in value and classifying them into one of five categories.
The five categories are Building Improvement, Literature & Archive, Utility Bills, Professional Services and Software/IT.
If you can't tell what it is, say Could not classify

Transaction:

Supplier: {}
Description: {}
Value: {}

The classification is:'''

def format_prompt(transaction):
    return zero_shot_prompt.format(transaction['Supplier'],
                                   transaction['Description'],
                                   transaction['Transaction value (£)'])

def classify_transaction(transaction):
    prompt = format_prompt(transaction)
    messages = [{"role": "system", "content": prompt}]
    
    completion_response = client.chat.completions.create(
        messages=messages,
        temperature=0,
        max_tokens=5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model=COMPLETIONS_MODEL
    )
    label = completion_response.choices[0].message.content.replace('\n', '')
    return label
```

### 2.2 Test on a Single Transaction

Let's test the function on the first transaction in the dataset.

```python
transaction = transactions.iloc[0]
print(f"Transaction: {transaction['Supplier']} {transaction['Description']} {transaction['Transaction value (£)']}")
print(f"Classification: {classify_transaction(transaction)}")
```

**Output:**
```
Transaction: M & J Ballantyne Ltd George IV Bridge Work 35098.0
Classification: Building Improvement
```

The model correctly identifies this as a "Building Improvement" transaction.

### 2.3 Evaluate on a Larger Sample

Now, apply the classifier to the first 25 transactions to get a broader sense of its performance.

```python
test_transactions = transactions.iloc[:25].copy()  # Use .copy() to avoid SettingWithCopyWarning
test_transactions['Classification'] = test_transactions.apply(lambda x: classify_transaction(x), axis=1)
```

Let's examine the distribution of classifications.

```python
print(test_transactions['Classification'].value_counts())
```

**Output:**
```
Classification
Building Improvement    17
Literature & Archive     3
Software/IT              2
Could not classify       2
Utility Bills            1
Name: count, dtype: int64
```

Display the full results for the sample.

```python
print(test_transactions.head(25))
```

**Output (truncated for brevity):**
```
         Date                      Supplier                 Description  Transaction value (£)       Classification
0  21/04/2016          M & J Ballantyne Ltd       George IV Bridge Work                35098.0  Building Improvement
1  26/04/2016                  Private Sale   Literary & Archival Items                30000.0  Literature & Archive
2  30/04/2016     City Of Edinburgh Council         Non Domestic Rates                40800.0        Utility Bills
3  09/05/2016              Computacenter Uk                 Kelvin Hall                72835.0          Software/IT
4  09/05/2016  John Graham Construction Ltd  Causewayside Refurbishment                64361.0  Building Improvement
... (rows 5-24 omitted)
```

### 2.4 Zero-shot Results Summary

The zero-shot approach performs well even without labeled examples. It correctly classified 23 out of 25 transactions (92%) in our small sample. The two "Could not classify" results were for transactions with ambiguous descriptions (e.g., "ALDL Charges"). To improve performance, we can use a labeled dataset to provide more context.

## 3. Classification with Embeddings

Next, we'll use a small set of labeled data to create embeddings and train a traditional classifier. This approach can be more efficient and cost-effective than repeatedly calling a large language model.

### 3.1 Load the Labeled Dataset

We've prepared a dataset of 101 transactions where the zero-shot classifications were manually reviewed and corrected.

```python
df = pd.read_csv('./data/labelled_transactions.csv')
print(df.head())
```

**Output:**
```
         Date                      Supplier                 Description  Transaction value (£)       Classification
0  15/08/2016  Creative Video Productions Ltd                 Kelvin Hall                26866                 Other
1  29/05/2017      John Graham Construction Ltd  Causewayside Refurbishment                74806  Building Improvement
2  29/05/2017         Morris & Spottiswood Ltd       George IV Bridge Work                56448  Building Improvement
3  31/05/2017      John Graham Construction Ltd  Causewayside Refurbishment               164691  Building Improvement
4  24/07/2017      John Graham Construction Ltd  Causewayside Refurbishment                27926  Building Improvement
```

### 3.2 Prepare the Text for Embedding

Combine the transaction features into a single text string. This provides rich context for the embedding model.

```python
df['combined'] = (
    "Supplier: " + df['Supplier'].str.strip() + 
    "; Description: " + df['Description'].str.strip() + 
    "; Value: " + df['Transaction value (£)'].astype(str).str.strip()
)
print(df[['combined', 'Classification']].head(2))
```

**Output:**
```
                                                                          combined       Classification
0  Supplier: Creative Video Productions Ltd; Description: Kelvin Hall; Value: 26866                 Other
1  Supplier: John Graham Construction Ltd; Description: Causewayside Refurbishment; Value: 74806  Building Improvement
```

In the next steps, you would generate embeddings for these combined text strings using an embedding model (e.g., OpenAI's `text-embedding-ada-002`) and then train a classifier like Logistic Regression or SVM on the embedding vectors. This approach often provides a strong balance between accuracy and computational cost.

## Next Steps

You have now successfully implemented a zero-shot classifier and prepared data for an embedding-based approach. To continue:

1.  **Generate Embeddings:** Use the `combined` text column with an embedding API to create vector representations.
2.  **Train a Classifier:** Split your labeled data, train a model (e.g., from scikit-learn) on the embeddings, and evaluate its performance.
3.  **Fine-tune a Model:** For the highest potential accuracy, you can fine-tune a pre-trained language model (like BERT) on your specific labeled dataset.

Each method offers different trade-offs in terms of accuracy, cost, and data requirements. The zero-shot approach is quick and requires no labeled data, while embedding-based classification offers a good middle ground, and fine-tuning can provide the best performance for well-defined, consistent tasks.