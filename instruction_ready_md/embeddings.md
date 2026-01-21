# Mistral Embeddings API Cookbook

This guide provides a step-by-step tutorial on using the Mistral Embeddings API to generate text embeddings and apply them to common NLP tasks. You'll learn how to measure semantic similarity, perform classification and clustering, and visualize embeddings.

## Prerequisites

Before starting, ensure you have the necessary libraries installed:

```bash
pip install mistralai seaborn numpy scikit-learn pandas
```

## Step 1: Initialize the Mistral Client

First, import the required libraries and set up your Mistral client with your API key.

```python
import os
from mistralai import Mistral

# Set your API key (store it securely in environment variables)
api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-embed"

client = Mistral(api_key=api_key)
```

## Step 2: Generate Your First Embeddings

The Mistral Embeddings API converts text into high-dimensional vectors that capture semantic meaning. Let's create embeddings for two sample sentences.

```python
embeddings_batch_response = client.embeddings.create(
    model=model,
    inputs=["Embed this sentence.", "As well as this one."],
)
```

The response contains an `EmbeddingResponse` object with the embeddings and token usage information. Each embedding is a vector of 1024 dimensions.

```python
# Check the dimension of the first embedding
len(embeddings_batch_response.data[0].embedding)
```

**Output:**
```
1024
```

## Step 3: Measure Semantic Distance Between Texts

Texts with similar meanings produce embeddings that are closer together in the vector space. Let's create a helper function to generate embeddings and measure distances.

```python
from sklearn.metrics.pairwise import euclidean_distances

def get_text_embedding(inputs):
    """Generate embeddings for a list of text inputs."""
    embeddings_batch_response = client.embeddings.create(
        model=model,
        inputs=inputs
    )
    return embeddings_batch_response.data[0].embedding
```

Now, compare how similar different sentences are to a reference sentence.

```python
sentences = [
    "A home without a cat — and a well-fed, well-petted and properly revered cat — may be a perfect home, perhaps, but how can it prove title?",
    "I think books are like people, in the sense that they’ll turn up in your life when you most need them"
]

# Generate embeddings for each sentence
embeddings = [get_text_embedding([t]) for t in sentences]

# Define a reference sentence about books
reference_sentence = "Books are mirrors: You only see in them what you already have inside you"
reference_embedding = get_text_embedding([reference_sentence])

# Calculate distances
for t, e in zip(sentences, embeddings):
    distance = euclidean_distances([e], [reference_embedding])
    print(f"Sentence: {t[:50]}...")
    print(f"Distance to reference: {distance[0][0]:.4f}\n")
```

The sentence about books will have a smaller distance to the reference than the sentence about cats, demonstrating semantic similarity.

## Step 4: Detect Paraphrases

Embeddings can identify when sentences convey similar meaning. Let's test this with a simple paraphrase detection example.

```python
sentences = [
    'Have a safe happy Memorial Day weekend everyone',
    'To all our friends at Whatsit Productions Films enjoy a safe happy Memorial Day weekend',
    'Where can I find the best cheese?'
]

# Generate embeddings
sentence_embeddings = [get_text_embedding([t]) for t in sentences]

# Compare all pairs
import itertools

sentence_embeddings_pairs = list(itertools.combinations(sentence_embeddings, 2))
sentence_pairs = list(itertools.combinations(sentences, 2))

for s, e in zip(sentence_pairs, sentence_embeddings_pairs):
    distance = euclidean_distances([e[0]], [e[1]])[0][0]
    print(f"Pair: {s[0][:30]}... | {s[1][:30]}...")
    print(f"Distance: {distance:.4f}\n")
```

The first two sentences (both about Memorial Day) will have a smaller distance than either paired with the third sentence about cheese.

## Step 5: Process Data in Batches

For larger datasets, process text in batches for efficiency. We'll use a medical symptom dataset to demonstrate.

```python
import pandas as pd

# Load the Symptom2Disease dataset
df = pd.read_csv("https://raw.githubusercontent.com/mistralai/cookbook/main/data/Symptom2Disease.csv", index_col=0)

def get_embeddings_by_chunks(data, chunk_size):
    """Process text in chunks to generate embeddings."""
    chunks = [data[x : x + chunk_size] for x in range(0, len(data), chunk_size)]
    embeddings_response = [
        client.embeddings.create(model=model, inputs=c) for c in chunks
    ]
    return [d.embedding for e in embeddings_response for d in e.data]

# Generate embeddings for all symptom descriptions
df['embeddings'] = get_embeddings_by_chunks(df['text'].tolist(), 50)

# View the first few rows
print(df.head())
```

## Step 6: Visualize Embeddings with t-SNE

Since embeddings are 1024-dimensional, we need dimensionality reduction to visualize them. t-SNE projects embeddings into 2D space while preserving relationships.

```python
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

# Convert embeddings to numpy array
embeddings_array = np.array(df['embeddings'].to_list())

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=0).fit_transform(embeddings_array)

# Create scatter plot colored by disease label
ax = sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=np.array(df['label'].to_list()))
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
```

You should see clusters forming based on disease categories, showing that similar symptoms produce similar embeddings.

## Step 7: Build a Classification Model

Use embeddings as features to train a classifier that predicts diseases from symptoms.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Split data into training and test sets
train_x, test_x, train_y, test_y = train_test_split(
    df['embeddings'], df["label"], test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x.to_list())
test_x = scaler.transform(test_x.to_list())

# Train a logistic regression classifier
clf = LogisticRegression(random_state=0, C=1.0, max_iter=500)
clf.fit(train_x, train_y.to_list())

# Evaluate performance
accuracy = 100 * np.mean(clf.predict(test_x) == test_y.to_list())
print(f"Classification Accuracy: {accuracy:.2f}%")
```

**Output:**
```
Classification Accuracy: 95.83%
```

Test the classifier with a new symptom description:

```python
text = "I've been experiencing frequent headaches and vision problems."
prediction = clf.predict([get_text_embedding([text])])
print(f"Predicted disease: {prediction[0]}")
```

## Step 8: Perform Clustering Without Labels

When labels aren't available, clustering can reveal natural groupings in your data.

```python
from sklearn.cluster import KMeans

# We know there are 24 disease categories
model = KMeans(n_clusters=24, max_iter=1000, random_state=42)
model.fit(df['embeddings'].to_list())

# Assign cluster labels
df["cluster"] = model.labels_

# Examine examples from one cluster
print("Examples from cluster 23:")
print(*df[df.cluster==23].text.head(3), sep='\n')
```

Symptoms within the same cluster should be semantically similar, even without using the disease labels.

## Step 9: Understand Retrieval Applications

Mistral embeddings are particularly effective for retrieval tasks, forming the foundation for Retrieval-Augmented Generation (RAG) systems. In a RAG pipeline:

1. **Knowledge Base Embedding**: Convert documents into embeddings and store them in a vector database.
2. **Query Processing**: Embed user queries to find the most similar document embeddings.
3. **Context Augmentation**: Feed retrieved documents to an LLM to generate informed responses.

For a complete RAG implementation tutorial, refer to the [Mistral RAG Guide](https://github.com/mistralai/cookbook/blob/main/mistral/rag/basic_RAG.ipynb).

## Summary

You've learned how to:
- Generate text embeddings using the Mistral API
- Measure semantic similarity between texts
- Detect paraphrases using embedding distances
- Process large datasets efficiently with batch operations
- Visualize high-dimensional embeddings with t-SNE
- Build classification models using embeddings as features
- Perform clustering to discover patterns in unlabeled data
- Understand how embeddings power retrieval systems

These techniques form the foundation for many NLP applications, from semantic search to intelligent document analysis.