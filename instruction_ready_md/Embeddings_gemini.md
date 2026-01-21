# Getting Started with Gemini Embedding Models

## Introduction

Embeddings are numerical representations that capture the semantic relationships between different pieces of text. They convert text into arrays of floating-point numbers (vectors), where similar meanings are represented by vectors that are close together in a high-dimensional space. The Gemini API provides state-of-the-art text embeddings that you can use for applications like semantic search, document retrieval, and similarity analysis.

This guide will walk you through the fundamentals of generating and using embeddings with the Gemini API.

## Prerequisites

Before you begin, ensure you have the following:

1.  **A Gemini API Key:** You need an API key to authenticate your requests. If you don't have one, follow the [authentication guide](https://ai.google.dev/gemini-api/docs/authentication).
2.  **Python Environment:** This tutorial assumes you are working in a Python environment, such as a Jupyter notebook or a local script.

## Step 1: Install the SDK and Set Up Your Client

First, install the official Python SDK for the Gemini API.

```bash
pip install -q -U "google-genai>=1.0.0"
```

Next, import the necessary library and initialize the client with your API key.

```python
from google import genai

# Replace 'YOUR_API_KEY' with your actual Gemini API key
GEMINI_API_KEY = 'YOUR_API_KEY'
client = genai.Client(api_key=GEMINI_API_KEY)
```

## Step 2: Generate Your First Embedding

With the client ready, you can generate an embedding for a piece of text. You must specify which embedding model to use. For this guide, we'll use `gemini-embedding-001`.

```python
MODEL_ID = "gemini-embedding-001"

text = ["Hello world"]
result = client.models.embed_content(model=MODEL_ID, contents=text)

# Extract the embedding from the result
[embedding] = result.embeddings

# The embedding is a list of floating-point numbers
print(f"Embedding length: {len(embedding.values)}")
print(f"First few values: {embedding.values[:5]} ...")
```

**Output:**
```
Embedding length: 3072
First few values: [-0.02342152, 0.01676572, 0.009261323, -0.06383, -0.007975887] ...
```

## Step 3: Generate Embeddings in Batches

For efficiency, you can embed multiple texts in a single API call.

```python
result = client.models.embed_content(
    model=MODEL_ID,
    contents=[
      'What is the meaning of life?',
      'How much wood would a woodchuck chuck?',
      'How does the brain work?'
    ]
)

for i, embedding in enumerate(result.embeddings):
    print(f"Embedding {i+1} length: {len(embedding.values)}")
```

## Step 4: Control Embedding Dimensionality

By default, embeddings are high-dimensional (e.g., 3072). You can reduce the size by specifying an `output_dimensionality`. This is useful for saving storage or speeding up downstream tasks, though it may slightly reduce representational power.

```python
from google.genai import types

text = ["Hello world"]

# Full-dimensional embedding
full_result = client.models.embed_content(model=MODEL_ID, contents=text)
[full_embedding] = full_result.embeddings

# Truncated embedding (10 dimensions)
truncated_result = client.models.embed_content(
    model=MODEL_ID,
    contents=text,
    config=types.EmbedContentConfig(output_dimensionality=10)
)
[truncated_embedding] = truncated_result.embeddings

print(f"Full embedding size: {len(full_embedding.values)}")
print(f"Truncated embedding size: {len(truncated_embedding.values)}")
```

## Step 5: Analyze Sentence Similarity

A common use case for embeddings is measuring semantic similarity. Let's create a small dataset of sentences and see which are most alike.

First, create a DataFrame with some sample sentences.

```python
import pandas as pd

sentences = [
    "I really enjoyed last night's movie",
    "we watched a lot of acrobatic scenes yesterday",
    "I had fun writing my first program in Python",
    "a tremendous feeling of relief to finally get my Nodejs scripts to run without errors!",
    "Oh Romeo, Romeo, wherefore art thou Romeo?"
]

df = pd.DataFrame(sentences, columns=["text"])
```

Next, generate an embedding for each sentence.

```python
df["embeddings"] = df.apply(
    lambda row: client.models.embed_content(
        model=MODEL_ID,
        contents=row['text']
    ).embeddings[0].values,
    axis=1
)
```

Now, calculate the pairwise cosine similarity between all sentence embeddings.

```python
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the similarity matrix
cos_sim_array = cosine_similarity(list(df.embeddings.values))

# Create a readable DataFrame
similarities_df = pd.DataFrame(cos_sim_array, index=sentences, columns=sentences)

# Visualize the similarities with a heatmap
plt.figure(figsize=(8, 6))
ax = sns.heatmap(similarities_df, annot=True, cmap="crest")
ax.xaxis.tick_top()
ax.set_xticklabels(sentences, rotation=90)
plt.title("Semantic Similarity Between Sentences")
plt.show()
```

You'll see that sentences about movies have high similarity with each other, as do sentences about programming. The Shakespeare quote is distinct from the others.

## Step 6: Use Task Types for Specialized Retrieval

For advanced applications like Retrieval-Augmented Generation (RAG), you can specify a `task_type` to get embeddings optimized for a specific role. This improves retrieval quality.

Common task types include:
*   `RETRIEVAL_QUERY`: For the user's search question.
*   `RETRIEVAL_DOCUMENT`: For the documents in your knowledge base.
*   `SEMANTIC_SIMILARITY`: For general similarity comparison.
*   `CLASSIFICATION`: For training a classifier.
*   `CLUSTERING`: For grouping similar documents.

Let's build a simple FAQ system to demonstrate `RETRIEVAL_QUERY` and `RETRIEVAL_DOCUMENT`.

First, create a small knowledge base.

```python
documents = [
    {
        "title": "Climate control system",
        "contents": "Operating the Climate Control System Your Googlecar has a climate control system... Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed..."
    },
    {
        "title": "Touchscreen",
        "contents": "Your Googlecar has a large touchscreen display that provides access to a variety of features..."
    },
    {
        "title": "Shifting gears",
        "contents": "Shifting Gears Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever..."
    }
]

docs_df = pd.DataFrame(documents)
```

Generate document embeddings using the `RETRIEVAL_DOCUMENT` task type.

```python
docs_df["embeddings"] = docs_df.apply(
    lambda row: client.models.embed_content(
        model=MODEL_ID,
        contents=row['contents'],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    ).embeddings[0].values,
    axis=1
)
```

Now, create a function that takes a user's question, embeds it as a `RETRIEVAL_QUERY`, and finds the most relevant document.

```python
import numpy as np

def find_best_passage(query: str, dataframe: pd.DataFrame, model: str) -> str:
    # 1. Embed the query
    query_embedding_result = client.models.embed_content(
        model=model,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_vector = query_embedding_result.embeddings[0].values

    # 2. Calculate similarity (dot product) with all documents
    doc_vectors = np.stack(dataframe.embeddings.values)
    similarity_scores = np.dot(doc_vectors, query_vector)

    # 3. Return the text of the most similar document
    best_index = np.argmax(similarity_scores)
    return dataframe.contents.iloc[best_index]
```

Test the retrieval function.

```python
question = "how to make the fan speed stronger on the car cooling system"
best_passage = find_best_passage(question, docs_df, MODEL_ID)
print("Most relevant passage found:")
print(best_passage[:200], "...") # Print first 200 chars
```

## Step 7: Generate a Final Answer with a Generative Model

The final step in a RAG pipeline is to use a generative model (like `gemini-1.5-flash`) to synthesize a friendly answer based on the retrieved document.

```python
# Switch to a generative model
GENERATIVE_MODEL_ID = "gemini-1.5-flash"

prompt = f"""
Your Role: You are a friendly AI assistant. Your purpose is to explain information to users who are not experts.

Your Task: Use the provided "Source Text" below to answer the user's question.

Guidelines:
- Be Clear and Simple: Explain any complicated ideas in easy-to-understand terms.
- Be Friendly: Write in a warm, conversational tone.
- Be Thorough: Construct a complete answer using all relevant information from the source.
- Stay on Topic: If the source text does not contain the answer, state that the information is not available.

QUESTION: {question}
SOURCE TEXT: {best_passage}
"""

response = client.models.generate_content(
    model=GENERATIVE_MODEL_ID,
    contents=prompt,
)

print("Assistant's Answer:")
print(response.text)
```

**Example Output:**
```
Hello there! To make the fan speed stronger in your Googlecar's climate control system, you just need to find the fan speed knob on the center console. Once you locate it, turn the knob clockwise. Turning it clockwise will increase the speed of the fan, giving you more airflow. That's all there is to it!
```

## Conclusion

You've learned the core operations for working with Gemini embedding models: generating single and batch embeddings, controlling dimensionality, measuring similarity, and using task-specific embeddings for retrieval. These techniques form the foundation for building powerful applications like semantic search, recommendation systems, and RAG pipelines.

## Next Steps

To dive deeper, explore these advanced topics:
*   **Search Reranking:** Use embeddings to improve the order of search results.
*   **Anomaly Detection:** Identify outliers in your text data using embeddings.
*   **Text Classification:** Train a classifier using embeddings as features.
*   **Vector Databases:** Integrate Gemini embeddings with databases like Chroma, Weaviate, or Qdrant for scalable retrieval.

Check the [official Gemini documentation](https://ai.google.dev/gemini-api/docs) for more models, detailed guides, and additional code examples.