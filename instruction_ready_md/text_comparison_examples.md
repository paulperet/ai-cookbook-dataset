# Cookbook: A Practical Guide to OpenAI Embeddings

## Introduction

OpenAI's embedding models convert text into high-dimensional vectors that capture semantic meaning. These vectors enable powerful applications like semantic search, question answering, and recommendation systems by measuring the similarity between pieces of text. This guide walks you through the core concepts and practical implementations.

## Prerequisites

Before starting, ensure you have the OpenAI Python package installed:

```bash
pip install openai
```

You'll also need an OpenAI API key. Set it as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or configure it directly in your Python code:

```python
import openai
openai.api_key = 'your-api-key-here'
```

## Understanding Embeddings

Embeddings are numerical representations of text. Sentences with similar meanings will have vectors that are close together in the embedding space. You can measure this closeness using cosine similarity or Euclidean distance.

OpenAI's embedding models (like `text-embedding-3-small`) support up to 8,191 tokens per input. For longer documents, you'll need to split them into chunks.

## 1. Implementing Semantic Search

Semantic search allows you to find relevant documents based on meaning rather than just keyword matching.

### Step 1: Precompute Document Embeddings

First, prepare your corpus by splitting documents and generating embeddings:

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Example documents
documents = [
    "The cat sat on the mat",
    "Dogs are great companions",
    "Felines enjoy resting on soft surfaces",
    "Canines make excellent pets"
]

# Generate embeddings for each document
document_embeddings = [get_embedding(doc) for doc in documents]
```

### Step 2: Search with Query Embeddings

When a user submits a query, embed it and find the most similar documents:

```python
from sklearn.metrics.pairwise import cosine_similarity

def search_documents(query, documents, document_embeddings, top_k=2):
    # Embed the query
    query_embedding = get_embedding(query)
    
    # Calculate similarities
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    
    # Get top results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'document': documents[idx],
            'similarity': similarities[idx]
        })
    
    return results

# Example search
query = "Tell me about cats"
results = search_documents(query, documents, document_embeddings)

for result in results:
    print(f"Similarity: {result['similarity']:.3f} - {result['document']}")
```

This approach forms the foundation for more advanced retrieval systems.

## 2. Building a Question Answering System

Combine semantic search with GPT models to create accurate question-answering systems.

### Step 1: Retrieve Relevant Context

Use semantic search to find the most relevant information for a question:

```python
def retrieve_context(question, documents, document_embeddings, top_k=3):
    results = search_documents(question, documents, document_embeddings, top_k)
    context = "\n\n".join([r['document'] for r in results])
    return context
```

### Step 2: Generate Answers with GPT

Feed the retrieved context to a GPT model for answer generation:

```python
def answer_question(question, documents, document_embeddings):
    # Retrieve relevant context
    context = retrieve_context(question, documents, document_embeddings)
    
    # Create prompt with context
    prompt = f"""Answer the question based on the context below.
    
Context: {context}

Question: {question}

Answer:"""
    
    # Generate answer
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You answer questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    
    return response.choices[0].message.content

# Example usage
question = "What do cats like to sit on?"
answer = answer_question(question, documents, document_embeddings)
print(f"Question: {question}")
print(f"Answer: {answer}")
```

This approach ensures answers are grounded in your source documents, improving accuracy and reducing hallucinations.

## 3. Creating Recommendation Systems

Embeddings can power recommendation engines by finding items similar to a user's preferences.

### Step 1: Prepare Item Embeddings

First, embed all items in your catalog:

```python
# Example product descriptions
products = [
    "Wireless Bluetooth headphones with noise cancellation",
    "Premium over-ear headphones with leather cushions",
    "True wireless earbuds with charging case",
    "Wired gaming headset with microphone",
    "Portable Bluetooth speaker with waterproof design"
]

product_embeddings = [get_embedding(product) for product in products]
```

### Step 2: Generate Recommendations

Find products similar to a user's preferred item:

```python
def recommend_products(preferred_product, products, product_embeddings, top_n=3):
    # Find index of preferred product
    try:
        pref_idx = products.index(preferred_product)
        pref_embedding = product_embeddings[pref_idx]
    except ValueError:
        # If product not in list, embed the description
        pref_embedding = get_embedding(preferred_product)
        all_embeddings = product_embeddings
    else:
        # Exclude the preferred product itself
        all_embeddings = [emb for i, emb in enumerate(product_embeddings) if i != pref_idx]
        all_products = [prod for i, prod in enumerate(products) if i != pref_idx]
    
    # Calculate similarities
    similarities = cosine_similarity([pref_embedding], all_embeddings)[0]
    
    # Get top recommendations
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    recommendations = []
    for idx in top_indices:
        recommendations.append({
            'product': all_products[idx],
            'similarity': similarities[idx]
        })
    
    return recommendations

# Example recommendation
preferred = "Noise cancelling headphones"
recommendations = recommend_products(preferred, products, product_embeddings)

print(f"Because you liked: {preferred}")
print("\nRecommended products:")
for rec in recommendations:
    print(f"- {rec['product']} (similarity: {rec['similarity']:.3f})")
```

## 4. Customizing Embeddings for Your Domain

While you can't fine-tune OpenAI's embedding models directly, you can train a transformation matrix to emphasize features relevant to your specific use case.

### Step 1: Prepare Training Data

Create pairs of text with similarity scores:

```python
# Example training pairs: (text1, text2, similarity_score)
training_pairs = [
    ("cat", "kitten", 0.9),
    ("dog", "puppy", 0.9),
    ("cat", "dog", 0.3),
    ("car", "vehicle", 0.8),
    ("car", "animal", 0.1)
]
```

### Step 2: Train a Custom Transformation

Learn a matrix that optimizes for your similarity judgments:

```python
from sklearn.linear_model import LinearRegression
import numpy as np

def train_custom_matrix(training_pairs, embedding_model="text-embedding-3-small"):
    # Prepare training data
    X = []
    y = []
    
    for text1, text2, similarity in training_pairs:
        emb1 = get_embedding(text1, model=embedding_model)
        emb2 = get_embedding(text2, model=embedding_model)
        
        # Use element-wise product as features (simplified approach)
        features = np.array(emb1) * np.array(emb2)
        X.append(features)
        y.append(similarity)
    
    X = np.array(X)
    y = np.array(y)
    
    # Train linear transformation
    model = LinearRegression()
    model.fit(X, y)
    
    return model

# Train custom transformation
custom_model = train_custom_matrix(training_pairs)

# Apply to new embeddings
def apply_custom_similarity(text1, text2, custom_model):
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    
    features = np.array(emb1) * np.array(emb2)
    predicted_similarity = custom_model.predict([features])[0]
    
    return max(0, min(1, predicted_similarity))  # Clip to [0, 1] range

# Test custom similarity
test_pair = ("cat", "feline")
custom_score = apply_custom_similarity(test_pair[0], test_pair[1], custom_model)
print(f"Custom similarity between '{test_pair[0]}' and '{test_pair[1]}': {custom_score:.3f}")
```

## Best Practices and Considerations

1. **Chunking Strategy**: For long documents, split them into coherent chunks (by paragraph, section, or fixed token size) before embedding.

2. **Storage Solutions**: For production systems, consider vector databases like Pinecone, Weaviate, or Qdrant for efficient similarity search.

3. **Cost Optimization**: Cache embeddings when possible, as generating them incurs API costs.

4. **Evaluation**: Always evaluate your embedding-based system with relevant metrics for your use case (precision@k, recall, MRR, etc.).

5. **Hybrid Approaches**: Combine embedding-based semantic search with traditional keyword search for robust retrieval systems.

## Next Steps

Explore the [OpenAI Cookbook repository](https://github.com/openai/openai-cookbook) for more advanced examples and techniques. For benchmarking against other models, refer to the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

Remember that embeddings are a powerful tool in your AI toolkit—they work best when combined with other techniques and tailored to your specific application needs.