# Implementing a Semantic Cache to Improve RAG Performance with FAISS

_Authored by: [Pere Martra](https://github.com/peremartra)_

## Introduction

This guide demonstrates how to enhance a typical Retrieval-Augmented Generation (RAG) system by integrating a semantic cache. A semantic cache stores previous user queries and their corresponding retrieved information. When a new, semantically similar query arrives, the system can retrieve the information from the cache instead of querying the vector database, significantly improving response times and reducing computational load.

We'll build a system using an open-source model (Gemma-2b-it), ChromaDB as the vector database, and FAISS to power the in-memory semantic cache.

## Prerequisites & Setup

First, install the required Python libraries.

```bash
pip install -q transformers==4.38.1
pip install -q accelerate==0.27.2
pip install -q sentence-transformers==2.5.1
pip install -q xformers==0.0.24
pip install -q chromadb==0.4.24
pip install -q datasets==2.17.1
pip install -q faiss-cpu==1.8.0
```

Now, import the necessary modules.

```python
import numpy as np
import pandas as pd
import chromadb
import faiss
import json
import time
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
```

## Step 1: Load and Prepare the Dataset

We'll use the `MedQuad-MedicalQnADataset` from Hugging Face. For demonstration purposes, we limit it to 15,000 rows.

```python
# Login to Hugging Face (required for Gemma and recommended for datasets)
from getpass import getpass
if 'hf_key' not in locals():
    hf_key = getpass("Your Hugging Face API Key: ")
!huggingface-cli login --token $hf_key
```

```python
# Load the dataset and convert it to a pandas DataFrame
data = load_dataset("keivalya/MedQuad-MedicalQnADataset", split='train')
data = data.to_pandas()
data["id"] = data.index

# Define constants and create a subset
MAX_ROWS = 15000
DOCUMENT = "Answer"
TOPIC = "qtype"
subset_data = data.head(MAX_ROWS)
```

## Step 2: Initialize the ChromaDB Vector Database

Create a persistent ChromaDB client and a collection to store our document embeddings.

```python
# Initialize the ChromaDB client
chroma_client = chromadb.PersistentClient(path="/path/to/persist/directory")

# Create or recreate the collection
collection_name = "news_collection"
if len(chroma_client.list_collections()) > 0 and collection_name in [chroma_client.list_collections()[0].name]:
    chroma_client.delete_collection(name=collection_name)

collection = chroma_client.create_collection(name=collection_name)
```

## Step 3: Populate the Vector Database

Add the dataset's answers and their metadata to the ChromaDB collection.

```python
collection.add(
    documents=subset_data[DOCUMENT].tolist(),
    metadatas=[{TOPIC: topic} for topic in subset_data[TOPIC].tolist()],
    ids=[f"id{x}" for x in range(MAX_ROWS)],
)
```

## Step 4: Create a Query Function for ChromaDB

Define a helper function to query the database.

```python
def query_database(query_text, n_results=10):
    results = collection.query(query_texts=query_text, n_results=n_results)
    return results
```

## Step 5: Build the Semantic Cache with FAISS

The core of our enhancement is the `semantic_cache` class. It uses FAISS to store embeddings of previous queries and retrieves cached results if a new query is sufficiently similar.

### 5.1 Initialize Cache Components

First, define functions to initialize the FAISS index and manage cache persistence.

```python
def init_cache():
    """Initializes the FAISS index and the sentence encoder."""
    index = faiss.IndexFlatL2(768)  # L2 distance index for 768-dim embeddings
    if index.is_trained:
        print('Index trained')
    encoder = SentenceTransformer('all-mpnet-base-v2')  # Embedding model
    return index, encoder

def retrieve_cache(json_file):
    """Loads the cache from a JSON file."""
    try:
        with open(json_file, 'r') as file:
            cache = json.load(file)
    except FileNotFoundError:
        cache = {'questions': [], 'embeddings': [], 'answers': [], 'response_text': []}
    return cache

def store_cache(json_file, cache):
    """Saves the cache to a JSON file."""
    with open(json_file, 'w') as file:
        json.dump(cache, file)
```

### 5.2 Define the Semantic Cache Class

The `semantic_cache` class handles querying the cache, querying ChromaDB if needed, and managing cache eviction.

```python
class semantic_cache:
    def __init__(self, json_file="cache_file.json", threshold=0.35, max_response=100, eviction_policy=None):
        """
        Initializes the semantic cache.

        Args:
            json_file (str): File to store/load the cache.
            threshold (float): Euclidean distance threshold for cache hits.
            max_response (int): Maximum number of items the cache can hold.
            eviction_policy (str): Eviction policy (e.g., 'FIFO').
        """
        self.index, self.encoder = init_cache()
        self.euclidean_threshold = threshold
        self.json_file = json_file
        self.cache = retrieve_cache(self.json_file)
        self.max_response = max_response
        self.eviction_policy = eviction_policy

    def evict(self):
        """Evicts items from the cache based on the defined policy."""
        if self.eviction_policy and len(self.cache["questions"]) > self.max_response:
            if self.eviction_policy == 'FIFO':
                # Remove the oldest item (First-In, First-Out)
                self.cache["questions"].pop(0)
                self.cache["embeddings"].pop(0)
                self.cache["answers"].pop(0)
                self.cache["response_text"].pop(0)

    def ask(self, question: str) -> str:
        """Retrieves an answer from the cache or queries ChromaDB."""
        start_time = time.time()
        try:
            # 1. Encode the question
            embedding = self.encoder.encode([question])

            # 2. Search the FAISS index for the nearest cached query
            self.index.nprobe = 8
            D, I = self.index.search(embedding, 1)  # D=distance, I=index

            # 3. Check if a cache hit occurs (valid index and distance under threshold)
            if D[0] >= 0:
                if I[0][0] >= 0 and D[0][0] <= self.euclidean_threshold:
                    row_id = int(I[0][0])
                    print('Answer recovered from Cache.')
                    print(f'{D[0][0]:.3f} smaller than {self.euclidean_threshold}')
                    print(f'Found cache in row: {row_id} with score {D[0][0]:.3f}')
                    print(f'response_text: ' + self.cache['response_text'][row_id])

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Time taken: {elapsed_time:.3f} seconds")
                    return self.cache['response_text'][row_id]

            # 4. Cache Miss: Query ChromaDB
            answer = query_database([question], 1)
            response_text = answer['documents'][0][0]

            # 5. Update the cache with the new Q&A pair
            self.cache['questions'].append(question)
            self.cache['embeddings'].append(embedding[0].tolist())
            self.cache['answers'].append(answer)
            self.cache['response_text'].append(response_text)

            print('Answer recovered from ChromaDB.')
            print(f'response_text: {response_text}')

            # Add the new embedding to the FAISS index
            self.index.add(embedding)

            # Apply eviction policy if cache is full
            self.evict()

            # Persist the updated cache
            store_cache(self.json_file, self.cache)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time:.3f} seconds")

            return response_text
        except Exception as e:
            raise RuntimeError(f"Error during 'ask' method: {e}")
```

## Step 6: Test the Semantic Cache

Let's instantiate the cache and run some queries to see it in action.

```python
# Initialize the cache
cache = semantic_cache('cache.json')
```

**First Query:** This will be a cache miss, retrieving from ChromaDB.

```python
results = cache.ask("How do vaccines work?")
```

**Second Query:** A different question, also a cache miss.

```python
results = cache.ask("Explain briefly what is a Sydenham chorea")
```

**Third Query:** A semantically similar question to the second one. This should be a cache hit.

```python
results = cache.ask("Briefly explain me what is a Sydenham chorea.")
```

**Fourth Query:** A rephrased version of the same question. It should still be a cache hit if the distance is under the threshold.

```python
question_def = "Write in 20 words what is a Sydenham chorea."
results = cache.ask(question_def)
```

## Step 7: Integrate with a Language Model (Gemma-2b-it)

Finally, we can use the retrieved context (from cache or ChromaDB) to generate an answer using a language model.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def generate_answer_with_context(question, context):
    """Generates a final answer using the LLM and the provided context."""
    prompt = f"""Answer the following question based on the provided context.

Context: {context}

Question: {question}

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the new generated part after the prompt
    answer = answer.split("Answer:")[-1].strip()
    return answer

# Example workflow
question = "What is Sydenham chorea?"
# 1. Retrieve context via the semantic cache
context = cache.ask(question)
# 2. Generate the final answer
final_answer = generate_answer_with_context(question, context)
print(f"Final Answer: {final_answer}")
```

## Conclusion

You have successfully built a RAG system enhanced with a semantic cache. This architecture:
1.  **Improves Latency:** Repeated or similar queries are served from a fast, in-memory cache (FAISS).
2.  **Reduces Load:** Decreases the number of queries to your primary vector database.
3.  **Maintains Control:** By caching retrieved documents rather than final LLM responses, you retain user influence over the answer's format and detail.

You can extend this system by experimenting with different FAISS indices, distance thresholds, eviction policies (like LRU), or by placing a second cache at the LLM response level for further optimization.