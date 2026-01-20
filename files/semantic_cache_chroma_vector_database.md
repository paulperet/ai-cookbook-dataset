# Implementing semantic cache to improve a RAG system with FAISS.

_Authored by:[Pere Martra](https://github.com/peremartra)_

In this notebook, we will explore a typical RAG solution where we will utilize an open-source model and the vector database Chroma DB. **However, we will integrate a semantic cache system that will store various user queries and decide whether to generate the prompt enriched with information from the vector database or the cache.**

A semantic caching system aims to identify similar or identical user requests. When a matching request is found, the system retrieves the corresponding information from the cache, reducing the need to fetch it from the original source.

As the comparison takes into account the semantic meaning of the requests, they don't have to be identical for the system to recognize them as the same question.  They can be formulated differently or contain inaccuracies, be they typographical or in the sentence structure, and we can identify that the user is actually requesting the same information.

For instance, queries like **What is the capital of France?**, **Tell me the name of the capital of France?**, and **What The capital of France is?** all convey the same intent and should be identified as the same question.

While the model's response may differ based on the request for a concise answer in the second example, the information retrieved from the vector database should be the same. This is why I'm placing the cache system between the user and the vector database, not between the user and the Large Language Model.

Most tutorials that guide you through creating a RAG system are designed for single-user use, meant to operate in a testing environment. In other words, within a notebook, interacting with a local vector database and making API calls or using a locally stored model.

This architecture quickly becomes insufficient when attempting to transition one of these models to production, where they might encounter from tens to thousands of recurrent requests.

One way to enhance performance is through one or multiple semantic caches. This cache retains the results of previous requests, and before resolving a new request, it checks if a similar one has been received before. If so, instead of re-executing the process, it retrieves the information from the cache.

In a RAG system, there are two points that are time consuming:
* Retrieve the information used to construct the enriched prompt:
* Call the Large Language Model to obtain the response.

In both points, a semantic cache system can be implemented, and we could even have two caches, one for each point.

Placing it at the model's response point may lead to a loss of influence over the obtained response. Our cache system could consider "Explain the French Revolution in 10 words" and "Explain the French Revolution in a hundred words" as the same query. If our cache system stores model responses, users might think that their instructions are not being followed accurately.

But both requests will require the same information to enrich the prompt. This is the main reason why I chose to place the semantic cache system between the user's request and the retrieval of information from the vector database.

However, this is a design decision. Depending on the type of responses and system requests, it can be placed at one point or another. It's evident that caching model responses would yield the most time savings, but as I've already explained, it comes at the cost of losing user influence over the response.

# Import and load the libraries.
To start we need to install the necesary Python packages.
* **[sentence transformers](https://www.sbert.net/)**. This library is necessary to transform the sentences into fixed-length vectors, also know as embeddings.
* **[xformers](https://github.com/facebookresearch/xformers)**. it's a package that provides libraries an utilities to facilitate the work with transformers models. We need to install in order to avoid an error when we work with the model and embeddings.  
* **[chromadb](https://www.trychroma.com/)**. This is our vector Database. ChromaDB is easy to use and open source, maybe the most used Vector Database used to store embeddings.
* **[accelerate](https://github.com/huggingface/accelerate)** Necesary to run the Model in a GPU.  

```python
!pip install -q transformers==4.38.1
!pip install -q accelerate==0.27.2
!pip install -q sentence-transformers==2.5.1
!pip install -q xformers==0.0.24
!pip install -q chromadb==0.4.24
!pip install -q datasets==2.17.1
```

```python
import numpy as np
import pandas as pd
```

# Load the Dataset
As we are working in a free and limited space, and we can use just a few GB of memory I limited the number of rows to use from the Dataset with the variable `MAX_ROWS`.

```python
#Login to Hugging Face. It is mandatory to use the Gemma Model,
#and recommended to acces public models and Datasets.
from getpass import getpass
if 'hf_key' not in locals():
  hf_key = getpass("Your Hugging Face API Key: ")
!huggingface-cli login --token $hf_key
```

```python
from datasets import load_dataset

data = load_dataset("keivalya/MedQuad-MedicalQnADataset", split='train')
```

ChromaDB requires that the data has a unique identifier. We can make it with this statement, which will create a new column called **Id**.

```python
data = data.to_pandas()
data["id"]=data.index
data.head(10)
```

```python
MAX_ROWS = 15000
DOCUMENT="Answer"
TOPIC="qtype"
```

```python
#Because it is just a sample we select a small portion of News.
subset_data = data.head(MAX_ROWS)
```

# Import and configure the Vector Database
To store the information, I've chosen to use ChromaDB, one of the most well-known and widely used open-source vector databases.

First we need to import ChromaDB.

```python
import chromadb
```

Now we only need to indicate the path where the vector database will be stored.

```python
chroma_client = chromadb.PersistentClient(path="/path/to/persist/directory")
```

# Filling and Querying the ChromaDB Database
The Data in ChromaDB is stored in collections. If the collection exist we need to delete it.

In the next lines, we are creating the collection by calling the `create_collection` function in the `chroma_client` created above.

```python
collection_name = "news_collection"
if len(chroma_client.list_collections()) > 0 and collection_name in [chroma_client.list_collections()[0].name]:
    chroma_client.delete_collection(name=collection_name)

collection = chroma_client.create_collection(name=collection_name)

```

We are now ready to add the data to the collection using the `add` function. This function requires three key pieces of information:

* In the **document** we store the content of the `Answer` column in the Dataset.
* In **metadatas**, we can inform a list of topics. I used the value in the column `qtype`.
* In **id** we need to inform an unique identificator for each row. I'm creating the ID using the range of `MAX_ROWS`.

```python
collection.add(
    documents=subset_data[DOCUMENT].tolist(),
    metadatas=[{TOPIC: topic} for topic in subset_data[TOPIC].tolist()],
    ids=[f"id{x}" for x in range(MAX_ROWS)],
)
```

Once we have the information in the Database we can query it, and ask for data that matches our needs. The search is done inside the content of the document, and it dosn't look for the exact word, or phrase. The results will be based on the similarity between the search terms and the content of documents.

Metadata isn't directly involved in the initial search process, it can be used to filter or refine the results after retrieval, enabling further customization and precision.

Let's define a function to query the ChromaDB Database.

```python
def query_database(query_text, n_results=10):
    results = collection.query(query_texts=query_text, n_results=n_results )
    return results
```

## Creating the semantic cache system
To implement the cache system, we will use Faiss, a library that allows storing embeddings in memory. It's quite similar to what Chroma does, but without its persistence.

For this purpose, we will create a class called `semantic_cache` that will work with its own encoder and provide the necessary functions for the user to perform queries.

In this class, we first query the cache implemented with Faiss, that contains the previous petitions, and if the returned results are above a specified threshold, it will return the content of the cache. Otherwise, it will fetch the result from the Chroma database.

The cache is stored in a .json file.

```python
!pip install -q faiss-cpu==1.8.0
```

```python
import faiss
from sentence_transformers import SentenceTransformer
import time
import json
```

The `init_cache()` function below initializes the semantic cache.

It employs the FlatLS index, which might not be the fastest but is ideal for small datasets. Depending on the characteristics of the data intended for the cache and the expected dataset size, another index such as HNSW or IVF could be utilized.

I chose this index because it aligns well with the example. It can be used with vectors of high dimensions, consumes minimal memory, and performs well with small datasets.

I outline the key features of the various indices available with Faiss.

* FlatL2 or FlatIP. Well-suited for small datasets, it may not be the fastest, but its memory consumption is not excessive.
* LSH. It works effectively with small datasets and is recommended for use with vectors of up to 128 dimensions.
* HNSW. Very fast but demands a substantial amount of RAM.
* IVF. Works well with large datasets without consuming much memory or compromising performance.

More information about the different indices available with Faiss can be found at this link: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

```python
def init_cache():
  index = faiss.IndexFlatL2(768)
  if index.is_trained:
    print('Index trained')

  # Initialize Sentence Transformer model
  encoder = SentenceTransformer('all-mpnet-base-v2')

  return index, encoder
```

In the `retrieve_cache` function, the .json file is retrieved from disk in case there is a need to reuse the cache across sessions.

```python
def retrieve_cache(json_file):
  try:
    with open(json_file, 'r') as file:
      cache = json.load(file)
  except FileNotFoundError:
      cache = {'questions': [], 'embeddings': [], 'answers': [], 'response_text': []}

  return cache
```

The `store_cache` function saves the file containing the cache data to disk.

```python
def store_cache(json_file, cache):
  with open(json_file, 'w') as file:
    json.dump(cache, file)
```

These functions will be used within the `SemanticCache` class, which includes the search function and its initialization function.

Even though the `ask` function has a substantial amount of code, its purpose is quite straightforward. It looks in the cache for the closest question to the one just made by the user.

Afterward, checks if it is within the specified threshold. If positive, it directly returns the response from the cache; otherwise, it calls the `query_database` function to retrieve the data from ChromaDB.

I've used Euclidean distance instead of Cosine, which is widely employed in vector comparisons. This choice is based on the fact that Euclidean distance is the default metric used by Faiss. Although Cosine distance can also be calculated, doing so adds complexity that may not significantly contribute to the final result.

I have included FIFO eviction policy in the semantic_cache class, which aims to improve its efficiency and flexibility. By introducing eviction policies, we provide users with the ability to control how the cache behaves when it reaches its maximum capacity. This is crucial for maintaining optimal cache performance and for handling situations where the available memory is constrained. 

Looking at the structure of the cache, the implementation of FIFO seemed straightforward. Whenever a new question-answer pair is added to the cache, it's appended to the end of the lists. Thus, the oldest (first-in) items are at the front of the lists. When the cache reaches its maximum size and you need to evict an item, you remove (pop) the first item from each list. This is the FIFO eviction policy. 

Another eviction policy is the Least Recently Used (LRU) policy, which is more complex because it requires knowledge of when each item in the cache was last accessed. However, this policy is not yet available and will be implemented later.

```python
class semantic_cache:
  def __init__(self, json_file="cache_file.json", thresold=0.35, max_response=100, eviction_policy=None):
    """Initializes the semantic cache.

    Args:
    json_file (str): The name of the JSON file where the cache is stored.
    thresold (float): The threshold for the Euclidean distance to determine if a question is similar.
    max_response (int): The maximum number of responses the cache can store.
    eviction_policy (str): The policy for evicting items from the cache. 
                            This can be any policy, but 'FIFO' (First In First Out) has been implemented for now.
                            If None, no eviction policy will be applied.
    """
       
    # Initialize Faiss index with Euclidean distance
    self.index, self.encoder = init_cache()

    # Set Euclidean distance threshold
    # a distance of 0 means identicals sentences
    # We only return from cache sentences under this thresold
    self.euclidean_threshold = thresold

    self.json_file = json_file
    self.cache = retrieve_cache(self.json_file)
    self.max_response = max_response
    self.eviction_policy = eviction_policy

  def evict(self):

    """Evicts an item from the cache based on the eviction policy."""
    if self.eviction_policy and len(self.cache["questions"]) > self.max_size:
        for _ in range((len(self.cache["questions"]) - self.max_response)):
            if self.eviction_policy == 'FIFO':
                self.cache["questions"].pop(0)
                self.cache["embeddings"].pop(0)
                self.cache["answers"].pop(0)
                self.cache["response_text"].pop(0)

  def ask(self, question: str) -> str:
      # Method to retrieve an answer from the cache or generate a new one
      start_time = time.time()
      try:
          #First we obtain the embeddings corresponding to the user question
          embedding = self.encoder.encode([question])

          # Search for the nearest neighbor in the index
          self.index.nprobe = 8
          D, I = self.index.search(embedding, 1)

          if D[0] >= 0:
              if I[0][0] >= 0 and D[0][0] <= self.euclidean_threshold:
                  row_id = int(I[0][0])

                  print('Answer recovered from Cache. ')
                  print(f'{D[0][0]:.3f} smaller than {self.euclidean_threshold}')
                  print(f'Found cache in row: {row_id} with score {D[0][0]:.3f}')
                  print(f'response_text: ' + self.cache['response_text'][row_id])

                  end_time = time.time()
                  elapsed_time = end_time - start_time
                  print(f"Time taken: {elapsed_time:.3f} seconds")
                  return self.cache['response_text'][row_id]

          # Handle the case when there are not enough results
          # or Euclidean distance is not met, asking to chromaDB.
          answer  = query_database([question], 1)
          response_text = answer['documents'][0][0]

          self.cache['questions'].append(question)
          self.cache['embeddings'].append(embedding[0].tolist())
          self.cache['answers'].append(answer)
          self.cache['response_text'].append(response_text)

          print('Answer recovered from ChromaDB. ')
          print(f'response_text: {response_text}')

          self.index.add(embedding)

          self.evict()

          store_cache(self.json_file, self.cache)
          
          end_time = time.time()
          elapsed_time = end_time - start_time
          print(f"Time taken: {elapsed_time:.3f} seconds")

          return response_text
      except Exception as e:
          raise RuntimeError(f"Error during 'ask' method: {e}")

```

### Testing the semantic_cache class.

```python
# Initialize the cache.
cache = semantic_cache('4cache.json')
```

```python
results = cache.ask("How do vaccines work?")
```

As expected, this response has been obtained from ChromaDB. The class then stores it in the cache.

Now, if we send a second question that is quite different, the response should also be retrieved from ChromaDB. This is because the question stored previously is so dissimilar that it would surpass the specified threshold in terms of Euclidean distance.

```python

results = cache.ask("Explain briefly what is a Sydenham chorea")
```

Perfect, the semantic cache system is behaving as expected.

Let's proceed to test it with a question very similar to the one we just asked.

In this case, the response should come directly from the cache without the need to access the ChromaDB database.

```python
results = cache.ask("Briefly explain me what is a Sydenham chorea.")
```

The two questions are so similar that their Euclidean distance is truly minimal, almost as if they were identical.

Now, let's try another question, this time a bit more distinct, and observe how the system behaves.

```python
question_def = "Write in 20 words what is a Sydenham chorea."
results = cache.ask(question_def)
```

We observe that the Euclidean distance has increased, but it still remains within the specified threshold. Therefore, it continues to return the response directly from the cache.

# Loading the model and creating the prompt
Time to use the library **transformers**, the most famous library from [hugging face](https://huggingface.co/) for working with language models.

We are importing:
* **Autotokenizer**: It is a utility class for tokenizing text inputs that are compatible with various pre-trained language models.
* **AutoModelForCausalLM**: it provides an interface to pre-trained language models specifically designed for language generation tasks using causal language modeling (e.g., GPT models), or the model used in this notebook [Gemma-2b-it](https://huggingface.co/google/gemma-2b-it).

Please, feel free to test [different