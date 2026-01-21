# Building a RAG System with Gemma, MongoDB, and Open-Source Models

This guide walks you through building a Retrieval-Augmented Generation (RAG) system using the open-source Gemma language model, MongoDB for vector storage and search, and Hugging Face's embedding models. You will load a movie dataset, generate embeddings, store them in MongoDB, perform vector searches, and generate contextual answers with Gemma.

## Prerequisites

Before you begin, ensure you have:
- A MongoDB Atlas cluster (free tier is sufficient).
- The connection URI for your cluster.
- A Python environment (Google Colab is recommended for GPU access).

## Step 1: Install Required Libraries

Begin by installing the necessary Python packages. These include libraries for data handling, Hugging Face models, and MongoDB interaction.

```bash
!pip install datasets pandas pymongo sentence_transformers
!pip install -U transformers
# Install below if using a GPU
!pip install accelerate
```

## Step 2: Load and Prepare the Dataset

You will use the `AIatMongoDB/embedded_movies` dataset from Hugging Face. This dataset contains movie information, including plots.

```python
from datasets import load_dataset
import pandas as pd

# Load the dataset from Hugging Face
dataset = load_dataset("AIatMongoDB/embedded_movies")
# Convert to a pandas DataFrame for easier manipulation
dataset_df = pd.DataFrame(dataset["train"])

# Display the first few rows to inspect the data
dataset_df.head(5)
```

Now, clean the data to ensure quality and remove any pre-existing embeddings, as you will generate new ones.

```python
# Remove data points where the 'fullplot' column is missing (NaN)
dataset_df = dataset_df.dropna(subset=["fullplot"])
print("Number of missing values in each column after removal:")
print(dataset_df.isnull().sum())

# Remove the existing 'plot_embedding' column, as you will create new embeddings
dataset_df = dataset_df.drop(columns=["plot_embedding"])
dataset_df.head(5)
```

## Step 3: Generate Embeddings with a Sentence Transformer

You will use the `gte-large` embedding model from Hugging Face to convert movie plots into vector embeddings.

```python
from sentence_transformers import SentenceTransformer

# Load the embedding model
embedding_model = SentenceTransformer("thenlper/gte-large")

def get_embedding(text: str) -> list[float]:
    """Generate an embedding for a given text string."""
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []
    # Encode the text to produce the embedding
    embedding = embedding_model.encode(text)
    # Convert the numpy array to a standard Python list
    return embedding.tolist()

# Apply the embedding function to each movie plot
dataset_df["embedding"] = dataset_df["fullplot"].apply(get_embedding)
dataset_df.head()
```

## Step 4: Set Up MongoDB Connection

To store your data, you need to connect to your MongoDB Atlas cluster. Ensure your cluster's IP is whitelisted and you have the connection URI.

```python
import pymongo
from google.colab import userdata  # For Colab secrets; adjust for your environment

def get_mongo_client(mongo_uri):
    """Establish a connection to the MongoDB cluster."""
    try:
        client = pymongo.MongoClient(mongo_uri)
        print("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None

# Retrieve your MongoDB URI from environment secrets
mongo_uri = userdata.get("MONGO_URI")
if not mongo_uri:
    print("MONGO_URI not set in environment variables")

# Create the MongoDB client
mongo_client = get_mongo_client(mongo_uri)

# Access the 'movies' database and 'movie_collection_2' collection
db = mongo_client["movies"]
collection = db["movie_collection_2"]
```

## Step 5: Ingest Data into MongoDB

Clear any existing data in the collection and insert the prepared DataFrame.

```python
# Delete any existing records in the collection
collection.delete_many({})

# Convert the DataFrame to a list of dictionaries (documents)
documents = dataset_df.to_dict("records")
# Insert all documents into the collection
collection.insert_many(documents)

print("Data ingestion into MongoDB completed")
```

## Step 6: Create a Vector Search Index in MongoDB Atlas

For efficient similarity search, you must create a vector search index on your MongoDB collection via the Atlas UI. The index definition should match the embedding dimensions of your model.

1.  In the Atlas UI, navigate to your cluster, select the "Atlas Search" tab, and create a new index on your collection.
2.  Use the following JSON configuration:

```json
{
 "fields": [{
     "numDimensions": 1024,
     "path": "embedding",
     "similarity": "cosine",
     "type": "vector"
   }]
}
```

**Note:** The `numDimensions` value (1024) corresponds to the output dimension of the `gte-large` model. If you use a different model (e.g., `gte-base`), adjust this value accordingly (768 for `gte-base`, 384 for `gte-small`).

## Step 7: Perform Vector Search on User Queries

Now, implement a function that takes a user's query, converts it to an embedding, and performs a vector search in MongoDB to find the most relevant movie plots.

```python
def vector_search(user_query, collection):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
        user_query (str): The user's query string.
        collection (MongoCollection): The MongoDB collection to search.

    Returns:
        list: A list of matching documents.
    """
    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)
    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    # Define the MongoDB aggregation pipeline for vector search
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",  # Name of your vector search index
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 150,  # Broad search scope
                "limit": 4,  # Return top 4 matches
            }
        },
        {
            "$project": {
                "_id": 0,  # Exclude the ID field
                "fullplot": 1,  # Include the plot
                "title": 1,    # Include the title
                "genres": 1,   # Include the genres
                "score": {"$meta": "vectorSearchScore"},  # Include the relevance score
            }
        },
    ]

    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)
```

Create a helper function to format the search results into a single string for the language model.

```python
def get_search_result(query, collection):
    """Retrieve and format search results for a given query."""
    get_knowledge = vector_search(query, collection)
    search_result = ""
    for result in get_knowledge:
        search_result += f"Title: {result.get('title', 'N/A')}, Plot: {result.get('fullplot', 'N/A')}\n"
    return search_result
```

## Step 8: Query the System and Load the Gemma Model

Test the retrieval system with a sample query.

```python
query = "What is the best romantic movie to watch and why?"
source_information = get_search_result(query, collection)
# Combine the query and retrieved context for the LLM
combined_information = f"Query: {query}\nContinue to answer the query by using the Search Results:\n{source_information}."

print(combined_information)
```

Now, load the Gemma-2B-Instruct model from Hugging Face to generate answers based on the retrieved context.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
# For GPU usage (recommended)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto")
# For CPU-only usage, comment the line above and uncomment the line below:
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")
```

## Step 9: Generate the Final Answer

Pass the combined query and context to Gemma to generate a coherent, context-aware response.

```python
# Tokenize the input and move it to the GPU
input_ids = tokenizer(combined_information, return_tensors="pt").to("cuda")
# Generate a response
response = model.generate(**input_ids, max_new_tokens=500)
# Decode and print the generated text
print(tokenizer.decode(response[0]))
```

## Conclusion

You have successfully built an end-to-end RAG system. The workflow includes:
1.  Loading and preprocessing a dataset.
2.  Generating embeddings with an open-source model.
3.  Storing vectors in MongoDB with a vector search index.
4.  Retrieving relevant context based on a user query.
5.  Generating an informed answer using the Gemma language model.

This system can be extended with more sophisticated query handling, chunking strategies for longer texts, or different LLM/embedding models.