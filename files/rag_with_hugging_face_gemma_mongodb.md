# Building A RAG System with Gemma, MongoDB and Open Source Models

Authored By: [Richmond Alake](https://huggingface.co/RichmondMongo)

## Step 1: Installing Libraries

The shell command sequence below installs libraries for leveraging open-source large language models (LLMs), embedding models, and database interaction functionalities. These libraries simplify the development of a RAG system, reducing the complexity to a small amount of code:

- PyMongo: A Python library for interacting with MongoDB that enables functionalities to connect to a cluster and query data stored in collections and documents.
- Pandas: Provides a data structure for efficient data processing and analysis using Python
- Hugging Face datasets: Holds audio, vision, and text datasets
- Hugging Face Accelerate: Abstracts the complexity of writing code that leverages hardware accelerators such as GPUs. Accelerate is leveraged in the implementation to utilise the Gemma model on GPU resources.
- Hugging Face Transformers: Access to a vast collection of pre-trained models
- Hugging Face Sentence Transformers: Provides access to sentence, text, and image embeddings.

```python
!pip install datasets pandas pymongo sentence_transformers
!pip install -U transformers
# Install below if using GPU
!pip install accelerate
```

## Step 2: Data sourcing and preparation

The data utilised in this tutorial is sourced from Hugging Face datasets, specifically the 
[AIatMongoDB/embedded_movies dataset](https://huggingface.co/datasets/AIatMongoDB/embedded_movies). 

```python
# Load Dataset
from datasets import load_dataset
import pandas as pd

# https://huggingface.co/datasets/AIatMongoDB/embedded_movies
dataset = load_dataset("AIatMongoDB/embedded_movies")

# Convert the dataset to a pandas dataframe
dataset_df = pd.DataFrame(dataset["train"])

dataset_df.head(5)
```

The operations within the following code snippet below focus on enforcing data integrity and quality. 
1. The first process ensures that each data point's `fullplot` attribute is not empty, as this is the primary data we utilise in the embedding process. 
2. This step also ensures we remove the `plot_embedding` attribute from all data points as this will be replaced by new embeddings created with a different embedding model, the `gte-large`.

```python
# Data Preparation

# Remove data point where plot coloumn is missing
dataset_df = dataset_df.dropna(subset=["fullplot"])
print("\nNumber of missing values in each column after removal:")
print(dataset_df.isnull().sum())

# Remove the plot_embedding from each data point in the dataset as we are going to create new embeddings with an open source embedding model from Hugging Face
dataset_df = dataset_df.drop(columns=["plot_embedding"])
dataset_df.head(5)
```

## Step 3: Generating embeddings

**The steps in the code snippets are as follows:**
1. Import the `SentenceTransformer` class to access the embedding models.
2. Load the embedding model using the `SentenceTransformer` constructor to instantiate the `gte-large` embedding model.
3. Define the `get_embedding` function, which takes a text string as input and returns a list of floats representing the embedding. The function first checks if the input text is not empty (after stripping whitespace). If the text is empty, it returns an empty list. Otherwise, it generates an embedding using the loaded model.
4. Generate embeddings by applying the `get_embedding` function to the "fullplot" column of the `dataset_df` DataFrame, generating embeddings for each movie's plot. The resulting list of embeddings is assigned to a new column named embedding.

*Note: It's not necessary to chunk the text in the full plot, as we can ensure that the text length remains within a manageable range.*

```python
from sentence_transformers import SentenceTransformer

# https://huggingface.co/thenlper/gte-large
embedding_model = SentenceTransformer("thenlper/gte-large")


def get_embedding(text: str) -> list[float]:
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []

    embedding = embedding_model.encode(text)

    return embedding.tolist()


dataset_df["embedding"] = dataset_df["fullplot"].apply(get_embedding)

dataset_df.head()
```

## Step 4: Database setup and connection

MongoDB acts as both an operational and a vector database. It offers a database solution that efficiently stores, queries and retrieves vector embeddings—the advantages of this lie in the simplicity of database maintenance, management and cost.

**To create a new MongoDB database, set up a database cluster:**

1. Head over to MongoDB official site and register for a [free MongoDB Atlas account](https://www.mongodb.com/cloud/atlas/register?utm_campaign=devrel&utm_source=community&utm_medium=cta&utm_content=Partner%20Cookbook&utm_term=richmond.alake), or for existing users, [sign into MongoDB Atlas](https://account.mongodb.com/account/login?utm_campaign=devrel&utm_source=community&utm_medium=cta&utm_content=Partner%20Cookbook&utm_term=richmond.alakee).

2. Select the 'Database' option on the left-hand pane, which will navigate to the Database Deployment page, where there is a deployment specification of any existing cluster. Create a new database cluster by clicking on the "+Create" button.

3.   Select all the applicable configurations for the database cluster. Once all the configuration options are selected, click the “Create Cluster” button to deploy the newly created cluster. MongoDB also enables the creation of free clusters on the “Shared Tab”.

 *Note: Don’t forget to whitelist the IP for the Python host or 0.0.0.0/0 for any IP when creating proof of concepts.*

4. After successfully creating and deploying the cluster, the cluster becomes accessible on the ‘Database Deployment’ page.

5. Click on the “Connect” button of the cluster to view the option to set up a connection to the cluster via various language drivers.

6. This tutorial only requires the cluster's URI(unique resource identifier). Grab the URI and copy it into the Google Colabs Secrets environment in a variable named `MONGO_URI` or place it in a .env file or equivalent.

### 4.1 Database and Collection Setup

Before moving forward, ensure the following prerequisites are met
- Database cluster set up on MongoDB Atlas
- Obtained the URI to your cluster

For assistance with database cluster setup and obtaining the URI, refer to our guide for [setting up a MongoDB cluster](https://www.mongodb.com/docs/guides/atlas/cluster/) and [getting your connection string](https://www.mongodb.com/docs/guides/atlas/connection-string/)

Once you have created a cluster, create the database and collection within the MongoDB Atlas cluster by clicking + Create Database in the cluster overview page. 

Here is a guide for [creating a database and collection](https://www.mongodb.com/basics/create-database)

**The database will be named `movies`.**

**The collection will be named `movie_collection_2`.**

## Step 5: Create a Vector Search Index

At this point make sure that your vector index is created via MongoDB Atlas.

This next step is mandatory for conducting efficient and accurate vector-based searches based on the vector embeddings stored within the documents in the `movie_collection_2` collection. 

Creating a Vector Search Index enables the ability to traverse the documents efficiently to retrieve documents with embeddings that match the query embedding based on vector similarity. 

Go here to read more about [MongoDB Vector Search Index](https://www.mongodb.com/docs/atlas/atlas-search/field-types/knn-vector/).

```
{
 "fields": [{
     "numDimensions": 1024,
     "path": "embedding",
     "similarity": "cosine",
     "type": "vector"
   }]
}

```

The `1024` value of the numDimension field corresponds to the dimension of the vector generated by the gte-large embedding model. If you use the `gte-base` or `gte-small` embedding models, the numDimension value in the vector search index must be set to 768 and 384, respectively.

## Step 6: Establish Data Connection

The code snippet below also utilises PyMongo to create a MongoDB client object, representing the connection to the cluster and enabling access to its databases and collections.

```python
import pymongo
from google.colab import userdata


def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB."""
    try:
        client = pymongo.MongoClient(mongo_uri)
        print("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None


mongo_uri = userdata.get("MONGO_URI")
if not mongo_uri:
    print("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(mongo_uri)

# Ingest data into MongoDB
db = mongo_client["movies"]
collection = db["movie_collection_2"]
```

```python
# Delete any existing records in the collection
collection.delete_many({})
```

Ingesting data into a MongoDB collection from a pandas DataFrame is a straightforward process that can be efficiently accomplished by converting the DataFrame into dictionaries and then utilising the `insert_many` method on the collection to pass the converted dataset records.

```python
documents = dataset_df.to_dict("records")
collection.insert_many(documents)

print("Data ingestion into MongoDB completed")
```

## Step 7: Perform Vector Search on User Queries

The following step implements a function that returns a vector search result by generating a query embedding and defining a MongoDB aggregation pipeline. 

The pipeline, consisting of the `$vectorSearch` and `$project` stages, executes queries using the generated vector and formats the results to include only the required information, such as plot, title, and genres while incorporating a search score for each result.

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

    # Define the vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 150,  # Number of candidate matches to consider
                "limit": 4,  # Return top 4 matches
            }
        },
        {
            "$project": {
                "_id": 0,  # Exclude the _id field
                "fullplot": 1,  # Include the plot field
                "title": 1,  # Include the title field
                "genres": 1,  # Include the genres field
                "score": {"$meta": "vectorSearchScore"},  # Include the search score
            }
        },
    ]

    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)
```

## Step 8: Handling user queries and loading Gemma

```python
def get_search_result(query, collection):

    get_knowledge = vector_search(query, collection)

    search_result = ""
    for result in get_knowledge:
        search_result += f"Title: {result.get('title', 'N/A')}, Plot: {result.get('fullplot', 'N/A')}\n"

    return search_result
```

```python
# Conduct query with retrival of sources
query = "What is the best romantic movie to watch and why?"
source_information = get_search_result(query, collection)
combined_information = f"Query: {query}\nContinue to answer the query by using the Search Results:\n{source_information}."

print(combined_information)
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
# CPU Enabled uncomment below 👇🏽
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")
# GPU Enabled use below 👇🏽
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto")
```

```python
# Moving tensors to GPU
input_ids = tokenizer(combined_information, return_tensors="pt").to("cuda")
response = model.generate(**input_ids, max_new_tokens=500)
print(tokenizer.decode(response[0]))
```

```python

```