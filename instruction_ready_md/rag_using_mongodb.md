# Build a RAG System with Claude 3 and MongoDB

This guide walks you through building a Retrieval-Augmented Generation (RAG) chatbot that acts as a Venture Capital Tech Analyst. The system uses a collection of tech news articles as its knowledge base, retrieves relevant context via MongoDB's vector search, and generates informed responses using Anthropic's Claude 3 model.

## Prerequisites

You will need API keys for the following services:
- **Claude API Key** (from [Anthropic Console](https://console.anthropic.com/settings/keys))
- **VoyageAI API Key** (from [VoyageAI](https://docs.voyageai.com/docs/quick-start))
- **Hugging Face Access Token** (from [Hugging Face Settings](https://huggingface.co/settings/tokens))

## Step 1: Setup and Data Preparation

### 1.1 Install Required Libraries

Begin by installing the necessary Python packages.

```bash
!pip install pymongo datasets pandas anthropic voyageai
```

### 1.2 Import Libraries and Define Helper Functions

Import the required modules and define a function to download and combine the dataset from Hugging Face.

```python
from io import BytesIO
import pandas as pd
import requests
from google.colab import userdata  # For Colab secrets; adjust for your environment

def download_and_combine_parquet_files(parquet_file_urls, hf_token):
    """
    Downloads Parquet files from the provided URLs using the given Hugging Face token,
    and returns a combined DataFrame.

    Parameters:
    - parquet_file_urls: List of strings, URLs to the Parquet files.
    - hf_token: String, Hugging Face authorization token.

    Returns:
    - combined_df: A pandas DataFrame containing the combined data from all Parquet files.
    """
    headers = {"Authorization": f"Bearer {hf_token}"}
    all_dataframes = []

    for parquet_file_url in parquet_file_urls:
        response = requests.get(parquet_file_url, headers=headers, timeout=60)
        if response.status_code == 200:
            parquet_bytes = BytesIO(response.content)
            df = pd.read_parquet(parquet_bytes)
            all_dataframes.append(df)
        else:
            print(f"Failed to download Parquet file from {parquet_file_url}: {response.status_code}")

    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        return combined_df
    else:
        print("No dataframes to concatenate.")
        return None
```

### 1.3 Load the Dataset

We'll use a subset of the `tech-news-embeddings` dataset. For this tutorial, we start with one file to keep things manageable. You can uncomment additional URLs to load more data.

```python
# Uncomment the links below to load more data
# Full list: https://huggingface.co/datasets/MongoDB/tech-news-embeddings/tree/refs%2Fconvert%2Fparquet/default/train
parquet_files = [
    "https://huggingface.co/api/datasets/AIatMongoDB/tech-news-embeddings/parquet/default/train/0000.parquet",
    # "https://huggingface.co/api/datasets/AIatMongoDB/tech-news-embeddings/parquet/default/train/0001.parquet",
    # "https://huggingface.co/api/datasets/AIatMongoDB/tech-news-embeddings/parquet/default/train/0002.parquet",
]

hf_token = userdata.get("HF_TOKEN")  # Retrieve your Hugging Face token from secrets
combined_df = download_and_combine_parquet_files(parquet_files, hf_token)
```

### 1.4 Clean and Prepare the Data

Prepare the DataFrame by removing unnecessary columns and limiting the number of documents for this demo to stay within API rate limits.

```python
# Remove the '_id' column from the initial dataset
combined_df = combined_df.drop(columns=["_id"])

# Remove the initial 'embedding' column as we will create new embeddings
combined_df = combined_df.drop(columns=["embedding"])

# Limit the number of documents to 500 for this demo due to VoyageAI API rate limits
max_documents = 500
if len(combined_df) > max_documents:
    combined_df = combined_df[:max_documents]
```

### 1.5 Generate New Embeddings with VoyageAI

We'll generate fresh embeddings for the article descriptions using VoyageAI's `voyage-large-2` model.

```python
import voyageai

vo = voyageai.Client(api_key=userdata.get("VOYAGE_API_KEY"))

def get_embedding(text: str) -> list[float]:
    """Generate an embedding for a given text string."""
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []

    embedding = vo.embed(text, model="voyage-large-2", input_type="document")
    return embedding.embeddings[0]

# Apply the embedding function to the 'description' column
combined_df["embedding"] = combined_df["description"].apply(get_embedding)
```

## Step 2: Set Up MongoDB

### 2.1 Create a MongoDB Atlas Cluster

1.  Register for a [free MongoDB Atlas account](https://www.mongodb.com/cloud/atlas/register) or sign in if you already have one.
2.  Click **"+ Create"** to deploy a new database cluster.
3.  Follow the setup wizard. For proof-of-concept projects, you can whitelist all IPs (`0.0.0.0/0`) in the network access settings.
4.  Once the cluster is deployed, click the **"Connect"** button.
5.  Choose **"Connect your application"** and copy the connection string (URI).

### 2.2 Create Database and Collection

Within your cluster in Atlas:
1.  Click **"+ Create Database"**.
2.  Name the database `tech_news`.
3.  Name the collection `hacker_noon_tech_news`.

### 2.3 Create a Vector Search Index

To enable semantic search on the embeddings, you must create a vector search index.

1.  In your Atlas cluster, navigate to the **"Search"** tab for your collection.
2.  Click **"Create Search Index"**.
3.  Select the **"JSON Editor"** method.
4.  Name the index `vector_index`.
5.  Use the following index definition:

```json
{
  "fields": [{
    "numDimensions": 1536,
    "path": "embedding",
    "similarity": "cosine",
    "type": "vector"
  }]
}
```

6.  Click **"Create Index"**. The index creation will take a few minutes.

## Step 3: Ingest Data into MongoDB

### 3.1 Connect to MongoDB

Now, let's connect to your MongoDB cluster from Python and prepare the collection.

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

# Retrieve your MongoDB URI from secrets
mongo_uri = userdata.get("MONGO_URI")
if not mongo_uri:
    print("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(mongo_uri)

DB_NAME = "tech_news"
COLLECTION_NAME = "hacker_noon_tech_news"

db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]
```

### 3.2 Clear Existing Data and Ingest New Data

We'll ensure we start with a fresh collection and then insert our prepared DataFrame.

```python
# Clear any existing records in the collection
collection.delete_many({})

# Convert the DataFrame to a list of dictionaries and insert into MongoDB
combined_df_json = combined_df.to_dict(orient="records")
collection.insert_many(combined_df_json)
```

## Step 4: Implement Vector Search

Create a function that performs a vector search on the MongoDB collection using a user's query.

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

    # Define the vector search aggregation pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 150,  # Number of candidate matches to consider
                "limit": 5,  # Return top 5 matches
            }
        },
        {
            "$project": {
                "_id": 0,  # Exclude the _id field
                "embedding": 0,  # Exclude the embedding field
                "score": {"$meta": "vectorSearchScore"},  # Include the search score
            }
        },
    ]

    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)
```

## Step 5: Build the RAG Query Handler

### 5.1 Initialize the Claude Client

Set up the Anthropic client to use the Claude 3 model.

```python
import anthropic

client = anthropic.Client(api_key=userdata.get("ANTHROPIC_API_KEY"))
```

### 5.2 Create the RAG Response Function

This is the core function that ties everything together: it retrieves relevant context and uses Claude to generate a final answer.

```python
def rag_response(user_query, collection):
    """
    Generate a response using RAG: retrieve relevant context and query Claude 3.
    """
    # 1. Perform vector search to find relevant articles
    search_results = vector_search(user_query, collection)

    # 2. Compile the search results into a context string
    search_result = ""
    for res in search_results:
        search_result += (
            f"Title: {res.get('title', 'N/A')}\n"
            f"Company Name: {res.get('company_name', 'N/A')}\n"
            f"URL: {res.get('url', 'N/A')}\n"
            f"Publication Date: {res.get('publication_date', 'N/A')}\n"
            f"Article URL: {res.get('article_url', 'N/A')}\n"
            f"Description: {res.get('description', 'N/A')}\n\n"
        )

    # 3. Construct the prompt for Claude
    system_prompt = """You are a Venture Capital Tech Analyst. You have access to a database of tech company articles and information.
    Use the provided context to answer the user's query accurately and concisely. If the context does not contain relevant information, state that clearly."""

    user_prompt = f"Context:\n{search_result}\n\nUser Query: {user_query}"

    # 4. Call the Claude API
    message = client.messages.create(
        model="claude-3-opus-20240229",  # You can use other Claude 3 models like 'claude-3-sonnet-20240229'
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )

    return message.content[0].text
```

## Step 6: Test Your RAG System

You can now test the system by asking a question related to tech news.

```python
# Example query
user_question = "What are the latest trends in quantum computing?"
answer = rag_response(user_question, collection)
print(answer)
```

## Summary

You have successfully built a RAG system that:
1.  Loads and prepares a tech news dataset.
2.  Generates embeddings using VoyageAI.
3.  Stores data in MongoDB Atlas with a vector search index.
4.  Retrieves relevant context via semantic search.
5.  Generates informed, analyst-style responses using Claude 3.

This architecture can be extended with more sophisticated retrieval strategies, hybrid search, or agentic workflows for more complex analyses.