# Building a Vector Search API with Pinecone and Retool for Custom GPTs

This guide walks you through creating a vector search system using Pinecone as a vector database and exposing it as an API via Retool. You'll then connect this API to a Custom GPT as an action, enabling ChatGPT to query your private knowledge base.

## Prerequisites

Before starting, ensure you have:
- A [Pinecone](https://www.pinecone.io/) account
- A [Retool](https://retool.com/) account
- A Custom GPT with actions enabled
- An OpenAI API key

## Setup

First, install the required Python libraries:

```bash
pip install -qU openai pinecone
```

Then, import the necessary modules and initialize the clients:

```python
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from openai import OpenAI

# Initialize clients
client = OpenAI()  # Uses OPENAI_API_KEY environment variable
pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")
```

## Step 1: Prepare Your Data

Define a sample dataset that you'll embed and store in Pinecone. This data will serve as the knowledge base for your GPT to query.

```python
data = [
    {"id": "vec1", "text": "OpenAI is a leading AI research organization focused on advancing artificial intelligence."},
    {"id": "vec2", "text": "The ChatGPT platform is renowned for its natural language processing capabilities."},
    {"id": "vec3", "text": "Many users leverage ChatGPT for tasks like creative writing, coding assistance, and customer support."},
    {"id": "vec4", "text": "OpenAI has revolutionized AI development with innovations like GPT-4 and its user-friendly APIs."},
    {"id": "vec5", "text": "ChatGPT makes AI-powered conversations accessible to millions, enhancing productivity and creativity."},
    {"id": "vec6", "text": "OpenAI was founded in December 2015 as an organization dedicated to advancing digital intelligence for the benefit of humanity."}
]
```

## Step 2: Create Embeddings

Create a function to convert text into vector embeddings using OpenAI's embedding model. This function will be used both for storing data and for querying.

```python
def embed(text):
    """Convert text to embeddings using OpenAI's text-embedding-3-large model."""
    text = text.replace("\n", " ")  # Clean text by removing newlines
    res = client.embeddings.create(input=[text], model="text-embedding-3-large")
    return res.data[0].embedding

# Generate embeddings for all documents
doc_embeds = [embed(d["text"]) for d in data]
```

## Step 3: Create a Pinecone Index

Create a Pinecone index programmatically. This index will store your vector embeddings and enable fast similarity searches.

```python
def create_index():
    """Create a Pinecone index if it doesn't already exist."""
    index_name = "openai-cookbook-pinecone-retool"

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=3072,  # Must match the embedding dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    return pc.Index(index_name)

index = create_index()
```

## Step 4: Populate the Index

Prepare your vectors by combining embeddings with their original text and metadata, then upsert them into the Pinecone index.

```python
def append_vectors(data, doc_embeds):
    """Combine document data with their embeddings for Pinecone storage."""
    vectors = []
    for d, e in zip(data, doc_embeds):
        vectors.append({
            "id": d['id'],
            "values": e,
            "metadata": {'text': d['text']}
        })
    return vectors

vectors = append_vectors(data, doc_embeds)

# Upsert vectors into the index with a namespace for organization
index.upsert(
    vectors=vectors,
    namespace="ns1"
)
```

## Step 5: Test the Search Functionality

Verify that your vector search works by querying the index with a sample question.

```python
query = "When was OpenAI founded?"
query_embedding = embed(query)

results = index.query(
    namespace="ns1",
    vector=query_embedding,
    top_k=1,  # Return only the top result
    include_values=False,
    include_metadata=True  # Include the original text in results
)

print(results)
```

## Step 6: Create a Retool Workflow

Now, create a Retool workflow that will serve as an API endpoint for querying your Pinecone index.

1. **Create a new workflow** in Retool and switch to Python mode.

2. **Add the following code** to the workflow's code block. This code:
   - Imports the necessary libraries
   - Initializes the OpenAI and Pinecone clients using Retool's configuration variables
   - Defines the embedding function
   - Queries the Pinecone index with the user's query

```python
from pinecone import Pinecone
from openai import OpenAI

# Initialize clients using Retool's secure configuration variables
client = OpenAI(api_key=retoolContext.configVars.openai_api_key) 
pc = Pinecone(api_key=retoolContext.configVars.pinecone_api_key)
index = pc.Index("openai-cookbook-pinecone-retool")

def embed(query):
    """Create embeddings for the query text."""
    res = client.embeddings.create(
        input=query,
        model="text-embedding-3-large"
    )
    doc_embeds = [r.embedding for r in res.data] 
    return doc_embeds 

# Generate embedding for the query passed from the workflow trigger
x = embed([startTrigger.data.query])

# Query Pinecone index
results = index.query(
    namespace="ns1",
    vector=x[0],
    top_k=2,  # Return top 2 matches
    include_values=False,
    include_metadata=True
)

return results.to_dict()['matches']
```

3. **Configure the workflow trigger:**
   - Go to Triggers and enable the Webhook
   - Add an alias like `vector_search` for a cleaner URL
   - Save your changes

4. **Deploy the workflow** by clicking the Deploy button. Your workflow is now accessible via API.

## Step 7: Create a Custom GPT Action

Finally, connect your Retool workflow to a Custom GPT as an action.

1. **Go to your GPT** and create a new action.

2. **Use the following OpenAPI specification**, replacing `YOUR_URL_HERE` with your Retool workflow URL:

```yaml
openapi: 3.1.0
info:
  title: Vector Search API
  description: An API for performing vector-based search queries.
  version: 1.0.0
servers:
  - url: YOUR_URL_HERE
    description: Sandbox server for the Vector Search API
paths:
  /url/vector-search:
    post:
      operationId: performVectorSearch
      summary: Perform a vector-based search query.
      description: Sends a query to the vector search API and retrieves results.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
                  description: The search query.
              required:
                - query
      responses:
        '200':
          description: Successful response containing search results.
        '400':
          description: Bad Request. The input data is invalid.
        '500':
          description: Internal Server Error. Something went wrong on the server side.
```

3. **Configure authentication:**
   - Set the auth method to API Key
   - Paste your Retool workflow API key
   - Set Auth Type to Custom
   - Set the Custom Header Name to `X-Workflow-Api-Key`

## Testing Your Setup

Your setup is now complete! You can test it by:

1. Sending a message to your Custom GPT asking a question related to your stored data
2. The GPT will use the action to query your Pinecone index via the Retool workflow
3. The GPT will receive the most relevant text snippets and use them to formulate an answer

This integration enables your Custom GPT to access and reference your private knowledge base, enhancing its capabilities with domain-specific information stored in your vector database.