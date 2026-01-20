

# Gemini API: Similarity Search using Qdrant

This notebook requires paid tier rate limits to run properly.  
(cf. pricing for more details).

## Overview

The Gemini API provides access to a family of generative AI models for generating content and solving problems. These models are designed and trained to handle both text and images as input.

Qdrant is a vector similarity search engine that offers an easy-to-use API for managing, storing, and searching vectors, with an additional payload. It is a production-ready service.

In this notebook, you'll learn how to perform a similarity search on data from a website with the help of Gemini API and Qdrant.

## Setup

First, you must install the packages and set the necessary environment variables.

### Installation

Install google's python client SDK for the Gemini API, `google-genai`. Next, install Qdrant's Python client SDK, `qdrant-client`.


```
%pip install -q "google-genai>=1.0.0"
%pip install -q protobuf==4.25.1 qdrant-client[fastembed]
```

## Configure your API key

To run the following cell, your API key must be stored it in a Colab Secret named `GOOGLE_API_KEY`. If you don't already have an API key, or you're not sure how to create a Colab Secret, see Authentication for an example.


```
from google.colab import userdata
from google import genai

GEMINI_API_KEY=userdata.get('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)
```

## Basic steps

Semantic search is the process using which search engines interpret and match keywords to a user's intent in organic search results. It goes beyond surface-level keyword matching. It uses the meaning of words, phrases, and context using advanced algorithms resulting in more relevant and user-friendly search experiences.

Semantic searches rely on vector embeddings which can best match the user query to the most similar result.

In this tutorial, you'll implement the three main components of semantic search:

1. Create an index

    Create and store the index for the data in the Qdrant vector store. You will use a Gemini API embedding model to create embedding vectors that can be stored in the Qdrant vector store.

2. Query the index

    Query the index using a query string to return the top `n` neighbors of the query.

You'll learn more about these stages in the upcoming sections while implementing the application.

## Import the required libraries


```
from bs4 import BeautifulSoup
from qdrant_client import models, QdrantClient
from urllib.request import urlopen
```

## 1. Create an index

In this stage, you will perform the following steps:

1. Read and parse the website data using Python's BeautifulSoup library.

2. Create embeddings of the website data.

3. Store the embeddings in Qdrant's vector database.
    
    Qdrant is a vector similarity search engine. Along with a convenient API to store, search, and manage points(i.e. vectors), it also provides an option to add an additional payload. The payloads are essentially extra bits of data that you can utilize to refine your search and obtain relevant information that you can then share with your users.

### Read and parse the website data

To read the website data as text, you will use the `BeautifulSoup` library from Python.


```
url = "https://blog.google/outreach-initiatives/sustainability/"\
      "report-ai-sustainability-google-cop28/"
html = urlopen(url).read()
soup = BeautifulSoup(html, features="html.parser")

# Remove all script and style elements
for script in soup(["script", "style"]):
    script.extract()    # Self-destruct

# Get the text
text_content = soup.get_text()
```

If you only want to select a specific portion of the website data to add context to the prompt, you can use regex, text slicing, or text splitting.

In this example, you'll use Python's `split()` function to extract the required portion of the text.


```
# The text content between the substrings "Later this month at COP28" to
# "POSTED IN:" is relevant for this tutorial. You can use Python's `split()`
# to select the required content.
text_content_1 = text_content.split("Later this month at COP28",1)[1]
final_text = text_content_1.split("POSTED IN:",1)[0]

texts = final_text.split(".")

documents = []

# Convert text into a chunk of 3 sentences.
for i in range(0, len(texts), 3):
  documents.append({"content": " ".join(texts[i:i+3])})
```

### Initialize the embedding model

To create the embeddings from the website data, you'll use the **gemini-embedding-001** model, which supports creating embeddings from text.

To use the embedding model, you have to use the `embed_content` function from the `google-genai` package. To learn more about the embedding model, read the model documentation.

One of the arguments passed to the embedding function is `task_type`. Specifying the `task_type` parameter ensures the model produces appropriate embeddingsfor the expected task and inputs. It is a string that can take on one of the following values:

| task_type	  |  Description |
|---|---|
| `RETRIEVAL_QUERY` | Specifies the given text is a query in a search or retrieval setting. |
| `RETRIEVAL_DOCUMENT` | Specifies the given text is a document in a search or retrieval setting. |  
| `SEMANTIC_SIMILARITY` | Specifies the given text will be used for Semantic Textual Similarity (STS). |  
| `CLASSIFICATION` | Specifies that the embeddings will be used for classification. |
| `CLUSTERING` | Specifies that the embeddings will be used for clustering. |


```
from google.genai import types

# Select embedding model
MODEL_ID = "gemini-embedding-001"  # @param ["gemini-embedding-001", "text-embedding-004"] {"allow-input": true, "isTemplate": true}


# Function to convert text to embeddings
def make_embed_text_fn(text, model=MODEL_ID,
                       task_type="retrieval_document"):
    embedding = client.models.embed_content(
        model=model,
        contents=text,
        config=types.EmbedContentConfig(
          task_type=task_type,
        )
    )
    return embedding.embeddings[0].values
```

### Store the data using Qdrant

 Next, you'll store the embeddings of the website data in Qdrant's vector store.

 First, you have to initiate a Qdrant client by creating an instance of `QdrantClient`. In this tutorial, you will store the embeddings in memory. To create an in-memory Qdrant client specify `:memory:` for the `location` argument of the `QdrantClient` class initializer. You can read more about the different types of storage in Qdrant in the storage reference guide.

After initializing the client, you have to create a Qdrant collection using the `recreate_collection` function of `QdrantClient`. You can specify your vector configuration inside the `recreate_collection` function. Pass an instance of `VectorParams` with the `size` set to `768` to match the embedding model and `distance` set to cosine.

**Note**: Since you will run the script several times during your experiments, `recreate_collection` is appropriate for this tutorial. `recreate_collection` will first try to remove an existing collection with the same name.


```
# Initialize Qdrant client.
qdrant = QdrantClient(":memory:")

# Create a collection named "GeminiCollection".
qdrant.create_collection(
    collection_name="GeminiCollection",
    vectors_config=models.VectorParams(
        size=3072,  # Vector size of `gemini-embedding-001`
        distance=models.Distance.COSINE,
    ),
)
```




    True



You will now insert the `documents` you parsed from the website data into the Qdrant collection you created earlier and index them using the `upsert` function of `QdrantClient`.

The `upsert` function takes the data to be stored and indexed as an array of `PointsStruct`s.

Points are the main entity in Qdrant operations. A point is a record consisting of a vector and an optional payload. You can perform a similarity search among the points in one collection. Read more about points in Qdrant's points documentation.

You'll create an array of points by enumerating over the documents you prepared earlier from the website data.


```
# Qdrant uses batch loading of points to optimize performance.
# You can create a batch in two ways - record-oriented and column-oriented.
# Here you are using the record-oriented approach.

qdrant.upsert(
    collection_name="GeminiCollection",
    points=[
        # Use PointStruct function to intialize the point.
        models.PointStruct(
            # Use `make_embed_text_fn` to convert text to embeddings.
            # Pass the same data as payload for a refined search.
            id=idx, vector=make_embed_text_fn(doc["content"]), payload = doc
        )
        for idx, doc in enumerate(documents)
    ]
)
```




    UpdateResult(operation_id=0, status=<UpdateStatus.COMPLETED: 'completed'>)



## 2. Query the index

You'll now query the Qdrant index you created earlier with a question related to the data contained in the website documents.
To query the index, you have to mention the collection name and the query vector. The query vector should be first converted to an embedding vector using the Gemini API embedding model you leveraged to create embedding vectors for the website data. Use the `make_embed_text_fn` you defined earlier for creating an embedding vector from your query. Since you are embedding a query string that is being used to search `retrieval_document` embeddings, the `task_type` must be set to `retrieval_query`.


```
hits = qdrant.search(
    collection_name="GeminiCollection",
    query_vector=make_embed_text_fn("How can AI address climate challenges?",
                                    task_type="retrieval_query"),
    limit=3,
)
for hit in hits:
    print("score:", hit.score, "- content:", hit.payload.get("content").replace("\n", ""))
```

    score: 0.8047714249217505 - content:  Already, it is starting to address climate challenges in three key areas: providing people and organizations with better information to make more sustainable choices, delivering improved predictions to help adapt to climate change, and finding recommendations to optimize climate action for high-impact applications Here’s a look at how, at Google, we’ve used AI to address climate challenges:Providing helpful information: People are looking for information to reduce their environmental footprint  Fuel-efficient routing in Google Maps uses AI to suggest routes that have fewer hills, less traffic, and constant speeds with the same or similar ETA
    score: 0.7745056851050607 - content: Managing the environmental impact of AIWhile scaling these applications of AI and finding new ways to use it to accelerate climate action is crucial, we need to build AI responsibly and manage the environmental impact associated with it As AI is at an inflection point, predicting the future growth of energy use and emissions from AI compute in our data centers is challenging  Historically, data center energy consumption has grown much more slowly than demand for computing power
    score: 0.7709980934840739 - content: Promoting environmentally and socially responsible deployment of AI Together, we can boldly and responsibly develop more tools and products that harness the power of AI to accelerate the climate progress we need 


    /tmp/ipykernel_47101/1026465469.py:1: DeprecationWarning: `search` method is deprecated and will be removed in the future. Use `query_points` instead.
      hits = qdrant.search(


## Conclusion

That's it. You have successfully performed a similarity search using Qdrant with the help of a Gemini API embedding model.