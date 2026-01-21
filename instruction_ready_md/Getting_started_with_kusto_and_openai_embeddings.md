# Using Azure Data Explorer (Kusto) as a Vector Database for AI Embeddings

This guide provides a step-by-step tutorial on using Azure Data Explorer (Kusto) as a vector database for semantic search with OpenAI embeddings. You will learn how to store precomputed embeddings in Kusto, generate new embeddings for search queries, and perform cosine similarity searches to find relevant documents.

## Prerequisites

Before you begin, ensure you have the following:

1.  An **Azure Data Explorer (Kusto)** cluster and database.
2.  **OpenAI API credentials** (either from Azure OpenAI or OpenAI directly).

## Setup: Install Required Libraries

First, install the necessary Python packages.

```bash
pip install wget openai azure-kusto-data
```

## Step 1: Download Precomputed Embeddings

To save time and computational resources, we'll use a precomputed dataset of Wikipedia article embeddings.

```python
import wget
import zipfile
import pandas as pd
from ast import literal_eval

# Download the embeddings file (~700 MB)
embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"
wget.download(embeddings_url)

# Extract the downloaded ZIP file
with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
    zip_ref.extractall("/lakehouse/default/Files/data")

# Load the CSV file into a pandas DataFrame
article_df = pd.read_csv('/lakehouse/default/Files/data/vector_database_wikipedia_articles_embedded.csv')

# Convert the string representations of vectors back into Python lists
article_df["title_vector"] = article_df.title_vector.apply(literal_eval)
article_df["content_vector"] = article_df.content_vector.apply(literal_eval)

# Inspect the first few rows
print(article_df.head())
```

## Step 2: Store Vectors in a Kusto Table

Now, we will write the DataFrame containing the embeddings to a Kusto table. The table will be created automatically if it doesn't exist.

1.  **Configure your Kusto connection details.** Replace the placeholder values with your own.

    ```python
    # Replace with your AAD Tenant ID, Kusto Cluster URI, Kusto DB name and Kusto Table
    AAD_TENANT_ID = ""
    KUSTO_CLUSTER =  ""
    KUSTO_DATABASE = "Vector"
    KUSTO_TABLE = "Wiki"

    kustoOptions = {
        "kustoCluster": KUSTO_CLUSTER,
        "kustoDatabase": KUSTO_DATABASE,
        "kustoTable": KUSTO_TABLE
    }
    ```

2.  **Authenticate and obtain an access token.** This example uses Synapse notebook utilities. Refer to the [authentication documentation](https://github.com/Azure/azure-kusto-spark/blob/master/docs/Authentication.md) for other methods.

    ```python
    access_token = mssparkutils.credentials.getToken(kustoOptions["kustoCluster"])
    ```

3.  **Convert the pandas DataFrame to a Spark DataFrame and write it to Kusto.**

    ```python
    # Convert pandas DataFrame to Spark DataFrame
    sparkDF = spark.createDataFrame(article_df)

    # Write data to a Kusto table
    (sparkDF.write
     .format("com.microsoft.kusto.spark.synapse.datasource")
     .option("kustoCluster", kustoOptions["kustoCluster"])
     .option("kustoDatabase", kustoOptions["kustoDatabase"])
     .option("kustoTable", kustoOptions["kustoTable"])
     .option("accessToken", access_token)
     .option("tableCreateOptions", "CreateIfNotExist")
     .mode("Append")
     .save())
    ```

## Step 3: Configure OpenAI for Embedding Generation

To search our vector database, we need to convert text queries into embedding vectors. Configure the OpenAI client based on your service provider.

**Important:** You must use the `text-embedding-3-small` model, as the precomputed embeddings were created with it.

```python
import openai

def embed(query):
    """Creates an embedding vector from a text query."""
    embedded_query = openai.Embedding.create(
        input=query,
        model="text-embedding-3-small", # Use this model for consistency
    )["data"][0]["embedding"]
    return embedded_query
```

### Option A: Using Azure OpenAI

If you are using Azure OpenAI, configure the client as follows:

```python
openai.api_version = '2022-12-01'
openai.api_base = '' # Add your Azure OpenAI endpoint here
openai.api_type = 'azure'
openai.api_key = ''  # Add your Azure OpenAI API key here

def embed(query):
    """Creates an embedding vector from a text query using Azure OpenAI."""
    embedded_query = openai.Embedding.create(
            input=query,
            deployment_id="embed", # Replace with your Azure OpenAI deployment ID
            chunk_size=1
    )["data"][0]["embedding"]
    return embedded_query
```

### Option B: Using OpenAI

If you are using OpenAI directly, configure the client as follows:

```python
openai.api_key = "" # Add your OpenAI API key here
# The `embed` function defined in the first code block above will work.
```

## Step 4: Perform a Semantic Search in Kusto

With our data stored and embedding function ready, we can now perform a similarity search.

1.  **Generate an embedding for your search query.**

    ```python
    search_query = "places where you worship"
    searchedEmbedding = embed(search_query)
    ```

2.  **Set up the Kusto query client.** We'll use the `series_cosine_similarity_fl` function for the similarity calculation. Ensure this User-Defined Function (UDF) is created in your Kusto database. [See the documentation for instructions](https://learn.microsoft.com/en-us/azure/data-explorer/kusto/functions-library/series-cosine-similarity-fl?tabs=query-defined).

    ```python
    from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
    from azure.kusto.data.helpers import dataframe_from_result_table

    # Build the connection string using AAD device authentication
    KCSB = KustoConnectionStringBuilder.with_aad_device_authentication(KUSTO_CLUSTER)
    KCSB.authority_id = AAD_TENANT_ID
    KUSTO_CLIENT = KustoClient(KCSB)
    ```

3.  **Execute the similarity search query.** This query calculates the cosine similarity between our query embedding and the `content_vector` column, returning the top 10 most similar articles.

    ```python
    # Construct and run the Kusto query
    KUSTO_QUERY = f"""
    Wiki
    | extend similarity = series_cosine_similarity_fl(dynamic({searchedEmbedding}), content_vector, 1, 1)
    | top 10 by similarity desc
    """

    RESPONSE = KUSTO_CLIENT.execute(KUSTO_DATABASE, KUSTO_QUERY)
    results_df = dataframe_from_result_table(RESPONSE.primary_results[0])
    print(results_df[['title', 'similarity', 'text']].head())
    ```

    The results will include articles about "Temple," "Christian worship," "Service of worship," etc., ranked by their similarity score.

## Step 5: Search by Title Embeddings

You can also perform searches against the `title_vector` column to find articles with similar titles.

```python
# Generate an embedding for a new query
search_query_2 = "unfortunate events in history"
searchedEmbedding2 = embed(search_query_2)

# Query against the title_vector column
KUSTO_QUERY_2 = f"""
Wiki
| extend similarity = series_cosine_similarity_fl(dynamic({searchedEmbedding2}), title_vector, 1, 1)
| top 10 by similarity desc
"""

RESPONSE_2 = KUSTO_CLIENT.execute(KUSTO_DATABASE, KUSTO_QUERY_2)
title_results_df = dataframe_from_result_table(RESPONSE_2.primary_results[0])
print(title_results_df[['title', 'similarity']].head())
```

## Summary

You have successfully built a semantic search system using Azure Data Explorer as a vector database. The process involved:
1.  Loading a dataset of precomputed embeddings.
2.  Storing the vector data in a Kusto table.
3.  Configuring the OpenAI API to generate new embeddings.
4.  Performing cosine similarity searches in Kusto to find documents relevant to a text query.

This architecture provides a scalable and powerful foundation for building Retrieval-Augmented Generation (RAG) applications and other AI-powered search solutions.