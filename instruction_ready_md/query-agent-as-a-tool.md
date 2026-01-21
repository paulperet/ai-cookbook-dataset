# Weaviate Query Agent with Gemini API: A Step-by-Step Guide

This guide demonstrates how to integrate the Weaviate Query Agent as a tool within the Gemini API. You will learn to connect a Weaviate Cloud instance to Google's Gemini model, enabling it to answer questions by querying your Weaviate database.

## Prerequisites

Before you begin, ensure you have the following:

1.  **Weaviate Cloud Instance (WCD):** The Query Agent is currently only accessible through Weaviate Cloud. You can create a [serverless cluster or a free 14-day sandbox](https://console.weaviate.cloud/).
2.  **Google Cloud Project & Gemini API Key:** Generate an API key from [Google AI Studio](https://aistudio.google.com/).
3.  **Data in Weaviate:** Your Weaviate cluster must contain data. If you need to populate it, you can follow the [Weaviate Import Example](integrations/Weaviate-Import-Example.ipynb) to import sample blog posts.

## Step 1: Setup and Installation

First, install the required Python libraries.

```bash
pip install -U google-genai
pip install -U "weaviate-client[agents]"
```

## Step 2: Import Libraries and Configure Environment

Create a new Python script or notebook and import the necessary modules. Then, set your API keys and Weaviate cluster URL as environment variables.

```python
import os
import weaviate
from weaviate_agents.query import QueryAgent
from google import genai
from google.genai import types

# Set your API keys and Weaviate URL
os.environ["WEAVIATE_URL"] = "your-weaviate-cluster-url"  # e.g., https://your-project.weaviate.network
os.environ["WEAVIATE_API_KEY"] = "your-weaviate-api-key"
os.environ["GOOGLE_API_KEY"] = "your-google-gemini-api-key"
```

## Step 3: Initialize the Gemini Client

Create a client instance to interact with the Gemini API.

```python
client = genai.Client()
```

## Step 4: Define the Query Agent Function

This function acts as the bridge between Gemini and your Weaviate database. It establishes a connection to your Weaviate cluster, initializes the Query Agent for a specific collection, and runs a natural language query.

```python
def query_agent_request(query: str) -> str:
    """
    Send a query to the database and get the response.

    Args:
        query (str): The question or query to search for in the database.
                     This can be any natural language question related to the content stored in the database.

    Returns:
        str: The response from the database containing relevant information.
    """

    # Connect to your Weaviate Cloud instance
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=os.getenv("WEAVIATE_URL"),
        auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
        headers={
            # Optional: Add headers for your model provider if needed.
            # Example: headers={"X-Goog-Studio-Api-Key": os.getenv("GEMINI_API_KEY")}
        }
    )

    # Initialize the Query Agent for your target collection(s)
    query_agent = QueryAgent(
        client=weaviate_client,
        collections=["WeaviateBlogChunks"]  # Replace with your collection name
    )
    # Run the query and return the final answer
    return query_agent.run(query).final_answer
```

**Note:** Replace `"WeaviateBlogChunks"` with the name of the collection you wish to query in your Weaviate instance.

## Step 5: Configure Gemini to Use the Tool

Now, you need to tell the Gemini model about your new `query_agent_request` function so it can call it when needed.

```python
config = types.GenerateContentConfig(tools=[query_agent_request])
```

## Step 6: Execute a Query

Finally, you can start a chat session with Gemini, providing it with a prompt. The model will automatically decide if it needs to use the Query Agent tool to fetch information from your database to formulate its answer.

```python
prompt = """
You are connected to a database that has a blog post on deploying Weaviate on Docker.
Can you answer how I can run Weaviate with Docker?
"""

# Create a chat session with the tool configuration
chat = client.chats.create(model='gemini-2.0-flash', config=config)

# Send the prompt and print the response
response = chat.send_message(prompt)
print(response.text)
```

When you run this code, Gemini will process your prompt, recognize the need for specific information about Docker deployment, call the `query_agent_request` function, and return a synthesized answer based on the data retrieved from your Weaviate collection.

**Example Output:**
```
To deploy Weaviate with Docker, you need to:

1.  Install Docker and Docker Compose.
2.  Obtain the Weaviate Docker image using:
    ```bash
    docker pull cr.weaviate.io/semitechnologies/weaviate:latest
    ```
3.  Prepare a `docker-compose.yml` file, which you can generate using the Weaviate configuration tool or example files from the documentation.
4.  Start Weaviate using either:
    *   Directly with Docker:
        ```bash
        docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest
        ```
    *   Using Docker Compose:
        ```bash
        docker-compose up -d
        ```
5.  Access Weaviate at `http://localhost:8080` and configure as needed.
6.  Check if Weaviate is ready by hitting the readiness endpoint:
    ```bash
    curl localhost:8080/v1/.well-known/ready
    ```
```

## Summary

You have successfully created an AI agent that combines the reasoning power of Google's Gemini with the precise data retrieval capabilities of Weaviate. By defining the Weaviate Query Agent as a tool, you enable Gemini to answer complex, data-driven questions by querying your private database directly. This pattern is powerful for building RAG (Retrieval-Augmented Generation) applications and knowledgeable AI assistants.