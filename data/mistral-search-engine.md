# Building a Mistral AI-Powered Search Engine: A Step-by-Step Guide

This guide walks you through building a search engine that uses Mistral AI for embeddings and a FAISS vector store. You'll learn how to scrape web results, process text, create embeddings, and query them using a conversational agent.

## Prerequisites & Setup

Before you begin, ensure you have a Mistral AI API key. Store it in a `.env` file as `MISTRAL_API_KEY`.

Install the required Python packages:

```bash
pip install aiohttp==3.9.5 beautifulsoup4==4.12.3 faiss_cpu==1.8.0 mistralai==0.4.0 nest_asyncio==1.6.0 numpy==1.26.4 pandas==2.2.2 python-dotenv==1.0.1 requests==2.32.3
```

Now, let's import the necessary modules and load your API key.

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
```

## Step 1: Define the Web Scraper and Processing Functions

First, we'll set up the core components for fetching search results, scraping web pages, and processing text into chunks.

```python
import aiohttp
import asyncio
import nest_asyncio
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import requests
import re
import pandas as pd
import faiss
import numpy as np
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Apply the nest_asyncio patch for async operations in environments like Jupyter
nest_asyncio.apply()

# Configuration
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36"
}
total_results_to_fetch = 10  # Total number of search results to scrape
chunk_size = 1000  # Size for splitting text into chunks
dataframe_out_path = 'temp.csv'
faiss_index_path = 'faiss_index.index'

# Initialize the Mistral client
mistral_api_key = MISTRAL_API_KEY
client = MistralClient(api_key=mistral_api_key)
```

### 1.1 Define Async Fetching Functions

These functions handle the asynchronous HTTP requests to Google Search and target web pages.

```python
async def fetch(session, url, params=None):
    """Generic async GET request."""
    async with session.get(url, params=params, headers=headers, timeout=30) as response:
        return await response.text()

async def fetch_page(session, params, page_num, results):
    """Fetch a single page of Google search results."""
    print(f"Fetching page: {page_num}")
    params["start"] = (page_num - 1) * params["num"]
    html = await fetch(session, "https://www.google.com/search", params)
    soup = BeautifulSoup(html, 'html.parser')

    for result in soup.select(".tF2Cxc"):
        if len(results) >= total_results_to_fetch:
            break
        title = result.select_one(".DKV0Md").text
        link = result.select_one(".yuRUbf a")["href"]
        results.append({"title": title, "links": link})
```

### 1.2 Define Content Scraping and Text Processing Functions

These functions extract raw text from URLs and split it into manageable chunks.

```python
def get_all_text_from_url(url):
    """Synchronously fetch and extract clean text from a URL."""
    response = requests.get(url, headers=headers, timeout=30)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

def split_text_into_chunks(text, chunk_size):
    """Split text into chunks of approximately `chunk_size` characters, respecting sentence boundaries."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if sum(len(s) for s in current_chunk) + len(sentence) + 1 > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
        else:
            current_chunk.append(sentence)

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

async def process_text_content(texts, chunk_size):
    """Process a list of texts into chunks asynchronously."""
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, split_text_into_chunks, text, chunk_size) for text in texts]
    return await asyncio.gather(*tasks)
```

### 1.3 Define the Embedding Function

This function sends text chunks to the Mistral API to generate vector embeddings.

```python
async def get_embeddings_from_mistral(client, text_chunks):
    """Generate embeddings for a list of text chunks using Mistral AI."""
    response = client.embeddings(model="mistral-embed", input=text_chunks)
    return [embedding.embedding for embedding in response.data]
```

## Step 2: Orchestrate the Data Pipeline

Now, we combine all the functions into a main pipeline that takes a search query, fetches results, processes them, and creates a vector store.

```python
async def fetch_and_process_data(search_query):
    """
    Main pipeline: Search Google, scrape pages, chunk text, create embeddings, and build a FAISS index.
    """
    params = {
        "q": search_query,
        "hl": "en",
        "gl": "uk",
        "start": 0,
        "num": 10
    }

    async with aiohttp.ClientSession() as session:
        # 1. Fetch search results
        page_num = 0
        results = []
        while len(results) < total_results_to_fetch:
            page_num += 1
            await fetch_page(session, params, page_num, results)

        urls = [result['links'] for result in results]

        # 2. Scrape content from each URL (using ThreadPool for synchronous requests)
        with ThreadPoolExecutor(max_workers=10) as executor:
            loop = asyncio.get_event_loop()
            texts = await asyncio.gather(
                *[loop.run_in_executor(executor, get_all_text_from_url, url) for url in urls]
            )

        # 3. Process texts into chunks
        chunks_list = await process_text_content(texts, chunk_size)

        # 4. Generate embeddings for each chunk
        embeddings_list = []
        for chunks in chunks_list:
            embeddings = await get_embeddings_from_mistral(client, chunks)
            embeddings_list.append(embeddings)

        # 5. Assemble data into a DataFrame
        data = []
        for i, result in enumerate(results):
            if i >= len(embeddings_list):
                print(f"Error: No embeddings returned for result {i}")
                continue
            for j, chunk in enumerate(chunks_list[i]):
                if j >= len(embeddings_list[i]):
                    print(f"Error: No embedding returned for chunk {j} of result {i}")
                    continue
                data.append({
                    'title': result['title'],
                    'url': result['links'],
                    'chunk': chunk,
                    'embedding': embeddings_list[i][j]
                })

        df = pd.DataFrame(data)
        df.to_csv(dataframe_out_path, index=False)

        # 6. Create and save the FAISS index
        dimension = len(embeddings_list[0][0])
        index = faiss.IndexFlatL2(dimension)
        embeddings = np.array([entry['embedding'] for entry in data], dtype=np.float32)
        index.add(embeddings)
        faiss.write_index(index, faiss_index_path)

# Example: Run the pipeline for a query
await fetch_and_process_data("What is the latest news about apple and openai?")
```

## Step 3: Query the Vector Store

With our data indexed, we need functions to query it. First, a function to convert a query string into an embedding, and a second to perform the similarity search.

```python
def query_embeddings(texts):
    """Convert a text string to an embedding using Mistral AI."""
    client = MistralClient(api_key=MISTRAL_API_KEY)
    # Note: The API expects a list. We wrap a single string in a list.
    response = client.embeddings(model="mistral-embed", input=[texts])
    return [embedding.embedding for embedding in response.data]

def query_vector_store(query_embedding, k=5):
    """
    Query the FAISS vector store and return the text results along with metadata.

    :param query_embedding: The embedding vector to query with.
    :param k: Number of nearest neighbors to retrieve.
    :return: List of dictionaries containing text chunks and metadata.
    """
    # Load the index and data
    index = faiss.read_index(faiss_index_path)
    df = pd.read_csv(dataframe_out_path)

    # Ensure the query embedding is in the correct format
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = np.array(query_embedding, dtype=np.float32)
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    # Perform the search
    distances, indices = index.search(query_embedding, k)

    # Retrieve the results
    results = []
    for idx in indices[0]:
        result = {
            'title': df.iloc[idx]['title'],
            'url': df.iloc[idx]['url'],
            'chunk': df.iloc[idx]['chunk']
        }
        results.append(result)
    return results

# Test the query system
embeddings = query_embeddings("AGI")
search_results = query_vector_store(embeddings[0], k=5)
print(search_results)
```

## Step 4: Define a Tool for the AI Agent

To integrate this search capability into a conversational AI, we define it as a "tool" that the model can call.

```python
import functools
import json

# Define the tool schema for the Mistral API
tools = [
    {
        "type": "function",
        "function": {
            "name": "mistral_web_search",
            "description": "Fetch and process data from Google search based on a query, store results in FAISS vector store, and retrieve results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "The search query to use for fetching data from Google search."
                    }
                },
                "required": ["search_query"]
            },
        },
    },
]

# Define the function that implements the tool
def mistral_web_search(search_query: str):
    """The callable function for the search tool. It runs the full pipeline and returns results."""
    async def run_search():
        await fetch_and_process_data(search_query)
        embeddings = query_embeddings(search_query)
        results_ = query_vector_store(embeddings[0], k=5)
        return results_
    return asyncio.run(run_search())

# Helper function to extract just the text from tool results
def tools_to_str(tools_output: list) -> str:
    """Extract and concatenate the 'chunk' text from a list of tool result dictionaries."""
    return '\n'.join([tool['chunk'] for tool in tools_output])

# Map the tool name to its function for the agent to call
names_to_functions = {
    'mistral_web_search': functools.partial(mistral_web_search),
}

# Test the tool
search_query = "mistral and openai"
results = mistral_web_search(search_query)
print("Tool Output:", results)
print("Extracted Text:\n", tools_to_str(results))
```

## Step 5: Integrate with the Mistral Chat API

Finally, we use the Mistral chat model with tool calling to create an interactive agent that can use our search tool.

### 5.1 Single Turn with Tool Calling

Let's see how a single user query triggers the tool and gets a final answer.

```python
# Initialize the chat
model = "mistral-large-latest"
client = MistralClient(api_key=MISTRAL_API_KEY)

messages = [
    ChatMessage(role="user", content="What happened during apple WWDC 2024?"),
]

# First API call: The model decides to use the tool
response = client.chat(model=model, messages=messages, tools=tools, tool_choice="any")
messages.append(response.choices[0].message)

# Extract the tool call details
tool_call = response.choices[0].message.tool_calls[0]
function_name = tool_call.function.name
function_params = json.loads(tool_call.function.arguments)

print(f"Model called function: {function_name}")
print(f"With parameters: {function_params}")

# Execute the tool
function_result_raw = names_to_functions[function_name](**function_params)
function_result_text = tools_to_str(function_result_raw)

# Provide the tool's result back to the model
messages.append(ChatMessage(role="tool", name=function_name, content=function_result_text, tool_call_id=tool_call.id))

# Final API call: The model synthesizes an answer using the search results
response = client.chat(model=model, messages=messages)
final_answer = response.choices[0].message.content
print("\nFinal Answer:\n", final_answer)
```

### 5.2 Interactive Chat Loop

For a complete user experience, here's a loop that handles the conversation flow automatically.

```python
messages = []
model = "mistral-large-latest"

while True:
    # Get user input
    user_input = input("\nAsk: ")
    if user_input.lower() in ['quit', 'exit']:
        break

    messages.append(ChatMessage(role="user", content=user_input))

    # Step 1: Model decides if/how to use the tool
    response = client.chat(model=model, messages=messages, tools=tools, tool_choice="any")
    assistant_message = response.choices[0].message
    messages.append(assistant_message)

    # Check if a tool was called
    if assistant_message.tool_calls:
        tool_call = assistant_message.tool_calls[0]
        function_name = tool_call.function.name
        function_params = json.loads(tool_call.function.arguments)

        # Execute the tool
        print("Searching for information...")
        function_result_raw = names_to_functions[function_name](**function_params)

        # Display sources
        print("Sources found:")
        for source in function_result_raw:
            print(f"  - {source['title']} ({source['url']})")

        # Format the tool result as text for the model
        function_result_text = tools_to_str(function_result_raw)
        messages.append(ChatMessage(role="tool", name=function_name, content=function_result_text, tool_call_id=tool_call.id))

        # Step 2: Get the final answer from the model
        response = client.chat(model=model, messages=messages)
        final_response = response.choices[0].message.content
        print("\nAnswer:", final_response)
        messages.append(ChatMessage(role="assistant", content=final_response))
    else:
        # If no tool was called, just display the model's response
        print(assistant_message.content)
```

## Summary

You've successfully built a Mistral AI-powered search engine that:
1.  **Scrapes** Google Search results and web page content.
2.  **Processes** text into chunks and generates embeddings using Mistral AI.
3.  **Indexes** the embeddings in a FAISS vector store for efficient similarity search.
4.  **Exposes** this capability as a tool for a Mistral chat model.
5.  **Creates** an interactive agent that can answer questions by searching the web and synthesizing the results.

This pipeline demonstrates a core RAG (Retrieval-Augmented Generation) pattern, combining external data retrieval with the reasoning power of a large language model.