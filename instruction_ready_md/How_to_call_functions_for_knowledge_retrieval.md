# Building an AI Agent with a Knowledge Base and Function Calling

This guide builds on the concept of function calling with chat models. You will create an agent that can access a knowledge base (arXiv) and intelligently choose between two functions to answer user questions about academic subjects.

The agent has access to two key functions:
1.  **`get_articles`**: Searches arXiv for papers on a given subject and returns a summarized list with links.
2.  **`read_article_and_summarize`**: Takes a specific article from a previous search, reads the full PDF, and provides a detailed summary of its core arguments, evidence, and conclusions.

This tutorial will walk you through creating a multi-function workflow where data from one function can be persisted and used by another.

## Prerequisites and Setup

First, install the required Python libraries.

```bash
pip install scipy tenacity tiktoken==0.3.3 termcolor openai arxiv pandas PyPDF2 tqdm
```

Now, import the necessary modules and set up your environment variables.

```python
import arxiv
import ast
import concurrent
import json
import os
import pandas as pd
import tiktoken
from csv import writer
from IPython.display import display, Markdown
from openai import OpenAI
from PyPDF2 import PdfReader
from scipy import spatial
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm
from termcolor import colored

# Configuration
GPT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"
client = OpenAI()
```

## Step 1: Create the Knowledge Base Utilities

We'll set up a local directory to store downloaded PDFs and a CSV file to cache paper details and embeddings for efficient retrieval.

### 1.1 Initialize the Data Directory

Create a directory to store downloaded papers and a CSV file to act as our paper library.

```python
# Define the directory for storing papers
data_dir = os.path.join(os.curdir, "data", "papers")
paper_dir_filepath = "./data/papers/arxiv_library.csv"

# Create the directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"Directory '{data_dir}' created successfully.")
else:
    print(f"Directory '{data_dir}' already exists.")

# Initialize a blank CSV file for the library
df = pd.DataFrame(list())
df.to_csv(paper_dir_filepath)
```

### 1.2 Define Core Helper Functions

We need a reliable function to generate text embeddings using the OpenAI API.

```python
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def embedding_request(text):
    """Generates an embedding for a given text string."""
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response
```

## Step 2: Implement the Agent's Functions

Now, we'll build the two main functions our agent can call.

### 2.1 The `get_articles` Function

This function searches arXiv, downloads the top papers, stores them locally, and caches their embeddings.

```python
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def get_articles(query, library=paper_dir_filepath, top_k=10):
    """
    Gets the top_k articles based on a user's query, sorted by relevance.
    Downloads the files and stores their metadata in the library CSV.
    """
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=top_k)
    result_list = []

    for result in client.results(search):
        result_dict = {
            "title": result.title,
            "summary": result.summary,
            "article_url": [x.href for x in result.links][0],  # Abstract page
            "pdf_url": [x.href for x in result.links][1],      # Direct PDF link
        }
        result_list.append(result_dict)

        # Download PDF and generate embedding for the title
        file_path = result.download_pdf(data_dir)
        response = embedding_request(text=result.title)
        file_reference = [result.title, file_path, response.data[0].embedding]

        # Append this paper's data to the library CSV
        with open(library, "a") as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(file_reference)

    return result_list
```

**Test the search function:**
```python
# Test the search functionality
result_output = get_articles("ppo reinforcement learning")
print(result_output[0]['title'])
print(result_output[0]['summary'][:200])  # Print first 200 chars of summary
```

### 2.2 Supporting Functions for Reading and Summarizing

Before building the second agent function, we need utilities for retrieving relevant papers, reading PDFs, chunking text, and summarizing.

```python
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100,
) -> list[str]:
    """Returns a list of filepaths for papers most related to the query."""
    query_embedding_response = embedding_request(query)
    query_embedding = query_embedding_response.data[0].embedding

    strings_and_relatednesses = [
        (row["filepath"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n]


def read_pdf(filepath):
    """Extracts and returns text from a PDF file."""
    reader = PdfReader(filepath)
    pdf_text = ""
    for page_number, page in enumerate(reader.pages, start=1):
        pdf_text += page.extract_text() + f"\nPage Number: {page_number}"
    return pdf_text


def create_chunks(text, n, tokenizer):
    """Splits text into n-sized chunks, preferably at sentence boundaries."""
    tokens = tokenizer.encode(text)
    i = 0
    while i < len(tokens):
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j


def extract_chunk(content, template_prompt):
    """Summarizes a chunk of text using the provided prompt."""
    prompt = template_prompt + content
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content
```

### 2.3 The `read_article_and_summarize` Function

This is the agent's second function. It finds the most relevant paper from the library, reads it, and provides a comprehensive summary.

```python
def summarize_text(query):
    """
    1. Reads the arXiv library CSV.
    2. Finds the paper most related to the query.
    3. Chunks the paper's text and summarizes each chunk in parallel.
    4. Produces a final, structured summary.
    """
    summary_prompt = "Summarize this text from an academic paper. Extract any key points with reasoning.\n\nContent:"

    # Load the library of papers
    library_df = pd.read_csv(paper_dir_filepath).reset_index()
    if len(library_df) == 0:
        print("No papers in library. Performing a search first.")
        get_articles(query)
        print("Papers downloaded, continuing.")
        library_df = pd.read_csv(paper_dir_filepath).reset_index()
    else:
        print(f"Existing library found with {len(library_df)} articles.")

    library_df.columns = ["title", "filepath", "embedding"]
    library_df["embedding"] = library_df["embedding"].apply(ast.literal_eval)

    # Find the most relevant paper
    strings = strings_ranked_by_relatedness(query, library_df, top_n=1)
    print("Chunking text from paper...")
    pdf_text = read_pdf(strings[0])

    # Chunk the document
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks = create_chunks(pdf_text, 1500, tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]

    print("Summarizing each chunk of text...")
    results = ""

    # Summarize chunks in parallel for speed
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(text_chunks)) as executor:
        futures = [executor.submit(extract_chunk, chunk, summary_prompt) for chunk in text_chunks]
        with tqdm(total=len(text_chunks)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
        for future in futures:
            data = future.result()
            results += data

    # Generate a final, cohesive summary
    print("Creating final summary...")
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": f"""Write a summary collated from this collection of key points extracted from an academic paper.
                        The summary should highlight the core argument, conclusions and evidence, and answer the user's query.
                        User query: {query}
                        The summary should be structured in bulleted lists following the headings Core Argument, Evidence, and Conclusions.
                        Key points:\n{results}\nSummary:\n""",
            }
        ],
        temperature=0,
    )
    return response
```

**Test the summarization function:**
```python
# Test the summarization pipeline
summary_response = summarize_text("PPO reinforcement learning sequence generation")
display(Markdown(summary_response.choices[0].message.content))
```

## Step 3: Configure the AI Agent

Now we'll set up the agent logic that decides when to call our functions.

### 3.1 Define the Chat Completion Handler

```python
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, model=GPT_MODEL):
    """Handles requests to the OpenAI ChatCompletion API."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            functions=functions,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
```

### 3.2 Create a Conversation Manager

```python
class Conversation:
    """A simple class to manage the conversation history."""
    def __init__(self):
        self.conversation_history = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

    def display_conversation(self):
        role_to_color = {"system": "red", "user": "green", "assistant": "blue", "function": "magenta"}
        for message in self.conversation_history:
            print(colored(f"{message['role']}: {message['content']}\n\n", role_to_color[message["role"]]))
```

### 3.3 Define the Functions for the Agent

We must describe our functions in the specific JSON format the OpenAI API expects.

```python
arxiv_functions = [
    {
        "name": "get_articles",
        "description": "Use this function to get academic papers from arXiv to answer user questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user's query about an academic topic. Responses should include summaries and article URLs.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_article_and_summarize",
        "description": "Use this function to read whole papers and provide a detailed summary. Only call this after get_articles has been used in the conversation.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A description of the article to summarize, based on the user's query.",
                }
            },
            "required": ["query"],
        },
    }
]
```

### 3.4 Create the Function Calling Engine

This is the core logic that interprets the model's request to call a function and executes the corresponding Python code.

```python
def chat_completion_with_function_execution(messages, functions=[None]):
    """Makes a ChatCompletion API call and executes a function if requested."""
    response = chat_completion_request(messages, functions)
    full_message = response.choices[0]

    if full_message.finish_reason == "function_call":
        print(f"Function generation requested, calling function...")
        return call_arxiv_function(messages, full_message)
    else:
        print(f"Function not required, responding to user directly.")
        return response


def call_arxiv_function(messages, full_message):
    """Executes the function specified by the model's function_call."""
    function_name = full_message.message.function_call.name

    if function_name == "get_articles":
        try:
            parsed_output = json.loads(full_message.message.function_call.arguments)
            print("Getting search results...")
            results = get_articles(parsed_output["query"])
        except Exception as e:
            print(f"Function execution failed for 'get_articles'")
            print(f"Error: {e}")
            raise

        # Append the function's result back to the conversation history
        messages.append({
            "role": "function",
            "name": function_name,
            "content": str(results),
        })

        # Let the model formulate a response based on the search results
        print("Got search results, summarizing content...")
        response = chat_completion_request(messages)
        return response

    elif function_name == "read_article_and_summarize":
        parsed_output = json.loads(full_message.message.function_call.arguments)
        print("Finding and reading paper...")
        summary = summarize_text(parsed_output["query"])
        return summary

    else:
        raise Exception(f"Function '{function_name}' does not exist.")
```

## Step 4: Run a Conversation with the Agent

Let's integrate all the components and have a conversation with our arXiv agent.

```python
# Initialize the conversation
conversation = Conversation()
system_message = """You are arXivGPT, a helpful assistant that pulls academic papers from arXiv to answer user questions.
You summarize the papers clearly so the customer can decide which to read to answer their question.
You always provide the article_url and title so the user can understand the source."""
conversation.add_message("system", system_message)

# Example User Query
user_query = "What are the latest advancements in reinforcement learning for robotics?"
conversation.add_message("user", user_query)

print("User Query:", user_query)
print("\n" + "="*50 + "\n")

# Get the agent's response, which may involve function calls
response = chat_completion_with_function_execution(
    conversation.conversation_history,
    functions=arxiv_functions
)

# Add the agent's response to the conversation history
if hasattr(response, 'choices'):
    # It's a direct ChatCompletion response
    assistant_message = response.choices[0].message.content
    conversation.add_message("assistant", assistant_message)
else:
    # It's a summary response from the `summarize_text` function
    conversation.add_message("assistant", response.choices[0].message.content)

# Display the conversation
conversation.display_conversation()
```

## How It Works: The Complete Flow

1.  **User Query:** You ask a question (e.g., about RL in robotics).
2.  **Agent Decision:** The LLM (GPT-4) reviews the conversation and the list of available functions (`arxiv_functions`). It decides if a function is needed to answer the question.
3.  **Function Call:** If needed, the model returns a `function_call` response specifying which function to use and with what arguments (e.g., `get_articles` with the query "reinforcement learning robotics").
4.  **Function Execution:** The `call_arxiv_function` handler executes the corresponding Python function, which searches arXiv or reads a PDF.
5.  **Result Integration:** The function's result is added to the conversation history as a `function` role message.
6.  **Final Response:** The model receives this new context and generates a final, informative answer for the user, incorporating the data retrieved by the function.

You can now extend this conversation by asking follow-up questions, such as "Can you read and summarize the third article from that list in detail?", which will trigger the `read_article_and_summarize` function.