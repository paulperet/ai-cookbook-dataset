# Building a Bring Your Own Browser (BYOB) Tool for Web Browsing and Summarization

**Disclaimer:** This guide is for educational purposes only. Ensure you comply with all applicable laws and service terms when using web search and scraping technologies. This example restricts searches to the `openai.com` domain to retrieve public information.

Large Language Models (LLMs) like GPT-4o have a knowledge cutoff date, meaning they lack information about events after that point. To provide accurate, up-to-date responses, we need to give the LLM access to current web information.

In this tutorial, you will build a Bring Your Own Browser (BYOB) tool using Python. This system will perform web searches, process the results, and use an LLM to generate a final, informed answer. We'll use Google's Custom Search API for searches and implement a Retrieval-Augmented Generation (RAG) pipeline.

**What you will build:**
1.  **A Search Engine:** Configure and use Google's Custom Search API to fetch relevant web results.
2.  **A Search Dictionary:** Scrape, summarize, and structure information from web pages.
3.  **A RAG Response:** Pass the structured data to an LLM to generate a final, cited answer.

**Use Case:** We will answer the query: *"List the latest OpenAI product launches in chronological order from latest to oldest in the past 2 years."* Without web access, GPT-4o cannot know about recent launches like the o1-preview model (September 2024). Our BYOB tool will solve this.

## Prerequisites

Before you begin, ensure you have the following:

1.  **Python 3.12 or later** installed.
2.  A **Google API Key** and a **Custom Search Engine ID (CSE ID)**. You can obtain these from the [Google Programmable Search Engine](https://developers.google.com/custom-search/v1/overview).
3.  An **OpenAI API Key** set as an environment variable (`OPENAI_API_KEY`).
4.  The necessary Python packages. Install them using the command below.

```bash
pip install requests beautifulsoup4 openai python-dotenv
```

## Step 1: Set Up the Search Engine

First, we'll configure a function to interact with Google's Custom Search API. This function will take a search term and return a list of relevant web pages.

### 1.1 Configure the Search Function

Create a function called `search`. It accepts the search term, your API keys, and optional parameters like the number of results and a domain filter.

```python
import requests

def search(search_item, api_key, cse_id, search_depth=10, site_filter=None):
    """
    Performs a web search using Google's Custom Search API.

    Args:
        search_item (str): The query to search for.
        api_key (str): Your Google API key.
        cse_id (str): Your Custom Search Engine ID.
        search_depth (int): Number of results to return (max 10).
        site_filter (str, optional): Domain to restrict results to (e.g., 'openai.com').

    Returns:
        list: A list of search result items (dictionaries).
    """
    service_url = 'https://www.googleapis.com/customsearch/v1'

    params = {
        'q': search_item,
        'key': api_key,
        'cx': cse_id,
        'num': search_depth
    }

    try:
        response = requests.get(service_url, params=params)
        response.raise_for_status()
        results = response.json()

        # Check if 'items' exists in the results
        if 'items' in results:
            if site_filter is not None:
                # Filter results to include only those with site_filter in the link
                filtered_results = [result for result in results['items'] if site_filter in result['link']]
                return filtered_results if filtered_results else []
            else:
                return results['items']
        else:
            print("No search results found.")
            return []

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the search: {e}")
        return []
```

### 1.2 Generate an Optimized Search Term

Search engines work better with concise keywords than with long, natural language questions. We'll use the LLM to convert the user's query into an effective search term, a process called **query expansion**.

First, let's see the limitation of the base model by asking it our target question directly.

```python
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables (including OPENAI_API_KEY)
load_dotenv('.env')

# Initialize the OpenAI client
client = OpenAI()

# Define our target query
search_query = "List the latest OpenAI product launches in chronological order from latest to oldest in the past 2 years"

# Ask the model without any web context
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": search_query}
    ]
).choices[0].message.content

print("Model's response without web access:")
print(response)
print("\n" + "="*80 + "\n")
```

As expected, the model's knowledge is outdated. Now, let's generate a better search term.

```python
# Use the LLM to create a concise, effective search term
search_term = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Provide a google search term based on search query provided below in 3-4 words"},
        {"role": "user", "content": search_query}
    ]
).choices[0].message.content

print(f"Generated search term: {search_term}")
```

### 1.3 Execute the Search

Now, use your API keys and the generated `search_term` to fetch results from the `openai.com` domain.

```python
# Load your Google API credentials from environment variables
api_key = os.getenv('API_KEY')
cse_id = os.getenv('CSE_ID')

# Perform the search, restricted to openai.com
search_items = search(search_item=search_term,
                      api_key=api_key,
                      cse_id=cse_id,
                      search_depth=10,
                      site_filter="https://openai.com")

# Inspect the raw search results
print(f"Found {len(search_items)} results.\n")
for item in search_items:
    print(f"Link: {item['link']}")
    print(f"Snippet: {item['snippet'][:150]}...\n")
```

## Step 2: Build a Search Dictionary

The search API returns links and brief snippets. For a high-quality answer, we need more detailed content from each page. In this step, we will:
1.  Scrape the main text from each URL.
2.  Use an LLM to summarize the scraped text in the context of our query.
3.  Organize everything into a structured list of dictionaries.

### 2.1 Define Helper Functions

We'll create three functions: one to scrape content, one to summarize it, and one to orchestrate the process for all search results.

```python
import requests
from bs4 import BeautifulSoup

# Constants
TRUNCATE_SCRAPED_TEXT = 50000  # Limit scraped text to fit model context
SEARCH_DEPTH = 5  # Number of top results to process in detail

def retrieve_content(url, max_tokens=TRUNCATE_SCRAPED_TEXT):
    """
    Fetches and cleans the main text content from a given URL.

    Args:
        url (str): The webpage URL.
        max_tokens (int): Approximate token limit for the scraped text.

    Returns:
        str: The cleaned text content, or None if retrieval fails.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        text = soup.get_text(separator=' ', strip=True)
        # Truncate text based on an approximate character limit
        characters = max_tokens * 4
        text = text[:characters]
        return text
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve {url}: {e}")
        return None

def summarize_content(content, search_term, character_limit=500):
    """
    Uses an LLM to generate a concise summary of content relevant to a search term.

    Args:
        content (str): The text to summarize.
        search_term (str): The original search term for context.
        character_limit (int): Target length for the summary.

    Returns:
        str: The generated summary, or None if summarization fails.
    """
    prompt = (
        f"You are an AI assistant tasked with summarizing content relevant to '{search_term}'. "
        f"Please provide a concise summary in {character_limit} characters or less."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using a smaller, cost-effective model for summarization
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content}
            ]
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        print(f"An error occurred during summarization: {e}")
        return None

def get_search_results(search_items, character_limit=500):
    """
    Processes a list of search items: scrapes content and generates summaries.

    Args:
        search_items (list): List of search result dictionaries from the API.
        character_limit (int): Target length for each summary.

    Returns:
        list: A list of dictionaries, each containing order, link, title, and summary.
    """
    results_list = []
    for idx, item in enumerate(search_items[:SEARCH_DEPTH], start=1): # Process top N results
        url = item.get('link')
        snippet = item.get('snippet', '')

        print(f"Processing result {idx}: {url}")
        web_content = retrieve_content(url, TRUNCATE_SCRAPED_TEXT)

        if web_content is None:
            print(f"  Error: Could not retrieve content. Skipping.\n")
        else:
            summary = summarize_content(web_content, search_term, character_limit)
            result_dict = {
                'order': idx,
                'link': url,
                'title': snippet,
                'Summary': summary
            }
            results_list.append(result_dict)
            print(f"  Summary generated.\n")
    return results_list
```

### 2.2 Execute the Processing Pipeline

Now, run the `get_search_results` function on your search items to build the structured dictionary.

```python
# Process the search results to get detailed summaries
results = get_search_results(search_items)

# Display the structured information
print("\n" + "="*80)
print("STRUCTURED SEARCH DICTIONARY")
print("="*80)
for result in results:
    print(f"Order: {result['order']}")
    print(f"Link: {result['link']}")
    print(f"Original Snippet: {result['title'][:100]}...")
    print(f"Generated Summary: {result['Summary']}")
    print('-' * 80)
```

## Step 3: Generate the Final RAG Response

We now have a rich, structured dataset (`results`). The final step is to pass this data, along with the original user query, to the LLM with instructions to synthesize an answer and cite its sources.

### 3.1 Construct the Final Prompt and Get the Answer

We'll create a system prompt that instructs the model to use the provided search data and format the response clearly.

```python
import json

# Construct the final instruction for the LLM
final_prompt = (
    f"The user will provide a dictionary of search results in JSON format for the search term '{search_term}'. "
    f"Based **only** on the search results provided, provide a detailed response to this query: **'{search_query}'**. "
    f"Present the answer as a chronological list from latest to oldest. "
    f"Make sure to cite the source for each item using the provided link at the end of your answer."
)

# Generate the final response using the main model (gpt-4o)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": final_prompt},
        {"role": "user", "content": json.dumps(results)}
    ],
    temperature=0  # Use low temperature for factual, consistent output
)

final_answer = response.choices[0].message.content

print("\n" + "="*80)
print("FINAL ANSWER (with citations from web data)")
print("="*80)
print(final_answer)
```

### Expected Output

Your final answer will now include recent product launches that were beyond the base model's knowledge cutoff, such as:
-   **OpenAI o1** (September 2024)
-   **SearchGPT** (July 2024)
-   **GPT-4o mini** (July 2024)
-   **Sora** (February 2024)

Each item will be accompanied by a citation to the source URL from your search dictionary.

## Summary

You have successfully built a Bring Your Own Browser (BYOB) tool that:
1.  **Interacts with a Search API:** You configured Google's Custom Search to fetch current web results.
2.  **Processes and Enriches Data:** You scraped web pages, generated focused summaries using an LLM, and created a structured knowledge base.
3.  **Generates a Cited, Up-to-Date Answer:** You used the enriched data in a RAG pipeline to instruct the LLM to produce a final, sourced response.

This pattern can be extended to answer a wide variety of questions requiring current information, making your AI applications more powerful and relevant. Remember to always respect `robots.txt` files and terms of service when scraping websites.