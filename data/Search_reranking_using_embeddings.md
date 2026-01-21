# Search Re-Ranking with Gemini Embeddings: A Step-by-Step Guide

This guide demonstrates how to implement a search re-ranking system using Gemini's embeddings and function calling. You will build an intelligent search assistant that queries Wikipedia, retrieves results, and re-ranks them by relevance using semantic similarity.

## Objectives
By the end of this tutorial, you will learn how to:
1.  Set up the Gemini API and necessary Python libraries.
2.  Use Gemini's function calling to interact with the Wikipedia API.
3.  Generate and embed search results.
4.  Re-rank search results based on semantic relevance.

## Prerequisites & Setup

First, install the required Python packages.

```bash
pip install -U -q google-genai wikipedia
```

Now, import the necessary libraries and configure your environment.

```python
import json
import textwrap
import numpy as np

from google import genai
from google.genai import types
import wikipedia
from wikipedia.exceptions import DisambiguationError, PageError
```

### Configure Your Gemini API Key
Before proceeding, you need a Gemini API key. You can obtain one from [Google AI Studio](https://aistudio.google.com/app/apikey).

Once you have your key, pass it to the Gemini client. The following example shows how to fetch it from an environment variable named `GEMINI_API_KEY`. In a Google Colab notebook, you could use `userdata.get('GEMINI_API_KEY')`.

```python
import os

# Fetch the API key from an environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=GEMINI_API_KEY)
```

### Select a Model
Choose the Gemini model you wish to use for generating content and embeddings.

```python
MODEL_ID = "gemini-2.5-flash"  # You can change this to other models like "gemini-2.5-pro"
```

## Step 1: Define the Wikipedia Search Tool

You will use Gemini's function calling capability to create a search tool. This function will:
1.  Accept a list of search queries.
2.  Use the `wikipedia` package to find relevant topics for each query.
3.  Fetch the top `n_topics` pages and use Gemini to extract information relevant to the original query.
4.  Avoid duplicate entries by maintaining a search history.

```python
def wikipedia_search(search_queries: list[str]) -> list[str]:
    """Search Wikipedia for each query and summarize relevant documents."""
    n_topics = 3
    search_history = set()  # Track searched topics to avoid duplicates
    search_urls = []
    summary_results = []

    for query in search_queries:
        print(f'Searching for "{query}"')
        search_terms = wikipedia.search(query)

        print(f"Related search terms: {search_terms[:n_topics]}")
        for search_term in search_terms[:n_topics]:  # Select first `n_topics` candidates
            if search_term in search_history:  # Skip if already covered
                continue

            print(f'Fetching page: "{search_term}"')
            search_history.add(search_term)

            try:
                # Fetch the Wikipedia page
                page = wikipedia.page(search_term, auto_suggest=False)
                url = page.url
                print(f"Information Source: {url}")
                search_urls.append(url)
                page_content = page.content

                # Use Gemini to extract relevant information from the page
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=textwrap.dedent(f"""\
                        Extract relevant information
                        about user's query: {query}
                        From this source:

                        {page_content}

                        Note: Do not summarize. Only extract and return the relevant information.
                    """)
                )

                urls = [url]
                # Handle any additional citations from the model's response
                if response.candidates[0].citation_metadata:
                    extra_citations = response.candidates[0].citation_metadata.citation_sources
                    extra_urls = [source.url for source in extra_citations]
                    urls.extend(extra_urls)
                    search_urls.extend(extra_urls)
                    print("Additional citations:", response.candidates[0].citation_metadata.citation_sources)

                try:
                    text = response.text
                except ValueError:
                    pass
                else:
                    summary_results.append(text + "\n\nBased on:\n  " + ',\n  '.join(urls))

            except DisambiguationError:
                print(f'Results for "{search_term}" (originally for "{query}") were ambiguous, skipping.')
                continue
            except PageError:
                print(f'"{search_term}" did not match any page, skipping.')
                continue
            except Exception:
                print(f'Error fetching "{search_term}", skipping.')
                continue

    print(f"Information Sources:")
    for url in search_urls:
        print('    ', url)

    return summary_results
```

### Test the Search Function
Let's test the function with a sample query.

```python
example_results = wikipedia_search(["What are LLMs?"])
```

The function will print the search process and return a list of extracted summaries. For example, it might return summaries for "Large language model", "Retrieval-augmented generation", and "Gemini (chatbot)".

## Step 2: Generate Supporting Search Queries

To improve search coverage, you can ask Gemini to generate a list of related queries based on the user's original question. This helps the system explore the topic from multiple angles.

Define an instruction prompt for the model.

```python
instructions = """You have access to the Wikipedia API which you will be using
to answer a user's query. Your job is to generate a list of search queries which
might answer a user's question. Be creative by using various key-phrases from
the user's query. To generate variety of queries, ask questions which are
related to the user's query that might help to find the answer. The more
queries you generate the better are the odds of you finding the correct answer.
Here is an example:

user: Tell me about Cricket World cup 2023 winners.

function_call: wikipedia_search(['What is the name of the team that
won the Cricket World Cup 2023?', 'Who was the captain of the Cricket World Cup
2023 winning team?', 'Which country hosted the Cricket World Cup 2023?', 'What
was the venue of the Cricket World Cup 2023 final match?', 'Cricket World cup 2023',
'Who lifted the Cricket World Cup 2023 trophy?'])

The search function will return a list of article summaries, use these to
answer the user's question.

Here is the user's query: {query}
"""
```

## Step 3: Enable Automatic Function Calling

Now, create a chat session with automatic function calling enabled. This allows the `ChatSession` to automatically execute the `wikipedia_search` function when the model decides it's needed.

Configure the generation settings with a higher temperature (e.g., 0.6) to encourage creative query generation.

```python
tools = [wikipedia_search]

config = types.GenerateContentConfig(
    temperature=0.6,
    tools=tools
)

# Create a new chat session
chat = client.chats.create(
    model="gemini-2.5-flash",
    config=config
)
```

## Step 4: Execute a Search and Retrieve Results

Send the user's query to the chat session. The model will use the provided instructions to generate related search queries, automatically call the `wikipedia_search` function, and compile an answer from the retrieved summaries.

```python
query = "Explain how deep-sea life survives."
response = chat.send_message(instructions.format(query=query))
```

The model will generate multiple search queries (e.g., "Deep-sea life survival strategies", "Hydrothermal vent ecosystems"), call the `wikipedia_search` function for each, and return a comprehensive answer.

You can view the final response.

```python
print(response.text)
```

The output will be a detailed explanation of deep-sea survival adaptations, synthesized from the retrieved Wikipedia summaries.

## Step 5: Re-Rank Search Results with Embeddings (Conceptual)

While the current system retrieves and summarizes information, the core re-ranking logic using embeddings is implied in the `wikipedia_search` function's selection of the top `n_topics`. For a more sophisticated re-ranking system, you would:

1.  **Generate Embeddings:** Use Gemini's embedding model to create vector representations for both the user's original query and each retrieved text summary.
2.  **Calculate Similarity:** Compute a similarity score (e.g., cosine similarity) between the query embedding and each summary embedding.
3.  **Re-rank:** Sort the summaries based on their similarity scores and return the most relevant ones.

This step would involve using `client.models.embed_content` and a library like `numpy` for vector math, building upon the search results you've already retrieved.

## Summary

You have successfully built a search assistant that:
*   Uses Gemini's function calling to query Wikipedia.
*   Generates related search queries to improve coverage.
*   Automatically executes searches and synthesizes answers.
*   Provides a foundation for implementing semantic re-ranking with embeddings.

This system demonstrates a powerful pattern for building retrieval-augmented generation (RAG) applications where external data sources are seamlessly integrated into an LLM's reasoning process.