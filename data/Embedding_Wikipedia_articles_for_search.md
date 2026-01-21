# Guide: Preparing a Wikipedia Dataset for Semantic Search

This guide walks you through creating a dataset of Wikipedia article sections, complete with text embeddings, for use in semantic search or Retrieval-Augmented Generation (RAG) applications. We'll download articles related to the 2022 Winter Olympics, process them into manageable chunks, generate embeddings using OpenAI's API, and store the results.

## Prerequisites

Before you begin, ensure you have the necessary libraries installed and your OpenAI API key configured.

### 1. Install Required Libraries
Run the following command in your terminal to install the required packages.

```bash
pip install mwclient mwparserfromhell openai pandas tiktoken
```

### 2. Set Your OpenAI API Key
The OpenAI client library reads your API key from the `OPENAI_API_KEY` environment variable. Set this variable in your environment. If you're unsure how, follow [OpenAI's API key safety guide](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).

### 3. Import Libraries and Initialize Client
Create a new Python script or notebook and start by importing the necessary modules and initializing the OpenAI client.

```python
import mwclient
import mwparserfromhell
from openai import OpenAI
import os
import pandas as pd
import re
import tiktoken

# Initialize the OpenAI client.
# It will automatically use the OPENAI_API_KEY environment variable.
client = OpenAI()
```

## Step 1: Collect Wikipedia Documents

We'll start by fetching article titles from a specific Wikipedia category. This function recursively collects page titles from the "Category:2022 Winter Olympics" and its immediate subcategories.

```python
# Define the target Wikipedia category and site.
CATEGORY_TITLE = "Category:2022 Winter Olympics"
WIKI_SITE = "en.wikipedia.org"

def titles_from_category(category: mwclient.listing.Category, max_depth: int) -> set[str]:
    """
    Recursively fetch page titles from a Wikipedia category and its subcategories.
    """
    titles = set()
    for member in category.members():
        if type(member) == mwclient.page.Page:
            titles.add(member.name)
        elif isinstance(member, mwclient.listing.Category) and max_depth > 0:
            deeper_titles = titles_from_category(member, max_depth=max_depth - 1)
            titles.update(deeper_titles)
    return titles

# Connect to Wikipedia and fetch titles.
site = mwclient.Site(WIKI_SITE)
category_page = site.pages[CATEGORY_TITLE]
article_titles = titles_from_category(category_page, max_depth=1)

print(f"Found {len(article_titles)} article titles in {CATEGORY_TITLE}.")
```

## Step 2: Chunk Documents into Sections

Large articles must be split into smaller, coherent sections for effective embedding and retrieval. We'll parse each article, ignore irrelevant sections (like "References"), clean the text, and split overly long sections.

### 2.1 Define Section Processing Functions
First, let's define helper functions to parse Wikipedia articles into a flat list of subsections.

```python
# Sections typically not useful for Q&A.
SECTIONS_TO_IGNORE = [
    "See also", "References", "External links", "Further reading",
    "Footnotes", "Bibliography", "Sources", "Citations", "Literature",
    "Notes and references", "Photo gallery", "Works cited", "Photos",
    "Gallery", "Notes", "References and sources", "References and notes",
]

def all_subsections_from_section(section, parent_titles, sections_to_ignore):
    """
    Flatten a Wikipedia section and all its nested subsections.
    Returns a list where each element is a tuple: (list_of_titles, section_text).
    """
    headings = [str(h) for h in section.filter_headings()]
    title = headings[0]
    # Skip ignored sections.
    if title.strip("= ").strip("=") in sections_to_ignore:
        return []

    titles = parent_titles + [title]
    full_text = str(section)
    # Extract text belonging to this specific section.
    section_text = full_text.split(title)[1]

    if len(headings) == 1:
        return [(titles, section_text)]
    else:
        first_subtitle = headings[1]
        section_text = section_text.split(first_subtitle)[0]
        results = [(titles, section_text)]
        # Recursively process child subsections.
        for subsection in section.get_sections(levels=[len(titles) + 1]):
            results.extend(all_subsections_from_section(subsection, titles, sections_to_ignore))
        return results

def all_subsections_from_title(title, sections_to_ignore=SECTIONS_TO_IGNORE, site_name=WIKI_SITE):
    """
    From a page title, return a flattened list of all its subsections.
    """
    site = mwclient.Site(site_name)
    page = site.pages[title]
    text = page.text()
    parsed_text = mwparserfromhell.parse(text)

    headings = [str(h) for h in parsed_text.filter_headings()]
    if headings:
        summary_text = str(parsed_text).split(headings[0])[0]
    else:
        summary_text = str(parsed_text)

    results = [([title], summary_text)]
    for subsection in parsed_text.get_sections(levels=[2]):
        results.extend(all_subsections_from_section(subsection, [title], sections_to_ignore))
    return results
```

### 2.2 Parse All Articles into Sections
Now, apply the functions to every article title we collected.

```python
# This may take about a minute per 100 articles.
wikipedia_sections = []
for title in article_titles:
    wikipedia_sections.extend(all_subsections_from_title(title))

print(f"Found {len(wikipedia_sections)} sections in {len(article_titles)} pages.")
```

### 2.3 Clean and Filter Sections
Clean the text by removing reference tags and filtering out very short sections.

```python
def clean_section(section):
    """Remove <ref> tags and strip extra whitespace."""
    titles, text = section
    text = re.sub(r"<ref.*?</ref>", "", text)
    text = text.strip()
    return (titles, text)

wikipedia_sections = [clean_section(ws) for ws in wikipedia_sections]

def keep_section(section):
    """Keep only sections with meaningful content."""
    titles, text = section
    return len(text) >= 16

original_num_sections = len(wikipedia_sections)
wikipedia_sections = [ws for ws in wikipedia_sections if keep_section(ws)]
print(f"Filtered out {original_num_sections - len(wikipedia_sections)} sections, leaving {len(wikipedia_sections)}.")
```

### 2.4 Split Long Sections
To respect context window limits, we recursively split sections longer than a token threshold (e.g., 1600 tokens). The splitter prefers natural boundaries like paragraphs.

```python
GPT_MODEL = "gpt-4o-mini"  # Determines which tokenizer to use.

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Count tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def halved_by_delimiter(string: str, delimiter: str = "\n") -> list:
    """Split a string in two, balancing token count around a delimiter."""
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]
    elif len(chunks) == 2:
        return chunks

    total_tokens = num_tokens(string)
    halfway = total_tokens // 2
    best_diff = halfway
    split_index = 0

    for i, chunk in enumerate(chunks):
        left = delimiter.join(chunks[: i + 1])
        left_tokens = num_tokens(left)
        diff = abs(halfway - left_tokens)
        if diff >= best_diff:
            break
        else:
            best_diff = diff
            split_index = i

    left = delimiter.join(chunks[:split_index])
    right = delimiter.join(chunks[split_index:])
    return [left, right]

def truncated_string(string: str, model: str, max_tokens: int, print_warning: bool = True) -> str:
    """Truncate a string to a maximum token count."""
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated

def split_strings_from_subsection(subsection, max_tokens=1000, model=GPT_MODEL, max_recursion=5):
    """
    Recursively split a subsection until all parts are under the token limit.
    """
    titles, text = subsection
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)

    if num_tokens_in_string <= max_tokens:
        return [string]
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    else:
        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(text, delimiter=delimiter)
            if left == "" or right == "":
                continue  # Try a finer-grained delimiter.
            else:
                results = []
                for half in [left, right]:
                    half_subsection = (titles, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
        # Fallback: truncate if no good split found.
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
```

Apply the splitting function to all sections.

```python
MAX_TOKENS = 1600
wikipedia_strings = []
for section in wikipedia_sections:
    wikipedia_strings.extend(split_strings_from_subsection(section, max_tokens=MAX_TOKENS))

print(f"{len(wikipedia_sections)} Wikipedia sections split into {len(wikipedia_strings)} strings.")
```

You can inspect an example chunk:

```python
print(wikipedia_strings[1])
```

## Step 3: Generate Embeddings

With our text chunks prepared, we can now generate vector embeddings using OpenAI's embedding model. We'll process in batches to stay within API limits.

```python
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 1000  # Max 2048 inputs per request.

embeddings = []
for batch_start in range(0, len(wikipedia_strings), BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    batch = wikipedia_strings[batch_start:batch_end]
    print(f"Processing batch {batch_start} to {batch_end-1}")

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
    # Ensure embeddings are in the same order as inputs.
    for i, data in enumerate(response.data):
        assert i == data.index

    batch_embeddings = [data.embedding for data in response.data]
    embeddings.extend(batch_embeddings)

# Create a DataFrame to hold the text and its corresponding embedding vector.
df = pd.DataFrame({"text": wikipedia_strings, "embedding": embeddings})
```

## Step 4: Store the Dataset

Finally, save the processed chunks and their embeddings to a CSV file. For larger-scale applications, consider using a dedicated vector database for better performance.

```python
SAVE_PATH = "data/winter_olympics_2022.csv"
df.to_csv(SAVE_PATH, index=False)
print(f"Dataset saved to {SAVE_PATH}")
```

## Next Steps

You now have a prepared dataset (`winter_olympics_2022.csv`) containing Wikipedia text chunks and their embeddings. This dataset is ready to be used in semantic search systems or as a knowledge base for RAG pipelines. In a follow-up tutorial, you can load this CSV, compute query embeddings, and find the most relevant text chunks for answering questions.