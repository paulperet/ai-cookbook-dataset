# Building a Question-Answering System: Collecting Wikipedia Data for the 2020 Summer Olympics

## Introduction

This guide is the first part of a three-part series on creating a question-answering (QA) model that can reliably determine when it has sufficient context to answer a question. The goal is to prevent the model from "confabulating" answers when the necessary information is not present in the provided text.

In this notebook, you will collect and preprocess a dataset of Wikipedia articles about the 2020 Summer Olympics. This topic was chosen because it represents recent information that GPT-3 likely did not see during its pre-training phase. You will extract relevant pages, split them into manageable sections, and prepare the data for the next stage of the project.

## Prerequisites

Before you begin, ensure you have the necessary Python libraries installed.

```bash
pip install pandas wikipedia nltk transformers
```

You will also need to download the NLTK sentence tokenizer data.

```python
import nltk
nltk.download('punkt')
```

Now, import the required modules.

```python
import pandas as pd
import wikipedia
import re
from typing import Set
import numpy as np
from nltk.tokenize import sent_tokenize
from transformers import GPT2TokenizerFast
```

## Step 1: Collect Wikipedia Pages

The first step is to gather all Wikipedia pages related to the 2020 Summer Olympics. You will use the Wikipedia API to recursively find pages linked from a starting point.

### 1.1 Define Helper Functions

Create functions to filter titles and fetch pages.

```python
def filter_olympic_2020_titles(titles):
    """
    Filter a list of Wikipedia titles to those related to the 2020 Olympics.
    """
    titles = [title for title in titles if '2020' in title and 'olympi' in title.lower()]
    return titles

def get_wiki_page(title):
    """
    Fetch a Wikipedia page by title, handling disambiguation and page errors.
    """
    try:
        return wikipedia.page(title)
    except wikipedia.exceptions.DisambiguationError as e:
        return wikipedia.page(e.options[0])
    except wikipedia.exceptions.PageError as e:
        return None
```

### 1.2 Recursively Find All Related Pages

This function starts from a seed title and follows links to build a comprehensive collection of pages.

```python
def recursively_find_all_pages(titles, titles_so_far=set()):
    """
    Recursively find all Wikipedia pages linked from the given list of titles.
    """
    all_pages = []
    
    # Filter titles to those we haven't processed and are about the 2020 Olympics
    titles = list(set(titles) - titles_so_far)
    titles = filter_olympic_2020_titles(titles)
    titles_so_far.update(titles)
    
    for title in titles:
        page = get_wiki_page(title)
        if page is None:
            continue
        all_pages.append(page)

        # Recursively process links from this page
        new_pages = recursively_find_all_pages(page.links, titles_so_far)
        for pg in new_pages:
            if pg.title not in [p.title for p in all_pages]:
                all_pages.append(pg)
        titles_so_far.update(page.links)
    return all_pages
```

### 1.3 Execute the Collection

Start the recursive search from the main "2020 Summer Olympics" page.

```python
pages = recursively_find_all_pages(["2020 Summer Olympics"])
print(f"Total pages collected: {len(pages)}")
```

## Step 2: Process Pages into Sections

Raw Wikipedia pages are long and contain non-informative parts (like references). You will split each page into sections based on headings, filter out low-value sections, and ensure each section is within a token limit.

### 2.1 Initialize Tokenizer and Define Utility Functions

You'll use the GPT-2 tokenizer to count tokens, which is a good proxy for GPT-3's tokenization.

```python
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    """Count the number of tokens in a string."""
    return len(tokenizer.encode(text))

def reduce_long(long_text: str, long_text_tokens: bool = False, max_len: int = 590) -> str:
    """
    Reduce a long text to a maximum of `max_len` tokens by cutting at sentence boundaries.
    """
    if not long_text_tokens:
        long_text_tokens = count_tokens(long_text)
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + count_tokens(sentence)
            if ntokens > max_len:
                return ". ".join(sentences[:i]) + "."
    return long_text
```

### 2.2 Define Sections to Discard

Create a list of section headings that typically don't contain useful textual information for QA.

```python
discard_categories = [
    'See also', 'References', 'External links', 'Further reading', "Footnotes",
    "Bibliography", "Sources", "Citations", "Literature", "Footnotes", "Notes and references",
    "Photo gallery", "Works cited", "Photos", "Gallery", "Notes", "References and sources",
    "References and notes",
]
```

### 2.3 Extract and Clean Sections

This core function processes a page's raw text, splits it by headings, and filters out unwanted sections.

```python
def extract_sections(
    wiki_text: str,
    title: str,
    max_len: int = 1500,
    discard_categories: Set[str] = discard_categories,
):
    """
    Extract sections from Wikipedia text, discarding references and other non-informative parts.
    Returns a list of tuples: (title, heading, content, token_count).
    """
    if len(wiki_text) == 0:
        return []

    # Find all headings and replace them with a delimiter
    headings = re.findall("==+ .* ==+", wiki_text)
    for heading in headings:
        wiki_text = wiki_text.replace(heading, "==+ !! ==+")
    contents = wiki_text.split("==+ !! ==+")
    contents = [c.strip() for c in contents]
    assert len(headings) == len(contents) - 1

    # The first content block is the summary (before any heading)
    cont = contents.pop(0).strip()
    outputs = [(title, "Summary", cont, count_tokens(cont)+4)]
    
    # Process each heading and its corresponding content
    max_level = 100
    keep_group_level = max_level
    remove_group_level = max_level
    nheadings, ncontents = [], []
    for heading, content in zip(headings, contents):
        plain_heading = " ".join(heading.split(" ")[1:-1])
        num_equals = len(heading.split(" ")[0])  # Number of '=' indicates heading level
        if num_equals <= keep_group_level:
            keep_group_level = max_level

        if num_equals > remove_group_level:
            if num_equals <= keep_group_level:
                continue
        keep_group_level = max_level
        if plain_heading in discard_categories:
            remove_group_level = num_equals
            keep_group_level = max_level
            continue
        nheadings.append(heading.replace("=", "").strip())
        ncontents.append(content)
        remove_group_level = max_level

    # Count tokens for each remaining section
    ncontent_ntokens = [
        count_tokens(c)
        + 3
        + count_tokens(" ".join(h.split(" ")[1:-1]))
        - (1 if len(c) == 0 else 0)
        for h, c in zip(nheadings, ncontents)
    ]

    # Create the final list of sections, truncating any that exceed the max_len
    for h, c, t in zip(nheadings, ncontents, ncontent_ntokens):
        if t < max_len:
            outputs.append((title, h, c, t))
        else:
            reduced_content = reduce_long(c, max_len)
            outputs.append((title, h, reduced_content, count_tokens(reduced_content)))
    
    return outputs
```

### 2.4 Test the Extraction

Let's test the function on a sample page to see the output format.

```python
bermuda_page = get_wiki_page('Bermuda at the 2020 Summer Olympics')
bermuda_sections = extract_sections(bermuda_page.content, bermuda_page.title)
print(f"Extracted {len(bermuda_sections)} sections from the Bermuda page.")
print("Example section:", bermuda_sections[-1])
```

## Step 3: Create the Full Dataset

Now, apply the section extraction to all collected pages and compile the results into a DataFrame.

### 3.1 Process All Pages

```python
sections = []
for page in pages:
    sections += extract_sections(page.content, page.title)

df = pd.DataFrame(sections, columns=["title", "heading", "content", "tokens"])
```

### 3.2 Filter and Clean the Data

Remove sections that are too short (less than 40 tokens) and drop any duplicate sections.

```python
df = df[df.tokens > 40]
df = df.drop_duplicates(['title', 'heading'])
df = df.reset_index(drop=True)
print(df.head())
```

### 3.3 Save the Dataset

Save the processed sections to a CSV file for use in the next notebook.

```python
df.to_csv('olympics-data/olympics_sections.csv', index=False)
print("Dataset saved to 'olympics-data/olympics_sections.csv'")
```

## Step 4: (Optional) Explore the Dataset

It's useful to understand the composition of your dataset. Let's perform some basic analysis.

### 4.1 Check the Most Common Titles

```python
print(df.title.value_counts().head())
```

### 4.2 Verify the Focus on Summer Olympics

Check how many titles explicitly mention "Summer" or "Winter".

```python
print("Titles containing 'Summer':", df.title.str.contains('Summer').value_counts())
print("Titles containing 'Winter':", df.title.str.contains('Winter').value_counts())
```

### 4.3 Visualize Token Distribution

Load the saved CSV and create a histogram of token counts per section.

```python
import pandas as pd
from matplotlib import pyplot as plt

df_loaded = pd.read_csv('olympics-data/olympics_sections.csv')
df_loaded[['tokens']].hist()
plt.xlabel('Number of tokens')
plt.ylabel('Number of Wikipedia sections')
plt.title('Distribution of number of tokens in Wikipedia sections')
plt.show()
```

## Summary

You have successfully collected and preprocessed a dataset of Wikipedia sections about the 2020 Summer Olympics. The dataset is now cleaned, filtered, and ready for the next step: generating question-answer pairs using an LLM.

**Next Steps:** Proceed to the [second notebook](olympics-2-create-qa.ipynb), where you will use this section data to create a training set of questions and answers.