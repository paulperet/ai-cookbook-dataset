# Building a Philosophy Quote Finder & Generator with Vector Embeddings

This guide walks you through building a "philosophy quote finder & generator" using OpenAI's vector embeddings and DataStax Astra DB as a vector store. You'll learn how to store, search, and generate philosophical quotes using semantic similarity.

## Prerequisites

Before starting, ensure you have:
- An **Astra DB** database (get your API Endpoint and Token from the [Astra Dashboard](https://docs.datastax.com/en/astra/astra-db-vector/)).
- An **OpenAI API key**.

## Setup

Install the required libraries:

```bash
pip install --quiet "astrapy>=0.6.0" "openai>=1.0.0" datasets
```

Import the necessary modules:

```python
from getpass import getpass
from collections import Counter

from astrapy.db import AstraDB
import openai
from datasets import load_dataset
```

## Step 1: Configure Your Credentials

Set your Astra DB and OpenAI credentials. You'll be prompted to enter them securely.

```python
ASTRA_DB_API_ENDPOINT = input("Please enter your API Endpoint: ")
ASTRA_DB_APPLICATION_TOKEN = getpass("Please enter your Token: ")
OPENAI_API_KEY = getpass("Please enter your OpenAI API Key: ")
```

## Step 2: Initialize Clients

Create an Astra DB client and an OpenAI client.

```python
astra_db = AstraDB(
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)

client = openai.OpenAI(api_key=OPENAI_API_KEY)
embedding_model_name = "text-embedding-3-small"
```

## Step 3: Create a Vector Collection

Create a collection in Astra DB to store your quote embeddings. The dimension `1536` matches the output size of OpenAI's `text-embedding-3-small` model.

```python
coll_name = "philosophers_astra_db"
collection = astra_db.create_collection(coll_name, dimension=1536)
```

## Step 4: Load and Inspect the Quote Dataset

Load a pre-processed dataset of philosopher quotes.

```python
philo_dataset = load_dataset("datastax/philosopher-quotes")["train"]
```

Inspect the dataset:

```python
print("An example entry:")
print(philo_dataset[16])

author_count = Counter(entry["author"] for entry in philo_dataset)
print(f"Total: {len(philo_dataset)} quotes. By author:")
for author, count in author_count.most_common():
    print(f"    {author:<20}: {count} quotes")
```

Output:
```
An example entry:
{'author': 'aristotle', 'quote': 'Love well, be loved and do something of value.', 'tags': 'love;ethics'}

Total: 450 quotes. By author:
    aristotle           : 50 quotes
    schopenhauer        : 50 quotes
    spinoza             : 50 quotes
    hegel               : 50 quotes
    freud               : 50 quotes
    nietzsche           : 50 quotes
    sartre              : 50 quotes
    plato               : 50 quotes
    kant                : 50 quotes
```

## Step 5: Compute Embeddings and Store Quotes

Compute vector embeddings for each quote in batches and insert them into your Astra DB collection. This optimizes API calls and insertion speed.

```python
BATCH_SIZE = 20
num_batches = ((len(philo_dataset) + BATCH_SIZE - 1) // BATCH_SIZE)

quotes_list = philo_dataset["quote"]
authors_list = philo_dataset["author"]
tags_list = philo_dataset["tags"]

print("Starting to store entries: ", end="")
for batch_i in range(num_batches):
    b_start = batch_i * BATCH_SIZE
    b_end = (batch_i + 1) * BATCH_SIZE
    
    # Compute embeddings for this batch
    b_emb_results = client.embeddings.create(
        input=quotes_list[b_start : b_end],
        model=embedding_model_name,
    )
    
    # Prepare documents for insertion
    b_docs = []
    for entry_idx, emb_result in zip(range(b_start, b_end), b_emb_results.data):
        if tags_list[entry_idx]:
            tags = {tag: True for tag in tags_list[entry_idx].split(";")}
        else:
            tags = {}
        
        b_docs.append({
            "quote": quotes_list[entry_idx],
            "$vector": emb_result.embedding,
            "author": authors_list[entry_idx],
            "tags": tags,
        })
    
    # Insert batch into collection
    collection.insert_many(b_docs)
    print(f"[{len(b_docs)}]", end="")

print("\nFinished storing entries.")
```

## Step 6: Build a Quote Search Engine

Create a function that finds similar quotes based on semantic similarity. You can optionally filter by author or tags.

```python
def find_quote_and_author(query_quote, n, author=None, tags=None):
    # Compute embedding for the query
    query_vector = client.embeddings.create(
        input=[query_quote],
        model=embedding_model_name,
    ).data[0].embedding
    
    # Build filter clause
    filter_clause = {}
    if author:
        filter_clause["author"] = author
    if tags:
        filter_clause["tags"] = {tag: True for tag in tags}
    
    # Perform vector search
    results = collection.vector_find(
        query_vector,
        limit=n,
        filter=filter_clause,
        fields=["quote", "author"]
    )
    
    return [(result["quote"], result["author"]) for result in results]
```

### Test the Search Function

**Search by quote similarity:**
```python
results = find_quote_and_author("We struggle all our life for nothing", 3)
for quote, author in results:
    print(f"- {quote} ({author})")
```

**Search restricted to a specific author:**
```python
results = find_quote_and_author("We struggle all our life for nothing", 2, author="nietzsche")
for quote, author in results:
    print(f"- {quote} ({author})")
```

**Search constrained by tags:**
```python
results = find_quote_and_author("We struggle all our life for nothing", 2, tags=["politics"])
for quote, author in results:
    print(f"- {quote} ({author})")
```

## Step 7: Implement Similarity Threshold Filtering

To filter out irrelevant results, you can apply a similarity threshold. The `$similarity` field returns a value between 0 and 1, where 1 indicates perfect similarity.

```python
quote = "Animals are our equals."
metric_threshold = 0.92

quote_vector = client.embeddings.create(
    input=[quote],
    model=embedding_model_name,
).data[0].embedding

results_full = collection.vector_find(
    quote_vector,
    limit=8,
    fields=["quote"]
)
results = [res for res in results_full if res["$similarity"] >= metric_threshold]

print(f"{len(results)} quotes within the threshold:")
for idx, result in enumerate(results):
    print(f"    {idx}. [similarity={result['$similarity']:.3f}] \"{result['quote'][:70]}...\"")
```

## Step 8: Build a Quote Generator

Create a function that generates new philosophical quotes inspired by existing ones. This uses an LLM (GPT-3.5-turbo) with a prompt template.

```python
completion_model_name = "gpt-3.5-turbo"

generation_prompt_template = """Generate a single short philosophical quote on the given topic,
similar in spirit and form to the provided actual example quotes.
Do not exceed 20-30 words in your quote.

REFERENCE TOPIC: "{topic}"

ACTUAL EXAMPLES:
{examples}
"""

def generate_quote(topic, n=2, author=None, tags=None):
    quotes = find_quote_and_author(query_quote=topic, n=n, author=author, tags=tags)
    
    if quotes:
        prompt = generation_prompt_template.format(
            topic=topic,
            examples="\n".join(f"  - {quote[0]}" for quote in quotes),
        )
        
        # Log the found quotes
        print("** quotes found:")
        for q, a in quotes:
            print(f"**    - {q} ({a})")
        print("** end of logging")
        
        # Generate new quote
        response = client.chat.completions.create(
            model=completion_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=320,
        )
        return response.choices[0].message.content.replace('"', '').strip()
    else:
        print("** no quotes found.")
        return None
```

### Test the Quote Generator

**Generate a quote on a general topic:**
```python
q_topic = generate_quote("politics and virtue")
print("\nA new generated quote:")
print(q_topic)
```

**Generate a quote inspired by a specific philosopher:**
```python
q_topic = generate_quote("animals", author="schopenhauer")
print("\nA new generated quote:")
print(q_topic)
```

## Step 9: Cleanup (Optional)

If you want to delete the collection and all its data, run:

```python
astra_db.delete_collection(coll_name)
```

**Warning:** This action is irreversible and will permanently delete your quote data.

## Conclusion

You've successfully built a philosophy quote finder and generator using vector embeddings. You learned how to:

1. Store quote embeddings in Astra DB
2. Perform semantic similarity searches
3. Filter results by metadata and similarity thresholds
4. Generate new quotes using an LLM inspired by existing ones

This pattern can be extended to other domains like document retrieval, recommendation systems, or creative writing assistants.