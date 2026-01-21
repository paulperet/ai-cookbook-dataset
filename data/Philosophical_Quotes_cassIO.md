# Philosophy with Vector Embeddings, OpenAI, and Cassandra/Astra DB

## Overview

In this guide, you will build a "philosophy quote finder & generator" using OpenAI's vector embeddings and Apache Cassandra® (or DataStax Astra DB through CQL) as the vector store. You will learn how to:
- Store and index vector embeddings for philosophical quotes.
- Build a semantic search engine to find similar quotes.
- Create an AI-powered quote generator that produces new philosophical statements.

This tutorial demonstrates standard vector search patterns and highlights the ease of using Cassandra's vector capabilities.

### Prerequisites

- An **Astra DB** database (or a Cassandra cluster with vector search support).
- An **OpenAI API key**.
- Basic familiarity with Python.

## 1. Setup

Begin by installing the required Python packages.

```bash
pip install --quiet "cassio>=0.1.3" "openai>=1.0.0" datasets
```

Now, import the necessary modules.

```python
from getpass import getpass
from collections import Counter

import cassio
from cassio.table import MetadataVectorCassandraTable

import openai
from datasets import load_dataset
```

## 2. Connect to Your Database

To connect to Astra DB through CQL, you need a **Database Administrator Token** and your **database ID**. Both can be found in the [Astra UI](https://astra.datastax.com).

If you are connecting to a Cassandra cluster instead, you would use `cassio.init(session=..., keyspace=...)` with your session and keyspace.

```python
astra_token = getpass("Please enter your Astra token ('AstraCS:...'): ")
database_id = input("Please enter your database ID ('3df2a5b6-...'): ")
```

Initialize the CassIO connection.

```python
cassio.init(token=astra_token, database_id=database_id)
```

## 3. Create the Vector Store

Create a table named `philosophers_cassio` to store vectors and their associated metadata. The vector dimension is 1536, matching OpenAI's embedding model.

```python
v_table = MetadataVectorCassandraTable(table="philosophers_cassio", vector_dimension=1536)
```

## 4. Connect to OpenAI

Set your OpenAI API key and initialize the client.

```python
OPENAI_API_KEY = getpass("Please enter your OpenAI API Key: ")
client = openai.OpenAI(api_key=OPENAI_API_KEY)
```

Define the embedding model you will use.

```python
embedding_model_name = "text-embedding-3-small"
```

### Test the Embedding Service

Quickly verify that you can obtain embeddings.

```python
result = client.embeddings.create(
    input=[
        "This is a sentence",
        "A second sentence"
    ],
    model=embedding_model_name,
)

print(f"Number of embeddings: {len(result.data)}")
print(f"First few values of the second embedding: {str(result.data[1].embedding)[:55]}...")
print(f"Embedding dimension: {len(result.data[1].embedding)}")
```

```
Number of embeddings: 2
First few values of the second embedding: [-0.010821706615388393, 0.001387271680869162, 0.0035479...
Embedding dimension: 1536
```

## 5. Load Quotes into the Vector Store

You will use a dataset of philosophical quotes. Load and inspect it.

```python
philo_dataset = load_dataset("datastax/philosopher-quotes")["train"]

print("An example entry:")
print(philo_dataset[16])
```

```
An example entry:
{'author': 'aristotle', 'quote': 'Love well, be loved and do something of value.', 'tags': 'love;ethics'}
```

Check the dataset size and distribution.

```python
author_count = Counter(entry["author"] for entry in philo_dataset)
print(f"Total: {len(philo_dataset)} quotes. By author:")
for author, count in author_count.most_common():
    print(f"    {author:<20}: {count} quotes")
```

```
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

### Insert Quotes with Embeddings

Now, compute embeddings for each quote in batches and insert them into the vector store. Each entry includes the quote text, its embedding vector, and metadata (author and tags).

```python
BATCH_SIZE = 50
num_batches = ((len(philo_dataset) + BATCH_SIZE - 1) // BATCH_SIZE)

quotes_list = philo_dataset["quote"]
authors_list = philo_dataset["author"]
tags_list = philo_dataset["tags"]

print("Starting to store entries:")
for batch_i in range(num_batches):
    b_start = batch_i * BATCH_SIZE
    b_end = (batch_i + 1) * BATCH_SIZE
    
    # Compute embeddings for this batch
    b_emb_results = client.embeddings.create(
        input=quotes_list[b_start:b_end],
        model=embedding_model_name,
    )
    
    # Insert each quote with its metadata
    print("B ", end="")
    for entry_idx, emb_result in zip(range(b_start, b_end), b_emb_results.data):
        if tags_list[entry_idx]:
            tags = {tag for tag in tags_list[entry_idx].split(";")}
        else:
            tags = set()
        
        author = authors_list[entry_idx]
        quote = quotes_list[entry_idx]
        
        v_table.put(
            row_id=f"q_{author}_{entry_idx}",
            body_blob=quote,
            vector=emb_result.embedding,
            metadata={**{tag: True for tag in tags}, **{"author": author}},
        )
        print("*", end="")
    print(f" done ({len(b_emb_results.data)})")

print("\nFinished storing entries.")
```

```
Starting to store entries:
B [**************************************************, done (50)]
B [**************************************************, done (50)]
B [**************************************************, done (50)]
B [**************************************************, done (50)]
B [**************************************************, done (50)]
B [**************************************************, done (50)]
B [**************************************************, done (50)]
B [**************************************************, done (50)]
B [**************************************************, done (50)]

Finished storing entries.
```

## 6. Use Case 1: Quote Search Engine

Create a search function that:
1. Embeds the query quote.
2. Queries the vector store for similar quotes.
3. Optionally filters by author or tags.

```python
def find_quote_and_author(query_quote, n, author=None, tags=None):
    query_vector = client.embeddings.create(
        input=[query_quote],
        model=embedding_model_name,
    ).data[0].embedding
    
    metadata = {}
    if author:
        metadata["author"] = author
    if tags:
        for tag in tags:
            metadata[tag] = True
    
    results = v_table.ann_search(
        query_vector,
        n=n,
        metadata=metadata,
    )
    return [
        (result["body_blob"], result["metadata"]["author"])
        for result in results
    ]
```

### Test the Search Engine

**Search by quote similarity:**

```python
find_quote_and_author("We struggle all our life for nothing", 3)
```

```
[('Life to the great majority is only a constant struggle for mere existence, with the certainty of losing it at last.',
  'schopenhauer'),
 ('We give up leisure in order that we may have leisure, just as we go to war in order that we may have peace.',
  'aristotle'),
 ('Perhaps the gods are kind to us, by making life more disagreeable as we grow older. In the end death seems less intolerable than the manifold burdens we carry',
  'freud')]
```

**Search restricted to a specific author:**

```python
find_quote_and_author("We struggle all our life for nothing", 2, author="nietzsche")
```

```
[('To live is to suffer, to survive is to find some meaning in the suffering.',
  'nietzsche'),
 ('What makes us heroic?--Confronting simultaneously our supreme suffering and our supreme hope.',
  'nietzsche')]
```

**Search constrained by a tag:**

```python
find_quote_and_author("We struggle all our life for nothing", 2, tags=["politics"])
```

```
[('Mankind will never see an end of trouble until lovers of wisdom come to hold political power, or the holders of power become lovers of wisdom',
  'plato'),
 ('Everything the State says is a lie, and everything it has it has stolen.',
  'nietzsche')]
```

### Filtering by Similarity Threshold

To avoid irrelevant results, you can set a cosine similarity threshold. Only vectors closer than this threshold will be returned.

```python
quote = "Animals are our equals."
metric_threshold = 0.84

quote_vector = client.embeddings.create(
    input=[quote],
    model=embedding_model_name,
).data[0].embedding

results = list(v_table.metric_ann_search(
    quote_vector,
    n=8,
    metric="cos",
    metric_threshold=metric_threshold,
))

print(f"{len(results)} quotes within the threshold:")
for idx, result in enumerate(results):
    print(f"    {idx}. [distance={result['distance']:.3f}] \"{result['body_blob'][:70]}...\"")
```

```
3 quotes within the threshold:
    0. [distance=0.855] "The assumption that animals are without rights, and the illusion that ..."
    1. [distance=0.843] "Animals are in possession of themselves; their soul is in possession o..."
    2. [distance=0.841] "At his best, man is the noblest of all animals; separated from law and..."
```

## 7. Use Case 2: Quote Generator

Now, build a quote generator that uses an LLM to create new philosophical quotes inspired by search results.

Define the completion model and a prompt template.

```python
completion_model_name = "gpt-3.5-turbo"

generation_prompt_template = """"Generate a single short philosophical quote on the given topic,
similar in spirit and form to the provided actual example quotes.
Do not exceed 20-30 words in your quote.

REFERENCE TOPIC: "{topic}"

ACTUAL EXAMPLES:
{examples}
"""
```

Create a generation function that:
1. Searches for similar quotes.
2. Constructs a prompt with the topic and examples.
3. Calls the LLM to generate a new quote.

```python
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

### Generate a New Quote

Test the generator with a topic.

```python
q_topic = generate_quote("politics and virtue")
print("\nA new generated quote:")
print(q_topic)
```

```
** quotes found:
**    - Happiness is the reward of virtue. (aristotle)
**    - Our moral virtues benefit mainly other people; intellectual virtues, on the other hand, benefit primarily ourselves; therefore the former make us universally popular, the latter unpopular. (schopenhauer)
** end of logging

A new generated quote:
Virtuous politics purifies society, while corrupt politics breeds
```

## Summary

You have successfully built a vector-powered philosophy quote application. You learned how to:
- Store and index text embeddings in Cassandra/Astra DB.
- Perform semantic similarity searches with optional metadata filtering.
- Generate new content using an LLM prompted by search results.

This pattern can be extended to other domains like document retrieval, recommendation systems, or creative writing assistants.