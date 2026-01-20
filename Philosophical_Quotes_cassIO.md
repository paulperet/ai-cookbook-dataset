# Philosophy with Vector Embeddings, OpenAI and Cassandra / Astra DB through CQL

### CassIO version

In this quickstart you will learn how to build a "philosophy quote finder & generator" using OpenAI's vector embeddings and [Apache Cassandra®](https://cassandra.apache.org), or equivalently DataStax [Astra DB through CQL](https://docs.datastax.com/en/astra-serverless/docs/vector-search/quickstart.html), as the vector store for data persistence.

The basic workflow of this notebook is outlined below. You will evaluate and store the vector embeddings for a number of quotes by famous philosophers, use them to build a powerful search engine and, after that, even a generator of new quotes!

The notebook exemplifies some of the standard usage patterns of vector search -- while showing how easy is it to get started with the vector capabilities of [Cassandra](https://cassandra.apache.org/doc/trunk/cassandra/vector-search/overview.html) / [Astra DB through CQL](https://docs.datastax.com/en/astra-serverless/docs/vector-search/quickstart.html).

For a background on using vector search and text embeddings to build a question-answering system, please check out this excellent hands-on notebook: [Question answering using embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb).

#### _Choose-your-framework_

Please note that this notebook uses the [CassIO library](https://cassio.org), but we cover other choices of technology to accomplish the same task. Check out this folder's [README](https://github.com/openai/openai-cookbook/tree/main/examples/vector_databases/cassandra_astradb) for other options. This notebook can run either as a Colab notebook or as a regular Jupyter notebook.

Table of contents:
- Setup
- Get DB connection
- Connect to OpenAI
- Load quotes into the Vector Store
- Use case 1: **quote search engine**
- Use case 2: **quote generator**
- (Optional) exploit partitioning in the Vector Store

### How it works

**Indexing**

Each quote is made into an embedding vector with OpenAI's `Embedding`. These are saved in the Vector Store for later use in searching. Some metadata, including the author's name and a few other pre-computed tags, are stored alongside, to allow for search customization.

**Search**

To find a quote similar to the provided search quote, the latter is made into an embedding vector on the fly, and this vector is used to query the store for similar vectors ... i.e. similar quotes that were previously indexed. The search can optionally be constrained by additional metadata ("find me quotes by Spinoza similar to this one ...").

The key point here is that "quotes similar in content" translates, in vector space, to vectors that are metrically close to each other: thus, vector similarity search effectively implements semantic similarity. _This is the key reason vector embeddings are so powerful._

The sketch below tries to convey this idea. Each quote, once it's made into a vector, is a point in space. Well, in this case it's on a sphere, since OpenAI's embedding vectors, as most others, are normalized to _unit length_. Oh, and the sphere is actually not three-dimensional, rather 1536-dimensional!

So, in essence, a similarity search in vector space returns the vectors that are closest to the query vector:

**Generation**

Given a suggestion (a topic or a tentative quote), the search step is performed, and the first returned results (quotes) are fed into an LLM prompt which asks the generative model to invent a new text along the lines of the passed examples _and_ the initial suggestion.

## Setup

First install some required packages:


```python
!pip install --quiet "cassio>=0.1.3" "openai>=1.0.0" datasets
```


```python
from getpass import getpass
from collections import Counter

import cassio
from cassio.table import MetadataVectorCassandraTable

import openai
from datasets import load_dataset
```

## Get DB connection

In order to connect to your Astra DB through CQL, you need two things:
- A Token, with role "Database Administrator" (it looks like `AstraCS:...`)
- the database ID (it looks like `3df2a5b6-...`)

    Make sure you have both strings -- which are obtained in the [Astra UI](https://astra.datastax.com) once you sign in. For more information, see here: [database ID](https://awesome-astra.github.io/docs/pages/astra/faq/#where-should-i-find-a-database-identifier) and [Token](https://awesome-astra.github.io/docs/pages/astra/create-token/#c-procedure).

If you want to _connect to a Cassandra cluster_ (which however must [support](https://cassandra.apache.org/doc/trunk/cassandra/vector-search/overview.html) Vector Search), replace with `cassio.init(session=..., keyspace=...)` with suitable Session and keyspace name for your cluster.


```python
astra_token = getpass("Please enter your Astra token ('AstraCS:...')")
database_id = input("Please enter your database id ('3df2a5b6-...')")
```

    Please enter your Astra token ('AstraCS:...') ········
    Please enter your database id ('3df2a5b6-...') 01234567-89ab-dcef-0123-456789abcdef



```python
cassio.init(token=astra_token, database_id=database_id)
```

### Creation of the DB connection

This is how you create a connection to Astra DB through CQL:

_(Incidentally, you could also use any Cassandra cluster (as long as it provides Vector capabilities), just by [changing the parameters](https://docs.datastax.com/en/developer/python-driver/latest/getting_started/#connecting-to-cassandra) to the following `Cluster` instantiation.)_

### Creation of the Vector Store through CassIO

You need a table which support vectors and is equipped with metadata. Call it "philosophers_cassio":


```python
v_table = MetadataVectorCassandraTable(table="philosophers_cassio", vector_dimension=1536)
```

## Connect to OpenAI

### Set up your secret key


```python
OPENAI_API_KEY = getpass("Please enter your OpenAI API Key: ")
```

    Please enter your OpenAI API Key:  ········


### A test call for embeddings

Quickly check how one can get the embedding vectors for a list of input texts:


```python
client = openai.OpenAI(api_key=OPENAI_API_KEY)
embedding_model_name = "text-embedding-3-small"

result = client.embeddings.create(
    input=[
        "This is a sentence",
        "A second sentence"
    ],
    model=embedding_model_name,
)
```

_Note: the above is the syntax for OpenAI v1.0+. If using previous versions, the code to get the embeddings will look different._


```python
print(f"len(result.data)              = {len(result.data)}")
print(f"result.data[1].embedding      = {str(result.data[1].embedding)[:55]}...")
print(f"len(result.data[1].embedding) = {len(result.data[1].embedding)}")
```

    len(result.data)              = 2
    result.data[1].embedding      = [-0.010821706615388393, 0.001387271680869162, 0.0035479...
    len(result.data[1].embedding) = 1536


## Load quotes into the Vector Store

_Note: the above is the syntax for OpenAI v1.0+. If using previous versions, the code to get the embeddings will look different._


```python
philo_dataset = load_dataset("datastax/philosopher-quotes")["train"]
```

A quick inspection:


```python
print("An example entry:")
print(philo_dataset[16])
```

    An example entry:
    {'author': 'aristotle', 'quote': 'Love well, be loved and do something of value.', 'tags': 'love;ethics'}


Check the dataset size:


```python
author_count = Counter(entry["author"] for entry in philo_dataset)
print(f"Total: {len(philo_dataset)} quotes. By author:")
for author, count in author_count.most_common():
    print(f"    {author:<20}: {count} quotes")
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


### Insert quotes into vector store

You will compute the embeddings for the quotes and save them into the Vector Store, along with the text itself and the metadata planned for later use. Note that the author is added as a metadata field along with the "tags" already found with the quote itself.

To optimize speed and reduce the calls, you'll perform batched calls to the embedding OpenAI service.

_(Note: for faster execution, Cassandra and CassIO would let you do concurrent inserts, which we don't do here for a more straightforward demo code.)_


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
    # compute the embedding vectors for this batch
    b_emb_results = client.embeddings.create(
        input=quotes_list[b_start : b_end],
        model=embedding_model_name,
    )
    # prepare the rows for insertion
    print("B ", end="")
    for entry_idx, emb_result in zip(range(b_start, b_end), b_emb_results.data):
        if tags_list[entry_idx]:
            tags = {
                tag
                for tag in tags_list[entry_idx].split(";")
            }
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


## Use case 1: **quote search engine**

For the quote-search functionality, you need first to make the input quote into a vector, and then use it to query the store (besides handling the optional metadata into the search call, that is).

Encapsulate the search-engine functionality into a function for ease of re-use:


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
    #
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

### Putting search to test

Passing just a quote:


```python
find_quote_and_author("We struggle all our life for nothing", 3)
```




    [('Life to the great majority is only a constant struggle for mere existence, with the certainty of losing it at last.',
      'schopenhauer'),
     ('We give up leisure in order that we may have leisure, just as we go to war in order that we may have peace.',
      'aristotle'),
     ('Perhaps the gods are kind to us, by making life more disagreeable as we grow older. In the end death seems less intolerable than the manifold burdens we carry',
      'freud')]



Search restricted to an author:


```python
find_quote_and_author("We struggle all our life for nothing", 2, author="nietzsche")
```




    [('To live is to suffer, to survive is to find some meaning in the suffering.',
      'nietzsche'),
     ('What makes us heroic?--Confronting simultaneously our supreme suffering and our supreme hope.',
      'nietzsche')]



Search constrained to a tag (out of those saved earlier with the quotes):


```python
find_quote_and_author("We struggle all our life for nothing", 2, tags=["politics"])
```




    [('Mankind will never see an end of trouble until lovers of wisdom come to hold political power, or the holders of power become lovers of wisdom',
      'plato'),
     ('Everything the State says is a lie, and everything it has it has stolen.',
      'nietzsche')]



### Cutting out irrelevant results

The vector similarity search generally returns the vectors that are closest to the query, even if that means results that might be somewhat irrelevant if there's nothing better.

To keep this issue under control, you can get the actual "distance" between the query and each result, and then set a cutoff on it, effectively discarding results that are beyond that threshold.
Tuning this threshold correctly is not an easy problem: here, we'll just show you the way.

To get a feeling on how this works, try the following query and play with the choice of quote and threshold to compare the results:

_Note (for the mathematically inclined): this "distance" is exactly the cosine similarity between the vectors, i.e. the scalar product divided by the product of the norms of the two vectors. As such, it is a number ranging from -1 to +1, where -1 is for exactly opposite-facing vectors and +1 for identically-oriented vectors. Elsewhere (e.g. in the "CQL" counterpart of this demo) you would get a rescaling of this quantity to fit the [0, 1] interval, which means the resulting numerical values and adequate thresholds there are transformed accordingly._


```python
quote = "Animals are our equals."
# quote = "Be good."
# quote = "This teapot is strange."

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

    3 quotes within the threshold:
        0. [distance=0.855] "The assumption that animals are without rights, and the illusion that ..."
        1. [distance=0.843] "Animals are in possession of themselves; their soul is in possession o..."
        2. [distance=0.841] "At his best, man is the noblest of all animals; separated from law and..."


## Use case 2: **quote generator**

For this task you need another component from OpenAI, namely an LLM to generate the quote for us (based on input obtained by querying the Vector Store).

You also need a template for the prompt that will be filled for the generate-quote LLM completion task.


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

Like for search, this functionality is best wrapped into a handy function (which internally uses search):


```python
def generate_quote(topic, n=2, author=None, tags=None):
    quotes = find_quote_and_author(query_quote=topic, n=n, author=author, tags=tags)
    if quotes:
        prompt = generation_prompt_template.format(
            topic=topic,
            examples="\n".join(f"  - {quote[0]}" for quote in quotes),
        )
        # a little logging:
        print("** quotes found:")
        for q, a in quotes:
            print(f"**    - {q} ({a})")
        print("** end of logging")
        #
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

_Note: similar to the case of the embedding computation, the code for the Chat Completion API would be slightly different for OpenAI prior to v1.0._

#### Putting quote generation to test

Just passing a text (a "quote", but one can actually just suggest a topic since its vector embedding will still end up at the right place in the vector space):


```python
q_topic = generate_quote("politics and virtue")
print("\nA new generated quote:")
print(q_topic)
```

    ** quotes found:
    **    - Happiness is the reward of virtue. (aristotle)
    **    - Our moral virtues benefit mainly other people; intellectual virtues, on the other hand, benefit primarily ourselves; therefore the former make us universally popular, the latter unpopular. (schopenhauer)
    ** end of logging
    
    A new generated quote:
    Virtuous politics purifies society, while corrupt politics breeds