# Semantic Search with Pinecone and OpenAI

In this guide you will learn how to use the OpenAI Embedding API to generate language embeddings, and then index those embeddings in the Pinecone vector database for fast and scalable vector search.

This is a powerful and common combination for building semantic search, question-answering, threat-detection, and other applications that rely on NLP and search over a large corpus of text data.

The basic workflow looks like this:

**Embed and index**

* Use the OpenAI Embedding API to generate vector embeddings of your documents (or any text data).
* Upload those vector embeddings into Pinecone, which can store and index millions/billions of these vector embeddings, and search through them at ultra-low latencies.

**Search**

* Pass your query text or document through the OpenAI Embedding API again.
* Take the resulting vector embedding and send it as a query to Pinecone.
* Get back semantically similar documents, even if they don't share any keywords with the query.

Let's get started...

## Setup

We first need to setup our environment and retrieve API keys for OpenAI and Pinecone. Let's start with our environment, we need HuggingFace *Datasets* for our data, and the OpenAI and Pinecone clients:


```python
!pip install -qU \
    pinecone-client==3.0.2 \
    openai==1.10.0 \
    datasets==2.16.1
```

[First Entry, ..., Last Entry]

### Creating Embeddings

Then we initialize our connection to OpenAI Embeddings *and* Pinecone vector DB. Sign up for an API key over at [OpenAI](https://platform.openai.com) and [Pinecone](https://app.pinecone.io).


```python
from openai import OpenAI

client = OpenAI(
    api_key="OPENAI_API_KEY"
)  # get API key from platform.openai.com
```

We can now create embeddings with the OpenAI Ada similarity model like so:


```python
MODEL = "text-embedding-3-small"

res = client.embeddings.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], model=MODEL
)
res
```




```python
print(f"vector 0: {len(res.data[0].embedding)}\nvector 1: {len(res.data[1].embedding)}")
```

    vector 0: 1536
    vector 1: 1536



```python
# we can extract embeddings to a list
embeds = [record.embedding for record in res.data]
len(embeds)
```




    2



Next, we initialize our index to store vector embeddings with Pinecone.


```python
len(embeds[0])
```




    1536



Initialize connection to Pinecone, you can get a free API key in the [Pinecone dashboard](https://app.pinecone.io).


```python
from pinecone import Pinecone

pc = Pinecone(api_key="...")
```




```python
import time
from pinecone import ServerlessSpec

spec = ServerlessSpec(cloud="aws", region="us-west-2")

index_name = 'semantic-search-openai'

# check if index already exists (if shouldn't if this is your first run)
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=len(embeds[0]),  # dimensionality of text-embed-3-small
        metric='dotproduct',
        spec=spec
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# connect to index
index = pc.Index(index_name)
time.sleep(1)
# view index stats
index.describe_index_stats()
```




    {'dimension': 1536,
     'index_fullness': 0.0,
     'namespaces': {},
     'total_vector_count': 0}



## Populating the Index

Now we will take 1K questions from the TREC dataset


```python
from datasets import load_dataset

# load the first 1K rows of the TREC dataset
trec = load_dataset('trec', split='train[:1000]')
trec
```




    Dataset({
        features: ['text', 'coarse_label', 'fine_label'],
        num_rows: 1000
    })




```python
trec[0]
```




    {'text': 'How did serfdom develop in and then leave Russia ?',
     'coarse_label': 2,
     'fine_label': 26}



Then we create a vector embedding for each phrase using OpenAI, and `upsert` the ID, vector embedding, and original text for each phrase to Pinecone.


```python
from tqdm.auto import tqdm

count = 0  # we'll use the count to create unique IDs
batch_size = 32  # process everything in batches of 32
for i in tqdm(range(0, len(trec['text']), batch_size)):
    # set end position of batch
    i_end = min(i+batch_size, len(trec['text']))
    # get batch of lines and IDs
    lines_batch = trec['text'][i: i+batch_size]
    ids_batch = [str(n) for n in range(i, i_end)]
    # create embeddings
    res = client.embeddings.create(input=lines_batch, model=MODEL)
    embeds = [record.embedding for record in res.data]
    # prep metadata and upsert batch
    meta = [{'text': line} for line in lines_batch]
    to_upsert = zip(ids_batch, embeds, meta)
    # upsert to Pinecone
    index.upsert(vectors=list(to_upsert))
```

[First Entry, ..., Last Entry]

---

# Querying

With our data indexed, we're now ready to move onto performing searches. This follows a similar process to indexing. We start with a text `query`, that we would like to use to find similar sentences. As before we encode this with OpenAI's text similarity Babbage model to create a *query vector* `xq`. We then use `xq` to query the Pinecone index.


```python
query = "What caused the 1929 Great Depression?"

xq = client.embeddings.create(input=query, model=MODEL).data[0].embedding
```

Now query...


```python
res = index.query(vector=[xq], top_k=5, include_metadata=True)
res
```




    {'matches': [{'id': '932',
                  'metadata': {'text': 'Why did the world enter a global '
                                       'depression in 1929 ?'},
                  'score': 0.751888752,
                  'values': []},
                 {'id': '787',
                  'metadata': {'text': "When was `` the Great Depression '' ?"},
                  'score': 0.597448647,
                  'values': []},
                 {'id': '400',
                  'metadata': {'text': 'What crop failure caused the Irish Famine '
                                       '?'},
                  'score': 0.367482603,
                  'values': []},
                 {'id': '835',
                  'metadata': {'text': 'What were popular songs and types of songs '
                                       'in the 1920s ?'},
                  'score': 0.324545294,
                  'values': []},
                 {'id': '262',
                  'metadata': {'text': 'When did World War I start ?'},
                  'score': 0.320995867,
                  'values': []}],
     'namespace': '',
     'usage': {'read_units': 6}}



The response from Pinecone includes our original text in the `metadata` field, let's print out the `top_k` most similar questions and their respective similarity scores.


```python
for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")
```

    0.75: Why did the world enter a global depression in 1929 ?
    0.60: When was `` the Great Depression '' ?
    0.37: What crop failure caused the Irish Famine ?
    0.32: What were popular songs and types of songs in the 1920s ?
    0.32: When did World War I start ?


Looks good, let's make it harder and replace *"depression"* with the incorrect term *"recession"*.


```python
query = "What was the cause of the major recession in the early 20th century?"

# create the query embedding
xq = client.embeddings.create(input=query, model=MODEL).data[0].embedding

# query, returning the top 5 most similar results
res = index.query(vector=[xq], top_k=5, include_metadata=True)

for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")
```

    0.63: Why did the world enter a global depression in 1929 ?
    0.55: When was `` the Great Depression '' ?
    0.34: What were popular songs and types of songs in the 1920s ?
    0.33: What crop failure caused the Irish Famine ?
    0.29: What is considered the costliest disaster the insurance industry has ever faced ?


And again...


```python
query = "Why was there a long-term economic downturn in the early 20th century?"

# create the query embedding
xq = client.embeddings.create(input=query, model=MODEL).data[0].embedding

# query, returning the top 5 most similar results
res = index.query(vector=[xq], top_k=5, include_metadata=True)

for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")
```

    0.62: Why did the world enter a global depression in 1929 ?
    0.54: When was `` the Great Depression '' ?
    0.34: What were popular songs and types of songs in the 1920s ?
    0.33: What crop failure caused the Irish Famine ?
    0.32: What do economists do ?


Looks great, our semantic search pipeline is clearly able to identify the meaning between each of our queries and return the most semantically similar questions from the already indexed questions.

Once we're finished with the index we delete it to save resources.


```python
pc.delete_index(index_name)
```

---