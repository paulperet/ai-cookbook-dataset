# Retrieval Augmentation for GPT-4 using Pinecone

#### Fixing LLMs that Hallucinate

In this notebook we will learn how to query relevant contexts to our queries from Pinecone, and pass these to a GPT-4 model to generate an answer backed by real data sources.

GPT-4 is a big step up from previous OpenAI completion models. It also exclusively uses the `ChatCompletion` endpoint, so we must use it in a slightly different way to usual. However, the power of the model makes the change worthwhile, particularly when augmented with an external knowledge base like the Pinecone vector database.

Required installs for this notebook are:


```python
!pip install -qU bs4 tiktoken openai langchain pinecone-client[grpc]
```

[First Entry, ..., Last Entry]

## Preparing the Data

In this example, we will download the LangChain docs from [langchain.readthedocs.io/](https://langchain.readthedocs.io/latest/en/). We get all `.html` files located on the site like so:


```python
!wget -r -A.html -P rtdocs https://python.langchain.com/en/latest/
```

This downloads all HTML into the `rtdocs` directory. Now we can use LangChain itself to process these docs. We do this using the `ReadTheDocsLoader` like so:


```python
from langchain.document_loaders import ReadTheDocsLoader

loader = ReadTheDocsLoader('rtdocs')
docs = loader.load()
len(docs)
```

This leaves us with hundreds of processed doc pages. Let's take a look at the format each one contains:


```python
docs[0]
```

We access the plaintext page content like so:


```python
print(docs[0].page_content)
```


```python
print(docs[5].page_content)
```

We can also find the source of each document:


```python
docs[5].metadata['source'].replace('rtdocs/', 'https://')
```

We can use these to create our `data` list:


```python
data = []

for doc in docs:
    data.append({
        'url': doc.metadata['source'].replace('rtdocs/', 'https://'),
        'text': doc.page_content
    })
```


```python
data[3]
```

It's pretty ugly but it's good enough for now. Let's see how we can process all of these. We will chunk everything into ~400 token chunks, we can do this easily with `langchain` and `tiktoken`:


```python
import tiktoken

tokenizer = tiktoken.get_encoding('p50k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)
```


```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)
```

Process the `data` into more chunks using this approach.


```python
from uuid import uuid4
from tqdm.auto import tqdm

chunks = []

for idx, record in enumerate(tqdm(data)):
    texts = text_splitter.split_text(record['text'])
    chunks.extend([{
        'id': str(uuid4()),
        'text': texts[i],
        'chunk': i,
        'url': record['url']
    } for i in range(len(texts))])
```

Our chunks are ready so now we move onto embedding and indexing everything.

## Initialize Embedding Model

We use `text-embedding-3-small` as the embedding model. We can embed text like so:


```python
import openai

# initialize openai API key
openai.api_key = "sk-..."

embed_model = "text-embedding-3-small"

res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=embed_model
)
```

In the response `res` we will find a JSON-like object containing our new embeddings within the `'data'` field.


```python
res.keys()
```

Inside `'data'` we will find two records, one for each of the two sentences we just embedded. Each vector embedding contains `1536` dimensions (the output dimensionality of the `text-embedding-3-small` model.


```python
len(res['data'])
```


```python
len(res['data'][0]['embedding']), len(res['data'][1]['embedding'])
```

We will apply this same embedding logic to the langchain docs dataset we've just scraped. But before doing so we must create a place to store the embeddings.

## Initializing the Index

Now we need a place to store these embeddings and enable a efficient vector search through them all. To do that we use Pinecone, we can get a [free API key](https://app.pinecone.io/) and enter it below where we will initialize our connection to Pinecone and create a new index.


```python
import pinecone

index_name = 'gpt-4-langchain-docs'

# initialize connection to pinecone
pinecone.init(
    api_key="PINECONE_API_KEY",  # app.pinecone.io (console)
    environment="PINECONE_ENVIRONMENT"  # next to API key in console
)

# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='dotproduct'
    )
# connect to index
index = pinecone.GRPCIndex(index_name)
# view index stats
index.describe_index_stats()
```

We can see the index is currently empty with a `total_vector_count` of `0`. We can begin populating it with OpenAI `text-embedding-3-small` built embeddings like so:


```python
from tqdm.auto import tqdm
import datetime
from time import sleep

batch_size = 100  # how many embeddings we create and insert at once

for i in tqdm(range(0, len(chunks), batch_size)):
    # find end of batch
    i_end = min(len(chunks), i+batch_size)
    meta_batch = chunks[i:i_end]
    # get ids
    ids_batch = [x['id'] for x in meta_batch]
    # get texts to encode
    texts = [x['text'] for x in meta_batch]
    # create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                done = True
            except:
                pass
    embeds = [record['embedding'] for record in res['data']]
    # cleanup metadata
    meta_batch = [{
        'text': x['text'],
        'chunk': x['chunk'],
        'url': x['url']
    } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)
```

Now we've added all of our langchain docs to the index. With that we can move on to retrieval and then answer generation using GPT-4.

## Retrieval

To search through our documents we first need to create a query vector `xq`. Using `xq` we will retrieve the most relevant chunks from the LangChain docs, like so:


```python
query = "how do I use the LLMChain in LangChain?"

res = openai.Embedding.create(
    input=[query],
    engine=embed_model
)

# retrieve from Pinecone
xq = res['data'][0]['embedding']

# get relevant contexts (including the questions)
res = index.query(xq, top_k=5, include_metadata=True)
```


```python
res
```

With retrieval complete, we move on to feeding these into GPT-4 to produce answers.

## Retrieval Augmented Generation

GPT-4 is currently accessed via the `ChatCompletions` endpoint of OpenAI. To add the information we retrieved into the model, we need to pass it into our user prompts *alongside* our original query. We can do that like so:


```python
# get list of retrieved text
contexts = [item['metadata']['text'] for item in res['matches']]

augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query
```


```python
print(augmented_query)
```

Now we ask the question:


```python
# system message to 'prime' the model
primer = f"""You are Q&A bot. A highly intelligent system that answers
user questions based on the information provided by the user above
each question. If the information can not be found in the information
provided by the user you truthfully say "I don't know".
"""

res = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ]
)
```

To display this response nicely, we will display it in markdown.


```python
from IPython.display import Markdown

display(Markdown(res['choices'][0]['message']['content']))
```

Let's compare this to a non-augmented query...


```python
res = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": query}
    ]
)
display(Markdown(res['choices'][0]['message']['content']))
```

If we drop the `"I don't know"` part of the `primer`?


```python
res = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are Q&A bot. A highly intelligent system that answers user questions"},
        {"role": "user", "content": query}
    ]
)
display(Markdown(res['choices'][0]['message']['content']))
```


```python

```