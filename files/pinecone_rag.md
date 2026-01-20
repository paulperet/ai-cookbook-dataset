# RAG with Mistral AI and Pinecone

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mistralai/cookbook/blob/main/third_party/Pinecone/pinecone_rag.ipynb) [![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/mistralai/cookbook/blob/main/third_party/Pinecone/pinecone_rag.ipynb)

To begin, we setup our prerequisite libraries.

```python
!pip3 install -qU datasets mistralai pinecone
```

## Data Preparation

We start by downloading a dataset that we will encode and store. The dataset [`jamescalam/ai-arxiv2-semantic-chunks`](https://huggingface.co/datasets/jamescalam/ai-arxiv2-semantic-chunks) contains scraped data from many popular ArXiv papers centred around LLMs and GenAI.

```python
from datasets import load_dataset

data = load_dataset(
    "jamescalam/ai-arxiv2-semantic-chunks",
    split="train[:10000]"
)
data
```

We have 200K chunks, where each chunk is roughly the length of 1-2 paragraphs in length. Here is an example of a single record:

```python
data[0]
```

Format the data into the format we need, this will contain `id`, `text` (which we will embed), and `metadata`.

```python
data = data.map(lambda x: {
    "id": x["id"],
    "metadata": {
        "title": x["title"],
        "content": x["content"],
    }
})
# drop unneeded columns
data = data.remove_columns([
    "title", "content", "prechunk_id",
    "postchunk_id", "arxiv_id", "references"
])
data
```

We need to define an embedding model to create our embedding vectors for retrieval, for that we will be using Mistral AI's `mistral-embed`. There is some cost associated with this model, so be aware of that (costs for running this notebook are <$1).

```python
import os
from mistralai import Mistral
import getpass  # console.mistral.ai/api-keys/

# get API key from left navbar in Mistral console
mistral_api_key = os.getenv("MISTRAL_API_KEY") or getpass.getpass("Enter your Mistral API key: ")

# initialize client
mistral = Mistral(api_key=mistral_api_key)
```

We can create embeddings now like so:

```python
embed_model = "mistral-embed"

embeds = mistral.embeddings.create(
    model=embed_model, inputs=["this is a test"]
)
```

We can view the dimensionality of our returned embeddings, which we'll need soon when initializing our vector index:

```python
dims = len(embeds.data[0].embedding)
dims
```

Now we create our vector DB to store our vectors. For this we need to get a [free Pinecone API key](https://app.pinecone.io) — the API key can be found in the "API Keys" button found in the left navbar of the Pinecone dashboard.

```python
from pinecone import Pinecone

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_API_KEY") or getpass.getpass("Enter your Pinecone API key: ")

# configure client
pc = Pinecone(api_key=api_key)
```

Now we setup our index specification, this allows us to define the cloud provider and region where we want to deploy our index. You can find a list of all [available providers and regions here](https://docs.pinecone.io/docs/projects).

```python
from pinecone import ServerlessSpec

spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
)
```

Creating an index, we set `dimension` equal to the dimensionality of `mistral-embed` (`1024`), and use a `metric` also compatible with `mistral-embed` (this can be either `cosine` or `dotproduct`). We also pass our `spec` to index initialization.

```python
import time

index_name = "mistral-rag"
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

# check if index already exists (it shouldn't if this is first time)
if index_name not in existing_indexes:
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=dims,  # dimensionality of mistral-embed
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

We will define an embedding function that will allow us to avoid throwing too many tokens into a single embedding batch (as of 21 May 2024 the limit is `16384` tokens).

```python
def embed(metadata: list[dict]):
    batch_size = len(metadata)
    while batch_size >= 1:  # Allow batch_size to go down to 1
        try:
            embeds = []
            for j in range(0, len(metadata), batch_size):
                j_end = min(len(metadata), j + batch_size)
                input_texts = [x["title"] + "\n" + x["content"] for x in metadata[j:j_end]]
                embed_response = mistral.embeddings.create(
                    inputs=input_texts,
                    model=embed_model
                )
                embeds.extend([x.embedding for x in embed_response.data])
            return embeds
        except Exception as e:
            batch_size = int(batch_size / 2)
            print(f"Hit an exception: {e}, attempting {batch_size=}")
    raise Exception("Failed to embed metadata after multiple attempts.")
```

We can see the index is currently empty with a `total_vector_count` of `0`. We can begin populating it with `mistral-embed` built embeddings like so:

**⚠️ WARNING: Embedding costs for the full dataset as of 3 Jan 2024 is ~$5.70**

```python
from tqdm.auto import tqdm

batch_size = 32  # how many embeddings we create and insert at once

for i in tqdm(range(0, len(data), batch_size)):
    # find end of batch
    i_end = min(len(data), i+batch_size)
    # create batch
    batch = data[i:i_end]
    # create embeddings
    embeds = embed(batch["metadata"])
    assert len(embeds) == (i_end-i)
    to_upsert = list(zip(batch["id"], embeds, batch["metadata"]))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)
```

Now let's test retrieval!

```python
def get_docs(query: str, top_k: int) -> list[str]:
    # encode query
    xq = mistral.embeddings.create(
        inputs=[query],
        model=embed_model
    ).data[0].embedding
    # search pinecone index
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    # get doc text
    docs = [x["metadata"]['content'] for x in res["matches"]]
    return docs
```

```python
query = "can you tell me about mistral LLM?"
docs = get_docs(query, top_k=5)
print("\n---\n".join(docs))
```

Our retrieval component works, now let's try feeding this into Mistral Large LLM to produce an answer.

```python
def generate(query: str, docs: list[str]):
    system_message = (
        "You are a helpful assistant that answers questions about AI using the "
        "context provided below.\n\n"
        "CONTEXT:\n"
        "\n---\n".join(docs)
    )
    messages = [
        {
            "role":"system", "content":system_message
        },
        {
            "role":"user", "content":query
        }
    ]
    # generate response
    chat_response = mistral.chat.complete(
        model="mistral-large-latest",
        messages=messages
    )
    return chat_response.choices[0].message.content
```

```python
out = generate(query=query, docs=docs)
print(out)
```

Don't forget to delete your index when you're done to save resources!

```python
pc.delete_index(index_name)
```