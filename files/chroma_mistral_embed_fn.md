# ✨ Using Mistral-Embed with ChromaDB

> **Author:** Grant Smith  
> **GitHub:** [ggsmith842](https://github.com/ggsmith842)

---

This notebook walks through how to use MistralAI's `mistral-embed` model as a custom embedding function in the ChromaDB vector database.

### Documentation Links
1. [ChromaDB Documentation](https://docs.trychroma.com/docs/overview/introduction)
2. [Mistral Embeddings](https://docs.mistral.ai/capabilities/embeddings/)

```python
%pip install chromadb mistralai -Uq
```

```python
import os
import getpass
import chromadb

from mistralai import Mistral
from datetime import datetime

from chromadb import Documents, EmbeddingFunction, Embeddings
```

```python
if os.environ.get('MISTRAL_API_KEY'):
  api_key = os.environ['MISTRAL_API_KEY']
else:
  api_key = getpass.getpass("Please provide your mistralai api key:")
```

```python
# create a temp client that only lasts for the current session
client = chromadb.EphemeralClient()

# create a persistent client that can be used after session ends
# client = chromadb.PersistentClient(path = os.getcwd())
```

```python
# create custom embedding function using mistral-embed
class MistralEmbedFn(EmbeddingFunction):

    def __init__(self, api_key: str = None) -> None:
        if api_key:
            self.api_key = api_key
        else:
          try:
            self.api_key = getpass.getpass("Please provide your MistralAi API Key:")
          except Exception as e:
            print(f'Error getting API key from user: {e}')

    def __call__(self, input: Documents) -> Embeddings:
        client = Mistral(api_key=self.api_key)
        try:
          embeddings = [e.embedding for e in (client.embeddings.create(model='mistral-embed', inputs = input)).data]
          return embeddings
        except Exception as e:
          print(f'An error occured getting embeddings from model: {e}')
```

```python
# instantiate embedding function to use in collection
embed_fn = MistralEmbedFn(api_key=api_key)

# create collection
collection = client.create_collection(
    name="quotes",
    embedding_function = embed_fn, #MistralEmbedFn(),
    metadata={
        "description": "Quotes about Computer Science",
        "created": str(datetime.now())
    }
)
```

```python
# add data to collection
collection.add(
    documents=[
        "A new, a vast, and a powerful language is developed for the future use of analysis, in which to wield its truths so that these may become of more speedy and accurate practical application for the purposes of mankind than the means hitherto in our possession have rendered possible.",
        "A computer would deserve to be called intelligent if it could deceive a human into believing that it was human."
    ],
    metadatas = [{"attribution": "Ada Lovelace"}, {"attribution": "Alan Turing"}],
    ids = [f'id{i}' for i in range(2)]
)
```

```python
# peek at collection
collection.peek()
```