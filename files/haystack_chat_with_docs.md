# Using Mistral AI with Haystack

In this cookbook, we will use Mistral embeddings and generative models in 2 [Haystack](https://github.com/deepset-ai/haystack) pipelines:

1) We will build an indexing pipeline that can create embeddings for the contents of URLs and indexes them into a vector database
2) We will build a retrieval-augmented chat pipeline to chat with the contents of the URLs

First, we install our dependencies

```python
!pip install mistral-haystack
!pip install trafilatura
```

```python
from haystack import version
version.__version__
```

Next, we need to set the `MISTRAL_API_KEY` environment variable 👇

```python
import os
from getpass import getpass

os.environ["MISTRAL_API_KEY"] = getpass("Mistral API Key:")
```

## Index URLs with Mistral Embeddings

Below, we are using `mistral-embed` in a full Haystack indexing pipeline. We create embeddings for the contents of the chosen URLs with `mistral-embed` and write them to an [`InMemoryDocumentStore`](https://docs.haystack.deepset.ai/v2.0/docs/inmemorydocumentstore) using the [`MistralDocumentEmbedder`](https://docs.haystack.deepset.ai/v2.0/docs/mistraldocumentembedder). 

> 💡This document store is the simplest to get started with as it has no requirements to setup. Feel free to change this document store to any of the [vector databases available for Haystack 2.0](https://haystack.deepset.ai/integrations?type=Document+Store) such as **Weaviate**, **Chroma**, **AstraDB** etc.

```python
from haystack import Pipeline
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.mistral.document_embedder import MistralDocumentEmbedder


document_store = InMemoryDocumentStore()
fetcher = LinkContentFetcher()
converter = HTMLToDocument()
embedder = MistralDocumentEmbedder()
writer = DocumentWriter(document_store=document_store)

indexing = Pipeline()

indexing.add_component(name="fetcher", instance=fetcher)
indexing.add_component(name="converter", instance=converter)
indexing.add_component(name="embedder", instance=embedder)
indexing.add_component(name="writer", instance=writer)

indexing.connect("fetcher", "converter")
indexing.connect("converter", "embedder")
indexing.connect("embedder", "writer")
```

```python
urls = ["https://mistral.ai/news/la-plateforme/", "https://mistral.ai/news/mixtral-of-experts"]

indexing.run({"fetcher": {"urls": urls}})
```

[Calculating embeddings: 1it [00:00,  3.69it/s], {'embedder': {'meta': {'model': 'mistral-embed', 'usage': {'prompt_tokens': 1658, 'total_tokens': 1658, 'completion_tokens': 0}}}, 'writer': {'documents_written': 2}}]

## Chat With the URLs with Mistral Generative Models

Now that we have indexed the contents and embeddings of various URLs, we can create a RAG pipeline that uses the [`MistralChatGenerator`](https://docs.haystack.deepset.ai/v2.0/docs/mistralchatgenerator) component with `mistral-small`.
A few more things to know about this pipeline:

- We are using the [`MistralTextEmbdder`](https://docs.haystack.deepset.ai/v2.0/docs/mistraltextembedder) to embed our question and retrieve the most relevant 1 document
- We are enabling streaming responses by providing a `streaming_callback`
- `documents` is being provided to the chat template by the retriever, while we provide `query` to the pipeline when we run it.

```python
from haystack.dataclasses import ChatMessage

chat_template = """Answer the following question based on the contents of the documents.\n
                Question: {{query}}\n
                Documents: {{documents[0].content}}
                """
user_message = ChatMessage.from_user(chat_template)
```

```python
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack_integrations.components.embedders.mistral.text_embedder import MistralTextEmbedder
from haystack_integrations.components.generators.mistral import MistralChatGenerator

text_embedder = MistralTextEmbedder()
retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=1)
prompt_builder = ChatPromptBuilder(template=user_message, variables=["query", "documents"], required_variables=["query", "documents"])
llm = MistralChatGenerator(model='mistral-small', streaming_callback=print_streaming_chunk)

rag_pipeline = Pipeline()
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)


rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder.prompt", "llm.messages")
```

```python
question = "What generative endpoints does the Mistral platform have?"

messages = [ChatMessage.from_user(chat_template)]

result = rag_pipeline.run(
    {
        "text_embedder": {"text": question},
        "prompt_builder": {"template": messages, "query": question},
        "llm": {"generation_kwargs": {"max_tokens": 165}},
    },
    include_outputs_from=["text_embedder", "retriever", "llm"],
)
```

The Mistral platform has three generative endpoints: mistral-tiny, mistral-small, and mistral-medium. Each endpoint serves a different model with varying performance and language support. Mistral-tiny serves Mistral 7B Instruct v0.2, which is the most cost-effective and only supports English. Mistral-small serves Mixtral 8x7B, which supports English, French, Italian, German, Spanish, and code. Mistral-medium serves a prototype model with higher performance, also supporting the same languages and code as Mistral-small. Additionally, the platform offers an embedding endpoint called Mistral-embed, which serves an embedding model with a 1024 embedding dimension designed for retrieval capabilities.