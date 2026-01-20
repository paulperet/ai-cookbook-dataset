# Building A RAG Ebook "Librarian" Using LlamaIndex

_Authored by: [Jonathan Jin](https://huggingface.co/jinnovation)_

## Introduction

This notebook demonstrates how to quickly build a RAG-based "librarian" for your local ebook library.

Think about the last time you visited a library and took advantage of the expertise of the knowledgeable staff there to help you find what you need out of the troves of textbooks, novels, and other resources at the library. Our RAG "librarian" will do the same for us, except for our own local collection of ebooks.

## Requirements

We'd like our librarian to be **lightweight** and **run locally as much as possible** with **minimal dependencies**. This means that we will leverage open-source to the fullest extent possible, as well as bias towards models that can be **executed locally on typical hardware, e.g. M1 Macbooks**.

## Components

Our solution will consist of the following components:

- [LlamaIndex], a data framework for LLM-based applications that's, unlike [LangChain], designed specifically for RAG;
- [Ollama], a user-friendly solution for running LLMs such as Llama 2 locally;
- The [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) embedding model, which performs [reasonably well and is reasonably lightweight in size](https://huggingface.co/spaces/mteb/leaderboard);
- [Llama 2], which we'll run via [Ollama].

[LlamaIndex]: https://docs.llamaindex.ai/en/stable/index.html
[LangChain]: https://python.langchain.com/docs/get_started/introduction
[Ollama]: https://ollama.com/
[Llama 2]: https://ollama.com/library/llama2

## Dependencies

First let's install our dependencies.

```python
%pip install -q \
    llama-index \
    EbookLib \
    html2text \
    llama-index-embeddings-huggingface \
    llama-index-llms-ollama
```

## Ollama installation

These dependencies help properly detect the GPU.

```python
!apt install pciutils lshw
```

Install Ollama.

```python
!curl -fsSL https://ollama.com/install.sh | sh
```

Run Ollama service in the background.

```python
get_ipython().system_raw('ollama serve &')
```

Pull Llama2 from the Ollama library.

```python
!ollama pull llama2
```

## Test Library Setup

Next, let's create our test "library."

For simplicity's sake, let's say that our "library" is simply a **nested directory of `.epub` files**. We can easily see this solution generalizing to, say, a Calibre library with a `metadata.db` database file. We'll leave that extension as an exercise for the reader. 😇

Let's pull two `.epub` files from [Project Gutenberg](https://www.gutenberg.org/) for our library.

```python
!mkdir -p "./test/library/jane-austen"
!mkdir -p "./test/library/victor-hugo"
!wget https://www.gutenberg.org/ebooks/1342.epub.noimages -O "./test/library/jane-austen/pride-and-prejudice.epub"
!wget https://www.gutenberg.org/ebooks/135.epub.noimages -O "./test/library/victor-hugo/les-miserables.epub"
```

## RAG with LlamaIndex

RAG with LlamaIndex, at its core, consists of the following broad phases:

1. **Loading**, in which you tell LlamaIndex where your data lives and how to load it;
2. **Indexing**, in which you augment your loaded data to facilitate querying, e.g. with vector embeddings;
3. **Querying**, in which you configure an LLM to act as the query interface for your indexed data.

This explanation only scratches at the surface of what's possible with LlamaIndex. For more in-depth details, I highly recommend reading the "High-Level Concepts" page of the LlamaIndex documentation.

### Loading

Naturally, let's start with the **loading** phase.

I mentioned before that LlamaIndex is designed specifically for RAG. This immediately becomes obvious from its `SimpleDirectoryReader` construct, which ✨ **magically** ✨ supports a whole host of multi-model file types for free. Conveniently for us, `.epub` is in the supported set.

```python
from llama_index.core import SimpleDirectoryReader

loader = SimpleDirectoryReader(
    input_dir="./test/",
    recursive=True,
    required_exts=[".epub"],
)

documents = loader.load_data()
```

`SimpleDirectoryReader.load_data()` converts our ebooks into a set of `Document`s for LlamaIndex to work with.

One important thing to note here is that the documents **have not been chunked at this stage** -- that will happen during indexing. Read on...

### Indexing

Next up after **loading** the data is to **index** it. This will allow our RAG pipeline to look up the relevant context for our query to pass to our LLM to **augment** their generated response. This is also where document chunking will take place.

`VectorStoreIndex` is a "default" entrypoint for indexing in LlamaIndex. By default, `VectorStoreIndex` uses a simple, in-memory dictionary to store the indices, but LlamaIndex also supports a wide variety of vector storage solutions for you to graduate to as you scale.

By default, LlamaIndex uses a chunk size of 1024 and a chunk overlap of 20. For more details, see the LlamaIndex documentation.

Like mentioned before, we'll use the `BAAI/bge-small-en-v1.5` to generate our embeddings. By default, LlamaIndex uses OpenAI (specifically `gpt-3.5-turbo`), which we'd like to avoid given our desire for a lightweight, locally-runnable end-to-end solution.

Thankfully, LlamaIndex supports retrieving embedding models from Hugging Face through the convenient `HuggingFaceEmbedding` class, so we'll use that here.

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

We'll pass that in to `VectorStoreIndex` as our embedding model to circumvent the OpenAI default behavior.

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embedding_model,
)
```

### Querying

Now for the final piece of the RAG puzzle -- wiring up the query layer.

We'll use Llama 2 for the purposes of this recipe, but I encourage readers to play around with different models to see which produces the "best" responses here.

First let's start up the Ollama server. Unfortunately, there is no support in the Ollama Python client for actually starting and stopping the server itself, so we'll have to pop out of Python land for this.

In a separate terminal, run: `ollama serve`. Remember to terminate this after we're done here!

Now let's hook Llama 2 up to LlamaIndex and use it as the basis of our query engine.

```python
from llama_index.llms.ollama import Ollama

llama = Ollama(
    model="llama2",
    request_timeout=40.0,
)

query_engine = index.as_query_engine(llm=llama)
```

## Final Result

With that, our basic RAG librarian is set up and we can start asking questions about our library. For example:

```python
print(query_engine.query("What are the titles of all the books available? Show me the context used to derive your answer."))
```

Based on the context provided, there are two books available:

1. "Pride and Prejudice" by Jane Austen
2. "Les Misérables" by Victor Hugo

The context used to derive this answer includes:

* The file path for each book, which provides information about the location of the book files on the computer.
* The titles of the books, which are mentioned in the context as being available for reading.
* A list of words associated with each book, such as "epub" and "notebooks", which provide additional information about the format and storage location of each book.

```python
print(query_engine.query("Who is the main character of 'Pride and Prejudice'?"))
```

The main character of 'Pride and Prejudice' is Elizabeth Bennet.

## Conclusion and Future Improvements

We've demonstrated how to build a basic RAG-based "librarian" that runs entirely locally, even on Apple silicon Macs. In doing so, we've also carried out a "grand tour" of LlamaIndex and how it streamlines the process of setting up RAG-based applications.

That said, we've really only scratched the surface of what's possible here. Here are some ideas of how to refine and build upon this foundation.

### Forcing Citations

To guard against the risk of our librarian hallucinating, how might we require that it provide citations for everything that it says?

### Using Extended Metadata

Ebook library management solutions like Calibre create additional metadata for ebooks in a library. This can provide information such as publisher or edition that might not be readily available in the text of the book itself. How could we extend our RAG pipeline to account for additional sources of information that aren't `.epub` files?

### Efficient Indexing

If we were to collect everything we built here into a script/executable, the resulting script would re-index our library on each invocation. For our tiny test library of two files, this is "fine," but for any library of non-trivial size this will very quickly become annoying for users. How could we persist the embedding indices and only update them when the contents of the library have meaningfully changed, e.g. new books have been added?