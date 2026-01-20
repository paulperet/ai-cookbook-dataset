# Building RAG with Custom Unstructured Data

_Authored by: [Maria Khalusova](https://github.com/MKhalusova)_

If you're new to RAG, please explore the basics of RAG first in [this other notebook](https://huggingface.co/learn/cookbook/rag_zephyr_langchain), and then come back here to learn about building RAG with custom data.

Whether you're building your own RAG-based personal assistant, a pet project, or an enterprise RAG system, you will quickly discover that a lot of important knowledge is stored in various formats like PDFs, emails, Markdown files, PowerPoint presentations, HTML pages, Word documents, and so on.

How do you preprocess all of this data in a way that you can use it for RAG?
In this quick tutorial, you'll learn how to build a RAG system that will incorporate data from multiple data types. You'll use [Unstructured](https://github.com/Unstructured-IO/unstructured) for data preprocessing, open-source models from Hugging Face Hub for embeddings and text generation, ChromaDB as a vector store, and LangChain for bringing everything together.

Let's go! We'll begin by installing the required dependencies:


```python
!pip install -q torch transformers accelerate bitsandbytes sentence-transformers unstructured[all-docs] langchain chromadb langchain_community
```

Next, let's get a mix of documents. Suppose, I want to build a RAG system that'll help me manage pests in my garden. For this purpose, I'll use diverse documents that cover the topic of IPM (integrated pest management):
* PDF: `https://www.gov.nl.ca/ecc/files/env-protection-pesticides-business-manuals-applic-chapter7.pdf`
* Powerpoint: `https://ipm.ifas.ufl.edu/pdfs/Citrus_IPM_090913.pptx`
* EPUB: `https://www.gutenberg.org/ebooks/45957`
* HTML: `https://blog.fifthroom.com/what-to-do-about-harmful-garden-and-plant-insects-and-pests.html`

Feel free to use your own documents for your topic of choice from the list of document types supported by Unstructured: `.eml`, `.html`, `.md`, `.msg`, `.rst`, `.rtf`, `.txt`, `.xml`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.heic`, `.csv`, `.doc`, `.docx`, `.odt`, `.pdf`, `.ppt`, `.pptx`, `.tsv`, `.xlsx`.


```python
!mkdir -p "./documents"
!wget https://www.gov.nl.ca/ecc/files/env-protection-pesticides-business-manuals-applic-chapter7.pdf -O "./documents/env-protection-pesticides-business-manuals-applic-chapter7.pdf"
!wget https://ipm.ifas.ufl.edu/pdfs/Citrus_IPM_090913.pptx -O "./documents/Citrus_IPM_090913.pptx"
!wget https://www.gutenberg.org/ebooks/45957.epub3.images -O "./documents/45957.epub"
!wget https://blog.fifthroom.com/what-to-do-about-harmful-garden-and-plant-insects-and-pests.html -O "./documents/what-to-do-about-harmful-garden-and-plant-insects-and-pests.html"
```

## Unstructured data preprocessing

You can use the Unstructured library to preprocess documents one by one, and write your own script to walk through a directory, but it's easier to use a Local source connector to ingest all documents in a given directory. Unstructured can ingest documents from local directories, S3 buckets, blob storage, SFTP, and many other places your documents might be stored in. The ingestion from those sources will be very similar differing mostly in authentication options.
Here you'll use Local source connector, but feel free to explore other options in the [Unstructured documentation](https://docs.unstructured.io/open-source/ingest/source-connectors/overview).

Optionally, you can also choose a [destination](https://docs.unstructured.io/open-source/ingest/destination-connectors/overview) for the processed documents - this could be MongoDB, Pinecone, Weaviate, etc. In this notebook, we'll keep everything local.


```python
# Optional cell to reduce the amount of logs

import logging

logger = logging.getLogger("unstructured.ingest")
logger.root.removeHandler(logger.root.handlers[0])
```


```python
import os

from unstructured.ingest.connector.local import SimpleLocalConfig
from unstructured.ingest.interfaces import PartitionConfig, ProcessorConfig, ReadConfig
from unstructured.ingest.runner import LocalRunner

output_path = "./local-ingest-output"

runner = LocalRunner(
    processor_config=ProcessorConfig(
        # logs verbosity
        verbose=True,
        # the local directory to store outputs
        output_dir=output_path,
        num_processes=2,
        ),
    read_config=ReadConfig(),
    partition_config=PartitionConfig(
        partition_by_api=True,
        api_key="YOUR_UNSTRUCTURED_API_KEY",
        ),
    connector_config=SimpleLocalConfig(
        input_path="./documents",
        # whether to get the documents recursively from given directory
        recursive=False,
        ),
    )
runner.run()

```

    [INFO: NumExpr defaulting to 2 threads., ..., 2024-06-04 13:09:41,469 MainProcess INFO     Running copy node to move content to desired output location]

Let's take a closer look at the configs that we have here.

`ProcessorConfig` controls various aspects of the processing pipeline, including output locations, number of workers, error handling behavior, logging verbosity and more. The only mandatory parameter here is the `output_dir` - the local directory where you want to store the outputs.

`ReadConfig` can be used to customize the data reading process for different scenarios, such as re-downloading data, preserving downloaded files, or limiting the number of documents processed. In most cases the default `ReadConfig` will work.

In the `PartitionConfig` you can choose whether to partition the documents locally or via API. This example uses API, and for this reason requires Unstructured API key. You can get yours [here](https://unstructured.io/api-key-free).  The free Unstructured API is capped at 1000 pages, and offers better OCR models for image-based documents than a local installation of Unstructured.
If you remove these two parameters, the documents will be processed locally, but you may need to install additional dependencies if the documents require OCR and/or document understanding models. Namely, you may need to install poppler and tesseract in this case, which you can get with brew:

```
!brew install poppler
!brew install tesseract
```

If you're on Windows, you can find alternative installation instructions in the [Unstructured docs](https://docs.unstructured.io/open-source/installation/full-installation). 

Finally, in the `SimpleLocalConfig` you need to specify where your original documents reside, and whether you want to walk through the directory recursively.

Once the documents are processed you'll find 4 json files in the `local-ingest-output` directory, one per document that was processed.
Unstructured partitions all types of documents in a uniform manner, and returns json with document elements.

[Document elements](https://docs.unstructured.io/api-reference/api-services/document-elements) have a type, e.g. `NarrativeText`, `Title`, or `Table`, they contain the extracted text, and metadata that Unstructured was able to obtain. Some metadata is common for all elements, such as filename of the document the element is from. Other metadata depends on file type or element type. For example, a `Table` element will contain table's representation as html in the metadata, and metadata for emails will contain information about senders and recipients.

Let's import element objects from these json files.


```python
from unstructured.staging.base import elements_from_json

elements = []

for filename in os.listdir(output_path):
    filepath = os.path.join(output_path, filename)
    elements.extend(elements_from_json(filepath))
```

Now that that you have extracted the elements from the documents, you can chunk them to fit the context window of the embeddings model.

## Chunking

If you are familiar with chunking methods that split long text documents into smaller chunks, you'll notice that Unstructured's chunking methods slightly differ, since the partitioning step already divides an entire document into its structural elements: titles, list items, tables, text, etc. By partitioning documents this way, you can avoid a situation where unrelated pieces of text end up in the same element, and then same chunk.  

Now, when you chunk the document elements with Unstructured, individual elements are already small so they will only be split if they exceed the desired maximum chunk size. Otherwise, they will remain as is. You can also optionally choose to combine consecutive text elements such as list items, for instance, that will together fit within chunk size limit.



```python
from unstructured.chunking.title import chunk_by_title

chunked_elements = chunk_by_title(elements,
                                  # maximum for chunk size
                                  max_characters=512,
                                  # You can choose to combine consecutive elements that are too small
                                  # e.g. individual list items
                                  combine_text_under_n_chars=200,
                                  )

```

The chunks are ready for RAG. To use them with LangChain, you can easily convert Unstructured elements to LangChain documents.


```python
from langchain_core.documents import Document

documents = []
for chunked_element in chunked_elements:
    metadata = chunked_element.metadata.to_dict()
    metadata["source"] = metadata["filename"]
    del metadata["languages"]
    documents.append(Document(page_content=chunked_element.text, metadata=metadata))
```

## Setting up the retriever

This example uses ChromaDB as a vector store and [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) embeddings model, feel free to use any other vector store.


```python
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import utils as chromautils

# ChromaDB doesn't support complex metadata, e.g. lists, so we drop it here.
# If you're using a different vector store, you may not need to do this
docs = chromautils.filter_complex_metadata(documents)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
```

If you plan to use a gated model from the Hugging Face Hub, be it an embeddings or text generation model, you'll need to authenticate yourself with your Hugging Face token, which you can get in your Hugging Face profile's settings.


```python
from huggingface_hub import notebook_login

notebook_login()
```

## RAG with LangChain

Let's bring everything together and build RAG with LangChain.
In this example we'll be using [`Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) from Meta. To make sure it can run smoothly in the free T4 runtime from Google Colab, you'll need to quantize it.


```python
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.chains import RetrievalQA
```


```python
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=200,
    eos_token_id=terminators,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt_template = """
<|start_header_id|>user<|end_header_id|>
You are an assistant for answering questions using provided context.
You are given the extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "I do not know." Don't make up an answer.
Question: {question}
Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)


qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)
```

## Results and next steps

Now that you have your RAG chain, let's ask it about aphids. Are they a pest in my garden?


```python
question = "Are aphids a pest?"

qa_chain.invoke(question)['result']
```

    [Setting `pad_token_id` to `eos_token_id`:128009 for open-end generation., ..., Yes, aphids are considered pests because they feed on the nutrient-rich liquids within plants, causing damage and potentially spreading disease. In fact, they're known to multiply quickly, which is why it's essential to control them promptly. As mentioned in the text, aphids can also attract ants, which are attracted to the sweet, sticky substance they produce called honeydew. So, yes, aphids are indeed a pest that requires attention to prevent further harm to your plants!]

This looks like a promising start! Now that you know the basics of preprocessing complex unstructured data for RAG, you can continue improving upon this example. Here are some ideas:

* You can connect to a different source to ingest the documents from, for example, an S3 bucket.
* You can add `return_source_documents=True` in the `qa_chain` arguments to make the chain return the documents that were passed to the prompt as context. This can be useful to understand what sources were used to generate the answer.
* If you want to leverage the elements metadata at the retrieval stage, consider using Hugging Face agents and creating a custom retriever tool as described in [this other notebook](https://huggingface.co/learn/cookbook/agents#2--rag-with-iterative-query-refinement--source-selection).
* There are many things you could do to improve search results. For instance, you could use Hybrid search instead of a single similarity-search retriever. Hybrid search combines multiple search algorithms to improve the accuracy and relevance of search results. Typically it's a combination of keyword-based search algorithms with vector search methods.

Have fun building RAG applications with Unstructured data!