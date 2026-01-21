# Building RAG with Custom Unstructured Data: A Step-by-Step Guide

_Authored by: [Maria Khalusova](https://github.com/MKhalusova)_

If you're new to Retrieval-Augmented Generation (RAG), it's recommended to first explore the basics in [this foundational guide](https://huggingface.co/learn/cookbook/rag_zephyr_langchain). Once you're comfortable, return here to learn how to build a RAG system that can handle diverse, real-world document formats.

In practice, valuable knowledge is often locked in various file types like PDFs, PowerPoint presentations, Word documents, emails, and HTML pages. This tutorial will show you how to preprocess this "unstructured" data and integrate it into a functional RAG pipeline.

You will use the following tools:
*   **[Unstructured](https://github.com/Unstructured-IO/unstructured)**: For ingesting and partitioning documents of various formats.
*   **Hugging Face Hub**: For open-source embedding and text generation models.
*   **ChromaDB**: As a local vector store.
*   **LangChain**: To orchestrate the components into a cohesive RAG chain.

## Prerequisites & Setup

Begin by installing the necessary Python libraries.

```bash
pip install -q torch transformers accelerate bitsandbytes sentence-transformers unstructured[all-docs] langchain chromadb langchain_community
```

## Step 1: Gather Your Documents

For this tutorial, we'll build a RAG system to answer questions about Integrated Pest Management (IPM). We'll download a mix of document types to simulate a real-world scenario.

Create a directory for your documents and download the example files.

```python
import os

# Create a directory for your documents
os.makedirs("./documents", exist_ok=True)

# Download example documents (you can replace these with your own)
!wget -q https://www.gov.nl.ca/ecc/files/env-protection-pesticides-business-manuals-applic-chapter7.pdf -O "./documents/env-protection-pesticides-business-manuals-applic-chapter7.pdf"
!wget -q https://ipm.ifas.ufl.edu/pdfs/Citrus_IPM_090913.pptx -O "./documents/Citrus_IPM_090913.pptx"
!wget -q https://www.gutenberg.org/ebooks/45957.epub3.images -O "./documents/45957.epub"
!wget -q https://blog.fifthroom.com/what-to-do-about-harmful-garden-and-plant-insects-and-pests.html -O "./documents/what-to-do-about-harmful-garden-and-plant-insects-and-pests.html"
```

**Note**: Unstructured supports many formats. Feel free to use your own documents from the supported list: `.pdf`, `.docx`, `.pptx`, `.html`, `.epub`, `.txt`, `.md`, `.eml`, `.jpg`, `.png`, `.csv`, `.xlsx`, and more.

## Step 2: Preprocess Documents with Unstructured

Instead of processing files individually, we'll use Unstructured's Local source connector to ingest all documents from our `./documents` directory. This approach handles multiple file types in one go.

First, let's reduce log verbosity for a cleaner output.

```python
import logging
logger = logging.getLogger("unstructured.ingest")
logger.root.removeHandler(logger.root.handlers[0])
```

Now, configure and run the ingestion pipeline.

```python
from unstructured.ingest.connector.local import SimpleLocalConfig
from unstructured.ingest.interfaces import PartitionConfig, ProcessorConfig, ReadConfig
from unstructured.ingest.runner import LocalRunner

output_path = "./local-ingest-output"

runner = LocalRunner(
    processor_config=ProcessorConfig(
        verbose=True,
        output_dir=output_path,  # Directory to store processed outputs
        num_processes=2,         # Use 2 processes for parallel ingestion
    ),
    read_config=ReadConfig(),    # Default settings for reading data
    partition_config=PartitionConfig(
        partition_by_api=True,   # Use the Unstructured API for better OCR/models
        api_key="YOUR_UNSTRUCTURED_API_KEY",  # Get a free key at https://unstructured.io/api-key-free
    ),
    connector_config=SimpleLocalConfig(
        input_path="./documents", # Path to your raw documents
        recursive=False,          # Set to True to search subdirectories
    ),
)
runner.run()
```

### Understanding the Configuration

*   **`ProcessorConfig`**: Controls the pipeline (output location, logging, parallel processes).
*   **`ReadConfig`**: Customizes data reading (e.g., re-downloading, limits). Defaults are usually fine.
*   **`PartitionConfig`**: Defines how documents are split into elements. Using `partition_by_api=True` leverages Unstructured's cloud API, which offers enhanced OCR and is free for up to 1000 pages. For local processing (requiring `poppler` and `tesseract`), omit the `api_key` parameter.
*   **`SimpleLocalConfig`**: Specifies the source directory for your documents.

After processing, you'll find JSON files in the `./local-ingest-output` directory—one per input document. Each file contains structured "elements" (like `Title`, `NarrativeText`, `Table`) extracted from the original document.

## Step 3: Load and Prepare Document Elements

Let's load the processed elements from the JSON files.

```python
from unstructured.staging.base import elements_from_json

elements = []
for filename in os.listdir(output_path):
    filepath = os.path.join(output_path, filename)
    elements.extend(elements_from_json(filepath))
```

## Step 4: Chunk the Elements for RAG

Unstructured's partitioning already breaks documents into logical elements. The `chunk_by_title` method respects these boundaries, only splitting an element if it exceeds a maximum size, and can combine small, consecutive elements (like list items) into coherent chunks.

```python
from unstructured.chunking.title import chunk_by_title

chunked_elements = chunk_by_title(
    elements,
    max_characters=512,        # Maximum characters per chunk
    combine_text_under_n_chars=200, # Combine small elements under this length
)
```

## Step 5: Convert to LangChain Documents

To use these chunks with LangChain, convert them into LangChain `Document` objects.

```python
from langchain_core.documents import Document

documents = []
for chunked_element in chunked_elements:
    metadata = chunked_element.metadata.to_dict()
    metadata["source"] = metadata["filename"]
    # Remove the 'languages' key to simplify metadata
    del metadata["languages"]
    documents.append(
        Document(page_content=chunked_element.text, metadata=metadata)
    )
```

## Step 6: Set Up the Vector Store and Retriever

We'll use ChromaDB as our vector store and the `BAAI/bge-base-en-v1.5` model for embeddings.

```python
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import utils as chromautils

# ChromaDB requires simple metadata (no lists/dicts). Filter if necessary.
filtered_docs = chromautils.filter_complex_metadata(documents)

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Create and persist the vector store from our documents
vectorstore = Chroma.from_documents(filtered_docs, embeddings)

# Create a retriever that fetches the top 3 most similar chunks
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
```

**Authentication Note**: If you plan to use a gated model from Hugging Face Hub, run `notebook_login()` from the `huggingface_hub` library and provide your token when prompted.

## Step 7: Build the RAG Chain with LangChain

We'll use the quantized `Llama-3-8B-Instruct` model to ensure it runs efficiently. The chain will retrieve relevant context and generate an answer.

First, set up the language model with quantization.

```python
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# Configure 4-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define stop tokens for the Llama 3 model
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Create a text generation pipeline
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

# Wrap the pipeline for LangChain
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
```

Next, define a prompt template that instructs the model to answer based solely on the provided context.

```python
from langchain.prompts import PromptTemplate

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
```

Finally, create the RetrievalQA chain that connects the retriever and the LLM.

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)
```

## Step 8: Query Your RAG System

Your RAG pipeline is now ready. Let's test it with a question about garden pests.

```python
question = "Are aphids a pest?"
result = qa_chain.invoke(question)
print(result['result'])
```

**Example Output**:
```
Yes, aphids are considered pests because they feed on the nutrient-rich liquids within plants, causing damage and potentially spreading disease. In fact, they're known to multiply quickly, which is why it's essential to control them promptly. As mentioned in the text, aphids can also attract ants, which are attracted to the sweet, sticky substance they produce called honeydew. So, yes, aphids are indeed a pest that requires attention to prevent further harm to your plants!
```

## Next Steps and Improvements

Congratulations! You've built a RAG system capable of handling multiple unstructured document formats. Here are some ideas to enhance it further:

1.  **Different Data Sources**: Modify the `connector_config` to ingest documents from cloud storage (S3, Blob Storage), databases, or SFTP servers.
2.  **Retrieve Source Documents**: Add `return_source_documents=True` to the `qa_chain` arguments to see which chunks were used to generate the answer, improving transparency.
3.  **Leverage Metadata**: Use element metadata (like document type or section titles) to create a more sophisticated retriever. Consider building a custom tool with Hugging Face agents, as shown in [this guide on iterative query refinement](https://huggingface.co/learn/cookbook/agents#2--rag-with-iterative-query-refinement--source-selection).
4.  **Improve Search**: Implement hybrid search, which combines vector similarity with keyword-based (BM25) search, to improve retrieval accuracy.

You now have a solid foundation for building powerful RAG applications with real-world, unstructured data. Happy building