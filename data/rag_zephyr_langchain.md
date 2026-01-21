# Building a Simple RAG for GitHub Issues with Hugging Face Zephyr and LangChain

_Authored by: [Maria Khalusova](https://github.com/MKhalusova)_

This guide demonstrates how to quickly build a Retrieval-Augmented Generation (RAG) system for a project's GitHub issues using the [`HuggingFaceH4/zephyr-7b-beta`](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) model and LangChain.

**What is RAG?**
RAG is a popular technique that enhances a Large Language Model's (LLM) responses by providing it with relevant, external context retrieved from a knowledge base. This is especially useful when the LLM lacks specific, proprietary, or frequently updated information in its training data. Unlike fine-tuning, RAG doesn't require retraining the model, making it faster, cheaper, and more flexible for swapping out models as needed.

## Prerequisites and Setup

Before you begin, install the required Python libraries.

```bash
pip install -q torch transformers accelerate bitsandbytes sentence-transformers faiss-gpu langchain langchain-community
```

If you encounter encoding issues during installation (common in Google Colab), run the following cell first:

```python
import locale
locale.getpreferredencoding = lambda: "UTF-8"
```

Now, let's start building the RAG pipeline.

## Step 1: Prepare the Data

We'll load GitHub issues from the [PEFT library's repository](https://github.com/huggingface/peft). First, you need a GitHub Personal Access Token.

### 1.1 Acquire GitHub Token

Securely input your token using `getpass`.

```python
from getpass import getpass
ACCESS_TOKEN = getpass("Enter your GitHub Personal Access Token: ")
```

### 1.2 Load GitHub Issues

Use LangChain's `GitHubIssuesLoader` to fetch all issues (open and closed) from the target repository, excluding pull requests.

```python
from langchain.document_loaders import GitHubIssuesLoader

loader = GitHubIssuesLoader(
    repo="huggingface/peft",
    access_token=ACCESS_TOKEN,
    include_prs=False,
    state="all"
)

docs = loader.load()
```

### 1.3 Chunk the Documents

GitHub issue content can be lengthy. We need to split it into smaller chunks suitable for our embedding model. We'll use the `RecursiveCharacterTextSplitter` with a chunk size of 512 characters and a 30-character overlap to preserve context.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
chunked_docs = splitter.split_documents(docs)
```

## Step 2: Create the Vector Database and Retriever

Now, we'll convert the document chunks into embeddings and store them in a vector database for efficient retrieval.

### 2.1 Generate Embeddings and Build the Vector Store

We'll use the `BAAI/bge-base-en-v1.5` embedding model via LangChain's `HuggingFaceEmbeddings` and store the vectors in a FAISS index.

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

db = FAISS.from_documents(
    chunked_docs,
    HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
)
```

### 2.2 Create the Retriever

Configure the vector store to act as a retriever, returning the top 4 most similar document chunks for any given query.

```python
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4}
)
```

## Step 3: Load the Quantized LLM

We'll use the `HuggingFaceH4/zephyr-7b-beta` model. To speed up inference and reduce memory usage, we load it in a quantized 4-bit format.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = 'HuggingFaceH4/zephyr-7b-beta'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

## Step 4: Assemble the LLM Chain

We need to create a pipeline for text generation and define a prompt template that instructs the model to use the provided context.

### 4.1 Create the Text Generation Pipeline

```python
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
```

### 4.2 Define the Prompt Template

The template must match the chat format expected by the Zephyr model.

```python
prompt_template = """
<|system|>
Answer the question based on your knowledge. Use the following context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>

"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = prompt | llm | StrOutputParser()
```

> **Note:** You can also use `tokenizer.apply_chat_template` to format a list of chat messages into the correct string format.

## Step 5: Create the RAG Chain

Finally, combine the retriever and the LLM chain. The RAG chain will first retrieve relevant context and then pass it, along with the original question, to the LLM for generation.

```python
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)
```

## Step 6: Evaluate the Results

Let's test the RAG system with a library-specific question and compare the answer to the base model's response without any retrieved context.

### 6.1 Define a Test Question

```python
question = "How do you combine multiple adapters?"
```

### 6.2 Query the Base Model (Without RAG)

First, let's see how the model answers without any external context.

```python
base_response = llm_chain.invoke({"context": "", "question": question})
print(base_response)
```

**Output:**
The model interprets "adapters" as physical computer hardware components, providing a generic answer about connecting USB and HDMI devices. This is incorrect in the context of the PEFT library, where "adapters" refer to parameter-efficient fine-tuning modules like LoRA.

### 6.3 Query the RAG System

Now, let's use our RAG chain, which will retrieve relevant context from GitHub issues before generating an answer.

```python
rag_response = rag_chain.invoke(question)
print(rag_response)
```

**Output:**
The model now provides a highly relevant answer, discussing techniques for combining multiple LoRA adapters, referencing community discussions, and noting it as an active area of research. The retrieved context from GitHub issues successfully guides the model to the correct domain.

## Conclusion

You have successfully built a RAG system for querying GitHub issues. The same model, when augmented with retrieved context, provides a dramatically more accurate and relevant answer compared to its standalone performance. This pipeline can be extended by adding more data sources (like documentation) or by experimenting with different embedding models and LLMs.