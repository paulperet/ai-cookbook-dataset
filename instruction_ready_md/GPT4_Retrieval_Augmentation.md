# Building a Retrieval-Augmented Generation (RAG) System with GPT-4 and Pinecone

## Introduction

Large Language Models (LLMs) like GPT-4 are powerful but can sometimes generate incorrect or "hallucinated" information. Retrieval-Augmented Generation (RAG) solves this by grounding the model's responses in factual data retrieved from an external knowledge base.

In this guide, you will build a RAG system that:
1.  Scrapes and processes documentation from the LangChain website.
2.  Creates vector embeddings and stores them in Pinecone.
3.  Retrieves relevant context for a user query.
4.  Uses GPT-4 to generate accurate, source-backed answers.

## Prerequisites & Setup

Ensure you have the necessary libraries installed.

```bash
pip install -qU bs4 tiktoken openai langchain pinecone-client[grpc]
```

You will also need API keys for:
*   **OpenAI:** To generate embeddings and use GPT-4.
*   **Pinecone:** To create and query the vector index.

## Step 1: Prepare the Data

First, you need a dataset. We'll use the LangChain documentation as our knowledge source.

### 1.1 Download the Documentation

Use `wget` to download all HTML pages from the LangChain documentation site.

```bash
wget -r -A.html -P rtdocs https://python.langchain.com/en/latest/
```
This command saves all `.html` files into a local directory named `rtdocs`.

### 1.2 Load and Inspect the Documents

Use LangChain's `ReadTheDocsLoader` to parse the downloaded HTML files into a structured format.

```python
from langchain.document_loaders import ReadTheDocsLoader

loader = ReadTheDocsLoader('rtdocs')
docs = loader.load()
print(f"Loaded {len(docs)} documents.")
```

Each document object contains `page_content` (the text) and `metadata` (including the source URL).

```python
# Inspect the first document's content
print(docs[0].page_content[:500]) # First 500 characters

# Inspect the source URL of the sixth document
print(docs[5].metadata['source'].replace('rtdocs/', 'https://'))
```

### 1.3 Structure the Raw Data

Create a clean list of dictionaries containing the URL and text for each document.

```python
data = []
for doc in docs:
    data.append({
        'url': doc.metadata['source'].replace('rtdocs/', 'https://'),
        'text': doc.page_content
    })
```

## Step 2: Chunk the Documents

LLMs and embedding models have context windows. To manage this, you must split long documents into smaller, overlapping chunks.

### 2.1 Initialize the Text Splitter

You'll use a tokenizer to measure text length and a recursive splitter to create chunks.

```python
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding('p50k_base')

def tiktoken_len(text):
    """Calculate the number of tokens for a given text."""
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,          # Target chunk size in tokens
    chunk_overlap=20,        # Overlap between chunks for context preservation
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""] # Split in this order
)
```

### 2.2 Process Documents into Chunks

Apply the splitter to each document's text and create a final list of chunks with unique IDs.

```python
from uuid import uuid4
from tqdm.auto import tqdm

chunks = []
for idx, record in enumerate(tqdm(data)):
    # Split the text
    texts = text_splitter.split_text(record['text'])
    # Create chunk records
    chunks.extend([{
        'id': str(uuid4()),
        'text': texts[i],
        'chunk': i,
        'url': record['url']
    } for i in range(len(texts))])

print(f"Created {len(chunks)} chunks from {len(data)} documents.")
```

## Step 3: Create Vector Embeddings

To enable semantic search, you must convert text chunks into numerical vector embeddings.

### 3.1 Initialize the Embedding Model

Use OpenAI's `text-embedding-3-small` model. First, set your API key.

```python
import openai

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"
embed_model = "text-embedding-3-small"

# Test the embedding model
res = openai.Embedding.create(
    input=["Sample document text goes here", "there will be several phrases in each batch"],
    engine=embed_model
)
# Check the dimensionality of the embeddings (should be 1536 for text-embedding-3-small)
embedding_dim = len(res['data'][0]['embedding'])
print(f"Embedding dimension: {embedding_dim}")
```

## Step 4: Initialize the Pinecone Vector Index

Pinecone will store your vectors and perform fast similarity searches.

### 4.1 Create and Connect to the Index

```python
import pinecone

index_name = 'gpt-4-langchain-docs'

# Initialize Pinecone
pinecone.init(
    api_key="YOUR_PINECONE_API_KEY",      # Find at app.pinecone.io
    environment="YOUR_PINECONE_ENVIRONMENT" # Found next to API key in console
)

# Create the index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=embedding_dim, # Must match the embedding model's output
        metric='dotproduct'
    )

# Connect to the index
index = pinecone.GRPCIndex(index_name)
print(index.describe_index_stats())
```

## Step 5: Populate the Index with Vectors

Now, embed your text chunks in batches and upload them to Pinecone.

```python
from tqdm.auto import tqdm
from time import sleep

batch_size = 100  # Process 100 chunks at a time

for i in tqdm(range(0, len(chunks), batch_size)):
    # Find the end of the current batch
    i_end = min(len(chunks), i + batch_size)
    meta_batch = chunks[i:i_end]

    # Prepare batch data
    ids_batch = [x['id'] for x in meta_batch]
    texts_batch = [x['text'] for x in meta_batch]

    # Create embeddings with retry logic for rate limits
    try:
        res = openai.Embedding.create(input=texts_batch, engine=embed_model)
    except Exception as e:
        # If we hit a rate limit, wait and retry
        sleep(5)
        res = openai.Embedding.create(input=texts_batch, engine=embed_model)

    embeds_batch = [record['embedding'] for record in res['data']]

    # Prepare metadata for storage
    meta_batch = [{
        'text': x['text'],
        'chunk': x['chunk'],
        'url': x['url']
    } for x in meta_batch]

    # Create (id, vector, metadata) tuples
    vectors_to_upsert = list(zip(ids_batch, embeds_batch, meta_batch))

    # Upload to Pinecone
    index.upsert(vectors=vectors_to_upsert)

print("Indexing complete.")
```

## Step 6: Retrieve Relevant Context

With your knowledge base indexed, you can now query it. The process involves embedding the user's query and finding the most similar text chunks in Pinecone.

```python
query = "how do I use the LLMChain in LangChain?"

# Embed the query
query_embed_res = openai.Embedding.create(input=[query], engine=embed_model)
query_vector = query_embed_res['data'][0]['embedding']

# Query Pinecone for the top 5 most similar chunks
retrieval_response = index.query(
    vector=query_vector,
    top_k=5,
    include_metadata=True  # Return the stored text and metadata
)

# Inspect the results
for match in retrieval_response['matches']:
    print(f"Score: {match['score']:.2f}")
    print(f"Text: {match['metadata']['text'][:200]}...")  # Preview
    print("-" * 50)
```

## Step 7: Generate Augmented Answers with GPT-4

This is the core RAG step. You will combine the retrieved context with the original query to give GPT-4 the information it needs to generate a factual answer.

### 7.1 Construct the Augmented Query

First, merge the retrieved context and the user query into a single prompt.

```python
# Extract the text from the retrieved chunks
contexts = [item['metadata']['text'] for item in retrieval_response['matches']]

# Combine contexts and query with clear separators
augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query

# Preview the augmented query
print(augmented_query[:1000])
```

### 7.2 Query GPT-4 with the Augmented Prompt

Define a system message to guide the model's behavior and make the API call.

```python
# System message primes the model to answer based only on the provided context
primer = """You are Q&A bot. A highly intelligent system that answers
user questions based on the information provided by the user above
each question. If the information can not be found in the information
provided by the user you truthfully say "I don't know".
"""

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": augmented_query}
    ],
    temperature=0  # For deterministic, factual answers
)

answer = response['choices'][0]['message']['content']
print("GPT-4 Answer (with RAG):")
print(answer)
```

### 7.3 Compare with a Non-Augmented Query

To appreciate the value of RAG, see how GPT-4 answers the same question *without* the retrieved context.

```python
# Query GPT-4 with the original question only
vanilla_response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": primer},
        {"role": "user", "content": query}  # No context provided
    ]
)
vanilla_answer = vanilla_response['choices'][0]['message']['content']

print("GPT-4 Answer (without RAG - may hallucinate):")
print(vanilla_answer)
```

## Conclusion

You have successfully built a RAG pipeline. By retrieving relevant information from a trusted source (the LangChain docs) and providing it to GPT-4, you significantly increase the likelihood of receiving accurate, verifiable answers.

**Key Takeaways:**
1.  **Data Preparation is Crucial:** Cleaning, chunking, and embedding your source material correctly forms the foundation of a good retrieval system.
2.  **Pinecone Enables Fast Search:** It allows you to quickly find the most relevant context from millions of vectors.
3.  **Prompt Engineering Guides the LLM:** The system message and the structure of the augmented query are essential for getting the desired "grounded" behavior from GPT-4.

**Next Steps:**
*   Experiment with different chunk sizes and overlap.
*   Try different embedding models.
*   Add a re-ranking step for even more precise retrieval.
*   Build a simple web interface to interact with your RAG system.