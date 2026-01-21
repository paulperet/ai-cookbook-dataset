# Build a RAG Pipeline with Hugging Face and Milvus

In this tutorial, you will build a Retrieval-Augmented Generation (RAG) system. This system combines a retrieval component—powered by the Milvus vector database—with a generative component using a Large Language Model (LLM) from Hugging Face. You will use a PDF document as your private knowledge base, embed its contents, store them in Milvus, and query the system to generate informed answers.

## Prerequisites

Before you begin, ensure you have the following:

1.  **Python Environment:** A Python environment (3.8 or later) is required.
2.  **Hugging Face Token:** A Hugging Face User Access Token. You can generate one from your [Hugging Face account settings](https://huggingface.co/settings/tokens). This is necessary for accessing certain models on the Hub.

## Step 1: Install Dependencies

First, install the required Python libraries.

```bash
pip install --upgrade pymilvus sentence-transformers huggingface-hub langchain_community langchain_text_splitters pypdf tqdm
```

> **Note for Google Colab Users:** After running the installation command, you may need to restart your runtime. Go to **Runtime > Restart session** in the menu.

## Step 2: Configure Your Environment

Set your Hugging Face token as an environment variable. This allows the `huggingface-hub` library to authenticate your requests.

```python
import os

# Replace 'hf_...' with your actual Hugging Face token
os.environ["HF_TOKEN"] = "hf_..."
```

## Step 3: Prepare Your Data

You will use the "AI Act" PDF as the source of private knowledge for your RAG system.

### 3.1 Download the PDF

Download the PDF file to your working directory.

```bash
# This command downloads the file if it doesn't already exist.
if [ ! -f "The-AI-Act.pdf" ]; then
    wget -q https://artificialintelligenceact.eu/wp-content/uploads/2021/08/The-AI-Act.pdf
fi
```

### 3.2 Load and Split the Document

Use LangChain's `PyPDFLoader` to extract text from the PDF and a `RecursiveCharacterTextSplitter` to divide the text into manageable chunks. This is crucial for effective retrieval, as it breaks down large documents into semantically meaningful pieces.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the PDF
loader = PyPDFLoader("The-AI-Act.pdf")
docs = loader.load()
print(f"Loaded {len(docs)} pages from the PDF.")

# 2. Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# 3. Extract the plain text from each chunk
text_lines = [chunk.page_content for chunk in chunks]
print(f"Created {len(text_lines)} text chunks.")
```

## Step 4: Prepare the Embedding Model

To perform semantic search, you need to convert text into vector embeddings. You'll use the `BAAI/bge-small-en-v1.5` model, a popular and effective choice.

```python
from sentence_transformers import SentenceTransformer

# Load the embedding model
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Define a helper function to generate embeddings
def emb_text(text):
    """Converts a text string into a normalized embedding vector."""
    return embedding_model.encode([text], normalize_embeddings=True).tolist()[0]

# Test the embedding function
test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)
print(f"Embedding dimension: {embedding_dim}")
print(f"First 10 elements of test embedding: {test_embedding[:10]}")
```

## Step 5: Load Data into Milvus

Milvus will store your document embeddings for fast, scalable similarity search.

### 5.1 Initialize the Milvus Client

You'll use Milvus Lite, which stores data in a local file, making it perfect for tutorials and development.

```python
from pymilvus import MilvusClient

# Initialize the client. Data will be stored in './hf_milvus_demo.db'
milvus_client = MilvusClient(uri="./hf_milvus_demo.db")
collection_name = "rag_collection"
```

> **Note on Deployment:** For production with large datasets (>1M vectors), consider deploying a standalone Milvus server or using Zilliz Cloud. You would then provide a connection URI like `http://localhost:19530` instead of a file path.

### 5.2 Create the Collection

A collection in Milvus is like a table in a traditional database. You'll create one to hold your embeddings.

```python
# Clean up: Drop the collection if it already exists
if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

# Create a new collection
milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,  # Must match the dimension of your embeddings
    metric_type="IP",         # Use Inner Product for similarity (works with normalized vectors)
    consistency_level="Strong",
)
print(f"Collection '{collection_name}' created.")
```

### 5.3 Insert the Document Chunks

Now, iterate through your text chunks, generate an embedding for each, and insert them into the Milvus collection.

```python
from tqdm import tqdm

data = []
print("Generating embeddings and preparing data for insertion...")
for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    # Create a dictionary for each chunk containing its ID, vector, and original text.
    # The 'text' field is a dynamic field stored in JSON format.
    data.append({"id": i, "vector": emb_text(line), "text": line})

# Perform the bulk insert
insert_res = milvus_client.insert(collection_name=collection_name, data=data)
print(f"Successfully inserted {insert_res['insert_count']} vectors.")
```

## Step 6: Build the RAG Query Pipeline

With your knowledge base stored in Milvus, you can now build the retrieval and generation pipeline.

### 6.1 Define a Query and Retrieve Relevant Context

Start by formulating a question and using Milvus to find the most semantically similar document chunks.

```python
question = "What is the legal basis for the proposal?"

# 1. Convert the question into an embedding vector
question_embedding = emb_text(question)

# 2. Search Milvus for the top 3 most similar chunks
search_res = milvus_client.search(
    collection_name=collection_name,
    data=[question_embedding],
    limit=3,  # Retrieve the top 3 matches
    search_params={"metric_type": "IP", "params": {}},
    output_fields=["text"],  # We want the 'text' field returned
)

# 3. Inspect the results
import json
retrieved_lines_with_distances = [
    (res["entity"]["text"], res["distance"]) for res in search_res[0]
]
print("Top retrieved document chunks:")
print(json.dumps(retrieved_lines_with_distances, indent=4))
```

### 6.2 Prepare the Context for the LLM

Combine the retrieved text chunks into a single context string that will be provided to the language model.

```python
context = "\n".join([chunk_text for chunk_text, _ in retrieved_lines_with_distances])
```

### 6.3 Configure the Language Model

You will use the `Mixtral-8x7B-Instruct-v0.1` model via the Hugging Face Inference API. The `InferenceClient` handles the API call.

```python
from huggingface_hub import InferenceClient

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm_client = InferenceClient(model=repo_id, timeout=120)  # Set a generous timeout
```

### 6.4 Construct the Prompt and Generate the Answer

Craft a prompt that instructs the LLM to answer the question using *only* the provided context.

```python
# Define the prompt template
PROMPT = """
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
<context>
{context}
</context>
<question>
{question}
</question>
"""

# Format the prompt with the actual context and question
prompt = PROMPT.format(context=context, question=question)

# Generate the final answer
answer = llm_client.text_generation(
    prompt,
    max_new_tokens=1000,
).strip()

print("\n" + "="*50)
print("QUESTION:", question)
print("="*50)
print("GENERATED ANSWER:\n")
print(answer)
print("="*50)
```

## Conclusion

Congratulations! You have successfully built a functional RAG pipeline. You learned how to:

1.  Process and chunk a PDF document.
2.  Generate text embeddings using a Sentence Transformer model.
3.  Store and index those embeddings in the Milvus vector database.
4.  Retrieve semantically relevant context for a user's query.
5.  Synthesize a final answer using a powerful LLM from Hugging Face, grounded in the retrieved documents.

This pipeline forms the backbone of many modern AI applications, from intelligent chatbots to advanced research assistants. You can extend it by experimenting with different embedding models, chunking strategies, LLMs, or prompt templates.