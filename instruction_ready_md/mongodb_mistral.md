# Building a RAG System with Mistral AI and MongoDB

This guide walks you through building a Retrieval-Augmented Generation (RAG) application that combines Mistral AI's language models with MongoDB's vector search capabilities. You'll create a system that can ingest PDF documents, generate embeddings, store them in MongoDB, and answer questions using retrieved context.

## Prerequisites

Before starting, ensure you have:
- A Mistral AI API key
- A MongoDB Atlas cluster
- Python 3.8 or higher

## Setup

### 1. Configure Environment Variables

Set your API keys as environment variables:

```bash
export MONGO_URI="your_mongodb_connection_string"
export MISTRAL_API_KEY="your_mistral_api_key"
```

### 2. Install Required Libraries

```python
!pip install mistralai==0.0.8
!pip install pymongo==4.3.3
!pip install gradio==4.10.0
!pip install langchain==0.0.348
!pip install pandas==2.0.3
```

### 3. Import Dependencies

```python
import os
import pymongo
import pandas as pd
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
```

### 4. Verify API Keys

```python
# Load environment variables
mistral_api_key = os.environ["MISTRAL_API_KEY"]
mongo_url = os.environ["MONGO_URI"]
```

## Step 1: Connect to MongoDB

First, create a function to establish a connection to your MongoDB database:

```python
def connect_mongodb():
    """Establish connection to MongoDB and return the collection."""
    client = pymongo.MongoClient(mongo_url)
    db = client["mistralpdf"]
    collection = db["pdfRAG"]
    return collection
```

This function connects to a database named `mistralpdf` and a collection named `pdfRAG`. You can modify these names to match your specific use case.

## Step 2: Create Embedding Generation Function

Next, implement a function to generate embeddings using Mistral AI's embedding model:

```python
def get_embedding(text, client):
    """Generate embeddings for text using Mistral AI."""
    text = text.replace("\n", " ")
    embeddings_batch_response = client.embeddings(
        model="mistral-embed",
        input=text,
    )
    return embeddings_batch_response.data[0].embedding
```

This function cleans the text by removing newline characters and calls Mistral AI's embedding endpoint to generate a 1536-dimensional vector representation of the text.

## Step 3: Prepare MongoDB Vector Search Index

Before running vector searches, you need to create a vector search index in MongoDB Atlas. Create an index with the following configuration:

```json
{
  "type": "vectorSearch",
  "fields": [
    {
      "numDimensions": 1536,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

**Important:** Name this index `vector_index` as it will be referenced in the search function. You can create this index through the MongoDB Atlas UI or using the MongoDB shell.

## Step 4: Implement Document Processing Pipeline

Now, create the main function to process PDF documents, generate embeddings, and store them in MongoDB:

```python
def data_prep(file):
    """Process PDF file, generate embeddings, and store in MongoDB."""
    # Initialize Mistral client
    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)
    
    # Load and split PDF document
    loader = PyPDFLoader(file.name)
    pages = loader.load_and_split()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
    )
    docs = text_splitter.split_documents(pages)
    
    # Extract text chunks and generate embeddings
    text_chunks = [text.page_content for text in docs]
    df = pd.DataFrame({'text_chunks': text_chunks})
    df['embedding'] = df.text_chunks.apply(lambda x: get_embedding(x, client))
    
    # Store in MongoDB
    collection = connect_mongodb()
    df_dict = df.to_dict(orient='records')
    collection.insert_many(df_dict)
    
    return "PDF processed and data stored in MongoDB."
```

This function:
1. Loads a PDF file using LangChain's PyPDFLoader
2. Splits the document into manageable chunks (100 characters with 20-character overlap)
3. Generates embeddings for each chunk using Mistral AI
4. Stores both the text and embeddings in MongoDB

## Step 5: Create Document Retrieval Function

Implement a function to find documents similar to a query using vector search:

```python
def find_similar_documents(embedding):
    """Find documents similar to the query embedding using vector search."""
    collection = connect_mongodb()
    
    documents = list(
        collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 20,
                    "limit": 10
                }
            },
            {"$project": {"_id": 0, "text_chunks": 1}}
        ]))
    
    return documents
```

This function uses MongoDB's `$vectorSearch` aggregation pipeline to find the 10 most similar documents to the query embedding, considering 20 candidates during the search.

## Step 6: Build the Question-Answering Function

The core of the RAG system is the question-answering function that combines retrieval with generation:

```python
def qna(users_question):
    """Answer questions using retrieved context from MongoDB."""
    # Initialize Mistral client
    api_key = os.environ["MISTRAL_API_KEY"]
    client = MistralClient(api_key=api_key)
    
    # Generate embedding for the question
    question_embedding = get_embedding(users_question, client)
    
    # Retrieve similar documents
    documents = find_similar_documents(question_embedding)
    
    # Clean and prepare context
    for doc in documents:
        doc['text_chunks'] = doc['text_chunks'].replace('\n', ' ')
    
    # Combine retrieved documents into context
    context = " ".join([doc["text_chunks"] for doc in documents])
    
    # Create prompt with context and question
    template = f"""
    You are an expert who loves to help people! Given the following context sections, answer the
    question using only the given context. If you are unsure and the answer is not
    explicitly written in the documentation, say "Sorry, I don't know how to help with that."

    Context sections:
    {context}

    Question:
    {users_question}

    Answer:
    """
    
    # Generate response using Mistral AI
    messages = [ChatMessage(role="user", content=template)]
    chat_response = client.chat(
        model="mistral-large-latest",
        messages=messages,
    )
    
    # Format documents for display
    formatted_documents = '\n'.join([doc['text_chunks'] for doc in documents])
    
    return chat_response.choices[0].message.content, formatted_documents
```

This function:
1. Generates an embedding for the user's question
2. Retrieves similar documents from MongoDB using vector search
3. Combines the retrieved documents into a context string
4. Creates a prompt that instructs the model to answer based only on the provided context
5. Calls Mistral AI's chat completion endpoint to generate an answer
6. Returns both the answer and the retrieved documents for transparency

## Step 7: Create a Simple Interface (Optional)

To test your RAG system, you can create a simple command-line interface:

```python
def main():
    """Simple command-line interface for the RAG system."""
    print("Mistral AI + MongoDB RAG System")
    print("=" * 40)
    
    # Process a PDF file
    pdf_path = input("Enter path to PDF file to process: ")
    if os.path.exists(pdf_path):
        result = data_prep(open(pdf_path, 'rb'))
        print(f"\n{result}")
    
    # Ask questions
    print("\nYou can now ask questions about the document. Type 'quit' to exit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break
        
        answer, sources = qna(question)
        print(f"\nAnswer: {answer}")
        print(f"\nSources used:\n{sources}")

if __name__ == "__main__":
    main()
```

## Usage Example

Here's how to use the complete system:

```python
# 1. Process a PDF document
from io import BytesIO

# Create a file-like object from a PDF
pdf_content = b"..."  # Your PDF bytes
file_obj = BytesIO(pdf_content)
file_obj.name = "document.pdf"

# Process the document
result = data_prep(file_obj)
print(result)  # "PDF processed and data stored in MongoDB."

# 2. Ask a question
question = "What are the main topics covered in this document?"
answer, sources = qna(question)

print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"\nRetrieved context:\n{sources}")
```

## Key Considerations

1. **Chunk Size**: The current chunk size of 100 characters is quite small. You may want to experiment with larger chunks (500-1000 characters) for better context retention.

2. **Embedding Model**: The system uses `mistral-embed` which generates 1536-dimensional vectors. Ensure your MongoDB vector index matches this dimension.

3. **Error Handling**: Consider adding error handling for API calls, database connections, and file processing.

4. **Scalability**: For production use, consider implementing batch processing for embeddings and adding caching mechanisms.

5. **Prompt Engineering**: The current prompt instructs the model to only use provided context. You can modify this template to suit your specific use case.

## Troubleshooting

- **Vector Search Errors**: Ensure your MongoDB vector search index is properly configured and named `vector_index`.
- **API Key Issues**: Verify that your environment variables are set correctly in your current shell session.
- **Connection Problems**: Check that your MongoDB Atlas cluster allows connections from your IP address.

This RAG system provides a foundation for building intelligent document question-answering applications. You can extend it by adding support for multiple document types, implementing more sophisticated chunking strategies, or creating a web interface using Gradio or Streamlit.