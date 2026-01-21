# Building a Question Answering System with LangChain, Qdrant, and OpenAI

This guide walks you through creating a fully functional Question Answering (QA) system. You'll use **LangChain** as the orchestration framework, **Qdrant** as a vector database to store your knowledge base, and **OpenAI** for generating text embeddings and powering the Large Language Model (LLM). By the end, you'll have a system that can retrieve relevant information from stored documents and generate precise answers.

## Prerequisites

Before you begin, ensure you have the following:

1.  **Docker & Docker Compose:** Required to run the local Qdrant server.
2.  **An OpenAI API Key:** Sign up and get your key from the [OpenAI platform](https://platform.openai.com/api-keys).

## Step 1: Setup Your Environment

### 1.1 Start the Qdrant Server

We'll run Qdrant locally using Docker. Create a `docker-compose.yaml` file with the following content:

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

Then, start the server in detached mode.

```bash
docker-compose up -d
```

Verify the server is running by checking its health endpoint.

```bash
curl http://localhost:6333
```

You should receive a JSON response confirming the service is operational.

### 1.2 Install Required Python Packages

Install the necessary libraries using pip.

```bash
pip install openai qdrant-client "langchain==0.0.100" wget
```

### 1.3 Configure Your OpenAI API Key

Set your API key as an environment variable. This is the most secure and recommended method.

```bash
export OPENAI_API_KEY="your-api-key-here"
```

To verify the key is accessible within your Python environment, run the following script.

```python
import os

if os.getenv("OPENAI_API_KEY") is not None:
    print("✅ OPENAI_API_KEY is ready")
else:
    print("❌ OPENAI_API_KEY environment variable not found")
```

**Note:** If you are running this in a notebook, you may need to restart your kernel after setting the environment variable for it to take effect.

## Step 2: Load and Prepare Your Data

We'll use a sample dataset from Google's Natural Questions research. This dataset contains question-answer pairs which will form our knowledge base.

First, download the sample data files.

```python
import wget
import json

# Download question and answer files
wget.download("https://storage.googleapis.com/dataset-natural-questions/questions.json")
wget.download("https://storage.googleapis.com/dataset-natural-questions/answers.json")

# Load the data into Python objects
with open("questions.json", "r") as fp:
    questions = json.load(fp)

with open("answers.json", "r") as fp:
    answers = json.load(fp)

# Let's inspect the first entry to understand the data structure
print("Sample Question:", questions[0])
print("\nSample Answer:", answers[0])
```

The `questions` list contains natural language queries, and the `answers` list contains the corresponding text passages that answer them.

## Step 3: Build the Knowledge Base with Qdrant

Now, we'll create embeddings for our answers and store them in Qdrant. LangChain provides a convenient abstraction for this.

```python
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings

# Initialize the embedding model (this uses your OpenAI API key)
embeddings = OpenAIEmbeddings()

# Create and populate the Qdrant collection in one step.
# This method chunks the texts, generates embeddings, and uploads them.
doc_store = Qdrant.from_texts(
    texts=answers,          # Our list of answer documents
    embedding=embeddings,   # The embedding model to use
    host="localhost",       # Location of our Qdrant server
    prefer_grpc=False       # Use HTTP for communication
)
```

At this point, your Qdrant instance contains a vector collection where each vector represents an answer from your dataset. The system can now perform semantic search over this knowledge base.

## Step 4: Create the QA Chain

A QA chain in LangChain links the retrieval step (finding relevant documents in Qdrant) with the generation step (asking the LLM to synthesize an answer). We'll use the `VectorDBQA` chain with the "stuff" method, which simply concatenates all retrieved documents into the LLM's context window.

```python
from langchain import VectorDBQA, OpenAI

# Initialize the LLM (OpenAI's GPT-3 by default)
llm = OpenAI()

# Create the QA chain
qa = VectorDBQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    vectorstore=doc_store,
    return_source_documents=False, # Set to True if you want to see which docs were retrieved
)
```

The chain is now ready. When you ask a question, it will:
1.  Convert the question into an embedding.
2.  Query Qdrant for the most semantically similar answer documents.
3.  Pass those documents as context, along with the original question, to the LLM.
4.  Return the LLM's final answer.

## Step 5: Query Your Knowledge Base

Let's test the system with a few randomly selected questions from our dataset.

```python
import random

random.seed(52)  # Set seed for reproducibility
selected_questions = random.choices(questions, k=5)

for question in selected_questions:
    print(f"❓ Question: {question}")
    answer = qa.run(question)
    print(f"🤖 Answer: {answer}\n")
```

You should see the LLM providing answers based on the context it retrieved from your Qdrant knowledge base.

## Step 6: Customize the Prompt (Advanced)

The default prompt used by the "stuff" chain is effective but generic. You can customize it to change the LLM's behavior, such as its response format or fallback actions.

### 6.1 Define a Custom Prompt Template

Let's create a prompt that instructs the model to give concise, single-sentence answers. If the answer isn't in the provided context, instead of admitting ignorance, it should suggest a random song title.

```python
from langchain.prompts import PromptTemplate

custom_prompt_text = """
Use the following pieces of context to answer the question at the end. Please provide
a short single-sentence summary answer only. If you don't know the answer or if it's
not present in given context, don't try to make up an answer, but suggest me a random
unrelated song title I could listen to.
Context: {context}
Question: {question}
Helpful Answer:
"""

custom_prompt_template = PromptTemplate(
    template=custom_prompt_text,
    input_variables=["context", "question"]
)
```

### 6.2 Create a New QA Chain with the Custom Prompt

Pass your custom template to the chain via the `chain_type_kwargs` parameter.

```python
custom_qa = VectorDBQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    vectorstore=doc_store,
    return_source_documents=False,
    chain_type_kwargs={"prompt": custom_prompt_template},
)
```

### 6.3 Test the Customized Chain

Now, test the new chain. For questions outside the knowledge base, expect a creative musical suggestion.

```python
random.seed(41)
for question in random.choices(questions, k=5):
    print(f"❓ Question: {question}")
    answer = custom_qa.run(question)
    print(f"🎵 Custom Answer: {answer}\n")
```

## Conclusion

You have successfully built a Retrieval-Augmented Generation (RAG) system. You've learned how to:
1.  Set up a local Qdrant vector database.
2.  Use LangChain to ingest documents and create a searchable knowledge base.
3.  Construct a QA chain that retrieves relevant context and uses an LLM to generate answers.
4.  Customize the LLM's instructions by modifying the prompt template.

This architecture forms the foundation for many advanced applications like chatbots, research assistants, and enterprise search tools. You can extend it by adding more sophisticated retrieval strategies, using different embedding models, or integrating other LLMs.