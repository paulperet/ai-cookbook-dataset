# AI-Powered Question Answering with Embeddings Search

## Introduction

Large Language Models (LLMs) like GPT excel at answering questions based on their training data. However, they cannot answer questions about:
*   Events after their knowledge cutoff date (e.g., October 2023 for `gpt-4o-mini`).
*   Your private or non-public documents.
*   Information from past conversations.

This guide demonstrates a robust **Search-Ask** method to overcome this limitation. The core idea is simple:
1.  **Search:** Query a library of reference text to find relevant sections.
2.  **Ask:** Insert the retrieved text into a prompt for the LLM and ask your question.

### Why Search Beats Fine-Tuning for Knowledge

Think of an LLM's knowledge in two ways:
*   **Model Weights (Long-term Memory):** Fine-tuning injects knowledge here. It's like studying for an exam days in advance—details can be forgotten or misremembered.
*   **Model Inputs (Short-term Memory):** Providing context in the prompt is like taking an exam with open notes. The model has direct access to the facts, leading to more accurate and reliable answers.

While fine-tuning is excellent for teaching new tasks or styles, using search to provide context is the recommended method for factual recall. The main constraint is the model's context window (e.g., 128,000 tokens for `gpt-4o`), which limits how much reference text you can provide at once.

## Prerequisites & Setup

Before you begin, ensure you have the necessary Python libraries installed and your OpenAI API key configured.

### 1. Install Required Libraries
Run the following command in your terminal:
```bash
pip install openai pandas tiktoken scipy
```

### 2. Import Libraries and Configure Client
Create a new Python script or notebook and start with the following imports and configuration.

```python
import ast  # For converting saved embeddings strings back to arrays
import os  # For reading environment variables
from openai import OpenAI  # For calling the OpenAI API
import pandas as pd  # For storing text and embeddings
import tiktoken  # For counting tokens
from scipy import spatial  # For calculating vector similarities

# Define models
GPT_MODEL = "gpt-4o"  # Choose "gpt-4o-mini" for lower cost
EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize the OpenAI client
# The client will automatically look for your API key in the `OPENAI_API_KEY` environment variable.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

**Important:** Set your `OPENAI_API_KEY` as an environment variable. If you haven't done this, the code will fail. You can set it temporarily in your terminal:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Part 1: Demonstrating the Knowledge Gap

Let's first confirm that the base LLM lacks knowledge about recent events, such as the 2024 Summer Olympics.

```python
query = 'Which athletes won the most number of gold medals in 2024 Summer Olympics?'

response = client.chat.completions.create(
    model=GPT_MODEL,
    messages=[
        {'role': 'system', 'content': 'You answer questions about the 2024 Games or latest events.'},
        {'role': 'user', 'content': query},
    ],
    temperature=0,
)

print(response.choices[0].message.content)
```

The model will respond that it cannot answer because its knowledge is limited to events before October 2023.

## Part 2: Providing Knowledge via Context

We can solve this by giving the model the necessary information directly in the prompt. Let's use a snippet from the Wikipedia article on the 2024 Summer Olympics.

```python
# Text excerpt from the 2024 Summer Olympics Wikipedia page
wikipedia_article = """2024 Summer Olympics
The United States topped the medal table for the fourth consecutive Summer Games and 19th time overall, with 40 gold and 126 total medals.
China tied with the United States on gold (40), but finished second due to having fewer silvers; the nation won 91 medals overall.
... [Additional context from the article] ...
"""

# Now ask the same question, but provide the context
query = 'Which athletes won the most number of gold medals in 2024 Summer Olympics?'

response = client.chat.completions.create(
    model=GPT_MODEL,
    messages=[
        {'role': 'system', 'content': 'You answer questions about the 2024 Games. Use the provided context.'},
        {'role': 'user', 'content': f"Context: {wikipedia_article}\n\nQuestion: {query}"},
    ],
    temperature=0,
)

print(response.choices[0].message.content)
```

This time, the model will successfully answer based on the provided text, noting that the United States and China each won 40 gold medals.

**This manual approach works but doesn't scale.** You can't paste entire document libraries into every prompt. The solution is to automate the search for relevant context.

## Part 3: Building a Search-Ask System

We will build a system that:
1.  Prepares a searchable database from documents.
2.  Finds the most relevant text for a user's question.
3.  Asks the LLM the question using only that relevant text.

### Step 1: Prepare the Search Data

This is a one-time setup process for your document library.

#### 1.1 Collect and Chunk Documents
Documents must be split into meaningful chunks. Here, we'll simulate this with a simple list of text sections.

```python
# In a real scenario, you would load and split documents from files (PDFs, docs, etc.).
# For this example, we'll use a list of pre-defined text chunks about the Olympics.
documents = [
    "The 2024 Summer Olympics were held in Paris, France.",
    "The United States won the most gold medals (40) at the 2024 Games.",
    "China also won 40 gold medals but had fewer silver medals than the US.",
    "The opening ceremony was held along the Seine river.",
    "Surfing events took place in Tahiti, French Polynesia."
]
```

#### 1.2 Generate and Store Embeddings
Embeddings are numerical representations of text that capture semantic meaning. Similar texts have similar embeddings.

```python
def get_embedding(text, model=EMBEDDING_MODEL):
    """Helper function to get an embedding for a given text string."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# Create a DataFrame to store our documents and their embeddings
df = pd.DataFrame({'text': documents})
df['embedding'] = df['text'].apply(lambda x: get_embedding(x))

# Inspect the DataFrame
print(df.head())
```

Your `df` DataFrame now contains the original text and its corresponding vector embedding.

**For production:** With large datasets, use a dedicated vector database (e.g., Pinecone, Weaviate, pgvector) for efficient storage and similarity search.

### Step 2: Search for Relevant Context

When a user asks a question, we need to find the most relevant document chunks.

#### 2.1 Define the Search Function
This function calculates the cosine similarity between the question's embedding and all document embeddings to find the best matches.

```python
def strings_ranked_by_relatedness(query, df, relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y), top_n=3):
    """
    Returns the top_n most related text chunks from the dataframe to the query.
    """
    # Get the embedding for the query
    query_embedding = get_embedding(query)
    
    # Calculate relatedness scores
    df['relatedness'] = df['embedding'].apply(lambda emb: relatedness_fn(query_embedding, emb))
    
    # Sort by relatedness and return the top N text chunks
    results = df.sort_values('relatedness', ascending=False).head(top_n)
    return results[['text', 'relatedness']]
```

#### 2.2 Test the Search
Let's search our small document library with a sample question.

```python
query = "Who won the most gold medals in 2024?"
results = strings_ranked_by_relatedness(query, df, top_n=2)
print("Top search results:")
print(results)
```

The function should return the chunks about the United States and China winning gold medals, as they are most semantically related to the question.

### Step 3: Ask the Question with Context

Finally, we construct a prompt that includes the retrieved context and the user's question for the LLM.

#### 3.1 Construct the Prompt
We need to count tokens to ensure we don't exceed the model's context limit. We'll use a helper function from `tiktoken`.

```python
def num_tokens(text, model=GPT_MODEL):
    """Return the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(query, df, model=GPT_MODEL, token_budget=4096):
    """
    Builds a prompt message for the LLM with relevant context and the user's query.
    """
    # Get the most related text chunks
    results = strings_ranked_by_relatedness(query, df)
    
    # Build the context string
    context = "\n\n".join(results['text'].tolist())
    
    # Construct the system message with context and instructions
    system_message = {
        'role': 'system',
        'content': f"""You answer questions based on the provided context. The context is a set of relevant facts.
        
        Context:
        {context}
        
        If the answer cannot be found in the context, say "I cannot answer based on the provided information."
        """
    }
    
    user_message = {'role': 'user', 'content': query}
    
    # Simple token check (for a robust system, implement iterative trimming)
    message_tokens = num_tokens(system_message['content']) + num_tokens(user_message['content'])
    if message_tokens > token_budget:
        print(f"Warning: Message length ({message_tokens} tokens) may exceed budget.")
    
    return [system_message, user_message]
```

#### 3.2 Ask the LLM
Now, we can use the constructed messages to get a final answer.

```python
def ask(query, df, model=GPT_MODEL):
    """Searches for relevant context and asks the LLM the question."""
    messages = query_message(query, df, model=model)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content

# Ask our question
final_answer = ask("Who won the most gold medals in 2024?", df)
print("Answer:", final_answer)
```

The model will now correctly answer the question using only the retrieved context from your document library, effectively bypassing its original knowledge cutoff.

## Summary & Next Steps

You have successfully built a basic **Search-Ask** pipeline:
1.  **Data Preparation:** Chunk documents and generate embeddings.
2.  **Semantic Search:** Find text chunks most related to a user's question using cosine similarity on embeddings.
3.  **Contextual Question Answering:** Provide the LLM with the retrieved context to generate accurate answers.

**To productionize this system:**

*   **Scale Storage:** Replace the Pandas DataFrame with a proper vector database for large document sets.
*   **Improve Search:** Combine embedding search with keyword (lexical) search for better recall (Hybrid Search).
*   **Optimize Chunking:** Implement more sophisticated text splitting to preserve context (e.g., using LangChain's text splitters).
*   **Manage Context Windows:** Implement logic to dynamically select the number of chunks (`top_n`) based on token counts to fully utilize the model's context window.

This pattern is the foundation for advanced applications like Retrieval-Augmented Generation (RAG) systems and AI agents that can interact with your private data.