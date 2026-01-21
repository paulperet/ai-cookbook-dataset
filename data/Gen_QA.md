# Retrieval-Augmented Generation (RAG) for Factual Question Answering

## Introduction
Large Language Models (LLMs) like GPT-3.5 are powerful, but they can sometimes generate incorrect or "hallucinated" information, especially when asked about specific or niche topics. In this guide, you'll learn how to build a Retrieval-Augmented Generation (RAG) system that combines the generative capabilities of OpenAI's models with a vector database (Pinecone) acting as an external knowledge base. This approach ensures answers are grounded in factual data.

## Prerequisites

Before you begin, ensure you have the following:

- An OpenAI API key
- A Pinecone API key (get one for free at [app.pinecone.io](https://app.pinecone.io))

Install the required Python libraries:

```bash
pip install -qU openai pinecone-client datasets
```

## Step 1: Set Up Your Environment

Import the necessary libraries and configure your API keys.

```python
import openai
import pinecone
from datasets import load_dataset
from tqdm.auto import tqdm
from time import sleep

# Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Initialize Pinecone
pinecone.init(
    api_key="YOUR_PINECONE_API_KEY",
    environment="us-east1-gcp"  # Check your Pinecone dashboard for the correct environment
)
```

## Step 2: Understand the Problem with Unassisted LLMs

First, let's see how a standard LLM handles a factual question without any external context. We'll create a helper function to query the model.

```python
def complete(prompt):
    """Query GPT-3.5-turbo-instruct with a given prompt."""
    res = openai.Completion.create(
        engine='gpt-3.5-turbo-instruct',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()
```

Now, ask a specific technical question about training sentence transformers.

```python
query = (
    "Which training method should I use for sentence transformers when "
    "I only have pairs of related sentences?"
)

answer = complete(query)
print(answer)
```

The model might return a plausible-sounding but incorrect answer, such as recommending "Masked Language Model (MLM) training," which is not suitable for this scenario. This demonstrates the need for augmenting the LLM with external knowledge.

## Step 3: Prepare the Knowledge Base Data

We'll use a dataset of YouTube video transcriptions related to machine learning and AI as our knowledge source.

```python
# Load the dataset from Hugging Face
data = load_dataset('jamescalam/youtube-transcriptions', split='train')
print(f"Dataset loaded: {len(data)} entries")
```

The dataset contains many short text snippets. To create more meaningful chunks, we'll merge consecutive snippets with a sliding window.

```python
new_data = []
window = 20  # Number of sentences to combine
stride = 4   # Stride to create overlapping chunks

for i in tqdm(range(0, len(data), stride)):
    i_end = min(len(data)-1, i+window)
    # Skip if the chunk spans multiple videos
    if data[i]['title'] != data[i_end]['title']:
        continue
    # Merge text from the window
    text = ' '.join(data[i:i_end]['text'])
    new_data.append({
        'start': data[i]['start'],
        'end': data[i_end]['end'],
        'title': data[i]['title'],
        'text': text,
        'id': data[i]['id'],
        'url': data[i]['url'],
        'published': data[i]['published'],
        'channel_id': data[i]['channel_id']
    })

print(f"Created {len(new_data)} merged text chunks.")
```

## Step 4: Create and Store Embeddings with Pinecone

We'll use OpenAI's `text-embedding-ada-002` model to generate vector embeddings for each text chunk, then store them in Pinecone for efficient similarity search.

First, create a Pinecone index if it doesn't exist.

```python
index_name = 'openai-youtube-transcriptions'
embed_model = "text-embedding-ada-002"

# Check if index exists, create if not
if index_name not in pinecone.list_indexes():
    # Create a sample embedding to get the dimension size
    sample_res = openai.Embedding.create(
        input=["sample text"],
        engine=embed_model
    )
    dimension = len(sample_res['data'][0]['embedding'])
    
    pinecone.create_index(
        index_name,
        dimension=dimension,
        metric='cosine',
        metadata_config={'indexed': ['channel_id', 'published']}
    )

# Connect to the index
index = pinecone.Index(index_name)
print(index.describe_index_stats())
```

Now, embed the text chunks in batches and upsert them to Pinecone.

```python
batch_size = 100

for i in tqdm(range(0, len(new_data), batch_size)):
    i_end = min(len(new_data), i+batch_size)
    meta_batch = new_data[i:i_end]
    
    # Extract IDs and texts
    ids_batch = [x['id'] for x in meta_batch]
    texts = [x['text'] for x in meta_batch]
    
    # Create embeddings with retry logic for rate limits
    done = False
    while not done:
        try:
            res = openai.Embedding.create(input=texts, engine=embed_model)
            done = True
        except Exception as e:
            sleep(5)
    
    embeds = [record['embedding'] for record in res['data']]
    
    # Prepare metadata
    meta_batch = [{
        'start': x['start'],
        'end': x['end'],
        'title': x['title'],
        'text': x['text'],
        'url': x['url'],
        'published': x['published'],
        'channel_id': x['channel_id']
    } for x in meta_batch]
    
    # Create vectors for upsert
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    
    # Upsert to Pinecone
    index.upsert(vectors=to_upsert)

print("Data ingestion complete.")
```

## Step 5: Retrieve Relevant Context

With our knowledge base populated, we can now retrieve relevant context for a given query.

```python
# Embed the query
query = (
    "Which training method should I use for sentence transformers when "
    "I only have pairs of related sentences?"
)

res = openai.Embedding.create(input=[query], engine=embed_model)
xq = res['data'][0]['embedding']

# Query Pinecone for the top 2 most relevant chunks
search_res = index.query(xq, top_k=2, include_metadata=True)

# Display the retrieved contexts
for match in search_res['matches']:
    print(f"Title: {match['metadata']['title']}")
    print(f"Text snippet: {match['metadata']['text'][:200]}...")
    print(f"Relevance score: {match['score']:.2f}")
    print("-" * 80)
```

## Step 6: Generate an Augmented Answer

Finally, we'll construct a prompt that includes the retrieved context and ask the LLM to generate an answer based on it.

```python
# Build a context string from the retrieved results
context = ""
for match in search_res['matches']:
    context += match['metadata']['text'] + "\n\n"

# Create the augmented prompt
prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer: """

# Generate the answer
augmented_answer = complete(prompt)
print("Augmented Answer:")
print(augmented_answer)
```

The model should now provide a correct and specific answer, such as "Multiple Negatives Ranking (MNR) loss," which is the appropriate training method for sentence transformers when you only have pairs of related sentences.

## Conclusion

You've successfully built a RAG system that:

1. **Identifies the limitation** of standalone LLMs in providing factual answers.
2. **Prepares a knowledge base** by chunking and embedding relevant textual data.
3. **Stores and retrieves** information efficiently using Pinecone's vector database.
4. **Augments the LLM's generation** with retrieved context to produce accurate, grounded answers.

This approach can be extended to any domain where you have a corpus of reference material, enabling you to build LLM applications that are both knowledgeable and reliable.