# Guide: Implementing Retrieval-Augmented Generation (RAG) with Pinecone

This guide demonstrates how to build a Retrieval-Augmented Generation (RAG) pipeline by connecting Claude with a Pinecone vector database. You will learn to embed a dataset, store it for semantic search, and use Claude to generate answers grounded in retrieved information.

## Prerequisites

You will need API keys for the following services:
*   **Claude API Key:** Get started at [docs.claude.com](https://docs.claude.com/claude/reference/getting-started-with-the-api).
*   **Pinecone API Key:** Get a free key from [Pinecone](https://docs.pinecone.io/docs/quickstart).
*   **Voyage AI API Key:** Get a free key from [Voyage AI](https://docs.voyageai.com/install/).

## Step 1: Environment Setup

Begin by installing the required Python libraries and configuring your API keys.

```bash
pip install anthropic datasets pinecone-client voyageai
```

```python
# Insert your API keys here
ANTHROPIC_API_KEY = "<YOUR_ANTHROPIC_API_KEY>"
PINECONE_API_KEY = "<YOUR_PINECONE_API_KEY>"
VOYAGE_API_KEY = "<YOUR_VOYAGE_API_KEY>"
```

## Step 2: Load the Dataset

We'll use an Amazon products dataset containing over 10,000 product descriptions. This will serve as the knowledge base for our RAG system.

```python
import pandas as pd

# Download the JSONL file
!wget https://www-cdn.anthropic.com/48affa556a5af1de657d426bcc1506cdf7e2f68e/amazon-products.jsonl

data = []
with open("amazon-products.jsonl") as file:
    for line in file:
        try:
            data.append(eval(line))
        except (SyntaxError, ValueError):
            # Skip malformed lines in the dataset
            pass

df = pd.DataFrame(data)
print(f"Dataset loaded with {len(df)} entries.")
print(df.head())
```

## Step 3: Initialize the Vector Database

We'll use Pinecone as our vector database. First, initialize the Pinecone client and create an index.

```python
from pinecone import Pinecone, ServerlessSpec
import time

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define the serverless specification
spec = ServerlessSpec(cloud="aws", region="us-west-2")

# Create the index
index_name = "amazon-products"
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    # Create the index if it doesn't exist
    pc.create_index(
        index_name,
        dimension=1024,  # Dimensionality of Voyage's voyage-2 embeddings
        metric="dotproduct",
        spec=spec,
    )
    # Wait for index to be fully initialized
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

# Connect to the index
index = pc.Index(index_name)
time.sleep(1)

# View index statistics
print(index.describe_index_stats())
```

The index should report a `total_vector_count` of 0, as we haven't added any vectors yet.

## Step 4: Configure the Embedding Model

We'll use Voyage AI's `voyage-2` model to generate embeddings for our product descriptions.

```python
import voyageai

# Initialize the Voyage AI client
vo = voyageai.Client(api_key=VOYAGE_API_KEY)

# Test the embedding model
texts = ["Sample text 1", "Sample text 2"]
result = vo.embed(texts, model="voyage-2", input_type="document")
print(f"Sample embedding dimension: {len(result.embeddings[0])}")
```

## Step 5: Populate the Vector Index

Now, we'll embed all product descriptions and upload them to our Pinecone index in batches.

```python
from tqdm.auto import tqdm
from time import sleep

descriptions = df["text"].tolist()
batch_size = 100  # Number of embeddings to create and insert at once

for i in tqdm(range(0, len(descriptions), batch_size)):
    # Find the end of the current batch
    i_end = min(len(descriptions), i + batch_size)
    descriptions_batch = descriptions[i:i_end]
    
    # Create embeddings with retry logic for rate limits
    done = False
    while not done:
        try:
            res = vo.embed(descriptions_batch, model="voyage-2", input_type="document")
            done = True
        except Exception:
            sleep(5)
    
    embeds = [record for record in res.embeddings]
    # Create unique IDs for each text
    ids_batch = [f"description_{idx}" for idx in range(i, i_end)]
    
    # Create metadata dictionaries for each text
    metadata_batch = [{"description": description} for description in descriptions_batch]
    
    # Prepare data for upsert
    to_upsert = list(zip(ids_batch, embeds, metadata_batch, strict=False))
    
    # Upsert to Pinecone
    index.upsert(vectors=to_upsert)

print("Index population complete.")
```

## Step 6: Perform a Basic Semantic Search

With the index populated, we can perform semantic search by embedding a user query and finding the most similar product descriptions.

```python
USER_QUESTION = "I want to get my daughter more interested in science. What kind of gifts should I get her?"

# Embed the user's question
question_embed = vo.embed([USER_QUESTION], model="voyage-2", input_type="query")

# Query the index
results = index.query(vector=question_embed.embeddings, top_k=5, include_metadata=True)

# Display the top results
for match in results['matches']:
    print(f"Score: {match['score']:.4f}")
    print(f"Description: {match['metadata']['description'][:200]}...")
    print("-" * 80)
```

## Step 7: Optimize Search with Query Expansion

To improve retrieval diversity, we can use Claude to generate multiple search keywords from the original question.

```python
import anthropic
import json

# Initialize the Anthropic client
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def get_completion(prompt):
    """Helper function to get completions from Claude."""
    completion = client.completions.create(
        model="claude-2.1",
        prompt=prompt,
        max_tokens_to_sample=1024,
    )
    return completion.completion

def create_keyword_prompt(question):
    """Create a prompt for generating search keywords."""
    return f"""\n\nHuman: Given a question, generate a list of 5 very diverse search keywords that can be used to search for products on Amazon.

The question is: {question}

Output your keywords as a JSON that has one property "keywords" that is a list of strings. Only output valid JSON.\n\nAssistant:{{"""

# Generate keywords from the user's question
keyword_json = "{" + get_completion(create_keyword_prompt(USER_QUESTION))
data = json.loads(keyword_json)
keywords_list = data["keywords"]

print("Generated keywords:", keywords_list)
```

## Step 8: Retrieve Results Using Expanded Keywords

Now, let's search for each keyword and collect the top results.

```python
results_list = []
for keyword in keywords_list:
    # Embed the keyword
    query_embed = vo.embed([keyword], model="voyage-2", input_type="query")
    # Search the index
    search_results = index.query(vector=query_embed.embeddings, top_k=3, include_metadata=True)
    # Collect the descriptions
    for search_result in search_results.matches:
        results_list.append(search_result["metadata"]["description"])

print(f"Retrieved {len(results_list)} product descriptions.")
```

## Step 9: Generate an Answer with Claude

Finally, we'll format the retrieved results and ask Claude to generate an answer based on them.

```python
def format_results(extracted: list[str]) -> str:
    """Format search results for Claude's prompt."""
    result = "\n".join(
        [
            f'<item index="{i + 1}">\n<page_content>\n{r}\n</page_content>\n</item>'
            for i, r in enumerate(extracted)
        ]
    )
    return f"\n<search_results>\n{result}\n</search_results>"

def create_answer_prompt(results_list, question):
    """Create the final prompt for Claude."""
    return f"""\n\nHuman: {format_results(results_list)} Using the search results provided within the <search_results></search_results> tags, please answer the following question <question>{question}</question>. Do not reference the search results in your answer.\n\nAssistant:"""

# Generate the final answer
answer = get_completion(create_answer_prompt(results_list, USER_QUESTION))
print("Claude's Answer:")
print(answer)
```

**Example Output:**
```
To get your daughter more interested in science, I would recommend getting her an age-appropriate science kit or set that allows for hands-on exploration and experimentation. For example, for a younger child you could try a beginner chemistry set, magnet set, or crystal growing kit. For an older child, look for kits that tackle more advanced scientific principles like physics, engineering, robotics, etc. The key is choosing something that sparks her natural curiosity and lets her actively investigate concepts through activities, observations, and discovery. Supplement the kits with science books, museum visits, documentaries, and conversations about science she encounters in everyday life. Making science fun and engaging is crucial for building her interest.
```

## Conclusion

You have successfully built a complete RAG pipeline. This system:
1.  Embeds a knowledge base (Amazon product descriptions) into a vector database.
2.  Performs semantic search to find relevant information for a user query.
3.  Uses query expansion to improve retrieval diversity.
4.  Generates a grounded, informative answer using Claude.

This architecture can be adapted for various applications, including customer support, research assistance, and content generation.