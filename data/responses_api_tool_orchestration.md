# Multi-Tool Orchestration with RAG using OpenAI's Responses API

This guide demonstrates how to build a dynamic, multi-tool workflow using OpenAI's Responses API. You will implement a Retrieval-Augmented Generation (RAG) system that intelligently routes user queries to the appropriate tool—be it a built-in web search or an external vector database like Pinecone. By the end, you'll have a functional pipeline that can answer general questions with live web data and domain-specific questions using an internal knowledge base.

## Prerequisites & Setup

Ensure you have the necessary Python libraries installed and your API keys configured.

```bash
pip install datasets tqdm pandas pinecone openai --quiet
```

```python
import os
import time
from tqdm.auto import tqdm
from pandas import DataFrame
from datasets import load_dataset
import random
import string

# Import OpenAI client and initialize with your API key.
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Import Pinecone client.
from pinecone import Pinecone
from pinecone import ServerlessSpec
```

## Step 1: Load and Prepare the Dataset

We'll use a medical reasoning dataset from Hugging Face. The goal is to merge question-answer pairs into a single text string for embedding.

```python
# Load the dataset (ensure you're logged in with huggingface-cli if needed)
ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split='train[:100]', trust_remote_code=True)
ds_dataframe = DataFrame(ds)

# Merge the Question and Response columns into a single string.
ds_dataframe['merged'] = ds_dataframe.apply(
    lambda row: f"Question: {row['Question']} Answer: {row['Response']}", axis=1
)
print("Example merged text:", ds_dataframe['merged'].iloc[0])
```

## Step 2: Create a Pinecone Index

First, determine the embedding dimension by creating a sample embedding. Then, create a Pinecone index with that dimension.

```python
MODEL = "text-embedding-3-small"  # Replace with your production embedding model if needed

# Compute an embedding for the first document to obtain the embedding dimension.
sample_embedding_resp = client.embeddings.create(
    input=[ds_dataframe['merged'].iloc[0]],
    model=MODEL
)
embed_dim = len(sample_embedding_resp.data[0].embedding)
print(f"Embedding dimension: {embed_dim}")

# Initialize Pinecone using your API key.
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define the Pinecone serverless specification.
AWS_REGION = "us-east-1"
spec = ServerlessSpec(cloud="aws", region=AWS_REGION)

# Create a random index name.
index_name = 'pinecone-index-' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))

# Create the index if it doesn't already exist.
if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=embed_dim,
        metric='dotproduct',
        spec=spec
    )

# Connect to the index.
index = pc.Index(index_name)
time.sleep(1)
print("Index stats:", index.describe_index_stats())
```

## Step 3: Populate the Index with Data

Process the dataset in batches, generate embeddings, and upsert them into the Pinecone index along with metadata.

```python
batch_size = 32
for i in tqdm(range(0, len(ds_dataframe['merged']), batch_size), desc="Upserting to Pinecone"):
    i_end = min(i + batch_size, len(ds_dataframe['merged']))
    lines_batch = ds_dataframe['merged'][i: i_end]
    ids_batch = [str(n) for n in range(i, i_end)]
    
    # Create embeddings for the current batch.
    res = client.embeddings.create(input=[line for line in lines_batch], model=MODEL)
    embeds = [record.embedding for record in res.data]
    
    # Prepare metadata by extracting original Question and Answer.
    meta = []
    for record in ds_dataframe.iloc[i:i_end].to_dict('records'):
        q_text = record['Question']
        a_text = record['Response']
        meta.append({"Question": q_text, "Answer": a_text})
    
    # Upsert the batch into Pinecone.
    vectors = list(zip(ids_batch, embeds, meta))
    index.upsert(vectors=vectors)
```

## Step 4: Query the Index

Define a helper function to query the Pinecone index with a natural language question and retrieve the most relevant documents.

```python
def query_pinecone_index(client, index, model, query_text):
    # Generate an embedding for the query.
    query_embedding = client.embeddings.create(input=query_text, model=model).data[0].embedding

    # Query the index and return top 5 matches.
    res = index.query(vector=[query_embedding], top_k=5, include_metadata=True)
    print("Query Results:")
    for match in res['matches']:
        print(f"{match['score']:.2f}: {match['metadata'].get('Question', 'N/A')} - {match['metadata'].get('Answer', 'N/A')}")
    return res
```

Test the query function with an example medical question.

```python
# Example usage with a different query from the train/test set
query = (
    "A 45-year-old man with a history of alcohol use presents with symptoms including confusion, ataxia, and ophthalmoplegia. "
    "What is the most likely diagnosis and the recommended treatment?"
)
query_pinecone_index(client, index, MODEL, query)
```

## Step 5: Generate a Response with Retrieved Context

Use the top matches from the query to construct a context and generate a final answer using the Responses API.

```python
# Retrieve and concatenate top 3 match contexts.
matches = index.query(
    vector=[client.embeddings.create(input=query, model=MODEL).data[0].embedding],
    top_k=3,
    include_metadata=True
)['matches']

context = "\n\n".join(
    f"Question: {m['metadata'].get('Question', '')}\nAnswer: {m['metadata'].get('Answer', '')}"
    for m in matches
)

# Use the context to generate a final answer.
response = client.responses.create(
    model="gpt-4o",
    input=f"Provide the answer based on the context: {context} and the question: {query} as per the internal knowledge base",
)
print("\nFinal Answer:")
print(response.output_text)
```

## Step 6: Define Tools for Multi-Tool Orchestration

Now, define the tools available to the Responses API. This includes a built-in web search tool and a custom function to query the Pinecone index.

```python
# Tools definition: The list of tools includes:
# - A web search preview tool.
# - A Pinecone search tool for retrieving medical documents.

# Define available tools.
tools = [   
    {"type": "web_search_preview",
      "user_location": {
        "type": "approximate",
        "country": "US",
        "region": "California",
        "city": "SF"
      },
      "search_context_size": "medium"},
    {
        "type": "function",
        "name": "PineconeSearchDocuments",
        "description": "Search for relevant documents based on the medical question asked by the user that is stored within the vector database using a semantic query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language query to search the vector database."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return.",
                    "default": 3
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
]
```

## Step 7: Process Queries Dynamically

Create a list of sample queries and process each one. The model will decide which tool to call based on the query content.

```python
# Example queries that the model should route appropriately.
queries = [
    {"query": "Who won the cricket world cup in 1983?"},
    {"query": "What is the most common cause of death in the United States according to the internet?"},
    {"query": ("A 7-year-old boy with sickle cell disease is experiencing knee and hip pain, "
               "has been admitted for pain crises in the past, and now walks with a limp. "
               "His exam shows a normal, cool hip with decreased range of motion and pain with ambulation. "
               "What is the most appropriate next step in management according to the internal knowledge base?")}
]

# Process each query dynamically.
for item in queries:
    input_messages = [{"role": "user", "content": item["query"]}]
    print("\n🌟--- Processing Query ---🌟")
    print(f"🔍 **User Query:** {item['query']}")
    
    # Call the Responses API with tools enabled and allow parallel tool calls.
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": "When prompted with a question, select the right tool to use based on the question."},
            {"role": "user", "content": item["query"]}
        ],
        tools=tools,
        parallel_tool_calls=True
    )
    
    print("\n✨ **Initial Response Output:**")
    print(response.output)
    
    # Determine if a tool call is needed and process accordingly.
    if response.output:
        tool_call = response.output[0]
        if tool_call.type in ["web_search_preview", "function_call"]:
            tool_name = tool_call.name if tool_call.type == "function_call" else "web_search_preview"
            print(f"\n🔧 **Model triggered a tool call:** {tool_name}")
            
            if tool_name == "PineconeSearchDocuments":
                print("🔍 **Invoking PineconeSearchDocuments tool...**")
                res = query_pinecone_index(client, index, MODEL, item["query"])
                if res["matches"]:
                    best_match = res["matches"][0]["metadata"]
                    result = f"**Question:** {best_match.get('Question', 'N/A')}\n**Answer:** {best_match.get('Answer', 'N/A')}"
                else:
                    result = "**No matching documents found in the index.**"
                print("✅ **PineconeSearchDocuments tool invoked successfully.**")
            else:
                print("🔍 **Invoking simulated web search tool...**")
                result = "**Simulated web search result.**"
                print("✅ **Simulated web search tool invoked successfully.**")
            
            # Append the tool call and its output back into the conversation.
            input_messages.append(tool_call)
            input_messages.append({
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": str(result)
            })
            
            # Get the final answer incorporating the tool's result.
            final_response = client.responses.create(
                model="gpt-4o",
                input=input_messages,
                tools=tools,
                parallel_tool_calls=True
            )
            print("\n💡 **Final Answer:**")
            print(final_response.output_text)
        else:
            # If no tool call is triggered, print the response directly.
            print("💡 **Final Answer:**")
            print(response.output_text)
```

The model will route general knowledge queries (like sports results) to the web search tool, while specific medical inquiries will trigger the Pinecone search function to retrieve relevant context from your internal knowledge base.

## Step 8: Orchestrate Sequential Tool Calls

You can also instruct the model to call tools in a specific sequence. For example, you might want it to first perform a web search and then query the internal database.

```python
# Process one query with explicit sequential tool calling instructions.
item = "What is the most common cause of death in the United States"

# Initialize input messages with the user's query.
input_messages = [{"role": "user", "content": item}]
print("\n🌟--- Processing Query ---🌟")
print(f"🔍 **User Query:** {item}")
    
print("\n🔧 **Calling Responses API with Tools Enabled**")
print("\n🕵️‍♂️ **Step 1: Web Search Call**")
print("   - Initiating web search to gather initial information.")
print("\n📚 **Step 2: Pinecone Search Call**")
print("   - Querying Pinecone to find relevant examples from the internal knowledge base.")
    
response = client.responses.create(
    model="gpt-4o",
    input=[
        {"role": "system", "content": "Every time it's prompted with a question, first call the web search tool for results, then call `PineconeSearchDocuments` to find real examples in the internal knowledge base."},
        {"role": "user", "content": item}
    ],
    tools=tools,
    parallel_tool_calls=True
)
    
print("\n✨ **Initial Response Output:**")
print(response.output)
```

## Step 9: Inspect and Process Tool Calls

You can inspect the tool calls returned by the API and manually append their results to the conversation for a final answer.

```python
# Let's assume the response triggered two tool calls.
# For demonstration, we'll simulate processing the second tool call (PineconeSearchDocuments).

# Extract the second tool call (function call).
tool_call_2 = response.output[2]  # Adjust index based on actual response structure
print(f"Tool Call 2: {tool_call_2}")
print(f"Call ID: {tool_call_2.call_id}")

# Simulate a result from the Pinecone search.
result = "**Question:** What are the leading causes of death in the US?\n**Answer:** Heart disease and cancer are the top two causes."

# Append the tool call and its output back into the conversation.
input_messages.append(tool_call_2)
input_messages.append({
    "type": "function_call_output",
    "call_id": tool_call_2.call_id,
    "output": str(result)
})

# Get the final answer incorporating the tool's result.
print("\n🔧 **Calling Responses API for Final Answer**")
response_2 = client.responses.create(
    model="gpt-4o",
    input=input_messages,
)
print("\n💡 **Final Answer:**")
print(response_2.output_text)
```

## Conclusion

You have successfully built a multi-tool orchestration system using OpenAI's Responses API. This system can:
1.  **Route queries intelligently:** Decide whether a question requires a web search or internal document retrieval.
2.  **Perform RAG:** Retrieve relevant context from a Pinecone vector database and generate accurate answers.
3.  **Sequence tool calls:** Execute tools in a specified order to combine information from multiple sources.

This framework is highly adaptable. You can extend it by adding more tools (e.g., database queries, API calls) and refining the routing logic. For further exploration, consider the official OpenAI cookbook for examples on file search and other advanced features.

Happy coding!