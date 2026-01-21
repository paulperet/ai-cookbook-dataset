# Web Search with References using Mistral AI

## Introduction
This guide demonstrates how to use the latest Mistral Large 2 model to conduct web searches and incorporate accurate source references into responses. A common challenge with chatbots and RAG systems is their tendency to hallucinate sources or improperly format URLs. Mistral's advanced capabilities address these issues, ensuring reliable information retrieval with proper attribution.

## Prerequisites
Before starting, ensure you have:
- A Mistral AI API key from the [Mistral API dashboard](https://console.mistral.ai/api-keys/)
- Python installed on your system

## Setup
Install the required Python packages:

```bash
pip install mistralai==1.2.3 wikipedia==1.4.0
```

## Step 1: Initialize the Mistral Client
First, import the necessary modules and initialize the Mistral client with your API key. **Note**: New API keys may take up to 1 minute to activate.

```python
from mistralai import Mistral
from mistralai.models import UserMessage, SystemMessage
import os

# Initialize the client with your API key
client = Mistral(
    api_key=os.environ["MISTRAL_API_KEY"],
)

# Define your query
query = "Who won the Nobel Peace Prize in 2024?"

# Set up the chat history with system instructions
chat_history = [
    SystemMessage(content="You are a helpful assistant that can search the web for information. Use context to answer the question."),
    UserMessage(content=query),
]
```

## Step 2: Define the Function Calling Tool
Function calling allows Mistral models to connect to external tools. We'll create a web search tool that the model can call when it needs additional information.

```python
web_search_tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for a query for which you do not know the answer",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query to search the web in keyword form.",
                }
            },
            "required": ["query"],
        },
    },
}
```

Now, make an initial chat completion request to see if the model wants to use our tool:

```python
chat_response = client.chat.complete(
    model="mistral-large-latest",
    messages=chat_history,
    tools=[web_search_tool],
)

# Check if the model wants to call a tool
if hasattr(chat_response.choices[0].message, 'tool_calls'):
    tool_call = chat_response.choices[0].message.tool_calls[0]
    chat_history.append(chat_response.choices[0].message)
    print(f"Tool call requested: {tool_call.function.name}")
else:
    print("No tool call found in the response")
```

## Step 3: Create the Wikipedia Search Function
We need to implement the actual search function that will be called when the model requests web search. This function searches Wikipedia and returns results in a specific format.

```python
import wikipedia
import json
from datetime import datetime

def get_wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for a query and return the results in a specific format.
    """
    result = wikipedia.search(query, results=5)
    data = {}
    
    for i, res in enumerate(result):
        try:
            pg = wikipedia.page(res, auto_suggest=False)
            data[i] = {
                "url": pg.url,
                "title": pg.title,
                "snippets": [pg.summary.split('.')],
                "description": None,
                "date": datetime.now().isoformat(),
                "source": "wikipedia"
            }
        except:
            # Skip pages that cause errors
            continue
    
    return json.dumps(data, indent=2)
```

## Step 4: Execute the Tool Call
Now that we have the tool call request, we can execute the actual Wikipedia search:

```python
import json
from mistralai import ToolMessage

# Extract the query from the tool call
query = json.loads(tool_call.function.arguments)["query"]
wb_result = get_wikipedia_search(query)

# Create a tool message with the results
tool_call_result = ToolMessage(
    content=wb_result,
    tool_call_id=tool_call.id,
    name=tool_call.function.name,
)

# Append the tool call result to the chat history
chat_history.append(tool_call_result)

# View the search results
print(json.dumps(json.loads(wb_result), indent=2))
```

## Step 5: Generate the Final Answer with References
The chat history now contains all the context needed for the model to generate an answer with proper references. Let's create a helper function to format the response nicely:

```python
from mistralai.models import TextChunk, ReferenceChunk

def format_response(chat_response: list, wb_result: dict):
    print("\n🤖 Answer:\n")
    refs_used = []
    
    # Print the main response
    for chunk in chat_response.choices[0].message.content:
        if isinstance(chunk, TextChunk):
            print(chunk.text, end="")
        elif isinstance(chunk, ReferenceChunk):
            refs_used += chunk.reference_ids
    
    # Print references
    if refs_used:
        print("\n\n📚 Sources:")
        for i, ref in enumerate(set(refs_used), 1):
            reference = json.loads(wb_result)[str(ref)]
            print(f"\n{i}. {reference['title']}: {reference['url']}")

# Get the final answer from Mistral
chat_response = client.chat.complete(
    model="mistral-large-latest",
    messages=chat_history,
    tools=[web_search_tool],
)

# Format and display the response
format_response(chat_response, wb_result)
```

## Step 6: Streaming Completion with References (Optional)
For real-time responses, you can use streaming completion. This shows the answer as it's being generated, with references inserted at the appropriate points:

```python
stream_response = client.chat.stream(
    model="mistral-large-latest",
    messages=chat_history,
    tools=[web_search_tool],
)

last_reference_index = 0
if stream_response is not None:
    for event in stream_response:
        chunk = event.data.choices[0]
        if chunk.delta.content:
            if isinstance(chunk.delta.content, list):
                # Process reference chunks
                references_ids = [
                    ref_id
                    for chunk_elem in chunk.delta.content
                    if chunk_elem.TYPE == "reference"
                    for ref_id in chunk_elem.reference_ids
                ]
                last_reference_index += len(references_ids)
                
                # Map reference IDs to actual URLs
                references_data = [json.loads(wb_result)[str(ref_id)] for ref_id in references_ids]
                urls = " " + ", ".join(
                    [
                        f"[{i}]({reference['url']})"
                        for i, reference in enumerate(
                            references_data,
                            start=last_reference_index - len(references_ids) + 1,
                        )
                    ]
                )
                print(urls, end="")
            else:
                # Print regular text chunks
                print(chunk.delta.content, end="")
```

## Conclusion
You've successfully implemented a web search system using Mistral AI that:
1. Detects when information needs to be retrieved from external sources
2. Performs Wikipedia searches through function calling
3. Incorporates accurate references into the final response
4. Supports both regular and streaming completion modes

This approach ensures that your AI applications provide reliable, well-sourced information while maintaining a natural conversational flow. You can extend this pattern to integrate other data sources or APIs by defining additional tools and corresponding functions.