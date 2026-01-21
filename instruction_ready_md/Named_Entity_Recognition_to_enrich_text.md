# Guide: Enriching Text with Named Entity Recognition and Wikipedia Links

This guide demonstrates how to use OpenAI's Chat Completions API with function calling to perform Named Entity Recognition (NER) and automatically enrich text with relevant Wikipedia links. You'll learn to extract structured entities from raw text and enhance them with contextual knowledge base references.

## Prerequisites

Ensure you have the following installed and configured:

### 1. Install Required Packages
Run the following commands in your terminal or notebook environment:

```bash
pip install --upgrade openai nlpia2-wikipedia tenacity
```

### 2. Set Up Your OpenAI API Key
Generate an API key from the [OpenAI Platform](https://platform.openai.com/account/api-keys) and set it as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Step 1: Initialize Your Environment

First, import the necessary libraries and configure your OpenAI client:

```python
import json
import logging
import os
from typing import Optional

import openai
import wikipedia
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Configure logging
logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Set your preferred model
OPENAI_MODEL = 'gpt-3.5-turbo-0613'
```

## Step 2: Define Entity Labels

Create a list of standard NER labels that the model will recognize. While we'll use all these for demonstration, only a subset is needed for Wikipedia linking:

```python
labels = [
    "person",      # people, including fictional characters
    "fac",         # buildings, airports, highways, bridges
    "org",         # organizations, companies, agencies, institutions
    "gpe",         # geopolitical entities like countries, cities, states
    "loc",         # non-gpe locations
    "product",     # vehicles, foods, apparel, appliances, software, toys 
    "event",       # named sports, scientific milestones, historical events
    "work_of_art", # titles of books, songs, movies
    "law",         # named laws, acts, or legislations
    "language",    # any named language
    "date",        # absolute or relative dates or periods
    "time",        # time units smaller than a day
    "percent",     # percentage (e.g., "twenty percent", "18%")
    "money",       # monetary values, including unit
    "quantity",    # measurements, e.g., weight or distance
]
```

## Step 3: Prepare Chat Messages

The Chat Completions API requires properly formatted messages. We'll create three types of messages to guide the model.

### System Message
This defines the assistant's role and task:

```python
def system_message(labels):
    return f"""
You are an expert in Natural Language Processing. Your task is to identify common Named Entities (NER) in a given text.
The possible common Named Entities (NER) types are exclusively: ({", ".join(labels)})."""
```

### Assistant Message (One-Shot Example)
This provides an example to improve the model's accuracy:

```python
def assistant_message():
    return """
EXAMPLE:
    Text: 'In Germany, in 1440, goldsmith Johannes Gutenberg invented the movable-type printing press. His work led to an information revolution and the unprecedented mass-spread of literature throughout Europe. Modelled on the design of the existing screw presses, a single Renaissance movable-type printing press could produce up to 3,600 pages per workday.'
    {
        "gpe": ["Germany", "Europe"],
        "date": ["1440"],
        "person": ["Johannes Gutenberg"],
        "product": ["movable-type printing press"],
        "event": ["Renaissance"],
        "quantity": ["3,600 pages"],
        "time": ["workday"]
    }
--"""
```

### User Message
This contains the actual text to process:

```python
def user_message(text):
    return f"""
TASK:
    Text: {text}
"""
```

## Step 4: Create Wikipedia Link Functions

Now, build the functions that will find and apply Wikipedia links to identified entities.

### Find Wikipedia Links
This function searches Wikipedia for an entity and returns the most relevant page URL:

```python
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def find_link(entity: str) -> Optional[str]:
    """
    Finds a Wikipedia link for a given entity.
    """
    try:
        titles = wikipedia.search(entity)
        if titles:
            # Naively consider the first result as the best
            page = wikipedia.page(titles[0])
            return page.url
    except wikipedia.exceptions.WikipediaException as ex:
        logging.error(f'Error occurred while searching for Wikipedia link for entity {entity}: {str(ex)}')
    
    return None
```

### Find Links for All Entities
This function processes all entities from specific label categories:

```python
def find_all_links(label_entities: dict) -> dict:
    """ 
    Finds all Wikipedia links for the dictionary entities in the whitelist label list.
    """
    whitelist = ['event', 'gpe', 'org', 'person', 'product', 'work_of_art']
    
    return {e: find_link(e) for label, entities in label_entities.items() 
                            for e in entities
                            if label in whitelist}
```

### Enrich Text with Links
This replaces entity mentions in the text with clickable Wikipedia links:

```python
def enrich_entities(text: str, label_entities: dict) -> str:
    """
    Enriches text with knowledge base links.
    """
    entity_link_dict = find_all_links(label_entities)
    logging.info(f"entity_link_dict: {entity_link_dict}")
    
    for entity, link in entity_link_dict.items():
        text = text.replace(entity, f"[{entity}]({link})")
    
    return text
```

## Step 5: Configure OpenAI Function Calling

Define the function schema that the model will use to structure its response:

```python
def generate_functions(labels: list) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "enrich_entities",
                "description": "Enrich Text with Knowledge Base Links",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "r'^(?:' + '|'.join({labels}) + ')$'": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "additionalProperties": False
                },
            }
        }
    ]
```

## Step 6: Execute the NER Pipeline

Create the main function that orchestrates the entire process:

```python
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def run_openai_task(labels, text):
    # Prepare messages
    messages = [
        {"role": "system", "content": system_message(labels=labels)},
        {"role": "assistant", "content": assistant_message()},
        {"role": "user", "content": user_message(text=text)}
    ]
    
    # Call the OpenAI API
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        tools=generate_functions(labels),
        tool_choice={"type": "function", "function": {"name": "enrich_entities"}},
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
    )
    
    # Parse the response
    response_message = response.choices[0].message
    
    # Map available functions
    available_functions = {"enrich_entities": enrich_entities}
    function_name = response_message.tool_calls[0].function.name
    
    # Get the function to call
    function_to_call = available_functions[function_name]
    logging.info(f"function_to_call: {function_to_call}")
    
    # Parse function arguments
    function_args = json.loads(response_message.tool_calls[0].function.arguments)
    logging.info(f"function_args: {function_args}")
    
    # Execute the function
    function_response = function_to_call(text, function_args)
    
    return {
        "model_response": response,
        "function_response": function_response
    }
```

## Step 7: Run the Example

Now let's test the pipeline with sample text:

```python
# Define your text
text = """The Beatles were an English rock band formed in Liverpool in 1960, comprising John Lennon, Paul McCartney, George Harrison, and Ringo Starr."""

# Run the NER and enrichment process
result = run_openai_task(labels, text)

# Display the results
print("Original Text:")
print(text)
print("\nEnriched Text:")
print(result['function_response'])
```

## Step 8: Estimate Inference Costs

Monitor your API usage and estimate costs:

```python
# Extract token usage from the response
i_tokens = result["model_response"].usage.prompt_tokens
o_tokens = result["model_response"].usage.completion_tokens

# Calculate costs (using gpt-3.5-turbo 4K context pricing)
i_cost = (i_tokens / 1000) * 0.0015
o_cost = (o_tokens / 1000) * 0.002

print(f"""Token Usage:
    Prompt: {i_tokens} tokens
    Completion: {o_tokens} tokens
    Cost estimation: ${round(i_cost + o_cost, 5)}""")
```

## Expected Output

When you run the example with The Beatles text, you should get output similar to:

```
Original Text:
The Beatles were an English rock band formed in Liverpool in 1960, comprising John Lennon, Paul McCartney, George Harrison, and Ringo Starr.

Enriched Text:
[The Beatles](https://en.wikipedia.org/wiki/The_Beatles) were an English rock band formed in [Liverpool](https://en.wikipedia.org/wiki/Liverpool) in 1960, comprising [John Lennon](https://en.wikipedia.org/wiki/John_Lennon), [Paul McCartney](https://en.wikipedia.org/wiki/Paul_McCartney), [George Harrison](https://en.wikipedia.org/wiki/George_Harrison), and [Ringo Starr](https://en.wikipedia.org/wiki/Ringo_Starr).
```

## Best Practices and Considerations

1. **Error Handling**: The Wikipedia search may fail for some entities. Consider implementing fallback strategies or manual verification for critical applications.

2. **Cost Optimization**: For production use, consider caching Wikipedia lookups and implementing rate limiting.

3. **Model Selection**: While this guide uses `gpt-3.5-turbo-0613`, you can experiment with `gpt-4-0613` for potentially better accuracy with more complex texts.

4. **Entity Whitelist**: Adjust the `whitelist` in `find_all_links()` based on which entity types you want to link to Wikipedia.

5. **Real-World Applications**: This technique can be extended to enrich documents for knowledge graphs, enhance educational content, or improve content discoverability in content management systems.

By following this guide, you've learned how to combine OpenAI's function calling capability with external APIs to create a powerful text enrichment pipeline that transforms unstructured text into linked, knowledge-rich content.