# Structured JSON Extraction with Claude and Tool Use

## Overview
This guide demonstrates how to use Claude's tool use feature to extract structured JSON data from various text inputs. You'll learn to define custom tools that prompt Claude to generate well-formatted JSON for tasks like summarization, entity extraction, sentiment analysis, and more.

## Prerequisites

First, install the required Python packages:

```bash
pip install anthropic requests beautifulsoup4
```

Then, import the necessary libraries:

```python
import json
import requests
from anthropic import Anthropic
from bs4 import BeautifulSoup

# Initialize the Anthropic client
client = Anthropic()
MODEL_NAME = "claude-3-haiku-20240307"  # Updated to actual model name
```

## Example 1: Article Summarization

In this example, you'll create a tool that extracts key information from an article and returns it as structured JSON.

### Step 1: Define the Summarization Tool

Create a tool with a schema that specifies the exact JSON structure you want:

```python
tools = [
    {
        "name": "print_summary",
        "description": "Prints a summary of the article.",
        "input_schema": {
            "type": "object",
            "properties": {
                "author": {
                    "type": "string",
                    "description": "Name of the article author"
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of topics, e.g. ['tech', 'politics']. Should be as specific as possible."
                },
                "summary": {
                    "type": "string",
                    "description": "Summary of the article. One or two paragraphs max."
                },
                "coherence": {
                    "type": "integer",
                    "description": "Coherence of the article's key points, 0-100 (inclusive)"
                },
                "persuasion": {
                    "type": "number",
                    "description": "Article's persuasion score, 0.0-1.0 (inclusive)"
                },
                "counterpoint": {
                    "type": "string",
                    "description": "A counterpoint or alternative perspective on the article's main argument"
                }
            },
            "required": ["author", "topics", "summary", "coherence", "persuasion", "counterpoint"]
        }
    }
]
```

### Step 2: Fetch and Process Article Content

Now, fetch an article from a URL and extract its text content:

```python
# Fetch article content
url = "https://www.anthropic.com/news/third-party-testing"
response = requests.get(url, timeout=30)
soup = BeautifulSoup(response.text, "html.parser")
article = " ".join([p.text for p in soup.find_all("p")])

# Prepare the query for Claude
query = f"""
<article>
{article}
</article>

Use the `print_summary` tool to extract a structured summary.
"""
```

### Step 3: Call Claude and Extract the JSON

Send the query to Claude and parse the response to extract the structured JSON:

```python
response = client.messages.create(
    model=MODEL_NAME,
    max_tokens=4096,
    tools=tools,
    messages=[{"role": "user", "content": query}]
)

# Extract the JSON from the tool use response
json_summary = None
for content in response.content:
    if content.type == "tool_use" and content.name == "print_summary":
        json_summary = content.input
        break

# Display the result
if json_summary:
    print("JSON Summary:")
    print(json.dumps(json_summary, indent=2))
else:
    print("No JSON summary found in the response.")
```

## Example 2: Named Entity Recognition

This example shows how to extract named entities from text with structured types and context.

### Step 1: Define the Entity Extraction Tool

```python
tools = [
    {
        "name": "print_entities",
        "description": "Prints extracted named entities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The extracted entity name."
                            },
                            "type": {
                                "type": "string",
                                "description": "The entity type (e.g., PERSON, ORGANIZATION, LOCATION)."
                            },
                            "context": {
                                "type": "string",
                                "description": "The context in which the entity appears in the text."
                            }
                        },
                        "required": ["name", "type", "context"]
                    }
                }
            },
            "required": ["entities"]
        }
    }
]
```

### Step 2: Prepare and Send the Query

```python
text = "John works at Google in New York. He met with Sarah, the CEO of Acme Inc., last week in San Francisco."

query = f"""
<document>
{text}
</document>

Use the print_entities tool to extract all named entities.
"""

response = client.messages.create(
    model=MODEL_NAME,
    max_tokens=4096,
    tools=tools,
    messages=[{"role": "user", "content": query}]
)

# Extract and display the entities
json_entities = None
for content in response.content:
    if content.type == "tool_use" and content.name == "print_entities":
        json_entities = content.input
        break

if json_entities:
    print("Extracted Entities (JSON):")
    print(json.dumps(json_entities, indent=2))
else:
    print("No entities found in the response.")
```

## Example 3: Sentiment Analysis

Learn how to extract sentiment scores from text using a structured JSON schema.

### Step 1: Define the Sentiment Analysis Tool

```python
tools = [
    {
        "name": "print_sentiment_scores",
        "description": "Prints the sentiment scores of a given text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "positive_score": {
                    "type": "number",
                    "description": "The positive sentiment score, ranging from 0.0 to 1.0."
                },
                "negative_score": {
                    "type": "number",
                    "description": "The negative sentiment score, ranging from 0.0 to 1.0."
                },
                "neutral_score": {
                    "type": "number",
                    "description": "The neutral sentiment score, ranging from 0.0 to 1.0."
                }
            },
            "required": ["positive_score", "negative_score", "neutral_score"]
        }
    }
]
```

### Step 2: Analyze Text Sentiment

```python
text = "The product was okay, but the customer service was terrible. I probably won't buy from them again."

query = f"""
<text>
{text}
</text>

Use the print_sentiment_scores tool to analyze the sentiment.
"""

response = client.messages.create(
    model=MODEL_NAME,
    max_tokens=4096,
    tools=tools,
    messages=[{"role": "user", "content": query}]
)

# Extract and display sentiment scores
json_sentiment = None
for content in response.content:
    if content.type == "tool_use" and content.name == "print_sentiment_scores":
        json_sentiment = content.input
        break

if json_sentiment:
    print("Sentiment Analysis (JSON):")
    print(json.dumps(json_sentiment, indent=2))
else:
    print("No sentiment analysis found in the response.")
```

## Example 4: Text Classification

This example demonstrates how to classify text into predefined categories with confidence scores.

### Step 1: Define the Classification Tool

```python
tools = [
    {
        "name": "print_classification",
        "description": "Prints the classification results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The category name."
                            },
                            "score": {
                                "type": "number",
                                "description": "The classification score for the category, ranging from 0.0 to 1.0."
                            }
                        },
                        "required": ["name", "score"]
                    }
                }
            },
            "required": ["categories"]
        }
    }
]
```

### Step 2: Classify the Text

```python
text = "The new quantum computing breakthrough could revolutionize the tech industry."

query = f"""
<document>
{text}
</document>

Use the print_classification tool. The categories can be Politics, Sports, Technology, Entertainment, Business.
"""

response = client.messages.create(
    model=MODEL_NAME,
    max_tokens=4096,
    tools=tools,
    messages=[{"role": "user", "content": query}]
)

# Extract and display classification results
json_classification = None
for content in response.content:
    if content.type == "tool_use" and content.name == "print_classification":
        json_classification = content.input
        break

if json_classification:
    print("Text Classification (JSON):")
    print(json.dumps(json_classification, indent=2))
else:
    print("No text classification found in the response.")
```

## Example 5: Working with Unknown Keys

Sometimes you need flexibility in your JSON structure. This example shows how to use an open-ended schema.

### Step 1: Define a Flexible Tool Schema

```python
tools = [
    {
        "name": "print_all_characteristics",
        "description": "Prints all characteristics which are provided.",
        "input_schema": {
            "type": "object",
            "additionalProperties": True
        }
    }
]
```

### Step 2: Use the Tool with Detailed Instructions

```python
query = """Given a description of a character, your task is to extract all the characteristics of the character and print them using the print_all_characteristics tool.

The print_all_characteristics tool takes an arbitrary number of inputs where the key is the characteristic name and the value is the characteristic value (age: 28 or eye_color: green).

<description>
The man is tall, with a beard and a scar on his left cheek. He has a deep voice and wears a black leather jacket.
</description>

Now use the print_all_characteristics tool."""

response = client.messages.create(
    model=MODEL_NAME,
    max_tokens=4096,
    tools=tools,
    tool_choice={"type": "tool", "name": "print_all_characteristics"},
    messages=[{"role": "user", "content": query}]
)

# Extract and display the flexible JSON output
tool_output = None
for content in response.content:
    if content.type == "tool_use" and content.name == "print_all_characteristics":
        tool_output = content.input
        break

if tool_output:
    print("Characteristics (JSON):")
    print(json.dumps(tool_output, indent=2))
else:
    print("Something went wrong.")
```

## Key Takeaways

1. **Structured Output**: Tool use allows you to define exact JSON schemas for Claude's responses, ensuring consistent, parseable output.

2. **Flexible Schemas**: You can create both strict schemas (with predefined properties) and flexible schemas (with `additionalProperties: True`) depending on your needs.

3. **Prompt Guidance**: Clear instructions in your query help Claude understand how to use the tools effectively.

4. **Response Parsing**: Always check the response content for `tool_use` items to extract your structured JSON data.

By following these patterns, you can adapt Claude's tool use feature for a wide variety of structured data extraction tasks in your applications.