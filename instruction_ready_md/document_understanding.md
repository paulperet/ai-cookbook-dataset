# Chat with Your Documents Using Built-in Document Q&A

This guide demonstrates how to use Mistral's built-in Optical Character Recognition (OCR) feature to enable any Large Language Model (LLM) to understand and answer questions about text-based documents. You can provide URLs to documents (like PDFs, images, or screenshots), and the model will process the text content for you.

## Prerequisites

First, install the required library and set up your client.

```bash
pip install mistralai
```

```python
from mistralai import Mistral

# Replace with your API key from https://console.mistral.ai/api-keys/
api_key = "YOUR_API_KEY"
client = Mistral(api_key=api_key)

# Specify the model you want to use
text_model = "mistral-small-latest"
```

## Step 1: Define the System Prompt

Since this workflow doesn't require tool calls, you can use a straightforward system prompt.

```python
system_prompt = "You are an AI Assistant with document understanding via URLs. You may be provided with URLs, followed by their corresponding OCR."
```

## Step 2: Extract URLs from User Queries

You'll need to extract document URLs from the user's message. For simplicity, this example uses a regular expression to find URLs, assuming they point to PDFs.

```python
import re

def extract_urls(text: str) -> list:
    """Extract all URLs from a given string."""
    url_pattern = r'\b((?:https?|ftp)://(?:www\.)?[^\s/$.?#].[^\s]*)\b'
    urls = re.findall(url_pattern, text)
    return urls

# Test the function
example_text = "Hi there, you can visit our docs at https://docs.mistral.ai/"
print(extract_urls(example_text))
# Output: ['https://docs.mistral.ai/']
```

## Step 3: Build the Chat Loop

Now, create an interactive loop that:
1.  Takes user input.
2.  Extracts any document URLs.
3.  Formats the message with the extracted URLs for the model.
4.  Sends the request and prints the assistant's response.

```python
import json

# Initialize the conversation with the system prompt
messages = [{"role": "system", "content": system_prompt}]

print("Chat started. Type 'quit' to exit.")
print("You can ask questions about documents by including their URLs.")
print("Example: 'Could you summarize this research paper? https://arxiv.org/pdf/2410.07073'")

while True:
    # Get user input
    user_input = input("\nUser > ")
    if user_input.lower() == "quit":
        break

    # Step 1: Extract URLs from the query
    document_urls = extract_urls(user_input)

    # Step 2: Structure the user message content
    # Start with the text query
    user_message_content = [{"type": "text", "text": user_input}]
    # Append each extracted URL as a document_url type
    for url in document_urls:
        user_message_content.append({"type": "document_url", "document_url": url})

    # Add the structured user message to the conversation history
    messages.append({"role": "user", "content": user_message_content})

    # Step 3: Call the Mistral API
    response = client.chat.complete(
        model=text_model,
        messages=messages,
        temperature=0  # Set to 0 for deterministic, factual responses
    )

    # Get the assistant's reply
    assistant_reply = response.choices[0].message.content

    # Step 4: Add the assistant's reply to history and print it
    messages.append({"role": "assistant", "content": assistant_reply})
    print(f"Assistant > {assistant_reply}")
```

## How It Works

1.  **User Query:** The user sends a message containing a question and one or more document URLs.
2.  **URL Extraction:** The `extract_urls` function identifies all URLs in the query.
3.  **Message Formatting:** The query is formatted into a list of content blocks. The initial text is provided as `"type": "text"`, and each extracted URL is added as a `"type": "document_url"`. This structure tells the Mistral API to process the linked documents using its built-in OCR.
4.  **Model Processing:** The model receives the text and the processed document content, then generates a relevant answer.
5.  **Response:** The assistant's answer is displayed and added to the conversation history for context.

## Example Interaction

```
User > Could you summarize the key findings in this paper? https://arxiv.org/pdf/2410.07073
Assistant > [The model will provide a summary based on the OCR-extracted text from the provided PDF URL.]
```

## Next Steps

-   Explore the [official Document Q&A documentation](https://docs.mistral.ai/capabilities/OCR/document_qna/) for advanced features and best practices.
-   Extend the URL extraction logic to handle different file types or local file uploads.
-   Integrate this functionality into a larger application, such as a chatbot or a document analysis tool.

You now have a functional workflow to chat with any document via URL using Mistral's built-in OCR capabilities.