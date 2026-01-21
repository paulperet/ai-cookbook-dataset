# Guide: Document Comprehension with Any Model via Tool Usage and OCR

## Introduction
Optical Character Recognition (OCR) transforms text-based documents and images into pure text. By leveraging OCR, you can enable any Large Language Model (LLM) to reliably understand documents—such as PDFs, photos, or screenshots—efficiently and cost-effectively.

In this tutorial, you will build a system that uses Mistral's models and Tool Usage to fetch and process documents from URLs on demand. The model will decide when to perform OCR and use the extracted text to answer your questions.

> **Note:** Mistral also offers a built-in Document Understanding feature. For more details, see the [official documentation](https://docs.mistral.ai/capabilities/OCR/document_understanding/).

---

## Prerequisites
Ensure you have a Mistral API key. You can create one on the [Mistral Platform](https://console.mistral.ai/api-keys/).

### Step 1: Install the Mistral Client
Begin by installing the required Python package.

```bash
pip install mistralai
```

### Step 2: Import Libraries and Initialize the Client
Import the necessary modules and set up your client with the API key.

```python
from mistralai import Mistral
import json

api_key = "YOUR_API_KEY"  # Replace with your actual API key
client = Mistral(api_key=api_key)

# Define the models you'll use
text_model = "mistral-small-latest"
ocr_model = "mistral-ocr-latest"
```

---

## Step 3: Define the System Prompt
Provide the model with clear instructions about its role and the tools available.

```python
system = """You are an AI Assistant with document understanding via URLs. You will be provided with URLs, and you must answer any questions related to those documents.

# OPEN URLS INSTRUCTIONS
You can open URLs by using the `open_urls` tool. It will open webpages and apply OCR to them, retrieving the contents. Use those contents to answer the user.
Only URLs pointing to PDFs and images are supported; you may encounter an error if they are not; provide that information to the user if required."""
```

---

## Step 4: Create the OCR Function
Define a helper function that takes a URL, attempts to perform OCR using the Mistral OCR API, and returns the extracted markdown text.

```python
def _perform_ocr(url: str) -> str:
    try:
        # First, try to process the URL as a document (PDF)
        response = client.ocr.process(
            model=ocr_model,
            document={
                "type": "document_url",
                "document_url": url
            }
        )
    except Exception:
        try:
            # If PDF fails, try as an image
            response = client.ocr.process(
                model=ocr_model,
                document={
                    "type": "image_url",
                    "image_url": url
                }
            )
        except Exception as e:
            # Return the error if both attempts fail
            return str(e)
    
    # Format the response: each page's markdown under a header
    pages_markdown = []
    for i in range(len(response.pages)):
        pages_markdown.append(f"### Page {i+1}\n{response.pages[i].markdown}")
    return "\n\n".join(pages_markdown)
```

---

## Step 5: Build the Tool Function
Create the main tool function that iterates over a list of URLs, calls the OCR function for each, and returns a consolidated string of all document contents.

```python
def open_urls(urls: list) -> str:
    contents = "# Documents"
    for url in urls:
        contents += f"\n\n## URL: {url}\n{_perform_ocr(url)}"
    return contents
```

---

## Step 6: Define the Tool Schema
Following Mistral's [Tool Usage documentation](https://docs.mistral.ai/capabilities/function_calling/), define the tool schema that will be passed to the model.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "open_urls",
            "description": "Open URLs websites (PDFs and Images) and perform OCR on them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "description": "The URLs list.",
                    }
                },
                "required": ["urls"],
            },
        },
    },
]
```

---

## Step 7: Map Function Names to Implementations
Create a dictionary to map the tool name to the actual Python function.

```python
names_to_functions = {
    'open_urls': open_urls
}
```

---

## Step 8: Implement the Interactive Chat Loop
Now, create the main loop that handles user input, tool calls, and model responses. The model will automatically call the `open_urls` tool when it detects a URL in the conversation.

```python
# Initialize the conversation with the system prompt
messages = [{"role": "system", "content": system}]

while True:
    # Get user input
    user_input = input("User > ")
    if user_input == "quit":
        break
    
    # Add user message to the conversation history
    messages.append({"role": "user", "content": user_input})

    # Handle potential tool calls in a loop
    while True:
        # Call the model with the current messages and available tools
        response = client.chat.complete(
            model=text_model,
            messages=messages,
            temperature=0,
            tools=tools
        )
        
        # Extract the assistant's message and any tool calls
        assistant_message = response.choices[0].message
        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": assistant_message.tool_calls
        })

        # If a tool was called, execute it and continue the loop
        if assistant_message.tool_calls:
            tool_call = assistant_message.tool_calls[0]
            function_name = tool_call.function.name
            function_params = json.loads(tool_call.function.arguments)
            
            # Call the corresponding function
            function_result = names_to_functions[function_name](**function_params)
            
            # Append the tool result to the conversation
            messages.append({
                "role": "tool",
                "name": function_name,
                "content": function_result,
                "tool_call_id": tool_call.id
            })
        else:
            # No tool was called; break the inner loop and print the final response
            break

    print("Assistant >", assistant_message.content)
```

---

## Step 9: Test the System
Run the script and try the following example prompts:

1. **Summarize a research paper:**  
   `Could you summarize what this research paper talks about? https://arxiv.org/pdf/2410.07073`

2. **Extract text from an image:**  
   `What is written here: https://jeroen.github.io/images/testocr.png`

The model will automatically detect the URLs, call the `open_urls` tool, perform OCR, and use the extracted text to answer your question.

---

## Conclusion
You have successfully built a document comprehension system using Mistral's Tool Usage and OCR capabilities. This approach allows any LLM to understand and reason about documents fetched from URLs, enabling powerful applications like summarization, Q&A, and data extraction.

To extend this system, you could:
- Add more tools (e.g., web search, database queries).
- Customize the OCR processing for specific document types.
- Integrate the workflow into a larger application or API.

For further reading, explore Mistral's [Tool Usage](https://docs.mistral.ai/capabilities/function_calling/) and [Document Understanding](https://docs.mistral.ai/capabilities/OCR/document_understanding/) documentation.