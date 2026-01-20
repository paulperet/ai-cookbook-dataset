# Chat with your Documents with built-in Document QnA

---

## Use our built-in Document QnA feature

Optical Character Recognition (OCR) transforms text-based documents and images into pure text outputs and markdown. By leveraging this feature, you can enable any Large Language Model (LLM) to reliably understand documents efficiently and cost-effectively.

In this guide, we will demonstrate how to use OCR with our models to discuss any text-based document, whether it's a PDF, photo, or screenshot, via URLs and our built-in feature.

---

### Method
This method will make use of our built-in feature that leverages OCR, we will extract the URLs with regex and call our models with this feature.

## Built-In
Mistral provides a built-in feature that leverages OCR with all models. By providing a URL pointing to a document, you can extract text data that will be provided to the model.

Following, there is a simple, quick, example of how to make use of this feature by extracting PDF URLs with regex and uploading them as a `document_url`.

Learn more about Document QnA [here](https://docs.mistral.ai/capabilities/OCR/document_qna/).

### Setup
First, let's install `mistralai`

```python
!pip install mistralai
```

[Collecting mistralai, ..., Successfully installed eval-type-backport-0.2.2 mistralai-1.7.0]

We can now set up our client. You can create an API key on our [Plateforme](https://console.mistral.ai/api-keys/).

```python
from mistralai import Mistral

api_key = "API_KEY"
client = Mistral(api_key=api_key)
text_model = "mistral-small-latest"
```

### System and Regex
Let's define a simple system prompt, since there is no tool call required for this demo we can be fairly straightforward.

```python
system = "You are an AI Assistant with document understanding via URLs. You may be provided with URLs, followed by their corresponding OCR."
```

To extract the URLs, we will use regex to extract any URL pattern from the user query.

*Note: We will assume there will only be PDF files for simplicity.*

```python
import re

def extract_urls(text: str) -> list:
    url_pattern = r'\b((?:https?|ftp)://(?:www\.)?[^\s/$.?#].[^\s]*)\b'
    urls = re.findall(url_pattern, text)
    return urls

# Example
extract_urls("Hi there, you can visit our docs in our website https://docs.mistral.ai/, we cannot wait to see what you will build with us.")
```

### Test
We can now try it out, we setup so that for each query all urls are extracted and added to the query properly.

#### Example Prompts ( PDFs )
- Could you summarize what this research paper talks about? https://arxiv.org/pdf/2410.07073
- Explain this architecture: https://arxiv.org/abs/2401.04088

```python
import json

messages = [{"role": "system", "content": system}]
while True:
    user_input = input("User > ")
    if user_input.lower() == "quit":
        break

    # Extract URLs from the user input, assuming they are always PDFs
    document_urls = extract_urls(user_input)
    user_message_content = [{"type": "text", "text": user_input}]
    for url in document_urls:
        user_message_content.append({"type": "document_url", "document_url": url})
    messages.append({"role": "user", "content": user_message_content})

    # Send the messages to the model and get a response
    response = client.chat.complete(
        model=text_model,
        messages=messages,
        temperature=0
    )
    messages.append({"role": "assistant", "content": response.choices[0].message.content})

    print("Assistant >", response.choices[0].message.content)
```