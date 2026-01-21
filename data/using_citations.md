# Implementing Citations with Claude API

## Overview

Claude's citation feature enables the model to provide detailed citations when answering questions about documents. This capability helps users track and verify information sources in LLM-powered applications. Citations are supported on `claude-sonnet-4-5` and `claude-3-5-haiku-20241022`.

Compared to prompt-based citation techniques, this feature offers several advantages:
- Reduces output tokens and costs by avoiding full quotes
- Ensures citations only point to valid provided sources
- Provides higher recall and precision in citation generation

## Prerequisites

First, install the required library and set up your Anthropic client:

```bash
pip install anthropic
```

```python
import json
import os
import anthropic

# Set your API key
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
# ANTHROPIC_API_KEY = ""  # Or set your key directly here

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
```

## Understanding Document Types

Claude supports three document types for citations, each with different location formats:

1. **Plain text documents** → Character location format
2. **PDF documents** → Page location format  
3. **Custom content documents** → Content block location format

## 1. Working with Plain Text Documents

Plain text documents are automatically chunked into sentences. Claude can cite these sentences appropriately, grouping multiple sentences together when needed, but won't cite text smaller than a sentence.

### Step 1: Prepare Your Documents

Let's create a customer support chatbot for "PetWorld" using help center articles:

```python
# Read all help center articles and create a list of documents
articles_dir = "./data/help_center_articles"
documents = []

for filename in sorted(os.listdir(articles_dir)):
    if filename.endswith(".txt"):
        with open(os.path.join(articles_dir, filename)) as f:
            content = f.read()
            # Split into title and body
            title_line, body = content.split("\n", 1)
            title = title_line.replace("title: ", "")
            documents.append(
                {
                    "type": "document",
                    "source": {"type": "text", "media_type": "text/plain", "data": body},
                    "title": title,
                    "citations": {"enabled": True},
                }
            )
```

### Step 2: Query Claude with Citations

Now, let's ask a question about order tracking:

```python
QUESTION = "I just checked out, where is my order tracking number? Track package is not available on the website yet for my order."

response = client.messages.create(
    model="claude-sonnet-4-5",
    temperature=0.0,
    max_tokens=1024,
    system="You are a customer support bot working for PetWorld. Your task is to provide short, helpful answers to user questions. Since you are in a chat interface avoid providing extra details. You will be given access to PetWorld's help center articles to help you answer questions.",
    messages=[
        {"role": "user", "content": documents},
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Here is the user's question: {QUESTION}"}],
        },
    ],
)
```

### Step 3: Visualize the Raw Response

To understand the structure of Claude's response with citations, use this visualization function:

```python
def visualize_raw_response(response):
    raw_response = {"content": []}

    print("\n" + "=" * 80 + "\nRaw response:\n" + "=" * 80)

    for content in response.content:
        if content.type == "text":
            block = {"type": "text", "text": content.text}
            if hasattr(content, "citations") and content.citations:
                block["citations"] = []
                for citation in content.citations:
                    citation_dict = {
                        "type": citation.type,
                        "cited_text": citation.cited_text,
                        "document_title": citation.document_title,
                    }
                    if citation.type == "page_location":
                        citation_dict.update(
                            {
                                "start_page_number": citation.start_page_number,
                                "end_page_number": citation.end_page_number,
                            }
                        )
                    block["citations"].append(citation_dict)
            raw_response["content"].append(block)

    return json.dumps(raw_response, indent=2)

print(visualize_raw_response(response))
```

### Step 4: Create User-Friendly Citation Formatting

For better readability, transform Claude's structured citations into a numbered reference format:

```python
def visualize_citations(response):
    """
    Takes a response object and returns a string with numbered citations.
    Example output: "here is the plain text answer [1][2] here is some more text [3]"
    with a list of citations below.
    """
    # Dictionary to store unique citations
    citations_dict = {}
    citation_counter = 1

    # Final formatted text
    formatted_text = ""
    citations_list = []

    print("\n" + "=" * 80 + "\nFormatted response:\n" + "=" * 80)

    for content in response.content:
        if content.type == "text":
            text = content.text
            if hasattr(content, "citations") and content.citations:
                # Sort citations by their appearance in the text
                def get_sort_key(citation):
                    if hasattr(citation, "start_char_index"):
                        return citation.start_char_index
                    elif hasattr(citation, "start_page_number"):
                        return citation.start_page_number
                    elif hasattr(citation, "start_block_index"):
                        return citation.start_block_index
                    return 0  # fallback

                sorted_citations = sorted(content.citations, key=get_sort_key)

                # Process each citation
                for citation in sorted_citations:
                    doc_title = citation.document_title
                    cited_text = citation.cited_text.replace("\n", " ").replace("\r", " ")
                    # Remove any multiple spaces that might have been created
                    cited_text = " ".join(cited_text.split())

                    # Create a unique key for this citation
                    citation_key = f"{doc_title}:{cited_text}"

                    # If this is a new citation, add it to our dictionary
                    if citation_key not in citations_dict:
                        citations_dict[citation_key] = citation_counter
                        citations_list.append(
                            f'[{citation_counter}] "{cited_text}" found in "{doc_title}"'
                        )
                        citation_counter += 1

                    # Add the citation number to the text
                    citation_num = citations_dict[citation_key]
                    text += f" [{citation_num}]"

            formatted_text += text

    # Combine the formatted text with the citations list
    final_output = formatted_text + "\n\n" + "\n".join(citations_list)
    return final_output

formatted_response = visualize_citations(response)
print(formatted_response)
```

This creates output similar to academic papers, with numbered citations in the text and a reference list at the end.

## 2. Working with PDF Documents

PDF citations reference specific page numbers, making it easy to track information sources. Text is automatically chunked into sentences, and citations include 1-indexed page numbers.

### Step 1: Prepare and Query a PDF

```python
import base64

# Read and encode the PDF
pdf_path = "data/Constitutional AI.pdf"
with open(pdf_path, "rb") as f:
    pdf_data = base64.b64encode(f.read()).decode()

pdf_response = client.messages.create(
    model="claude-sonnet-4-5",
    temperature=0.0,
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_data},
                    "title": "Constitutional AI Paper",
                    "citations": {"enabled": True},
                },
                {"type": "text", "text": "What is the main idea of Constitutional AI?"},
            ],
        }
    ],
)

print(visualize_raw_response(pdf_response))
print(visualize_citations(pdf_response))
```

## 3. Working with Custom Content Documents

Custom content documents give you complete control over citation granularity. You define your own chunks of any size, which is useful for documents that don't work well with sentence chunking.

### Step 1: Create Custom Content Documents

```python
# Read all help center articles and create a list of custom content documents
articles_dir = "./data/help_center_articles"
documents = []

for filename in sorted(os.listdir(articles_dir)):
    if filename.endswith(".txt"):
        with open(os.path.join(articles_dir, filename)) as f:
            content = f.read()
            # Split into title and body
            title_line, body = content.split("\n", 1)
            title = title_line.replace("title: ", "")

            documents.append(
                {
                    "type": "document",
                    "source": {"type": "content", "content": [{"type": "text", "text": body}]},
                    "title": title,
                    "citations": {"enabled": True},
                }
            )

QUESTION = "I just checked out, where is my order tracking number? Track package is not available on the website yet for my order."

custom_content_response = client.messages.create(
    model="claude-sonnet-4-5",
    temperature=0.0,
    max_tokens=1024,
    system="You are a customer support bot working for PetWorld. Your task is to provide short, helpful answers to user questions. Since you are in a chat interface avoid providing extra details. You will be given access to PetWorld's help center articles to help you answer questions.",
    messages=[
        {"role": "user", "content": documents},
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Here is the user's question: {QUESTION}"}],
        },
    ],
)

print(visualize_raw_response(custom_content_response))
print(visualize_citations(custom_content_response))
```

Notice that with custom content documents, the `cited_text` will be the entire article rather than individual sentences.

## 4. Using the Context Field

The `context` field allows you to provide additional information that Claude can use when generating responses, but that won't be cited. This is useful for metadata, contextual retrieval, or usage instructions.

### Step 1: Add Context to a Document

```python
# Create a document with context field
document = {
    "type": "document",
    "source": {
        "type": "text",
        "media_type": "text/plain",
        "data": "PetWorld offers a loyalty program where customers earn 1 point for every dollar spent. Once you accumulate 100 points, you'll receive a $5 reward that can be used on your next purchase. Points expire 12 months after they are earned. You can check your point balance in your account dashboard or by asking customer service.",
    },
    "title": "Loyalty Program Details",
    "context": "WARNING: This article has not been updated in 12 months. Content may be out of date. Be sure to inform the user this content may be incorrect after providing guidance.",
    "citations": {"enabled": True},
}

QUESTION = "How does PetWorld's loyalty program work? When do points expire?"

context_response = client.messages.create(
    model="claude-sonnet-4-5",
    temperature=0.0,
    max_tokens=1024,
    messages=[{"role": "user", "content": [document, {"type": "text", "text": QUESTION}]}],
)

print(visualize_raw_response(context_response))
print(visualize_citations(context_response))
```

Claude will use the context information to inform its response (like warning about outdated content) but won't cite the context field itself.

## 5. PDF Highlighting with PyMuPDF

While PDF citations return page numbers, you can use third-party libraries to highlight the cited text in the PDF. This example uses PyMuPDF to create an annotated PDF.

### Step 1: Install PyMuPDF

```bash
pip install PyMuPDF
```

### Step 2: Query and Highlight PDF Citations

```python
import fitz  # PyMuPDF

# Setup paths and read PDF
pdf_path = "data/Amazon-com-Inc-2023-Shareholder-Letter.pdf"
output_pdf_path = "data/Amazon-com-Inc-2023-Shareholder-Letter-highlighted.pdf"

# Read and encode the PDF
with open(pdf_path, "rb") as f:
    pdf_data = base64.b64encode(f.read()).decode()

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": pdf_data},
                    "title": "Amazon 2023 Shareholder Letter",
                    "citations": {"enabled": True},
                },
                {
                    "type": "text",
                    "text": "What was Amazon's total revenue in 2023 and how much did it grow year-over-year?",
                },
            ],
        }
    ],
)

print(visualize_raw_response(response))

# Collect PDF citations
pdf_citations = []
for content in response.content:
    if hasattr(content, "citations") and content.citations:
        for citation in content.citations:
            if citation.type == "page_location":
                pdf_citations.append(citation)

doc = fitz.open(pdf_path)

# Process each citation
for citation in pdf_citations:
    if citation.type == "page_location":
        text_to_find = citation.cited_text.replace("\u0002", "")
        start_page = citation.start_page_number - 1  # Convert to 0-based index
        end_page = citation.end_page_number - 2

        # Process each page in the citation range
        for page_num in range(start_page, end_page + 1):
            page = doc[page_num]

            text_instances = page.search_for(text_to_find.strip())

            if text_instances:
                print(f"Found cited text on page {page_num + 1}")
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors({"stroke": (1, 1, 0)})  # Yellow highlight
                    highlight.update()
            else:
                print(f"{text_to_find} not found on page {page_num + 1}")

# Save the new PDF
doc.save(output_pdf_path)
doc.close()

print(f"\nCreated highlighted PDF at: {output_pdf_path}")
```

## Summary

Claude's citation feature provides a powerful way to build transparent, verifiable AI applications. By following this guide, you can:

1. Implement citations with plain text, PDF, and custom content documents
2. Format citations for user-friendly display
3. Add contextual information without affecting citations
4. Create highlighted PDFs to visualize cited content

Remember to choose the appropriate document type based on your needs: plain text for sentence-level citations, PDF for page references, or custom content for granular control over citation units.