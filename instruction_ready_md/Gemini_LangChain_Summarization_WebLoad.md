# Summarize Large Documents with Gemini and LangChain

## Overview

This guide demonstrates how to build a document summarization application using Google's Gemini models and the LangChain framework. You'll learn to load web content, structure prompts, and chain components together to generate concise summaries.

## Prerequisites

Before you begin, ensure you have:
- A Google AI Studio API key.
- Basic knowledge of Python.

## Setup

### 1. Install Required Packages

Install the necessary LangChain libraries and the Google Gemini integration.

```bash
pip install langchain-core==0.1.23
pip install langchain==0.1.1
pip install langchain-google-genai==0.0.6
pip install -U langchain-community==0.0.20
```

### 2. Import Libraries

Import the core modules you'll use throughout this tutorial.

```python
from langchain import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain_google_genai import ChatGoogleGenerativeAI
```

### 3. Configure Your API Key

Set your Gemini API key as an environment variable. Replace `'YOUR_API_KEY'` with your actual key.

```python
import os

# Set your API key
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
```

## Tutorial: Summarize a Web Article

Follow these steps to create a summarization pipeline.

### Step 1: Load the Document

Use LangChain's `WebBaseLoader` to fetch and parse content from a webpage.

```python
# Initialize the loader with a target URL
loader = WebBaseLoader("https://blog.google/technology/ai/google-gemini-ai/#sundar-note")

# Load the document
docs = loader.load()
```

### Step 2: Initialize the Gemini Model

Create an instance of the Gemini model. This example uses the efficient `gemini-2.5-flash` model, which is well-suited for summarization tasks.

```python
# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Optional: Configure model parameters like temperature
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
```

### Step 3: Create Prompt Templates

Define two prompt templates:
1.  **Document Prompt:** Extracts raw text from the loaded document.
2.  **LLM Prompt:** Instructs the model to summarize the provided text.

```python
# Template to extract page content from the document
doc_prompt = PromptTemplate.from_template("{page_content}")

# Template to instruct the model for summarization
llm_prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""
llm_prompt = PromptTemplate.from_template(llm_prompt_template)

print(f"LLM Prompt variables: {llm_prompt.input_variables}")
```

### Step 4: Build the Summarization Chain

Construct a **Stuff Documents Chain** using LangChain Expression Language (LCEL). This chain combines all document content into a single prompt, sends it to the LLM, and parses the output.

```python
# Create the summarization pipeline
stuff_chain = (
    # Step 1: Extract and combine text from all document parts
    {
        "text": lambda docs: "\n\n".join(
            format_document(doc, doc_prompt) for doc in docs
        )
    }
    | llm_prompt          # Step 2: Format the combined text into the prompt
    | llm                 # Step 3: Send the prompt to the Gemini model
    | StrOutputParser()   # Step 4: Parse the model's text response
)
```

**How the Chain Works:**
1.  The lambda function extracts the `page_content` from each document segment and joins them with blank lines.
2.  This combined text is injected into the `llm_prompt` template.
3.  The fully formed prompt is sent to the Gemini model for processing.
4.  The `StrOutputParser()` ensures the response is returned as a clean string.

### Step 5: Generate the Summary

Invoke the chain with the loaded documents to produce the final summary.

```python
# Run the chain
summary = stuff_chain.invoke(docs)
print(summary)
```

**Example Output:**
> Google has introduced Gemini, its most capable AI model yet. Gemini is multimodal, meaning it can understand and interact with various forms of information, including text, code, audio, images, and video. It comes in three sizes: Ultra (for complex tasks), Pro (for a wide range of tasks), and Nano (for on-device tasks). Gemini surpasses existing models in performance benchmarks across various domains, including natural language understanding, reasoning, and coding.
>
> Google emphasizes Gemini's safety and responsibility features, including comprehensive bias and toxicity evaluation, adversarial testing, and collaboration with external experts.
>
> Gemini is being integrated into various Google products, such as Bard, Pixel, Search, and Ads, and will be available to developers through APIs.
>
> The release of Gemini marks a significant milestone in AI development, opening up new possibilities for innovation and enhancing human capabilities in various areas.

## Conclusion

You have successfully built a document summarization application. This pipeline can be adapted to summarize content from various sources (PDFs, databases) by swapping the `WebBaseLoader` with other LangChain document loaders. Experiment with different prompt templates and model parameters (like `temperature`) to adjust the style and creativity of your summaries.