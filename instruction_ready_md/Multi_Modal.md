# Multi-Modal Image Understanding with Anthropic and LlamaIndex

This guide demonstrates how to use the Anthropic Multi-Modal LLM through LlamaIndex to analyze and extract structured information from images.

## Prerequisites

First, install the required packages.

```bash
pip install llama-index
pip install llama-index-multi-modal-llms-anthropic
pip install llama-index-embeddings-huggingface
pip install llama-index-vector-stores-qdrant
pip install matplotlib
```

## Setup

Set your Anthropic API key as an environment variable.

```python
import os

os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"
```

## Step 1: Load a Local Image for Analysis

We'll start by downloading a sample image and loading it for analysis.

```python
# Download a sample image
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/images/prometheus_paper_card.png' -O 'prometheus_paper_card.png'
```

Now, let's load the image document using LlamaIndex's `SimpleDirectoryReader`.

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal

# Load the image as a document
image_documents = SimpleDirectoryReader(input_files=["prometheus_paper_card.png"]).load_data()

# Initialize the Anthropic Multi-Modal LLM
anthropic_mm_llm = AnthropicMultiModal(max_tokens=300)
```

## Step 2: Generate a Textual Description

With the model initialized, you can now prompt it to describe the image content.

```python
response = anthropic_mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_documents,
)

print(response)
```

**Expected Output:**
```
The image is a diagram titled "Prometheus: Inducing Fine-Grained Evaluation Capability In Language Models". It outlines the key components and workflow of the Prometheus system.
...
```

## Step 3: Analyze Images from URLs

You can also analyze images directly from URLs without downloading them locally.

First, let's define a helper function to load image URLs.

```python
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls

image_urls = [
    "https://venturebeat.com/wp-content/uploads/2024/03/Screenshot-2024-03-04-at-12.49.41%E2%80%AFAM.png",
]

# Load the image from the URL as a document
image_url_documents = load_image_urls(image_urls)
```

Now, query the model with the remote image.

```python
response = anthropic_mm_llm.complete(
    prompt="Describe the images as an alternative text",
    image_documents=image_url_documents,
)

print(response)
```

**Expected Output:**
```
The image shows a table comparing the benchmark scores of various Claude 3 AI models...
```

## Step 4: Extract Structured Data from an Image

For more advanced use cases, you can extract structured, typed information from images using Pydantic models.

First, download a sample image containing financial data.

```bash
!wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/images/ark_email_sample.PNG' -O 'ark_email_sample.png'
```

Define the Pydantic schema that represents the data structure you want to extract.

```python
from pydantic import BaseModel

class TickerInfo(BaseModel):
    """Information for a single stock ticker."""
    direction: str
    ticker: str
    company: str
    shares_traded: int
    percent_of_total_etf: float

class TickerList(BaseModel):
    """List of stock tickers for a fund."""
    fund: str
    tickers: list[TickerInfo]
```

Load the target image.

```python
from llama_index.core import SimpleDirectoryReader

image_documents = SimpleDirectoryReader(input_files=["ark_email_sample.png"]).load_data()
```

Now, create a `MultiModalLLMCompletionProgram` to parse the image into the defined schema.

```python
from llama_index.core.program import MultiModalLLMCompletionProgram

prompt_template_str = """\
Can you get the stock information in the image \
and return the answer? Pick just one fund.

Make sure the answer is a JSON format corresponding to a Pydantic schema. The Pydantic schema is given below.
"""

llm_program = MultiModalLLMCompletionProgram.from_defaults(
    output_cls=TickerList,
    image_documents=image_documents,
    prompt_template_str=prompt_template_str,
    multi_modal_llm=anthropic_mm_llm,
    verbose=True,
)
```

Run the program to extract the structured data.

```python
response = llm_program()
print(response)
```

**Expected Output:**
```
fund='ARKK' tickers=[TickerInfo(direction='Buy', ticker='TSLA', company='TESLA INC', shares_traded=93664, percent_of_total_etf=0.2453)]
```

## Summary

You've successfully used the Anthropic Multi-Modal LLM to:
1. Generate descriptive text from local images.
2. Analyze images directly from URLs.
3. Extract structured, typed data from images using Pydantic models.

This workflow enables powerful image understanding and data extraction capabilities for building advanced multi-modal AI applications.