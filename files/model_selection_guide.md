# Practical Guide for Model Selection for Real‑World Use Cases

## Purpose & Audience

This cookbook serves as your practical guide to selecting, prompting, and deploying the right OpenAI model (between GPT 4.1, o3, and o4-mini) for specific workloads. Instead of exhaustive documentation, we provide actionable decision frameworks and real-world examples that help Solutions Engineers, Technical Account Managers, Partner Architects, and semi-technical practitioners quickly build working solutions. The content focuses on current model capabilities, vertical-specific implementations, and today's industry needs, with clear pathways from model selection to production deployment. Each section offers concise, adaptable code examples that you can immediately apply to your use cases while pointing to existing resources for deeper dives into specific topics.

> Note: The below prescriptive guidance and experimentation has been conducted with latest SOTA models available today. These metrics are bound to change in the future with different scenarios and timeline into consideration.

## How to Use This Cookbook

This cookbook is organized into distinct sections to help you quickly find the information you need. Each section covers a specific aspect of model selection, implementation, and deployment.

1. **[Purpose & Audience](#purpose-audience)**: An overview of who this cookbook is for and what it covers.
2. **[Model Guide](#model-guide)**: A quick reference to help you select the right model for your needs, including model comparisons and evolution diagrams based on mapping different use-case scenarios.
3. **Use Cases**:
   - **[3A. Long-Context RAG for Legal Q&A](#3a-use-case-long-context-rag-for-legal-qa)**: Building an agentic system to answer questions from complex legal documents.
   - **[3B. AI Co-Scientist for Pharma R&D](#3b-use-case-ai-co-scientist-for-pharma-rd)**: Accelerating experimental design in pharmaceutical research with multi-agent systems.
   - **[3C. Insurance Claim Processing](#3c-use-case-insurance-claim-processing)**: Digitizing and validating handwritten insurance forms with vision and reasoning.
4. **[Prototype to Production](#prototype-to-production)**: A checklist to help you transition from prototype to production.
5. **[Adaptation Decision Tree](#adaptation-decision-tree)**: A flowchart to guide your model selection based on specific requirements.
6. **[Appendices](#appendices)**: Reference materials including pricing, latency, prompt patterns, and links to external resources.

For quick decisions, focus on the Model Guide and Adaptation Decision Tree sections. For implementation details, explore the specific use cases relevant to your needs.


================================================================================



## Model Guide

## 2.1 Model‑Intro Matrix

| Model | Core strength | Ideal first reach‑for | Watch‑outs | Escalate / Downgrade path |
| :---- | :---- | :---- | :---- | :---- |
| GPT‑4o | Real‑time voice / vision chat | Live multimodal agents | Slightly below 4.1 on text SOTA (state-of-the-art) | Need deep reasoning → o4‑mini |
| GPT‑4.1 | 1 M‑token text accuracy king | Long‑doc analytics, code review | Cannot natively reason; higher cost than minis | Tight budget → 4.1‑mini / nano |
| o3 | Deep tool‑using agent | High‑stakes, multi‑step reasoning | Latency & price | Cost/latency → o4‑mini |
| o4‑mini | Cheap, fast reasoning | High‑volume "good‑enough" logic | Depth ceiling vs o3 | Accuracy critical → o3 |

*(Full price and utility table → [Section 6.1](#appendices))*

## 2.2 Model Evolution at a Glance

OpenAI's model lineup has evolved to address specialized needs across different dimensions. These diagrams showcase the current model families and their relationships.

### Fundamental Differences: "o-series" vs "GPT" Models

OpenAI offers two distinct model families, each with unique strengths:

- **GPT Models (4o, 4.1)**: Optimized for general-purpose tasks with excellent instruction following. GPT-4.1 excels with long contexts (1M tokens) while GPT-4o has variants for realtime speech, text-to-speech, and speech-to-text. GPT-4.1 also comes in a mini, and nano variant, while GPT-4o has a mini variant. These variants are cheaper and faster than their full-size counterparts.

- **o-series Models (o3, o4-mini)**: Specialized for deep reasoning and step-by-step problem solving. These models excel at complex, multi-stage tasks requiring logical thinking and tool use. Choose these when accuracy and reasoning depth are paramount. These models also have an optional `reasoning_effort` parameter (that can be set to `low`, `medium`, or `high`), which allows users to control the amount of tokens used for reasoning.

### OpenAI Model Evolution 

### Key Characteristics

- **GPT-4.1 Family**: Optimized for long context processing with 1M token context window.
- **o3**: Specialized for deep multi-step reasoning. 
- **o4-mini**: Combines reasoning capabilities with vision at lower cost.

Each model excels in different scenarios, with complementary strengths that can be combined for complex workflows.

In this cookbook we only experimented with the GPT-4.1 series models, o3, and o4-mini. We didn't experiment with the GPT-4o series models.

================================================================================



## 3A. Use Case: Long-Context RAG for Legal Q&A

## 🗂️ TL;DR Matrix

This table summarizes the core technology choices and their rationale for **this specific Long-Context Agentic RAG implementation**.

| Layer | Choice | Utility |
| :---- | :---- | :---- |
| **Chunking** | Sentence-aware Splitter | Splits document into 20 equal chunks, respecting sentence boundaries. |
| **Routing** | `gpt-4.1-mini` | Uses natural language understanding to identify relevant chunks without embedding index. |
| **Path Selection** | `select(ids=[...])` and `scratchpad(text="...")` | Records reasoning while drilling down through document hierarchy. |
| **Citation** | Paragraph-level | Balances precision with cost; provides meaningful context for answers. |
| **Synthesis** | `gpt-4.1` (Structured Output) | Generates answers directly from selected paragraphs with citations. |
| **Verification** | `o4-mini` (LLM-as-Judge) | Validates factual accuracy and citation correctness. |

*Note: Prices and model identifiers accurate as of April 2025, subject to change.*

This section outlines the construction of a Retrieval-Augmented Generation (RAG) system designed to accurately answer questions about complex and lengthy procedural texts, using the *Trademark Trial and Appeal Board Manual of Procedure (TBMP)* as a representative case. The TBMP is an essential legal resource detailing the procedures governing trademark litigation before the USPTO's Trademark Trial and Appeal Board, and is frequently consulted by intellectual property attorneys and legal professionals. By leveraging the latest OpenAI models, the system enhances understanding and interpretability of dense legal content, enabling precise, contextually aware responses through advanced language understanding and dynamic retrieval capabilities.

These approaches can also be applied to other use cases that require precise information retrieval from complex documentation, such as healthcare compliance manuals, financial regulatory frameworks, or technical documentation systems where accuracy, citation, and auditability are mission-critical requirements.

## 1\. Scenario Snapshot

* **Corpus:** The primary document is the [Trademark Trial and Appeal Board Manual of Procedure (TBMP, 2024 version)](https://www.uspto.gov/sites/default/files/documents/tbmp-Master-June2024.pdf). This manual contains detailed procedural rules and guidelines, coming to 1194 pages total.  
* **Users:** The target users are intellectual property (IP) litigation associates and paralegals who need quick, accurate answers to procedural questions based *only* on the TBMP.  
* **Typical Asks:** Users pose questions requiring synthesis and citation, such as:  
  1. "What are the requirements for filing a motion to compel discovery according to the TBMP?"  
  2. "What deadlines apply to discovery conferences as specified in the manual?"  
  3. "Explain how the Board handles claims of attorney-client privilege during depositions according to the TBMP."  
  4. "Enumerate the Fed. R. Civ. P. 11 sanctions the Board can invoke according to the TBMP."  

*Note: Depending on your specific deployment environment, you may need to adapt some implementation steps to match your infrastructure requirements.*

> While OpenAI's File Search tool offers a good starting point for many use cases, this section introduces a different approach that takes advantage of million-token context windows to process large documents without any preprocessing or vector database. The agentic approach described here enables zero-latency ingestion, dynamic granularity of retrieval, and fine-grained citation traceability.

## 2\. Agentic RAG Flow

Before diving into the implementation, let's understand the overall approach:

1. **Load the entire document** into the context window
2. **Split into 20 chunks** that respect sentence boundaries
3. **Ask the model** which chunks might contain relevant information
4. **Drill down** into selected chunks by splitting them further
5. **Repeat** until we reach paragraph-level content
6. **Generate an answer** based on the selected paragraphs
7. **Verify the answer** for factual accuracy

This hierarchical navigation approach mimics how a human might skim a document, focus on relevant chapters, then specific sections, and finally read only the most relevant paragraphs.

## Agentic RAG System: Model Usage

| Process Stage | Model Used | Purpose |
|---------------|------------|---------|
| Initial Routing | `gpt-4.1-mini` | Identifies which document chunks might contain relevant information |
| Hierarchical Navigation | `gpt-4.1-mini` | Continues drilling down to find most relevant paragraphs |
| Answer Generation | `gpt-4.1` | Creates structured response with citations from selected paragraphs |
| Answer Verification | `o4-mini` | Validates factual accuracy and proper citation usage |

This zero-preprocessing approach leverages large context windows to navigate documents on-the-fly, mimicking how a human would skim a document to find relevant information. 

## 3\. Implementation

Let's implement this approach step by step.

Start by installing the required packages.

```python
%pip install tiktoken pypdf nltk openai pydantic --quiet
```

### 3.1 Document Loading

First, let's load the document and check its size. For this guide, we'll focus on sections 100-900, which cover the core procedural aspects through Review of Decision of Board. Sections 1000 and beyond (Interferences, Concurrent Use Proceedings, Ex Parte Appeals) are specialized procedures outside our current scope.

```python
import requests
from io import BytesIO
from pypdf import PdfReader
import re
import tiktoken
from nltk.tokenize import sent_tokenize
import nltk
from typing import List, Dict, Any

# Download nltk data if not already present
nltk.download('punkt_tab')

def load_document(url: str) -> str:
    """Load a document from a URL and return its text content."""
    print(f"Downloading document from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    pdf_bytes = BytesIO(response.content)
    pdf_reader = PdfReader(pdf_bytes)
    
    full_text = ""
    

    max_page = 920  # Page cutoff before section 1000 (Interferences)
    for i, page in enumerate(pdf_reader.pages):
        if i >= max_page:
            break
        full_text += page.extract_text() + "\n"
    
    # Count words and tokens
    word_count = len(re.findall(r'\b\w+\b', full_text))
    
    tokenizer = tiktoken.get_encoding("o200k_base")
    token_count = len(tokenizer.encode(full_text))
    
    print(f"Document loaded: {len(pdf_reader.pages)} pages, {word_count} words, {token_count} tokens")
    return full_text

# Load the document
tbmp_url = "https://www.uspto.gov/sites/default/files/documents/tbmp-Master-June2024.pdf"
document_text = load_document(tbmp_url)

# Show the first 500 characters
print("\nDocument preview (first 500 chars):")
print("-" * 50)
print(document_text[:500])
print("-" * 50)
```

We can see that the document is over 900k tokens long! While we could fit that into GPT 4.1's context length, we also want to have verifiable citations, so we're going to proceed with a recursive chunking strategy.

### 3.2 Improved 20-Chunk Splitter with Minimum Token Size

Now, let's create an improved function to split the document into 20 chunks, ensuring each has a minimum token size and respecting sentence boundaries.

> 20 is an empirically chosen number for this specific document/task and it might need tuning for other documents based on size and structure (The higher the number, the more fine-grained the chunks). The key principle here however is splitting sections of the document up, in order to let the language model decide relevant components. This same reasoning also applies to the `max_depth` parameter which will be introduced later on in the cookbook.

```python
# Global tokenizer name to use consistently throughout the code
TOKENIZER_NAME = "o200k_base"

def split_into_20_chunks(text: str, min_tokens: int = 500) -> List[Dict[str, Any]]:
    """
    Split text into up to 20 chunks, respecting sentence boundaries and ensuring
    each chunk has at least min_tokens (unless it's the last chunk).
    
    Args:
        text: The text to split
        min_tokens: The minimum number of tokens per chunk (default: 500)
    
    Returns:
        A list of dictionaries where each dictionary has:
        - id: The chunk ID (0-19)
        - text: The chunk text content
    """
    # First, split the text into sentences
    sentences = sent_tokenize(text)
    
    # Get tokenizer for counting tokens
    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    
    # Create chunks that respect sentence boundaries and minimum token count
    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0
    
    for sentence in sentences:
        # Count tokens in this sentence
        sentence_tokens = len(tokenizer.encode(sentence))
        
        # If adding this sentence would make the chunk too large AND we already have the minimum tokens,
        # finalize the current chunk and start a new one
        if (current_chunk_tokens + sentence_tokens > min_tokens * 2) and current_chunk_tokens >= min_tokens:
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                "id": len(chunks),  # Integer ID instead of string
                "text": chunk_text
            })
            current_chunk_sentences = [sentence]
            current_chunk_tokens = sentence_tokens
        else:
            # Add this sentence to the current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens
    
    # Add the last chunk if there's anything left
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        chunks.append({
            "id": len(chunks),  # Integer ID instead of string
            "text": chunk_text
        })
    
    # If we have more than 20 chunks, consolidate them
    if len(chunks) > 20:
        # Recombine all text
        all_text = " ".join(chunk["text"] for chunk in chunks)
        # Re-split into exactly 20 chunks, without minimum token requirement
        sentences = sent_tokenize(all_text)
        sentences_per_chunk = len(sentences) // 20 + (1 if len(sentences) % 20 > 0 else 0)
        
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            # Get the sentences for this chunk
            chunk_sentences = sentences[i:i+sentences_per_chunk]
            # Join the sentences into a single text
            chunk_text = " ".join(chunk_sentences)
            # Create a chunk object with ID and text
            chunks.append({
                "id": len(chunks),  # Integer ID instead of string
                "text": chunk_text
            })
    
    # Print chunk statistics
    print(f"Split document into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        token_count = len(tokenizer.encode(chunk["text"]))
        print(f"Chunk {i}: {token_count} tokens")
    
    return chunks

# Split the document into 20 chunks with minimum token size
document_chunks = split_into_20_chunks(document_text, min_tokens=500)
```

### 3.3 Router Function with Improved Tool Schema

Now, let's create the router function that will select relevant chunks and maintain a scratchpad.

> Maintaining a scratchpad allows the model to track decision criteria and reasoning over time. This implementation uses a two-pass approach with GPT-4.1-mini: first requiring the model to update the scratchpad via a tool call (tool_choice="required"), then requesting structured JSON output for chunk selection. This approach provides better visibility into the model's reasoning process while ensuring consistent structured outputs for downstream processing.

```python
from openai import OpenAI
import json
from typing import List, Dict, Any

# Initialize OpenAI client
client = OpenAI()

def route_chunks(question: str, chunks: List[Dict[str, Any]], 
                depth: int, scratchpad: str = "") -> Dict[str, Any]:
    """
    Ask the model which chunks contain information relevant to the question.
    Maintains a scratchpad for the model's reasoning.
    Uses structured output for chunk selection and required tool calls for scratchpad.
    
    Args:
        question: The user's question
        chunks: List of chunks to evaluate
        depth: Current depth in the navigation hierarchy
        scratchpad: Current scratchpad content
    
    Returns:
        Dictionary with selected IDs and updated scratchpad
    """
    print(f"\n==== ROUTING AT DEPTH {depth} ====")
    print(f"Evaluating {len(chunks)} chunks for relevance")
    
    # Build system message
    system_message = """You are an expert document navigator. Your task is to:
1. Identify which text chunks might contain information to answer the user's question
2. Record your reasoning in a scratchpad for later reference
3. Choose chunks that are most likely relevant. Be selective, but thorough. Choose as many chunks as you need to answer the question, but avoid selecting