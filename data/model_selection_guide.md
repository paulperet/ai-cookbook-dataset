# Practical Guide for Model Selection for Real‑World Use Cases

## Purpose & Audience

This guide serves as a practical resource for selecting, prompting, and deploying the right OpenAI model (GPT‑4.1, o3, or o4‑mini) for specific workloads. It provides actionable decision frameworks and real-world examples to help Solutions Engineers, Technical Account Managers, Partner Architects, and semi-technical practitioners quickly build working solutions. The content focuses on current model capabilities, vertical-specific implementations, and clear pathways from model selection to production deployment.

> **Note:** The guidance and experimental results are based on the latest state-of-the-art models available at the time of writing. Metrics and recommendations may change in the future.

## How to Use This Guide

This guide is organized into distinct sections:
1.  **Model Guide**: A quick reference for selecting the right model.
2.  **Use Cases**: Detailed implementations for specific scenarios.
3.  **Prototype to Production**: A checklist for transitioning from prototype to production.
4.  **Adaptation Decision Tree**: A flowchart to guide model selection.
5.  **Appendices**: Reference materials including pricing, latency, and prompt patterns.

For quick decisions, focus on the **Model Guide** and **Adaptation Decision Tree**. For implementation details, explore the relevant **Use Cases**.

---

## Model Guide

### 2.1 Model‑Intro Matrix

| Model | Core Strength | Ideal First Reach‑For | Watch‑Outs | Escalate / Downgrade Path |
| :---- | :------------ | :-------------------- | :--------- | :------------------------ |
| GPT‑4o | Real‑time voice / vision chat | Live multimodal agents | Slightly below GPT‑4.1 on text SOTA | Need deep reasoning → o4‑mini |
| GPT‑4.1 | 1M‑token text accuracy | Long‑document analytics, code review | Cannot natively reason; higher cost than minis | Tight budget → GPT‑4.1‑mini / nano |
| o3 | Deep tool‑using agent | High‑stakes, multi‑step reasoning | Latency & price | Cost/latency → o4‑mini |
| o4‑mini | Cheap, fast reasoning | High‑volume "good‑enough" logic | Depth ceiling vs. o3 | Accuracy critical → o3 |

*(Full price and utility details are in [Section 6.1](#appendices)).*

### 2.2 Model Evolution at a Glance

OpenAI offers two distinct model families, each with unique strengths:

*   **GPT Models (4o, 4.1)**: Optimized for general-purpose tasks with excellent instruction following.
    *   **GPT‑4.1**: Excels with long contexts (1M tokens).
    *   **GPT‑4o**: Has variants for realtime speech, text-to-speech, and speech-to-text.
    *   Both have cheaper, faster variants (mini, nano).
*   **o‑series Models (o3, o4‑mini)**: Specialized for deep reasoning and step-by-step problem solving. They excel at complex, multi-stage tasks requiring logical thinking and tool use. These models include an optional `reasoning_effort` parameter (`low`, `medium`, `high`) to control token usage for reasoning.

**Key Characteristics:**
*   **GPT‑4.1 Family**: Optimized for long context processing.
*   **o3**: Specialized for deep multi-step reasoning.
*   **o4‑mini**: Combines reasoning capabilities with vision at a lower cost.

> **Note for This Guide:** The following use cases experiment with the GPT‑4.1 series, o3, and o4‑mini models. The GPT‑4o series is not covered.

---

## Use Case 3A: Long-Context RAG for Legal Q&A

This section details the construction of a Retrieval-Augmented Generation (RAG) system designed to answer questions about complex, lengthy procedural texts, using the *Trademark Trial and Appeal Board Manual of Procedure (TBMP)* as a case study. This approach can be adapted for other domains requiring precise information retrieval from dense documentation, such as healthcare compliance or financial regulations.

### 🗂️ TL;DR Implementation Matrix

| Layer | Choice | Utility |
| :---- | :---- | :---- |
| **Chunking** | Sentence-aware Splitter | Splits document into ~20 equal chunks, respecting sentence boundaries. |
| **Routing** | `gpt-4.1-mini` | Uses natural language understanding to identify relevant chunks without an embedding index. |
| **Path Selection** | `select(ids=[...])` and `scratchpad(text="...")` | Records reasoning while drilling down through the document hierarchy. |
| **Citation** | Paragraph-level | Balances precision with cost; provides meaningful context for answers. |
| **Synthesis** | `gpt-4.1` (Structured Output) | Generates answers directly from selected paragraphs with citations. |
| **Verification** | `o4-mini` (LLM-as-Judge) | Validates factual accuracy and citation correctness. |

*Note: Prices and model identifiers are accurate as of April 2025 and are subject to change.*

### Scenario Snapshot

*   **Corpus:** The [Trademark Trial and Appeal Board Manual of Procedure (TBMP, 2024 version)](https://www.uspto.gov/sites/default/files/documents/tbmp-Master-June2024.pdf). This manual contains detailed procedural rules and guidelines across 1194 pages.
*   **Users:** Intellectual property (IP) litigation associates and paralegals.
*   **Typical Questions:**
    1.  "What are the requirements for filing a motion to compel discovery according to the TBMP?"
    2.  "What deadlines apply to discovery conferences as specified in the manual?"
    3.  "Explain how the Board handles claims of attorney-client privilege during depositions according to the TBMP."

> **Note:** While OpenAI's File Search tool is a good starting point, this approach leverages million-token context windows to process large documents without preprocessing or a vector database. It enables zero-latency ingestion, dynamic retrieval granularity, and fine-grained citation traceability.

### Agentic RAG Flow

The system mimics how a human skims a document:
1.  Load the entire document into the context window.
2.  Split it into ~20 chunks that respect sentence boundaries.
3.  Ask the model which chunks might contain relevant information.
4.  Drill down into selected chunks by splitting them further.
5.  Repeat until reaching paragraph-level content.
6.  Generate an answer based on the selected paragraphs.
7.  Verify the answer for factual accuracy.

### Agentic RAG System: Model Usage

| Process Stage | Model Used | Purpose |
| :------------ | :--------- | :------ |
| Initial Routing | `gpt-4.1-mini` | Identifies which document chunks might contain relevant information. |
| Hierarchical Navigation | `gpt-4.1-mini` | Continues drilling down to find the most relevant paragraphs. |
| Answer Generation | `gpt-4.1` | Creates a structured response with citations from selected paragraphs. |
| Answer Verification | `o4-mini` | Validates factual accuracy and proper citation usage. |

### Implementation

#### Step 1: Setup and Installation

Begin by installing the required Python packages.

```bash
pip install tiktoken pypdf nltk openai pydantic --quiet
```

#### Step 2: Document Loading

Load the document and check its size. This guide focuses on sections 100-900, which cover core procedural aspects.

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

The output will show the document is over 900k tokens long. While this could fit into GPT‑4.1's context, we'll use a recursive chunking strategy to enable verifiable citations.

#### Step 3: Improved 20-Chunk Splitter

Create a function to split the document into approximately 20 chunks, ensuring each has a minimum token size while respecting sentence boundaries.

> **Note:** 20 is an empirically chosen number for this specific document and task. It may need tuning for other documents based on size and structure. The key principle is to split the document into sections, allowing the language model to decide which components are relevant.

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

#### Step 4: Router Function with Improved Tool Schema

Create the router function that selects relevant chunks and maintains a scratchpad for reasoning.

> **Note:** Maintaining a scratchpad allows the model to track decision criteria over time. This implementation uses a two-pass approach with `gpt-4.1-mini`: first requiring the model to update the scratchpad via a tool call (`tool_choice="required"`), then requesting structured JSON output for chunk selection. This provides better visibility into the model's reasoning while ensuring consistent structured outputs.

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
3. Choose chunks that are most likely relevant. Be selective, but thorough. Choose as many chunks as you need to answer the question, but avoid selecting"""
```