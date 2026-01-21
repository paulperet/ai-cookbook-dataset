# Building a Temporally-Aware Knowledge Graph: A Practical Guide

## Overview
This guide provides a hands-on tutorial for building **temporally-aware knowledge graphs** and performing **multi-hop retrieval** directly over those graphs. You'll learn how to systematically update and validate your knowledge base as new data arrives, and how to traverse complex relationships to answer sophisticated queries.

## Who This Guide Is For
This tutorial is designed for engineers, architects, and analysts working with structured data who want to:
- Build knowledge graphs that stay current and relevant
- Implement multi-step reasoning over connected facts
- Move from prototype to production with scalable architectures

## What You'll Build
1. **Temporally-aware knowledge graph construction**: A pipeline that ingests raw data and produces time-stamped triplets, with systematic validation and updating
2. **Multi-hop retrieval using knowledge graphs**: Combining OpenAI models with structured graph queries to traverse relationships across multiple steps

## Prerequisites

Before starting, ensure you have:
- Python 3.12 or later
- An OpenAI API key
- Basic understanding of knowledge graphs and LLMs

## Setup

First, install the required dependencies:

```bash
pip install --upgrade pip
pip install chonkie datetime ipykernel jinja2 matplotlib networkx numpy openai plotly pydantic rapidfuzz scipy tenacity tiktoken pandas
pip install "datasets<3.0"
```

Next, set up your OpenAI API key:

```python
import os

if "OPENAI_API_KEY" not in os.environ:
    import getpass
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Paste your OpenAI API key here: ")
```

---

# Part 1: Creating a Temporally-Aware Knowledge Graph with a Temporal Agent

## Why Temporal Knowledge Graphs Matter

Traditional knowledge graphs treat facts as static, but real-world information evolves constantly. What was true last quarter may be outdated today, risking errors or misinformed decisions. Temporal knowledge graphs allow you to:
- Precisely answer questions like "What was true on a given date?"
- Analyze how facts and relationships have shifted over time
- Ensure decisions are always based on the most relevant context

### Real-World Examples

| Industry | Example Question | Risk Without Temporal Context |
|----------|------------------|--------------------------------|
| **Financial Services** | "How has Moody's long-term rating for Bank YY evolved since Feb 2023?" | Mispricing credit risk by mixing historical & current ratings |
| **Manufacturing** | "Which ECU firmware was deployed in model Q3 cars shipped between 2022-05 and 2023-03?" | Misdiagnosing field failures due to firmware drift |

## Understanding the Temporal Agent

A **Temporal Agent** is a pipeline component that ingests raw data and produces time-stamped triplets for your knowledge graph. This enables precise time-based querying, timeline construction, and trend analysis.

### Key Enhancements in This Implementation

1. **Temporal validity extraction**: Identifies temporal spans and episodic context without requiring auxiliary reference statements
2. **Fact invalidation logic**: Introduces bidirectionality checks and constrains comparisons by episodic type
3. **Temporal & episodic typing**: Differentiates between `Fact`, `Opinion`, `Prediction`, and temporal classes `Static`, `Dynamic`, `Atemporal`
4. **Multi-event extraction**: Handles compound sentences and nested date references in a single pass

## The Temporal Agent Pipeline

The Temporal Agent processes incoming statements through three stages:

1. **Temporal Classification**: Labels each statement as **Atemporal**, **Static**, or **Dynamic**
2. **Temporal Event Extraction**: Identifies relative or partial dates and resolves them to absolute dates
3. **Temporal Validity Check**: Ensures every statement includes timestamps and detects contradictions

## Model Selection for Temporal Agents

When building systems with LLMs, start with larger models and optimize later. The GPT-4.1 series is particularly well-suited for Temporal Agents due to its strong instruction-following ability.

### Recommended Development Workflow

1. **Prototype with GPT-4.1**: Maximize correctness and reduce prompt-debug time
2. **Swap to smaller variants**: Once prompts and logic are stable, switch to GPT-4.1-mini or GPT-4.1-nano for lower latency and cost
3. **Distill onto smaller models**: Use OpenAI's Model Distillation to train smaller models with high-quality outputs from a larger teacher model

| Model | Relative Cost | Relative Latency | Intelligence | Ideal Role |
|-------|---------------|------------------|--------------|------------|
| GPT-4.1 | High | Medium | Highest | Ground-truth prototyping |
| GPT-4.1-mini | Medium | Low | Medium | Balanced production systems |
| GPT-4.1-nano | Lowest | Fastest | Basic | Cost-sensitive bulk processing |

## Building the Temporal Agent Pipeline

Now let's implement the pipeline step by step.

### Step 1: Load Transcripts

We'll use the "Earnings Calls Dataset" from HuggingFace, which contains 188 earnings call transcripts from 2016-2020. This dataset demonstrates temporal concepts well as companies discuss evolving topics across multiple calls.

```python
from datasets import load_dataset

# Load the earnings call transcripts dataset
hf_dataset_name = "jlh-ibm/earnings_call"
hf_dataset = load_dataset(hf_dataset_name, "transcripts")
my_dataset = hf_dataset["train"]

# Examine the dataset structure
print(f"Dataset features: {my_dataset.features}")
print(f"Number of rows: {len(my_dataset)}")

# View a sample entry
row = my_dataset[0]
print(f"Company: {row['company']}")
print(f"Date: {row['date']}")
print(f"Transcript preview: {row['transcript'][:200]}...")
```

### Step 2: Analyze Company Distribution

Let's examine which companies are represented in our dataset:

```python
from collections import Counter

company_counts = Counter(my_dataset["company"])
print("Company distribution:")
for company, count in company_counts.most_common():
    print(f"  {company}: {count} transcripts")
```

For this tutorial, we'll focus on processing transcripts from AMD and Nvidia, though the pipeline can scale to any company.

---

*Note: The tutorial continues with the remaining pipeline steps including semantic chunking, statement extraction, temporal range extraction, triplet creation, entity resolution, and the invalidation agent. Each section will provide clear, step-by-step instructions with code examples and explanations.*