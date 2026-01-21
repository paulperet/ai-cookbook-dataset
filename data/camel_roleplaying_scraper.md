# Building a Web Scraper & Knowledge Graph with CAMEL AI

This guide demonstrates how to build a sophisticated web scraping and knowledge graph generation pipeline using the CAMEL AI framework. You will learn to combine multiple tools—Firecrawl for scraping, Mistral for language processing, and Neo4j for graph storage—to extract, process, and structure information from the web.

## Prerequisites

Before you begin, ensure you have the following:
- Python installed on your system.
- API keys for Mistral AI, Firecrawl, and AgentOps (all offer free tiers).

## Step 1: Installation

First, install the CAMEL AI package with all necessary dependencies.

```bash
pip install camel-ai[all]==0.1.6.4
```

## Step 2: Configure API Keys

Set up your environment with the required API keys. This allows the tools to authenticate with their respective services.

### 2.1 AgentOps API Key
Get a free API key from [AgentOps](https://app.agentops.ai/signin).

```python
import os
from getpass import getpass

# Prompt for the AgentOps API key securely
agentops_api_key = getpass('Enter your AgentOps API key: ')
os.environ["AGENTOPS_API_KEY"] = agentops_api_key
```

### 2.2 Mistral AI API Key
Get an API key with free credits from [Mistral AI](https://console.mistral.ai/api-keys/).

```python
# Prompt for the Mistral AI API key securely
mistral_api_key = getpass('Enter your Mistral AI API key: ')
os.environ["MISTRAL_API_KEY"] = mistral_api_key
```

### 2.3 Configure the Mistral Model
Now, configure the specific Mistral model you will use for language processing.

```python
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import MistralConfig

# Set up the Mistral Large 2 model
mistral_large_2 = ModelFactory.create(
    model_platform=ModelPlatformType.MISTRAL,
    model_type=ModelType.MISTRAL_LARGE,
    model_config_dict=MistralConfig(temperature=0.2).as_dict(),
)
```

### 2.4 Firecrawl API Key
Get an API key with free credits from [Firecrawl](https://www.firecrawl.dev/).

```python
# Prompt for the Firecrawl API key securely
firecrawl_api_key = getpass('Enter your Firecrawl API key: ')
os.environ["FIRECRAWL_API_KEY"] = firecrawl_api_key
```

## Step 3: Scrape a Web Page with Firecrawl

Firecrawl is a tool that simplifies extracting and cleaning content from web pages. Let's test it by scraping a blog post from the CAMEL AI website.

```python
from camel.loaders import Firecrawl

# Initialize the Firecrawl client
firecrawl = Firecrawl()

# Scrape and clean content from a specified URL
response = firecrawl.tidy_scrape(
    url="https://www.camel-ai.org/post/crab"
)

# Print the cleaned text content
print(response)
```

**Expected Output:**
The `print` statement will output the main textual content of the scraped webpage, which in this example is an article about the "CRAB" (Cross-environment Agent Benchmark) research project. The output will be a long string of cleaned markdown/text.

## Summary

You have successfully set up the core components of the CAMEL AI pipeline. You have:
1.  Installed the necessary library.
2.  Securely configured API keys for external services.
3.  Initialized a language model (Mistral Large 2).
4.  Performed a test web scrape using Firecrawl to extract clean text.

In the next steps of this tutorial, you would typically use this scraped content to:
-   Perform Retrieval-Augmented Generation (RAG) using CAMEL's `AutoRetriever` with a vector database like Qdrant.
-   Employ CAMEL's multi-agent system to analyze the text and extract entities and relationships.
-   Store these structured relationships in a knowledge graph using Neo4j.

This setup provides a flexible foundation for building advanced information retrieval and analysis systems.