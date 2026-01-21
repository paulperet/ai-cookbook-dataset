# Guide: GitHub Marketplace Model Inference with Phi-4 Reasoning

This guide demonstrates how to use models from the GitHub Marketplace for inference, specifically Microsoft's Phi-4-reasoning model. You will learn to set up your environment, configure API access, and run inference tasks through practical examples.

## Prerequisites

Before you begin, ensure you have the following:

1.  A GitHub account with a Personal Access Token (PAT) that has access to the AI models service.
2.  An Azure account (optional, for the Azure OpenAI section).
3.  Python installed on your system.

## Step 1: Environment Setup

First, install the required Python packages.

```bash
pip install requests python-dotenv azure-ai-inference
```

## Step 2: Configuration

Create a configuration file named `local.env` in your project directory. This file will securely store your API credentials.

1.  Create a new file called `local.env`.
2.  Add the following environment variables, replacing the placeholder values with your actual credentials:

```bash
# GitHub Configuration
GITHUB_TOKEN=your_personal_access_token_here
GITHUB_INFERENCE_ENDPOINT=https://models.github.ai/inference
GITHUB_MODEL=microsoft/Phi-4-reasoning

# Azure OpenAI Configuration (Optional)
AZURE_API_KEY=your_azure_api_key_here
AZURE_OPENAI_ENDPOINT=your_azure_endpoint_here
AZURE_OPENAI_MODEL=Phi-4-reasoning
```

**Note:** You can change the `GITHUB_MODEL` to `microsoft/Phi-4-mini-reasoning` to use a smaller, faster model.

## Step 3: Load Configuration and Initialize

Now, you will write a Python script to load these environment variables and prepare for API calls.

Create a new Python file (e.g., `inference_demo.py`) and start by importing the necessary libraries and loading your configuration.

```python
import os
import requests
import json
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# Load variables from the local.env file
load_dotenv('local.env')

# Access the environment variables
endpoint = os.getenv("GITHUB_INFERENCE_ENDPOINT")
model = os.getenv("GITHUB_MODEL")
token = os.getenv("GITHUB_TOKEN")

# Optional: Load Azure variables if you plan to use that service
azuretoken = os.getenv("AZURE_KEY")
azureendpoint = os.getenv("AZURE_ENDPOINT")
azuremodel = os.getenv("AZURE_MODEL")

# Use fallback values if GitHub config is not found
if not endpoint:
    endpoint = "https://models.github.ai/inference"
    print("Warning: GITHUB_INFERENCE_ENDPOINT not found in local.env, using default value")

if not model:
    model = "microsoft/Phi-4-reasoning"
    print("Warning: GITHUB_MODEL not found in local.env, using default value")

if not token:
    raise ValueError("GITHUB_TOKEN not found in local.env file. Please add your GitHub token.")

print(f"Endpoint: {endpoint}")
print(f"Model: {model}")
print(f"Token available: {'Yes' if token else 'No'}")
```

## Step 4: Create Helper Functions for the GitHub API

To interact cleanly with the GitHub Inference API, you will define helper functions. These functions handle the HTTP requests and response parsing.

Add the following functions to your script:

```python
def generate_chat_completion(messages, model_id=model, temperature=0.7, max_tokens=1000):
    """Generate a completion using GitHub's chat completions API."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        api_url = f"{endpoint}/v1/chat/completions"
        print(f"Calling API at: {api_url}")

        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raises an exception for HTTP errors
        result = response.json()

        if 'choices' in result and len(result['choices']) > 0:
            if 'message' in result['choices'][0]:
                return result['choices'][0]['message']['content']
            else:
                return f"Error: Unexpected response format: {result['choices'][0]}"
        else:
            return f"Error: Unexpected response format: {result}"
    except Exception as e:
        print(f"Full error details: {str(e)}")
        return f"Error during API call: {str(e)}"
```

## Step 5: Run Your First Inference

Let's test the setup with a simple, playful query to see the model's reasoning capabilities in action.

Add this code to your script to define the conversation and call the API:

```python
# Example 1: A playful reasoning task
example1_messages = [
    {"role": "system", "content": "You are a helpful AI assistant that answers questions accurately and concisely."},
    {"role": "user", "content": "How many strawberries do I need to collect 9 r's?"}
]

print("=== Example 1: How many strawberries for 9 r's? ===")
print("Messages:")
for msg in example1_messages:
    print(f"{msg['role']}: {msg['content']}")
print("\nGenerating response...\n")

response1 = generate_chat_completion(example1_messages)
print("Response:")
print(response1)
```

When you run the script, the model will process the riddle. It understands that the word "strawberry" contains three 'r's. To collect 9 'r's, you would need 3 strawberries (3 strawberries * 3 r's = 9 r's). The model's detailed internal reasoning will be visible in its response.

## Step 6: Solve a Complex Pattern Recognition Riddle

Now, let's challenge the model with a more complex task involving pattern recognition across multiple examples.

Add this second example to your script:

```python
# Example 2: A pattern recognition riddle
example2_messages = [
    {"role": "system", "content": "You are a helpful AI assistant that solves riddles and finds patterns in sequences."},
    {"role": "user", "content": "I will give you a riddle to solve with a few examples, and something to complete at the end"},
    {"role": "user", "content": "nuno Δημήτρης evif Issis 4"},
    {"role": "user", "content": "ntres Inez neves Margot 4"},
    {"role": "user", "content": "ndrei Jordan evlewt Μαρία 9"},
    {"role": "user", "content": "nπέντε Kang-Yuk xis-ytnewt Nubia 21"},
    {"role": "user", "content": "nπέντε Κώστας eerht-ytnewt Μανώλης 18"},
    {"role": "user", "content": "nminus one-point-two Satya eno Bill X."},
    {"role": "user", "content": "What is a likely completion for X that is consistent with examples above?"}
]

print("\n" + "="*60)
print("=== Example 2: Solving a Pattern Riddle ===")
print("Messages:")
for msg in example2_messages:
    # Print a preview of long messages
    content_preview = msg['content'][:50] + '...' if len(msg['content']) > 50 else msg['content']
    print(f"{msg['role']}: {content_preview}")
print("\nGenerating response...\n")

# Use a lower temperature for more deterministic, pattern-focused output
response2 = generate_chat_completion(example2_messages, temperature=0.2, max_tokens=10000)
print("Response:")
print(response2)
```

This task requires the model to identify a transformation rule (like reversing words or converting numbers) across multilingual and encoded examples. The model will analyze the sequence and propose a value for `X` that fits the discovered pattern.

## Summary

You have successfully set up access to the GitHub Marketplace's inference API and used the Phi-4-reasoning model to perform tasks ranging from linguistic reasoning to complex pattern recognition. The key steps were:

1.  Configuring your environment with a `.env` file.
2.  Writing helper functions to call the chat completions API.
3.  Executing inference on different types of prompts to evaluate the model's capabilities.

You can extend this foundation by integrating the optional Azure OpenAI service, building a chat application, or implementing a Retrieval-Augmented Generation (RAG) pipeline.