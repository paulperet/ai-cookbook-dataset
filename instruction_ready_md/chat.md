# Guide: Using Azure OpenAI for Chat Completions

This guide walks you through setting up the Azure OpenAI client, authenticating, creating chat completions, and handling content filtering.

## Prerequisites

Before you begin, ensure you have:
- An Azure account with an OpenAI resource created.
- Access to the Azure Portal to retrieve your endpoint and keys.

## Step 1: Install Dependencies

Install the required Python packages.

```bash
pip install "openai>=1.0.0,<2.0.0"
pip install python-dotenv
```

## Step 2: Import Libraries and Load Environment Variables

Create a `.env` file to store your credentials securely, then load them in your script.

```python
import os
import openai
import dotenv

dotenv.load_dotenv()
```

## Step 3: Authenticate the Client

Azure OpenAI supports API key authentication and Azure Active Directory (Azure AD). Choose the method that fits your use case.

### Option A: Authenticate with an API Key

Set the `use_azure_active_directory` flag to `False`. Retrieve your endpoint and API key from the Azure Portal under **Keys and Endpoint** for your resource.

```python
use_azure_active_directory = False

if not use_azure_active_directory:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]

    client = openai.AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version="2023-09-01-preview"
    )
```

### Option B: Authenticate with Azure Active Directory

Set the `use_azure_active_directory` flag to `True`. Install the Azure Identity library, then use a token provider for automatic token caching and refresh.

```bash
pip install "azure-identity>=1.15.0"
```

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

use_azure_active_directory = True

if use_azure_active_directory:
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]

    client = openai.AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        ),
        api_version="2023-09-01-preview"
    )
```

> **Note:** The `AzureOpenAI` client can infer credentials from environment variables if not explicitly provided:
> - `api_key` from `AZURE_OPENAI_API_KEY`
> - `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
> - `api_version` from `OPENAI_API_VERSION`
> - `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`

## Step 4: Deploy a Model

In the Azure Portal, navigate to your OpenAI resource and open **Azure OpenAI Studio**. Under the **Deployments** tab, create a deployment for a GPT model (e.g., `gpt-35-turbo`). Note the deployment name you assign.

```python
deployment = "your-deployment-name"  # Replace with your deployment name
```

## Step 5: Create a Chat Completion

Now, use the client to generate a chat completion. The `messages` parameter defines the conversation history.

```python
response = client.chat.completions.create(
    model=deployment,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Knock knock."},
        {"role": "assistant", "content": "Who's there?"},
        {"role": "user", "content": "Orange."},
    ],
    temperature=0,
)

print(f"{response.choices[0].message.role}: {response.choices[0].message.content}")
```

## Step 6: Stream a Chat Completion

To receive the response incrementally, set `stream=True`. This is useful for real-time applications.

```python
response = client.chat.completions.create(
    model=deployment,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Knock knock."},
        {"role": "assistant", "content": "Who's there?"},
        {"role": "user", "content": "Orange."},
    ],
    temperature=0,
    stream=True
)

for chunk in response:
    if len(chunk.choices) > 0:
        delta = chunk.choices[0].delta
        if delta.role:
            print(delta.role + ": ", end="", flush=True)
        if delta.content:
            print(delta.content, end="", flush=True)
```

## Step 7: Handle Content Filtering

Azure OpenAI includes built-in content filtering. Learn more about configuring filters in the [Azure documentation](https://learn.microsoft.com/azure/ai-services/openai/concepts/content-filter).

### Detect a Flagged Prompt

If a prompt violates content policies, the API raises a `BadRequestError`. Catch this exception to inspect the filter results.

```python
import json

messages = [
    {"role": "system", "content": "You is a helpful assistant."},
    {"role": "user", "content": "<text violating the content policy>"}
]

try:
    completion = client.chat.completions.create(
        messages=messages,
        model=deployment,
    )
except openai.BadRequestError as e:
    err = json.loads(e.response.text)
    if err["error"]["code"] == "content_filter":
        print("Content filter triggered!")
        content_filter_result = err["error"]["innererror"]["content_filter_result"]
        for category, details in content_filter_result.items():
            print(f"{category}:\n filtered={details['filtered']}\n severity={details['severity']}")
```

### Inspect Filter Results for a Successful Request

For successful completions, you can access the filter results for both the prompt and the completion.

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the biggest city in Washington?"}
]

completion = client.chat.completions.create(
    messages=messages,
    model=deployment,
)
print(f"Answer: {completion.choices[0].message.content}")

# Prompt content filter results
prompt_filter_result = completion.model_extra["prompt_filter_results"][0]["content_filter_results"]
print("\nPrompt content filter results:")
for category, details in prompt_filter_result.items():
    print(f"{category}:\n filtered={details['filtered']}\n severity={details['severity']}")

# Completion content filter results
print("\nCompletion content filter results:")
completion_filter_result = completion.choices[0].model_extra["content_filter_results"]
for category, details in completion_filter_result.items():
    print(f"{category}:\n filtered={details['filtered']}\n severity={details['severity']}")
```

## Summary

You have now learned how to:
1. Set up the Azure OpenAI client with two authentication methods.
2. Deploy a model and use it for chat completions.
3. Stream responses for real-time interaction.
4. Handle and inspect content filtering results.

This provides a solid foundation for integrating Azure OpenAI's chat capabilities into your applications.