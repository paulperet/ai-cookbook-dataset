# Guide: Using Azure OpenAI Chat Models with Your Own Data

## Overview
This guide demonstrates how to use Azure OpenAI chat models (like GPT-3.5-Turbo and GPT-4) with your proprietary data. This "Azure OpenAI on your data" feature allows models to reference your specific documents when generating responses, providing more accurate, up-to-date answers without requiring model fine-tuning.

The solution integrates Azure OpenAI with Azure AI Search to retrieve relevant data from your sources, which is then used to augment the model's prompts.

## Prerequisites
Before starting, ensure you have the following Azure resources created:

1.  **Azure OpenAI Resource:** Access to Azure OpenAI with a deployed chat model (e.g., GPT-3.5-Turbo or GPT-4).
2.  **Azure AI Search Resource:** (formerly Azure Cognitive Search) to index and search your data.
3.  **Azure Blob Storage Resource:** To host your document files.
4.  **Your Documents:** Uploaded to Blob Storage and indexed in Azure AI Search. Supported formats include PDF, TXT, DOCX, and more.

For a detailed walkthrough on uploading documents and creating a search index via Azure AI Studio, follow the [official Quickstart](https://learn.microsoft.com/azure/ai-services/openai/use-your-data-quickstart?pivots=programming-language-studio&tabs=command-line).

## Setup and Installation

### 1. Install Required Packages
Begin by installing the necessary Python libraries.

```bash
pip install "openai>=1.0.0,<2.0.0"
pip install python-dotenv
```

### 2. Configure Environment Variables
Create a `.env` file in your project directory to store your credentials securely. Add the following variables:

```plaintext
AZURE_OPENAI_ENDPOINT=<your-azure-openai-endpoint>
AZURE_OPENAI_API_KEY=<your-azure-openai-api-key>
SEARCH_ENDPOINT=<your-ai-search-endpoint>
SEARCH_KEY=<your-ai-search-api-key>
SEARCH_INDEX_NAME=<your-search-index-name>
```

**Where to find these values:**
*   `AZURE_OPENAI_ENDPOINT` & `AZURE_OPENAI_API_KEY`: Located under **"Keys and Endpoint"** for your Azure OpenAI resource in the Azure Portal.
*   `SEARCH_ENDPOINT`: Found on the **"Overview"** page of your Azure AI Search resource.
*   `SEARCH_KEY`: Located under **"Keys"** for your Search resource.
*   `SEARCH_INDEX_NAME`: The name of the index you created containing your data.

### 3. Load Environment Variables and Import Libraries
In your Python script, load the environment variables.

```python
import os
import openai
import dotenv

dotenv.load_dotenv()
```

## Authentication
Azure OpenAI supports two primary authentication methods. Choose the one that fits your security setup.

### Option 1: API Key Authentication (Default)
This method uses the API key from your `.env` file.

```python
# Set your model deployment name
deployment = "<deployment-id-of-the-model-to-use>"

# Initialize the client with API key auth
client = openai.AzureOpenAI(
    base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT']}/openai/deployments/{deployment}/extensions",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2023-09-01-preview"
)
```

### Option 2: Azure Active Directory Authentication
For enhanced security using managed identities or service principals, use Azure AD authentication.

First, install the Azure Identity library:

```bash
pip install "azure-identity>=1.15.0"
```

Then, configure your client:

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Set your model deployment name
deployment = "<deployment-id-of-the-model-to-use>"

# Initialize the client with Azure AD auth
client = openai.AzureOpenAI(
    base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT']}/openai/deployments/{deployment}/extensions",
    azure_ad_token_provider=get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default"
    ),
    api_version="2023-09-01-preview"
)
```

> **Note:** The `AzureOpenAI` client can infer credentials from environment variables. For example, it will automatically use `AZURE_OPENAI_API_KEY` if the `api_key` parameter is not provided.

## Using Chat Completion with Your Data
With authentication configured, you can now query the model, grounding its responses in your indexed data.

### Example Context
For this tutorial, imagine you have indexed the Azure AI services documentation. The model will use this knowledge base to answer questions.

### Step 1: Make a Standard Request
Send a user query. The `extra_body` parameter connects the request to your Azure AI Search index.

```python
completion = client.chat.completions.create(
    model=deployment,
    messages=[
        {"role": "user", "content": "What are the differences between Azure Machine Learning and Azure AI services?"}
    ],
    extra_body={
        "dataSources": [
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": os.environ["SEARCH_ENDPOINT"],
                    "key": os.environ["SEARCH_KEY"],
                    "indexName": os.environ["SEARCH_INDEX_NAME"],
                }
            }
        ]
    }
)

# Print the assistant's response
print(f"{completion.choices[0].message.role}: {completion.choices[0].message.content}")

# Access the context (source data) used by the model
print(f"\nContext: {completion.choices[0].message.model_extra['context']['messages'][0]['content']}")
```

The response will include both the model's answer and the `context` property, which shows the specific data retrieved from your index to inform the response.

### Step 2: Make a Streaming Request (Optional)
For a real-time, token-by-token response, use the streaming option by adding `stream=True`.

```python
response = client.chat.completions.create(
    model=deployment,
    messages=[
        {"role": "user", "content": "What are the differences between Azure Machine Learning and Azure AI services?"}
    ],
    extra_body={
        "dataSources": [
            {
                "type": "AzureCognitiveSearch",
                "parameters": {
                    "endpoint": os.environ["SEARCH_ENDPOINT"],
                    "key": os.environ["SEARCH_KEY"],
                    "indexName": os.environ["SEARCH_INDEX_NAME"],
                }
            }
        ]
    },
    stream=True,  # Enable streaming
)

for chunk in response:
    delta = chunk.choices[0].delta

    if delta.role:
        print(f"\n{delta.role}: ", end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)
    # Context is also streamed in model_extra
    if delta.model_extra.get("context"):
        print(f"Context: {delta.model_extra['context']}", end="", flush=True)
```

## Next Steps
You have successfully configured an Azure OpenAI model to use your own data. To build upon this:
*   Experiment with different queries and document sets.
*   Explore the [Azure OpenAI on your data documentation](https://learn.microsoft.com/azure/ai-services/openai/concepts/use-your-data) for advanced configurations.
*   Review [data, privacy, and security guidelines](https://learn.microsoft.com/legal/cognitive-services/openai/data-privacy?context=%2Fazure%2Fai-services%2Fopenai%2Fcontext%2Fcontext) for production deployments.