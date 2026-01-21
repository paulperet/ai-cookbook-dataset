# Azure OpenAI Completions Guide

This guide demonstrates how to use the Azure OpenAI service to generate text completions using the legacy OpenAI Python library (`openai<1.0.0`).

## Prerequisites

Before you begin, ensure you have:
1. An active Azure subscription.
2. An Azure OpenAI resource provisioned.
3. The necessary permissions to access keys and create deployments.

## Step 1: Install the Required Library

Install the legacy OpenAI client library compatible with Azure OpenAI.

```bash
pip install "openai>=0.28.1,<1.0.0"
```

## Step 2: Configure Your Azure OpenAI Endpoint

First, import the library and set your API version and base endpoint.

```python
import os
import openai

# Set the API version (check for the latest stable version in the Azure portal)
openai.api_version = '2023-05-15'

# Set your endpoint URL
# Find this in the Azure Portal under your resource's "Keys and Endpoint" section
openai.api_base = 'https://your-resource-name.openai.azure.com/'  # Replace with your endpoint
```

## Step 3: Set Up Authentication

You can authenticate using either a key from the Azure portal or Microsoft Entra ID (formerly Azure Active Directory).

### Option A: Authenticate with an Azure Portal Key

This is the simplest method for development.

1.  **Get your key:** Navigate to your Azure OpenAI resource in the [Azure Portal](https://portal.azure.com). Go to **Resource Management > Keys and Endpoint** and copy one of the keys.
2.  **Configure the client:**

```python
openai.api_type = 'azure'
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your key as an environment variable
```

> **Best Practice:** For better security and configuration management, set these values as environment variables instead of hardcoding them.
> ```bash
> export OPENAI_API_BASE="https://your-resource.openai.azure.com/"
> export OPENAI_API_KEY="your-key-here"
> export OPENAI_API_TYPE="azure"
> export OPENAI_API_VERSION="2023-05-15"
> ```

### Option B: Authenticate with Microsoft Entra ID (Optional)

For production systems integrated with Azure identity management, you can use token-based authentication.

1.  Install the Azure Identity library:
    ```bash
    pip install azure-identity
    ```
2.  Use the following code to authenticate:

```python
from azure.identity import DefaultAzureCredential

# Get a credential object from the local environment (e.g., Azure CLI, VS Code, Managed Identity)
default_credential = DefaultAzureCredential()

# Request a token for the Azure Cognitive Services scope
token = default_credential.get_token("https://cognitiveservices.azure.com/.default")

# Configure the OpenAI client to use Azure AD authentication
openai.api_type = 'azure_ad'
openai.api_key = token.token
```

**Handling Token Refresh:** Tokens expire. To automatically refresh them, you can attach a custom authentication handler to the client's request session.

```python
import typing
import time
import requests
if typing.TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class TokenRefresh(requests.auth.AuthBase):
    """Auth class to automatically refresh Azure AD tokens."""
    def __init__(self, credential: "TokenCredential", scopes: typing.List[str]) -> None:
        self.credential = credential
        self.scopes = scopes
        self.cached_token: typing.Optional[str] = None

    def __call__(self, req):
        # Refresh token if it's expired or about to expire (within 5 minutes)
        if not self.cached_token or self.cached_token.expires_on - time.time() < 300:
            self.cached_token = self.credential.get_token(*self.scopes)
        req.headers["Authorization"] = f"Bearer {self.cached_token.token}"
        return req

# Create a session with the token refresh handler
session = requests.Session()
session.auth = TokenRefresh(default_credential, ["https://cognitiveservices.azure.com/.default"])

# Assign the session to the OpenAI client
openai.requestssession = session
```

## Step 4: Create a Model Deployment

Azure OpenAI requires you to deploy a model before you can use it.

1.  In the [Azure Portal](https://portal.azure.com), navigate to your Azure OpenAI resource.
2.  Go to **Resource Management > Model deployments**.
3.  Click **+ Create new deployment**.
4.  Select the `gpt-3.5-turbo-instruct` model from the dropdown, give it a unique deployment name (e.g., `gpt-35-turbo-instruct`), and create it.

Once deployed, note the **deployment name** you assigned.

```python
# Assign the deployment name you created in the portal
deployment_id = 'gpt-35-turbo-instruct'  # Replace with your deployment name
```

## Step 5: Generate a Completion

Now you are ready to send a prompt to your deployed model and receive a text completion.

```python
# Define your prompt
prompt = "The food was delicious and the waiter"

# Call the Completion API
completion = openai.Completion.create(
    deployment_id=deployment_id,
    prompt=prompt,
    stop=".",  # Stop generation when a period is encountered
    temperature=0  # Controls randomness; 0 makes outputs more deterministic
)

# Print the combined prompt and completion
generated_text = completion['choices'][0]['text']
print(f"{prompt}{generated_text}.")
```

**Expected Output:**
The model will complete the sentence. For example:
```
The food was delicious and the waiter provided excellent service.
```

## Summary

You have successfully:
1.  Installed and configured the legacy Azure OpenAI client.
2.  Set up authentication using either an API key or Microsoft Entra ID.
3.  Deployed a model (`gpt-3.5-turbo-instruct`) via the Azure Portal.
4.  Sent a completion request to your deployment and received a generated text response.

For moving to production, remember to use environment variables for configuration and consider the token refresh mechanism for Entra ID authentication.