# Azure DALL·E Image Generation Guide

This guide demonstrates how to generate images using Azure OpenAI's DALL·E service. You will learn to set up your environment, authenticate with Azure, and create and download AI-generated images.

## Prerequisites

Before you begin, ensure you have:
- An active Azure subscription.
- An Azure OpenAI resource created in the [Azure Portal](https://portal.azure.com).
- Your resource's endpoint and an API key (or Azure Active Directory credentials).

## Step 1: Environment Setup

First, install the required Python packages.

```bash
pip install "openai>=0.28.1,<1.0.0"
pip install requests
pip install pillow
# Optional: For Azure Active Directory authentication
pip install azure-identity
```

Import the necessary modules.

```python
import os
import openai
```

## Step 2: Configure the Azure OpenAI Client

Retrieve your endpoint from the Azure Portal under **Keys and Endpoint** in the **Resource Management** section. Then, configure the OpenAI SDK.

```python
# Set your Azure OpenAI endpoint
openai.api_base = ''  # Add your endpoint here

# DALL·E currently requires this specific API version
openai.api_version = '2023-06-01-preview'
```

## Step 3: Authenticate with Azure

Azure OpenAI supports two primary authentication methods: API keys and Azure Active Directory. Choose the method that fits your use case.

### Option A: Authenticate with an API Key

This is the simplest method. Set your API key from the Azure Portal.

```python
# Set the API type to 'azure'
openai.api_type = 'azure'
# Use an environment variable for security
openai.api_key = os.environ["OPENAI_API_KEY"]
```

> **Tip:** For better security and convenience, set these environment variables in your system:
> ```
> OPENAI_API_BASE=<your-endpoint>
> OPENAI_API_KEY=<your-key>
> OPENAI_API_TYPE=azure
> OPENAI_API_VERSION=2023-06-01-preview
> ```

### Option B: Authenticate with Azure Active Directory

Use this method for managed identity or service principal authentication.

```python
from azure.identity import DefaultAzureCredential

# Set a flag to control the authentication method
use_azure_active_directory = True  # Set to False to use API key method

if use_azure_active_directory:
    default_credential = DefaultAzureCredential()
    token = default_credential.get_token("https://cognitiveservices.azure.com/.default")

    openai.api_type = 'azure_ad'
    openai.api_key = token.token
```

Since Azure AD tokens expire, you can implement an automatic refresh mechanism for long-running sessions.

```python
import typing
import time
import requests

if typing.TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class TokenRefresh(requests.auth.AuthBase):
    """Auth class to automatically refresh an expiring Azure AD token."""
    def __init__(self, credential: "TokenCredential", scopes: typing.List[str]) -> None:
        self.credential = credential
        self.scopes = scopes
        self.cached_token: typing.Optional[str] = None

    def __call__(self, req):
        # Refresh token if it's expired or will expire in 5 minutes
        if not self.cached_token or self.cached_token.expires_on - time.time() < 300:
            self.cached_token = self.credential.get_token(*self.scopes)
        req.headers["Authorization"] = f"Bearer {self.cached_token.token}"
        return req

if use_azure_active_directory:
    # Attach the token refresh logic to the OpenAI client's session
    session = requests.Session()
    session.auth = TokenRefresh(default_credential, ["https://cognitiveservices.azure.com/.default"])
    openai.requestssession = session
```

## Step 4: Generate an Image

Now you are ready to create images. Use the `Image.create` method with a descriptive prompt.

```python
# Generate two 1024x1024 images based on your prompt
generation_response = openai.Image.create(
    prompt='A cyberpunk monkey hacker dreaming of a beautiful bunch of bananas, digital art',
    size='1024x1024',
    n=2  # Number of images to generate
)

print(generation_response)
```

The response is a JSON object containing URLs for your generated images.

## Step 5: Download and Display the Image

After generation, download the image from the provided URL and save it locally.

```python
import requests

# Create a directory for images if it doesn't exist
image_dir = os.path.join(os.curdir, 'images')
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

# Define the save path
image_path = os.path.join(image_dir, 'generated_image.png')

# Retrieve and save the first generated image
image_url = generation_response["data"][0]["url"]
generated_image = requests.get(image_url).content
with open(image_path, "wb") as image_file:
    image_file.write(generated_image)

print(f"Image saved to: {image_path}")
```

Finally, use the Pillow library to open and display the downloaded image.

```python
from PIL import Image

# Display the image
img = Image.open(image_path)
img.show()  # This will open the image in your default viewer
```

## Summary

You have successfully:
1. Installed the necessary Python packages.
2. Configured the Azure OpenAI client with your endpoint.
3. Authenticated using either an API key or Azure Active Directory.
4. Generated an AI image with DALL·E.
5. Downloaded and displayed the generated image.

You can now experiment with different prompts, sizes (`'256x256'`, `'512x512'`, `'1024x1024'`), and the number of images (`n`) to create various visual content.