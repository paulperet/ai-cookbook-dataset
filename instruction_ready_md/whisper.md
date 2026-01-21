# Guide: Transcribe Audio with Azure OpenAI Whisper (Preview)

This guide walks you through using the Azure OpenAI Whisper model to transcribe audio files. You'll learn how to set up your environment, authenticate with Azure, and run a transcription.

## Prerequisites

Before you begin, ensure you have:
*   An Azure account with an active subscription.
*   An Azure OpenAI resource with the Whisper model deployed. Follow the [official Microsoft guide](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal) to create one.
*   Your resource endpoint and a deployment ID for the Whisper model.

## Step 1: Install Dependencies

First, install the required Python libraries. We'll use a specific version of the OpenAI Python SDK compatible with the Azure API.

```bash
pip install "openai>=0.28.1,<1.0.0"
pip install python-dotenv
```

## Step 2: Configure the Azure OpenAI Client

Create a new Python script or notebook. Start by importing the necessary libraries and loading environment variables from a `.env` file.

```python
import os
import dotenv
import openai

# Load environment variables from a .env file
dotenv.load_dotenv()
```

Next, configure the OpenAI SDK to connect to your Azure endpoint. You need to set the API base URL, the API version, and specify your Whisper model's deployment ID.

> **Tip:** For better security in development, set these values as environment variables (`OPENAI_API_BASE`, `OPENAI_API_KEY`, `OPENAI_API_TYPE`, `OPENAI_API_VERSION`) instead of hardcoding them.

```python
# Set the endpoint for your Azure OpenAI resource
openai.api_base = os.environ["OPENAI_API_BASE"]

# Use the minimum API version that supports Whisper
openai.api_version = "2023-09-01-preview"

# Enter your specific Whisper model deployment ID here
deployment_id = "<deployment-id-for-your-whisper-model>"
```

## Step 3: Set Up Authentication

Azure OpenAI supports two primary authentication methods: API keys and Azure Active Directory (AAD). Choose one based on your security requirements.

### Option A: Authenticate with an API Key (Simpler)

This is the most straightforward method. Set the API type to `'azure'` and provide your key.

```python
# Set to False to use API Key authentication
use_azure_active_directory = False

if not use_azure_active_directory:
    openai.api_type = 'azure'
    openai.api_key = os.environ["OPENAI_API_KEY"]
```

### Option B: Authenticate with Azure Active Directory (More Secure)

For production systems, using Azure AD credentials is recommended. This method uses the `azure-identity` library to fetch a token.

First, install the identity package if you haven't already:
```bash
pip install azure-identity
```

Then, configure the client to use token-based authentication.

```python
from azure.identity import DefaultAzureCredential

# Set to True to use Azure Active Directory authentication
use_azure_active_directory = True

if use_azure_active_directory:
    # Get credentials from the local environment (e.g., VS Code, Azure CLI, Managed Identity)
    default_credential = DefaultAzureCredential()
    # Request a token for the Cognitive Services scope
    token = default_credential.get_token("https://cognitiveservices.azure.com/.default")

    openai.api_type = 'azure_ad'
    openai.api_key = token.token
```

**Handling Token Refresh:** Tokens expire. To automatically refresh an expiring token for all requests, you can attach a custom authentication handler to the SDK's session.

```python
import typing
import time
import requests

if typing.TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class TokenRefresh(requests.auth.AuthBase):
    """Custom auth class to refresh Azure AD tokens automatically."""
    def __init__(self, credential: "TokenCredential", scopes: typing.List[str]) -> None:
        self.credential = credential
        self.scopes = scopes
        self.cached_token: typing.Optional[str] = None

    def __call__(self, req):
        # Refresh token if it's expired or will expire in less than 5 minutes
        if not self.cached_token or self.cached_token.expires_on - time.time() < 300:
            self.cached_token = self.credential.get_token(*self.scopes)
        req.headers["Authorization"] = f"Bearer {self.cached_token.token}"
        return req

if use_azure_active_directory:
    # Create a session with the automatic token refresh mechanism
    session = requests.Session()
    session.auth = TokenRefresh(default_credential, ["https://cognitiveservices.azure.com/.default"])
    # Attach the session to the OpenAI client
    openai.requestssession = session
```

## Step 4: Transcribe an Audio File

With authentication configured, you can now transcribe audio. Let's start by downloading a sample audio file to test with.

```python
import requests

# URL to a sample audio file from Azure's Speech SDK samples
sample_audio_url = "https://github.com/Azure-Samples/cognitive-services-speech-sdk/raw/master/sampledata/audiofiles/wikipediaOcelot.wav"

# Download the file
audio_file = requests.get(sample_audio_url)
with open("wikipediaOcelot.wav", "wb") as f:
    f.write(audio_file.content)

print("Sample audio file downloaded.")
```

Now, use the `openai.Audio.transcribe` method. You must pass the audio file in binary read mode, specify the model name (`"whisper-1"`), and provide your `deployment_id`.

```python
# Open the audio file and send it to the Whisper model for transcription
transcription = openai.Audio.transcribe(
    file=open("wikipediaOcelot.wav", "rb"),
    model="whisper-1",
    deployment_id=deployment_id,
)

# Print the transcribed text
print("Transcription:")
print(transcription.text)
```

## Summary

You have successfully set up the Azure OpenAI Python SDK, configured authentication (using either an API key or Azure AD), and transcribed an audio file using the Whisper model. You can now adapt this code to process your own audio files by changing the file path in the `transcribe` method.

**Next Steps:**
*   Experiment with different audio formats (Whisper supports `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.m4a`, `.wav`, and `.webm`).
*   Explore the `transcribe` method's optional parameters, such as `language` or `prompt`, to improve accuracy for specific use cases.
*   Integrate this transcription logic into a larger application, such as a voice note processor or meeting summarizer.