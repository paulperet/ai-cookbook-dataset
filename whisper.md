# Azure audio whisper (preview) example

> Note: There is a newer version of the openai library available. See https://github.com/openai/openai-python/discussions/742

The example shows how to use the Azure OpenAI Whisper model to transcribe audio files.

## Setup

First, we install the necessary dependencies.

```python
! pip install "openai>=0.28.1,<1.0.0"
! pip install python-dotenv
```

Next, we'll import our libraries and configure the Python OpenAI SDK to work with the Azure OpenAI service.

> Note: In this example, we configured the library to use the Azure API by setting the variables in code. For development, consider setting the environment variables instead:

```
OPENAI_API_BASE
OPENAI_API_KEY
OPENAI_API_TYPE
OPENAI_API_VERSION
```

```python
import os
import dotenv
import openai


dotenv.load_dotenv()
```

To properly access the Azure OpenAI Service, we need to create the proper resources at the [Azure Portal](https://portal.azure.com) (you can check a detailed guide on how to do this in the [Microsoft Docs](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal))

Once the resource is created, the first thing we need to use is its endpoint. You can get the endpoint by looking at the *"Keys and Endpoints"* section under the *"Resource Management"* section. Having this, we will set up the SDK using this information:

```python
openai.api_base = os.environ["OPENAI_API_BASE"]

# Min API version that supports Whisper
openai.api_version = "2023-09-01-preview"

# Enter the deployment_id to use for the Whisper model
deployment_id = "<deployment-id-for-your-whisper-model>"
```

### Authentication

The Azure OpenAI service supports multiple authentication mechanisms that include API keys and Azure credentials.

```python
# set to True if using Azure Active Directory authentication
use_azure_active_directory = False
```

#### Authentication using API key

To set up the OpenAI SDK to use an *Azure API Key*, we need to set up the `api_type` to `azure` and set `api_key` to a key associated with your endpoint (you can find this key in *"Keys and Endpoints"* under *"Resource Management"* in the [Azure Portal](https://portal.azure.com))

```python
if not use_azure_active_directory:
    openai.api_type = 'azure'
    openai.api_key = os.environ["OPENAI_API_KEY"]
```

#### Authentication using Azure Active Directory
Let's now see how we can get a key via Microsoft Active Directory Authentication.

```python
from azure.identity import DefaultAzureCredential

if use_azure_active_directory:
    default_credential = DefaultAzureCredential()
    token = default_credential.get_token("https://cognitiveservices.azure.com/.default")

    openai.api_type = 'azure_ad'
    openai.api_key = token.token
```

A token is valid for a period of time, after which it will expire. To ensure a valid token is sent with every request, you can refresh an expiring token by hooking into requests.auth:

```python
import typing
import time
import requests

if typing.TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class TokenRefresh(requests.auth.AuthBase):

    def __init__(self, credential: "TokenCredential", scopes: typing.List[str]) -> None:
        self.credential = credential
        self.scopes = scopes
        self.cached_token: typing.Optional[str] = None

    def __call__(self, req):
        if not self.cached_token or self.cached_token.expires_on - time.time() < 300:
            self.cached_token = self.credential.get_token(*self.scopes)
        req.headers["Authorization"] = f"Bearer {self.cached_token.token}"
        return req

if use_azure_active_directory:
    session = requests.Session()
    session.auth = TokenRefresh(default_credential, ["https://cognitiveservices.azure.com/.default"])

    openai.requestssession = session
```

## Audio transcription

Audio transcription, or speech-to-text, is the process of converting spoken words into text. Use the `openai.Audio.transcribe` method to transcribe an audio file stream to text.

You can get sample audio files from the [Azure AI Speech SDK repository at GitHub](https://github.com/Azure-Samples/cognitive-services-speech-sdk/tree/master/sampledata/audiofiles).

```python
# download sample audio file
import requests

sample_audio_url = "https://github.com/Azure-Samples/cognitive-services-speech-sdk/raw/master/sampledata/audiofiles/wikipediaOcelot.wav"
audio_file = requests.get(sample_audio_url)
with open("wikipediaOcelot.wav", "wb") as f:
    f.write(audio_file.content)
```

```python
transcription = openai.Audio.transcribe(
    file=open("wikipediaOcelot.wav", "rb"),
    model="whisper-1",
    deployment_id=deployment_id,
)
print(transcription.text)
```