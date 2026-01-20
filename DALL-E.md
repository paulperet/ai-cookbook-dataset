# Azure DALL·E image generation example

> Note: There is a newer version of the openai library available. See https://github.com/openai/openai-python/discussions/742

This notebook shows how to generate images with the Azure OpenAI service.

## Setup

First, we install the necessary dependencies.

```python
! pip install "openai>=0.28.1,<1.0.0"
# We need requests to retrieve the generated image
! pip install requests
# We use Pillow to display the generated image
! pip install pillow 
# (Optional) If you want to use Microsoft Active Directory
! pip install azure-identity
```

```python
import os
import openai
```

Additionally, to properly access the Azure OpenAI Service, we need to create the proper resources at the [Azure Portal](https://portal.azure.com) (you can check a detailed guide on how to do this in the [Microsoft Docs](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal))

Once the resource is created, the first thing we need to use is its endpoint. You can get the endpoint by looking at the *"Keys and Endpoints"* section under the *"Resource Management"* section. Having this, we will set up the SDK using this information:

```python
openai.api_base = '' # Add your endpoint here

# At the moment DALL·E is only supported by the 2023-06-01-preview API version
openai.api_version = '2023-06-01-preview'
```

### Authentication

The Azure OpenAI service supports multiple authentication mechanisms that include API keys and Azure credentials.

```python
use_azure_active_directory = False
```

#### Authentication using API key

To set up the OpenAI SDK to use an *Azure API Key*, we need to set up the `api_type` to `azure` and set `api_key` to a key associated with your endpoint (you can find this key in *"Keys and Endpoints"* under *"Resource Management"* in the [Azure Portal](https://portal.azure.com))

```python
if not use_azure_active_directory:
    openai.api_type = 'azure'
    openai.api_key = os.environ["OPENAI_API_KEY"]
```

> Note: In this example, we configured the library to use the Azure API by setting the variables in code. For development, consider setting the environment variables instead:

```
OPENAI_API_BASE
OPENAI_API_KEY
OPENAI_API_TYPE
OPENAI_API_VERSION
```

#### Authentication using Microsoft Active Directory
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

## Generations

With setup and authentication complete, you can now generate images on the Azure OpenAI service and retrieve them from the returned URLs.

#### 1. Generate the images

The first step in this process is to actually generate the images:

```python
generation_response = openai.Image.create(
    prompt='A cyberpunk monkey hacker dreaming of a beautiful bunch of bananas, digital art',
    size='1024x1024',
    n=2
)

print(generation_response)
```

Having the response from the `Image.create` call, we download from the URL using `requests`.

```python
import os
import requests

# First a little setup
image_dir = os.path.join(os.curdir, 'images')
# If the directory doesn't exist, create it
if not os.path.isdir(image_dir):
    os.mkdir(image_dir)

# With the directory in place, we can initialize the image path (note that filetype should be png)
image_path = os.path.join(image_dir, 'generated_image.png')

# Now we can retrieve the generated image
image_url = generation_response["data"][0]["url"]  # extract image URL from response
generated_image = requests.get(image_url).content  # download the image
with open(image_path, "wb") as image_file:
    image_file.write(generated_image)
```

With the image downloaded, we use the [Pillow](https://pypi.org/project/Pillow/) library to open and display it:

```python
from PIL import Image 

display(Image.open(image_path))
```