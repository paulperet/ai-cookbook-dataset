# Azure embeddings example

> Note: There is a newer version of the openai library available. See https://github.com/openai/openai-python/discussions/742

This example will cover embeddings using the Azure OpenAI service.

## Setup

First, we install the necessary dependencies.

! pip install "openai>=0.28.1,<1.0.0"

For the following sections to work properly we first have to setup some things. Let's start with the `api_base` and `api_version`. To find your `api_base` go to https://portal.azure.com, find your resource and then under "Resource Management" -> "Keys and Endpoints" look for the "Endpoint" value.

```python
import os
import openai
```

```python
openai.api_version = '2023-05-15'
openai.api_base = '' # Please add your endpoint here
```

We next have to setup the `api_type` and `api_key`. We can either get the key from the portal or we can get it through Microsoft Active Directory Authentication. Depending on this the `api_type` is either `azure` or `azure_ad`.

### Setup: Portal
Let's first look at getting the key from the portal. Go to https://portal.azure.com, find your resource and then under "Resource Management" -> "Keys and Endpoints" look for one of the "Keys" values.

```python
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

### (Optional) Setup: Microsoft Active Directory Authentication
Let's now see how we can get a key via Microsoft Active Directory Authentication. Uncomment the following code if you want to use Active Directory Authentication instead of keys from the portal.

```python
# from azure.identity import DefaultAzureCredential

# default_credential = DefaultAzureCredential()
# token = default_credential.get_token("https://cognitiveservices.azure.com/.default")

# openai.api_type = 'azure_ad'
# openai.api_key = token.token
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

session = requests.Session()
session.auth = TokenRefresh(default_credential, ["https://cognitiveservices.azure.com/.default"])

openai.requestssession = session
```

## Deployments
In this section we are going to create a deployment that we can use to create embeddings.

### Deployments: Create manually
Let's create a deployment using the `text-similarity-curie-001` model. Create a new deployment by going to your Resource in your portal under "Resource Management" -> "Model deployments".

```python
deployment_id = '' # Fill in the deployment id from the portal here
```

### Deployments: Listing
Now because creating a new deployment takes a long time, let's look in the subscription for an already finished deployment that succeeded.

```python
print('While deployment running, selecting a completed one that supports embeddings.')
deployment_id = None
result = openai.Deployment.list()
for deployment in result.data:
    if deployment["status"] != "succeeded":
        continue
    
    model = openai.Model.retrieve(deployment["model"])
    if model["capabilities"]["embeddings"] != True:
        continue
    
    deployment_id = deployment["id"]
    break

if not deployment_id:
    print('No deployment with status: succeeded found.')
else:
    print(f'Found a succeeded deployment that supports embeddings with id: {deployment_id}.')
```

### Embeddings
Now let's send a sample embedding to the deployment.

```python
embeddings = openai.Embedding.create(deployment_id=deployment_id,
                                     input="The food was delicious and the waiter...")
                                
print(embeddings)
```