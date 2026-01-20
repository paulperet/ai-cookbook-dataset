# Inference Endpoints (dedicated) 
_Authored by: [Moritz Laurer](https://huggingface.co/MoritzLaurer)_

Have you ever wanted to create your own machine learning API? That's what we will do in this recipe with the [HF Dedicated Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index). Inference Endpoints enable you to pick any of the hundreds of thousands of models on the HF Hub, create your own API on a deployment platform you control, and on hardware you choose.

[Serverless Inference APIs](link-to-recipe) are great for initial testing, but they are limited to a pre-configured selection of popular models and they are rate limited, because the serverless API's hardware is used by many users at the same time. With a Dedicated Inference Endpoint, you can customize the deployment of your model and the hardware is exclusively dedicated to you. 

In this recipe, we will: 
- Create an Inference Endpoint via a simple UI and send standard HTTP requests to the Endpoint
- Create and manage different Inference Endpoints programmatically with the `huggingface_hub` library
- Cover three use-cases: text generation with an LLM, image generation with Stable Diffusion, and reasoning over images with Idefics2. 

## Install and login
In case you don't have a HF Account, you can create your account [here](https://huggingface.co/join). If you work in a larger team, you can also create a [HF Organization](https://huggingface.co/organizations) and manage all your models, datasets and Endpoints via this organization. Dedicated Inference Endpoints are a paid service and you will therefore need to add a credit card to the [billing settings](https://huggingface.co/settings/billing) of your personal HF account, or of your HF organization.  

You can then create a user access token [here](https://huggingface.co/docs/hub/security-tokens). A token with `read` or `write` permissions will work for this guide, but we encourage the use of fine-grained tokens for increased security. For this notebook, you'll need a fine-grained token with `User Permissions > Inference > Make calls to Inference Endpoints & Manage Inference Endpoints` and `Repository permissions > google/gemma-1.1-2b-it & HuggingFaceM4/idefics2-8b-chatty`.


```python
!pip install huggingface_hub~=0.23.3
!pip install transformers~=4.41.2
```


```python
# Login to the HF Hub. We recommend using this login method 
# to avoid the need for explicitly storing your HF token in variables 
import huggingface_hub
huggingface_hub.interpreter_login()
```

## Creating your first Endpoint

With this initial setup out of the way, we can now create our first Endpoint. Navigate to https://ui.endpoints.huggingface.co/ and click on `+ New` next to `Dedicated Endpoints`. You will then see the interface for creating a new Endpoint with the following options (see image below):

- **Model Repository**: Here you can insert the identifier of any model on the HF Hub. For this initial demonstration, we use [google/gemma-1.1-2b-it](https://huggingface.co/google/gemma-1.1-2b-it), a small generative LLM (2.5B parameters). 
- **Endpoint Name**: The Endpoint Name is automatically generated based on the model identifier, but you are free to change the name. Valid Endpoint names must only contain lower-case characters, numbers or hyphens ("-") and are between 4 to 32 characters long.
- **Instance Configuration**: Here you can choose from a wide range of CPUs or GPUs from all major cloud platforms. You can also adjust the region, for example if you need to host your Endpoint in the EU. 
- **Automatic Scale-to-Zero**: You can configure your Endpoint to scale to zero GPUs/CPUs after a certain amount of time. Scaled-to-zero Endpoints are not billed anymore. Note that restarting the Endpoint requires the model to be re-loaded into memory (and potentially re-downloaded), which can take several minutes for large models. 
- **Endpoint Security Level**: The standard security level is `Protected`, which requires an authorized HF token for accessing the Endpoint. `Public` Endpoints are accessible by anyone without token authentification. `Private` Endpoints are only available through an intra-region secured AWS or Azure PrivateLink connection.
- **Advanced configuration**: Here you can select some advanced options like the Docker container type. As Gemma is compatible with [Text Generation Inference (TGI)](https://huggingface.co/docs/text-generation-inference/index) containers, the system automatically selects TGI as the container type and other good default values.

For this guide, select the options in the image below and click on `Create Endpoint`. 

After roughly one minute, your Endpoint will be created and you will see a page similar to the image below. 

On the Endpoint's `Overview` page, will find the URL for querying the Endpoint, a Playground for testing the model and additional tabs on `Analytics`, `Usage & Cost`, `Logs`and `Settings`.  

### Creating and managing Endpoints programmatically

When moving into production, you don't always want to manually start, stop and modify your Endpoints. The `huggingface_hub` library provides good functionality for managing your Endpoints programmatically. See the docs [here](https://huggingface.co/docs/huggingface_hub/guides/inference_endpoints) and details on all functions [here](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_endpoints). Here are some key functions:



```python
# list all your inference endpoints
huggingface_hub.list_inference_endpoints()

# get an existing endpoint and check it's status
endpoint = huggingface_hub.get_inference_endpoint(
    name="gemma-1-1-2b-it-yci",  # the name of the endpoint 
    namespace="MoritzLaurer"  # your user name or organization name
)
print(endpoint)

# Pause endpoint to stop billing
endpoint.pause()

# Resume and wait until the endpoint is ready
#endpoint.resume()
#endpoint.wait()

# Update the endpoint to a different GPU
# You can find the correct arguments for different hardware types in this table: https://huggingface.co/docs/inference-endpoints/pricing#gpu-instances
#endpoint.update(
#    instance_size="x1",
#    instance_type="nvidia-a100",  # nvidia-a10g
#)
```

You can also create an inference Endpoint programmatically. Let's recreate the same `gemma` LLM Endpoint as the one created with the UI.


```python
from huggingface_hub import create_inference_endpoint


model_id = "google/gemma-1.1-2b-it"
endpoint_name = "gemma-1-1-2b-it-001"  # Valid Endpoint names must only contain lower-case characters, numbers or hyphens ("-") and are between 4 to 32 characters long.
namespace = "MoritzLaurer"  # your user or organization name


# check if endpoint with this name already exists from previous tests
available_endpoints_names = [endpoint.name for endpoint in huggingface_hub.list_inference_endpoints()]
if endpoint_name in available_endpoints_names:
    endpoint_exists = True
else: 
    endpoint_exists = False
print("Does the endpoint already exist?", endpoint_exists)
    

# create new endpoint
if not endpoint_exists:
    endpoint = create_inference_endpoint(
        endpoint_name,
        repository=model_id,
        namespace=namespace,
        framework="pytorch",
        task="text-generation",
        # see the available hardware options here: https://huggingface.co/docs/inference-endpoints/pricing#pricing
        accelerator="gpu",
        vendor="aws",
        region="us-east-1",
        instance_size="x1",
        instance_type="nvidia-a10g",
        min_replica=0,
        max_replica=1,
        type="protected",
        # since the LLM is compatible with TGI, we specify that we want to use the latest TGI image
        custom_image={
            "health_route": "/health",
            "env": {
                "MODEL_ID": "/repository"
            },
            "url": "ghcr.io/huggingface/text-generation-inference:latest",
        },
    )
    print("Waiting for endpoint to be created")
    endpoint.wait()
    print("Endpoint ready")

# if endpoint with this name already exists, get and resume existing endpoint
else:
    endpoint = huggingface_hub.get_inference_endpoint(name=endpoint_name, namespace=namespace)
    if endpoint.status in ["paused", "scaledToZero"]:
        print("Resuming endpoint")
        endpoint.resume()
    print("Waiting for endpoint to start")
    endpoint.wait()
    print("Endpoint ready")
```


```python
# access the endpoint url for API calls
print(endpoint.url)
```

## Querying your Endpoint

Now let's query this Endpoint like any other LLM API. First copy the Endpoint URL from the interface (or use `endpoint.url`) and assign it to `API_URL` below. We then use the standardised messages format for the text inputs, i.e. a dictionary of user and assistant messages, which you might know from other LLM API services. We then need to apply the chat template to the messages, which LLMs like Gemma, Llama-3 etc. have been trained to expect (see details on in the [docs](https://huggingface.co/docs/transformers/main/en/chat_templating)). For most recent generative LLMs, it is essential to apply this chat template, otherwise the model's performance will degrade without throwing an error. 


```python
import requests
from transformers import AutoTokenizer

# paste your endpoint URL here or reuse endpoint.url if you created the endpoint programmatically
API_URL = endpoint.url  # or paste link like "https://dz07884a53qjqb98.us-east-1.aws.endpoints.huggingface.cloud" 
HEADERS = {"Authorization": f"Bearer {huggingface_hub.get_token()}"}

# function for standard http requests
def query(payload=None, api_url=None):
    response = requests.post(api_url, headers=HEADERS, json=payload)
    return response.json()


# define conversation input in messages format
# you can also provide multiple turns between user and assistant
messages = [
    {"role": "user", "content": "Please write a short poem about open source for me."},
    #{"role": "assistant", "content": "I am not in the mood."},
    #{"role": "user", "content": "Can you please do this for me?"},
]

# apply the chat template for the respective model
model_id = "google/gemma-1.1-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id) 
messages_with_template = tokenizer.apply_chat_template(messages, tokenize=False)
print("Your text input looks like this, after the chat template has been applied:\n")
print(messages_with_template)
```

    Your text input looks like this, after the chat template has been applied:
    
    <bos><start_of_turn>user
    Please write a short poem about open source for me.<end_of_turn>
    



```python
# send standard http request to endpoint
output = query(
    payload = {
        "inputs": messages_with_template,
        "parameters": {"temperature": 0.2, "max_new_tokens": 100, "seed": 42, "return_full_text": False},
    },
    api_url = API_URL
)

print("The output from your API/Endpoint call:\n")
print(output)
```

    The output from your API/Endpoint call:
    
    [{'generated_text': "Free to use, free to share,\nA collaborative code, a community's care.\n\nCode transparent, bugs readily found,\nContributions welcome, stories unbound.\nOpen source, a gift to all,\nBuilding the future, one line at a call.\n\nSo join the movement, embrace the light,\nOpen source, shining ever so bright."}]


That's it, you've made the first request to your Endpoint - your very own API!

If you want the Endpoint to handle the chat template automatically and if your LLM runs on a TGI container, you can also use the [messages API](https://huggingface.co/docs/text-generation-inference/en/messages_api) by appending the `/v1/chat/completions` path to the URL. With the `/v1/chat/completions` path, the [TGI](https://huggingface.co/docs/text-generation-inference/index) container running on the Endpoint applies the chat template automatically and is fully compatible with OpenAI's API structure for easier interoperability. See the [TGI Swagger UI](https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/chat_completions) for all available parameters. Note that the parameters accepted by the default `/` path and by the `/v1/chat/completions` path are slightly different. Here is the slightly modified code for using the messages API:


```python
API_URL_CHAT = API_URL + "/v1/chat/completions"

output = query(
    payload = {
        "messages": messages,
        "model": "tgi",
        "parameters": {"temperature": 0.2, "max_tokens": 100, "seed": 42},
    },
    api_url = API_URL_CHAT
)

print("The output from your API/Endpoint call with the OpenAI-compatible messages API route:\n")
print(output)

```

    The output from your API/Endpoint call with the OpenAI-compatible messages API route:
    
    {'id': '', 'object': 'text_completion', 'created': 1718283608, 'model': '/repository', 'system_fingerprint': '2.0.5-dev0-sha-90184df', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '**Open Source**\n\nA license for the mind,\nTo share, distribute, and bind,\nIdeas freely given birth,\nFor the good of all to sort.\n\nCode transparent, eyes open wide,\nA permission for the wise,\nTo learn, to build, to use at will,\nA future bright, we help fill.\n\nFrom servers vast to candles low,\nOpen source, a guiding key,\nFor progress made, knowledge shared,\nA future brimming with'}, 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 20, 'completion_tokens': 100, 'total_tokens': 120}}


### Simplified Endpoint usage with the InferenceClient

You can also use the [`InferenceClient`](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client#huggingface_hub.InferenceClient) to easily send requests to your Endpoint. The client is a convenient utility available in the `huggingface_hub` Python library that allows you to easily make calls to both [Dedicated Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index) and the [Serverless Inference API](https://huggingface.co/docs/api-inference/index). See the [docs](https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client#inference) for details. 

This is the most succinct way of sending requests to your Endpoint:


```python
from huggingface_hub import InferenceClient

client = InferenceClient()

output = client.chat_completion(
    messages,  # the chat template is applied automatically, if your endpoint uses a TGI container
    model=API_URL, 
    temperature=0.2, max_tokens=100, seed=42,
)

print("The output from your API/Endpoint call with the InferenceClient:\n")
print(output)
```


```python
# pause the endpoint to stop billing
#endpoint.pause()
```

## Creating Endpoints for a wide variety of models
Following the same process, you can create Endpoints for any of the models on the HF Hub. Let's illustrate some other use-cases.

### Image generation with Stable Diffusion
We can create an image generation Endpoint with almost the exact same code as for the LLM. The only difference is that we do not use the TGI container in this case, as TGI is only designed for LLMs (and vision LMs). 


```python
!pip install Pillow  # for image processing
```

    [Collecting Pillow, ..., Successfully installed Pillow-10.3.0]



```python
from huggingface_hub import create_inference_endpoint

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
endpoint_name = "stable-diffusion-xl-base-1-0-001"  # Valid Endpoint names must only contain lower-case characters, numbers or hyphens ("-") and are between 4 to 32 characters long.
namespace = "MoritzLaurer"  # your user or organization name
task = "text-to-image"

# check if endpoint with this name already exists from previous tests
available_endpoints_names = [endpoint.name for endpoint in huggingface_hub.list_inference_endpoints()]
if endpoint_name in available_endpoints_names:
    endpoint_exists = True
else: 
    endpoint_exists = False
print("Does the endpoint already exist?", endpoint_exists)
    

# create new endpoint
if not endpoint_exists:
    endpoint = create_inference_endpoint(
        endpoint_name,
        repository=model_id,
        namespace=namespace,
        framework="pytorch",
        task=task,
        # see the available hardware options here: https://huggingface.co/docs/inference-endpoints/pricing#pricing
        accelerator="gpu",
        vendor="aws",
        region="us-east-1",
        instance_size="x1",
        instance_type="nvidia-a100",
        min_replica=0,
        max_replica=1,
        type="protected",
    )
    print("Waiting for endpoint to be created")
    endpoint.wait()
    print("Endpoint ready")

# if endpoint with this name already exists, get existing endpoint
else:
    endpoint = huggingface_hub.get_inference_endpoint(name=endpoint_name, namespace=namespace)
    if endpoint.status in ["paused", "scaledToZero"]:
