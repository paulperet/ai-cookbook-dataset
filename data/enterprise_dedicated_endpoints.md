# Creating and Managing Dedicated Inference Endpoints

## Introduction

Have you ever wanted to create your own machine learning API? In this guide, you'll learn how to use Hugging Face's Dedicated Inference Endpoints to deploy any model from the Hub as a production-ready API on hardware you control.

While Serverless Inference APIs are great for initial testing, they're limited to popular models and have rate limits. Dedicated Inference Endpoints give you full control over deployment configuration with hardware exclusively dedicated to your use case.

In this tutorial, you will:
- Create an Inference Endpoint via the UI and send HTTP requests
- Programmatically manage Endpoints using the `huggingface_hub` library
- Deploy three different model types: text generation, image generation, and multimodal reasoning

## Prerequisites

Before you begin, ensure you have:

1. A Hugging Face account (create one [here](https://huggingface.co/join))
2. Billing configured with a credit card (go to [Settings > Billing](https://huggingface.co/settings/billing))
3. A fine-grained access token with:
   - `User Permissions > Inference > Make calls to Inference Endpoints & Manage Inference Endpoints`
   - `Repository permissions > google/gemma-1.1-2b-it & HuggingFaceM4/idefics2-8b-chatty`

## Setup

First, install the required packages:

```bash
pip install huggingface_hub~=0.23.3
pip install transformers~=4.41.2
```

Then log in to the Hugging Face Hub:

```python
import huggingface_hub
huggingface_hub.interpreter_login()
```

## Creating Your First Endpoint via UI

Navigate to https://ui.endpoints.huggingface.co/ and click `+ New` next to `Dedicated Endpoints`. Configure your endpoint with these settings:

- **Model Repository**: `google/gemma-1.1-2b-it` (a 2.5B parameter generative LLM)
- **Endpoint Name**: Accept the auto-generated name or create your own (4-32 characters, lowercase, numbers, hyphens only)
- **Instance Configuration**: Choose your preferred GPU/CPU and region
- **Automatic Scale-to-Zero**: Configure if you want the endpoint to pause after inactivity
- **Endpoint Security Level**: `Protected` (requires HF token)
- **Advanced Configuration**: TGI container will be auto-selected for Gemma

Click `Create Endpoint`. After about a minute, your endpoint will be ready. On the Overview page, you'll find the API URL and a Playground for testing.

## Programmatic Endpoint Management

For production workflows, you'll want to manage endpoints programmatically. The `huggingface_hub` library provides comprehensive functionality.

### Listing and Managing Existing Endpoints

```python
# List all your inference endpoints
huggingface_hub.list_inference_endpoints()

# Get a specific endpoint and check its status
endpoint = huggingface_hub.get_inference_endpoint(
    name="gemma-1-1-2b-it-yci",  # your endpoint name
    namespace="MoritzLaurer"  # your username or organization
)
print(endpoint)

# Pause endpoint to stop billing
endpoint.pause()

# Resume and wait until ready
# endpoint.resume()
# endpoint.wait()

# Update endpoint hardware
# endpoint.update(
#     instance_size="x1",
#     instance_type="nvidia-a100",
# )
```

### Creating an Endpoint Programmatically

Let's recreate the same Gemma endpoint programmatically:

```python
from huggingface_hub import create_inference_endpoint

model_id = "google/gemma-1.1-2b-it"
endpoint_name = "gemma-1-1-2b-it-001"
namespace = "MoritzLaurer"  # your username or organization

# Check if endpoint already exists
available_endpoints_names = [endpoint.name for endpoint in huggingface_hub.list_inference_endpoints()]
endpoint_exists = endpoint_name in available_endpoints_names
print("Does the endpoint already exist?", endpoint_exists)

# Create new endpoint
if not endpoint_exists:
    endpoint = create_inference_endpoint(
        endpoint_name,
        repository=model_id,
        namespace=namespace,
        framework="pytorch",
        task="text-generation",
        accelerator="gpu",
        vendor="aws",
        region="us-east-1",
        instance_size="x1",
        instance_type="nvidia-a10g",
        min_replica=0,
        max_replica=1,
        type="protected",
        custom_image={
            "health_route": "/health",
            "env": {"MODEL_ID": "/repository"},
            "url": "ghcr.io/huggingface/text-generation-inference:latest",
        },
    )
    print("Waiting for endpoint to be created")
    endpoint.wait()
    print("Endpoint ready")
else:
    # Get and resume existing endpoint
    endpoint = huggingface_hub.get_inference_endpoint(name=endpoint_name, namespace=namespace)
    if endpoint.status in ["paused", "scaledToZero"]:
        print("Resuming endpoint")
        endpoint.resume()
    print("Waiting for endpoint to start")
    endpoint.wait()
    print("Endpoint ready")

# Access the endpoint URL
print(endpoint.url)
```

## Querying Your Endpoint

Now let's query your endpoint like any other LLM API.

### Standard HTTP Request

First, prepare your input using the standard messages format and apply the chat template:

```python
import requests
from transformers import AutoTokenizer

# Use your endpoint URL
API_URL = endpoint.url
HEADERS = {"Authorization": f"Bearer {huggingface_hub.get_token()}"}

def query(payload=None, api_url=None):
    response = requests.post(api_url, headers=HEADERS, json=payload)
    return response.json()

# Define conversation in messages format
messages = [
    {"role": "user", "content": "Please write a short poem about open source for me."},
]

# Apply chat template for the model
model_id = "google/gemma-1.1-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
messages_with_template = tokenizer.apply_chat_template(messages, tokenize=False)
print("Your text input after chat template:\n")
print(messages_with_template)
```

Now send the request:

```python
output = query(
    payload={
        "inputs": messages_with_template,
        "parameters": {"temperature": 0.2, "max_new_tokens": 100, "seed": 42, "return_full_text": False},
    },
    api_url=API_URL
)

print("The output from your API call:\n")
print(output)
```

### Using the OpenAI-Compatible Messages API

If your LLM runs on a TGI container, you can use the OpenAI-compatible API by appending `/v1/chat/completions` to your URL:

```python
API_URL_CHAT = API_URL + "/v1/chat/completions"

output = query(
    payload={
        "messages": messages,
        "model": "tgi",
        "parameters": {"temperature": 0.2, "max_tokens": 100, "seed": 42},
    },
    api_url=API_URL_CHAT
)

print("Output using OpenAI-compatible API:\n")
print(output)
```

### Simplified Usage with InferenceClient

The `InferenceClient` provides the most concise way to query your endpoint:

```python
from huggingface_hub import InferenceClient

client = InferenceClient()

output = client.chat_completion(
    messages,
    model=API_URL,
    temperature=0.2,
    max_tokens=100,
    seed=42,
)

print("Output using InferenceClient:\n")
print(output)
```

When you're done, pause the endpoint to stop billing:

```python
# endpoint.pause()
```

## Deploying Different Model Types

You can create endpoints for any model on the Hugging Face Hub. Let's explore two additional use cases.

### Image Generation with Stable Diffusion

First, install the image processing library:

```bash
pip install Pillow
```

Now create an image generation endpoint:

```python
from huggingface_hub import create_inference_endpoint

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
endpoint_name = "stable-diffusion-xl-base-1-0-001"
namespace = "MoritzLaurer"
task = "text-to-image"

# Check if endpoint exists
available_endpoints_names = [endpoint.name for endpoint in huggingface_hub.list_inference_endpoints()]
endpoint_exists = endpoint_name in available_endpoints_names
print("Does the endpoint already exist?", endpoint_exists)

# Create new endpoint
if not endpoint_exists:
    endpoint = create_inference_endpoint(
        endpoint_name,
        repository=model_id,
        namespace=namespace,
        framework="pytorch",
        task=task,
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
else:
    endpoint = huggingface_hub.get_inference_endpoint(name=endpoint_name, namespace=namespace)
    if endpoint.status in ["paused", "scaledToZero"]:
        print("Resuming endpoint")
        endpoint.resume()
    print("Waiting for endpoint to start")
    endpoint.wait()
    print("Endpoint ready")
```

## Conclusion

You've successfully learned how to create and manage Dedicated Inference Endpoints for various AI models. You can now:

1. Deploy any model from the Hugging Face Hub as a production API
2. Programmatically manage endpoints for CI/CD workflows
3. Query endpoints using standard HTTP requests, OpenAI-compatible APIs, or the InferenceClient
4. Deploy different types of models including text generation and image generation

Remember to pause endpoints when not in use to control costs, and explore the [Inference Endpoints documentation](https://huggingface.co/docs/inference-endpoints) for advanced configurations and best practices.