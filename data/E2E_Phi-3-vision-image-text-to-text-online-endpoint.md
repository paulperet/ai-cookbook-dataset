# Deploy Phi-3-vision-128k-instruct to an Azure ML Online Endpoint

This guide walks you through deploying the `Phi-3-vision-128k-instruct` model to an Azure Machine Learning Online Endpoint for real-time inference. You will set up the environment, deploy the model, test it with both standard and Azure OpenAI-style payloads, and clean up resources.

## Prerequisites

Before you begin, ensure you have:
* An Azure subscription
* An Azure Machine Learning workspace
* The necessary Python packages installed

## 1. Setup and Authentication

First, install the required Azure ML packages and authenticate with your workspace.

```bash
pip install azure-ai-ml azure-identity
```

Now, connect to your Azure ML workspace and the system registry.

```python
from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
)

# Authenticate with Azure
try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

# Connect to your workspace
try:
    workspace_ml_client = MLClient.from_config(credential)
    subscription_id = workspace_ml_client.subscription_id
    resource_group = workspace_ml_client.resource_group_name
    workspace_name = workspace_ml_client.workspace_name
except Exception as ex:
    print(ex)
    # Enter your workspace details manually if auto-config fails
    subscription_id = "<SUBSCRIPTION_ID>"
    resource_group = "<RESOURCE_GROUP>"
    workspace_name = "<WORKSPACE_NAME>"

workspace_ml_client = MLClient(
    credential, subscription_id, resource_group, workspace_name
)

# Connect to the AzureML system registry
registry_ml_client = MLClient(credential, registry_name="azureml")
```

## 2. Locate and Deploy the Model

### 2.1 Find the Model in the Registry

Check for the `Phi-3-vision-128k-instruct` model in the AzureML registry.

```python
model_name = "Phi-3-vision-128k-instruct"
version_list = list(registry_ml_client.models.list(model_name))

if len(version_list) == 0:
    print("Model not found in registry")
else:
    model_version = version_list[0].version
    foundation_model = registry_ml_client.models.get(model_name, model_version)
    print(
        f"\n\nUsing model name: {foundation_model.name}, version: {foundation_model.version}, id: {foundation_model.id} for inferencing"
    )
```

### 2.2 Create an Online Endpoint

Online endpoints provide a durable REST API for integrating models into applications.

```python
import time
from azure.ai.ml.entities import ManagedOnlineEndpoint

# Create a unique endpoint name
timestamp = int(time.time())
online_endpoint_name = model_name[:13] + str(timestamp)
print(f"Creating online endpoint with name: {online_endpoint_name}")

# Define the endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description=f"Online endpoint for {foundation_model.name}, for visual chat-completion task",
    auth_mode="key",
)

# Deploy the endpoint
workspace_ml_client.begin_create_or_update(endpoint).wait()
```

### 2.3 Create a Deployment for the Endpoint

Now, create a deployment that attaches the model to your endpoint. **Note:** The commented code below shows the full deployment configuration. Uncomment and adjust the `instance_type` based on your region's availability and requirements.

```python
from azure.ai.ml.entities import ManagedOnlineDeployment, OnlineRequestSettings, ProbeSettings

deployment_name = "phi-3-vision"

# Uncomment and use this block to create the deployment
"""
demo_deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name=online_endpoint_name,
    model=foundation_model.id,
    instance_type="Standard_NC48ads_A100_v4",  # Choose an appropriate SKU
    instance_count=1,
    request_settings=OnlineRequestSettings(
        request_timeout_ms=180000,
        max_queue_wait_ms=500,
    ),
    liveness_probe=ProbeSettings(
        failure_threshold=49,
        success_threshold=1,
        timeout=299,
        period=180,
        initial_delay=180,
    ),
    readiness_probe=ProbeSettings(
        failure_threshold=10,
        success_threshold=1,
        timeout=10,
        period=10,
        initial_delay=10,
    ),
)

workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()

# Route all traffic to the new deployment
endpoint.traffic = {deployment_name: 100}
workspace_ml_client.begin_create_or_update(endpoint).result()
"""
```

## 3. Test the Endpoint with Sample Data

### 3.1 Prepare a Test Payload

Create a JSON payload containing an image URL and a text prompt for the vision model.

```python
import json
import os

test_json = {
    "input_data": {
        "input_string": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://www.ilankelman.org/stopsigns/australia.jpg"
                        },
                    },
                    {
                        "type": "text",
                        "text": "What is shown in this image? Be extremely detailed and specific.",
                    },
                ],
            },
        ],
        "parameters": {"temperature": 0.7, "max_new_tokens": 2048},
    }
}

# Save the payload to a file
sample_score_file_path = os.path.join(".", "sample_chat_completions_score.json")
with open(sample_score_file_path, "w") as f:
    json.dump(test_json, f, indent=4)

print("Input payload:\n")
print(json.dumps(test_json, indent=4))
```

### 3.2 Invoke the Endpoint

Send the payload to your deployed endpoint and parse the response.

```python
import pandas as pd

# Invoke the endpoint
response = workspace_ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name=deployment_name,
    request_file=sample_score_file_path,
)

print("Raw JSON Response: \n", response, "\n")

# Parse and display the generated text
json_data = json.loads(response)
response_df = pd.DataFrame([json_data])
print("Generated Text:\n", response_df["output"].iloc[0])
```

## 4. Test with an Azure OpenAI-Style Payload

The endpoint also supports a payload format compatible with the Azure OpenAI API, which can be useful for client applications expecting that structure.

### 4.1 Create an Azure OpenAI-Style Payload

```python
aoai_test_json = {
    "model": foundation_model.name,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://www.ilankelman.org/stopsigns/australia.jpg"
                    },
                },
                {
                    "type": "text",
                    "text": "What is shown in this image? Be extremely detailed and specific.",
                },
            ],
        }
    ],
    "temperature": 0.7,
    "max_new_tokens": 2048,
}
```

### 4.2 Get Endpoint Credentials and URI

Retrieve the scoring URI and access key for your endpoint.

```python
# Get the standard scoring URI
scoring_uri = workspace_ml_client.online_endpoints.get(
    name=online_endpoint_name
).scoring_uri

# Modify the URI for Azure OpenAI-style requests
aoai_format_scoring_uri = scoring_uri.replace("/score", "/v1/chat/completions")

# Get the primary key for authentication
data_plane_token = workspace_ml_client.online_endpoints.get_keys(
    name=online_endpoint_name
).primary_key
```

### 4.3 Send the Request

Use the `urllib` library to send a direct HTTP request with the Azure OpenAI-style payload.

```python
import urllib.request

# Prepare the request
body = str.encode(json.dumps(aoai_test_json))
url = aoai_format_scoring_uri
api_key = data_plane_token

headers = {"Content-Type": "application/json", "Authorization": ("Bearer " + api_key)}
req = urllib.request.Request(url, body, headers)

# Send the request and handle the response
try:
    response = urllib.request.urlopen(req)
    result = response.read().decode("utf-8")
    print(result)
except urllib.error.HTTPError as error:
    print(f"The request failed with status code: {error.code}")
    print(error.info())
    print(error.read().decode("utf8", "ignore"))
```

## 5. Clean Up Resources

To avoid incurring unnecessary costs, delete the online endpoint and its associated compute resources when you are finished.

```python
workspace_ml_client.online_endpoints.begin_delete(name=online_endpoint_name).wait()
```

## Summary

You have successfully deployed the `Phi-3-vision-128k-instruct` model to an Azure ML Online Endpoint. You learned how to:
1. Authenticate and connect to your workspace.
2. Locate a model in the registry and deploy it to an endpoint.
3. Test the endpoint using both standard and Azure OpenAI-compatible payloads.
4. Clean up resources to stop billing.

This endpoint is now ready to serve real-time, vision-enabled chat completions for your applications.