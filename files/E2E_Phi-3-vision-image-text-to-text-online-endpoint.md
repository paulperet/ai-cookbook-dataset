## (Visual) Chat Completion inference using Online Endpoints

This sample shows how to deploy `Phi-3-vision-128k-instruct' to an online endpoint for inference.

### Outline
* Set up pre-requisites
* Pick a model to deploy
* Download and prepare data for inference
* Deploy the model for real time inference
* Test the endpoint
* Test the endpoint using Azure OpenAI style payload
* Clean up resources

### 1. Set up pre-requisites
* Install dependencies
* Connect to AzureML Workspace. Learn more at [set up SDK authentication](https://learn.microsoft.com/azure/machine-learning/how-to-setup-authentication?tabs=sdk). Replace  `<WORKSPACE_NAME>`, `<RESOURCE_GROUP>` and `<SUBSCRIPTION_ID>` below.
* Connect to `azureml` system registry


```python
# Import necessary modules
from azure.ai.ml import MLClient
from azure.identity import (
    DefaultAzureCredential,
    InteractiveBrowserCredential,
)

try:
    # Try to get the default Azure credential
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # If default credential is not available, use interactive browser credential
    credential = InteractiveBrowserCredential()

try:
    # Try to create an MLClient using the provided credential
    workspace_ml_client = MLClient.from_config(credential)
    subscription_id = workspace_ml_client.subscription_id
    resource_group = workspace_ml_client.resource_group_name
    workspace_name = workspace_ml_client.workspace_name
except Exception as ex:
    print(ex)
    # If MLClient creation fails, enter the details of your AML workspace manually
    subscription_id = "<SUBSCRIPTION_ID>"
    resource_group = "<RESOURCE_GROUP>"
    workspace_name = "<WORKSPACE_NAME>"

# Create an MLClient instance with the provided credentials and workspace details
workspace_ml_client = MLClient(
    credential, subscription_id, resource_group, workspace_name
)

# The models, fine tuning pipelines, and environments are available in the AzureML system registry, "azureml"
registry_ml_client = MLClient(credential, registry_name="azureml")
```

### 2. Deploy the model to an online endpoint
Online endpoints give a durable REST API that can be used to integrate with applications that need to use the model.


```python
# This code checks if the model with the specified name exists in the registry.
# If the model exists, it retrieves the first version of the model and prints its details.
# If the model does not exist, it prints a message indicating that the model was not found.

# model_name: Name of the model to check in the registry
model_name = "Phi-3-vision-128k-instruct"

# Get the list of versions for the specified model name
version_list = list(registry_ml_client.models.list(model_name))

# Check if any versions of the model exist in the registry
if len(version_list) == 0:
    print("Model not found in registry")
else:
    # Get the first version of the model
    model_version = version_list[0].version
    foundation_model = registry_ml_client.models.get(model_name, model_version)
    
    # Print the details of the model
    print(
        "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
            foundation_model.name, foundation_model.version, foundation_model.id
        )
    )
```


```python
# Import necessary modules
import time
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

# Create online endpoint - endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name
timestamp = int(time.time())
online_endpoint_name = model_name[:13] + str(timestamp)
print(f"Creating online endpoint with name: {online_endpoint_name}")

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description=f"Online endpoint for {foundation_model.name}, for visual chat-completion task",
    auth_mode="key",
)
workspace_ml_client.begin_create_or_update(endpoint).wait()
```


```python
# This code creates a deployment for the online endpoint.
# It sets the deployment name, endpoint name, model, instance type, instance count, and request settings.
# It also sets the liveness probe and readiness probe settings.
# Finally, it updates the traffic distribution for the endpoint.

"""
from azure.ai.ml.entities import OnlineRequestSettings, ProbeSettings

# create a deployment
deployment_name = "phi-3-vision"
demo_deployment = ManagedOnlineDeployment(
    name=deployment_name,
    endpoint_name=online_endpoint_name,
    model=foundation_model.id,
    instance_type="Standard_NC48ads_A100_v4",
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
endpoint.traffic = {deployment_name: 100}
workspace_ml_client.begin_create_or_update(endpoint).result()
"""
```

### 3. Test the endpoint with sample data

We will send a sample request to the model, using the json that we create below.


```python
# Import necessary modules
import json
import os

# Define the test JSON payload
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

# Save the JSON object to a file
sample_score_file_path = os.path.join(".", "sample_chat_completions_score.json")
with open(sample_score_file_path, "w") as f:
    json.dump(test_json, f, indent=4)

# Print the input payload
print("Input payload:\n")
print(test_json)
```


```python
# Import necessary modules
import pandas as pd

# score the sample_chat_completions_score.json file using the online endpoint with the azureml endpoint invoke method
response = workspace_ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name=deployment_name,
    request_file=sample_score_file_path,
)
print("Raw JSON Response: \n", response, "\n")

# Parse the JSON string
json_data = json.loads(response)

# Convert the parsed JSON to a DataFrame
response_df = pd.DataFrame([json_data])
print("Generated Text:\n", response_df["output"].iloc[0])
```

### 4. Test the endpoint using Azure OpenAI style payload

We will send a sample request with Azure OpenAI Style payload to the model.


```python
# This code defines a JSON payload for testing the online endpoint with Azure OpenAI style payload.
# It includes the model name, a list of messages with user role and content (image URL and text),
# temperature, and max_new_tokens.

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


```python
# Get the scoring uri
scoring_uri = workspace_ml_client.online_endpoints.get(
    name=online_endpoint_name
).scoring_uri
# Update the scoring uri to use for AOAI
aoai_format_scoring_uri = scoring_uri.replace("/score", "/v1/chat/completions")

# Get the key for data plane operation
data_plane_token = workspace_ml_client.online_endpoints.get_keys(
    name=online_endpoint_name
).primary_key
```


```python
import urllib.request
import json

# Prepare request
body = str.encode(json.dumps(aoai_test_json))
url = aoai_format_scoring_uri
api_key = data_plane_token

headers = {"Content-Type": "application/json", "Authorization": ("Bearer " + api_key)}
req = urllib.request.Request(url, body, headers)

# Send request & get response
try:
    response = urllib.request.urlopen(req)
    result = response.read().decode("utf-8")
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))
    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", "ignore"))
```

### 5. Delete the online endpoint
Don't forget to delete the online endpoint, else you will leave the billing meter running for the compute used by the endpoint.


```python
#Delete Workspace
workspace_ml_client.online_endpoints.begin_delete(name=online_endpoint_name).wait(#)
```