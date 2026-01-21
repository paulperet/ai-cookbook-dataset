# Fine-Tuning Phi-3-mini-4k-instruct for Chat Completion with Azure ML

This guide walks you through fine-tuning the **Phi-3-mini-4k-instruct** model for a conversational chat-completion task using the **ultrachat_200k** dataset. You will use the Azure Machine Learning SDK to orchestrate the fine-tuning pipeline, register the resulting model, deploy it to a managed online endpoint for real-time inference, and finally clean up resources.

## Prerequisites

Before you begin, ensure you have the following:
*   An active Azure subscription.
*   An Azure Machine Learning workspace.
*   Appropriate permissions to create compute resources, run jobs, and deploy models within the workspace.
*   Python 3.8 or later installed in your environment.

## 1. Setup and Installation

First, install the required Python libraries.

```bash
pip install azure-ai-ml
pip install azure-identity
pip install datasets==2.9.0
pip install mlflow
pip install azureml-mlflow
```

Next, configure your connection to the Azure ML workspace and registry.

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute
import time

# Authenticate to Azure
try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

# Connect to your Azure ML workspace
try:
    workspace_ml_client = MLClient.from_config(credential=credential)
except:
    workspace_ml_client = MLClient(
        credential,
        subscription_id="<SUBSCRIPTION_ID>",
        resource_group_name="<RESOURCE_GROUP>",
        workspace_name="<WORKSPACE_NAME>",
    )

# Connect to the Azure ML system registry
registry_ml_client = MLClient(credential, registry_name="azureml")

# Set experiment name and a unique timestamp for tracking
experiment_name = "chat_completion_Phi-3-mini-4k-instruct"
timestamp = str(int(time.time()))
```

## 2. Select a Foundation Model

You will fine-tune the `Phi-3-mini-4k-instruct` model from the Azure ML Model Catalog. This model is a lightweight, 3.8B parameter model designed for instruction-following tasks.

```python
model_name = "Phi-3-mini-4k-instruct"
foundation_model = registry_ml_client.models.get(model_name, label="latest")

print(
    "\n\nUsing model name: {0}, version: {1}, id: {2} for fine tuning".format(
        foundation_model.name, foundation_model.version, foundation_model.id
    )
)
```

## 3. Configure Compute for Fine-Tuning

Fine-tuning requires GPU compute. The following code checks the model's tags for a list of supported compute SKUs and validates your chosen cluster.

**Important:** Ensure your compute cluster is available in your region and quota. The size must match the model's requirements to avoid "CUDA Out of Memory" errors.

```python
import ast

# Check if the model specifies allowed compute SKUs
if "finetune_compute_allow_list" in foundation_model.tags:
    computes_allow_list = ast.literal_eval(
        foundation_model.tags["finetune_compute_allow_list"]
    )
    print(f"Please create a compute from the list: {computes_allow_list}")
else:
    computes_allow_list = None
    print("`finetune_compute_allow_list` tag not found in model metadata.")

# Define your compute cluster name and size
compute_cluster = "<YOUR_COMPUTE_CLUSTER_NAME>"
compute_cluster_size = "<YOUR_COMPUTE_SKU>"  # e.g., "Standard_NC24rs_v3"

# Validate the compute cluster
try:
    compute = workspace_ml_client.compute.get(compute_cluster)
except Exception as e:
    print(e)
    raise ValueError(
        f"Compute size {compute_cluster_size} not available in workspace"
    )

if compute.provisioning_state.lower() == "failed":
    raise ValueError(
        f"Provisioning failed for compute '{compute_cluster}'. Please try a different compute."
    )

# Validate against the allow list or a default deny list
if computes_allow_list is not None:
    computes_allow_list_lower_case = [x.lower() for x in computes_allow_list]
    if compute.size.lower() not in computes_allow_list_lower_case:
        raise ValueError(
            f"VM size {compute.size} is not in the allow-listed computes for finetuning"
        )
else:
    # Basic check for known unsupported SKUs
    unsupported_gpu_vm_list = [
        "standard_nc6",
        "standard_nc12",
        "standard_nc24",
        "standard_nc24r",
    ]
    if compute.size.lower() in unsupported_gpu_vm_list:
        raise ValueError(
            f"VM size {compute.size} is currently not supported for finetuning"
        )

# Determine the number of GPUs per node in the selected compute
gpu_count_found = False
workspace_compute_sku_list = workspace_ml_client.compute.list_sizes()
available_sku_sizes = []
for compute_sku in workspace_compute_sku_list:
    available_sku_sizes.append(compute_sku.name)
    if compute_sku.name.lower() == compute.size.lower():
        gpus_per_node = compute_sku.gpus
        gpu_count_found = True

if gpu_count_found:
    print(f"Number of GPUs in compute {compute.size}: {gpus_per_node}")
else:
    raise ValueError(
        f"GPU count for compute {compute.size} not found. Available SKUs: {available_sku_sizes}."
    )
```

## 4. Prepare the Training Dataset

You will use the `ultrachat_200k` dataset. To keep this example fast, you will download and use a 5% subset.

### 4.1 Download the Dataset

Run the helper script to download and format the data. Ensure `download-dataset.py` is in your working directory.

```python
import os

exit_status = os.system(
    "python ./download-dataset.py --dataset HuggingFaceH4/ultrachat_200k --download_dir ultrachat_200k_dataset --dataset_split_pc 5"
)

if exit_status != 0:
    raise Exception("Error downloading dataset")
```

### 4.2 Inspect the Data

The dataset is in JSON Lines format. Each entry contains a `prompt`, a list of `messages` (with `role` and `content`), and a `prompt_id`.

```python
import pandas as pd

pd.set_option("display.max_colwidth", 0)
df = pd.read_json("./ultrachat_200k_dataset/train_sft.jsonl", lines=True)
df.head()
```

**Example Data Structure:**
```json
{
  "prompt": "Create a fully-developed protagonist for a dystopian story...",
  "messages": [
    { "content": "Create a fully-developed protagonist...", "role": "user" },
    { "content": "Name: Ava\n\nAva was 16 when...", "role": "assistant" },
    { "content": "Can you provide more details?", "role": "user" },
    { "content": "Certainly! ...", "role": "assistant" }
  ],
  "prompt_id": "d938b65dfe31f05f80eb8572964c6673..."
}
```

## 5. Configure and Submit the Fine-Tuning Job

### 5.1 Define Fine-Tuning Parameters

Set the training and optimization parameters. Model-specific defaults from the registry will override these if present.

```python
# Default training parameters
training_parameters = dict(
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
)

# Default optimization parameters
optimization_parameters = dict(
    apply_lora="true",      # Use Low-Rank Adaptation for efficient tuning
    apply_deepspeed="true", # Use DeepSpeed for memory optimization
    deepspeed_stage=2,
)

# Combine parameters
finetune_parameters = {**training_parameters, **optimization_parameters}

# Apply model-specific defaults if they exist
if "model_specific_defaults" in foundation_model.tags:
    print("Warning! Model specific defaults exist and will be applied.")
    finetune_parameters.update(
        ast.literal_eval(foundation_model.tags["model_specific_defaults"])
    )

print(f"Fine-tuning parameters for the run: {finetune_parameters}")
```

### 5.2 Create the Pipeline Display Name

Generate a descriptive name for your pipeline run based on the parameters.

```python
def get_pipeline_display_name():
    batch_size = (
        int(finetune_parameters.get("per_device_train_batch_size", 1))
        * int(finetune_parameters.get("gradient_accumulation_steps", 1))
        * int(gpus_per_node)
        * int(finetune_parameters.get("num_nodes_finetune", 1))
    )
    scheduler = finetune_parameters.get("lr_scheduler_type", "linear")
    deepspeed = finetune_parameters.get("apply_deepspeed", "false")
    ds_stage = finetune_parameters.get("deepspeed_stage", "2")
    ds_string = f"ds{ds_stage}" if deepspeed == "true" else "nods"

    lora = finetune_parameters.get("apply_lora", "false")
    lora_string = "lora" if lora == "true" else "nolora"

    save_limit = finetune_parameters.get("save_total_limit", -1)
    seq_len = finetune_parameters.get("max_seq_length", -1)

    return (
        model_name
        + "-"
        + "ultrachat"
        + "-"
        + f"bs{batch_size}"
        + "-"
        + f"{scheduler}"
        + "-"
        + ds_string
        + "-"
        + lora_string
        + f"-save_limit{save_limit}"
        + f"-seqlen{seq_len}"
    )

pipeline_display_name = get_pipeline_display_name()
print(f"Display name for the run: {pipeline_display_name}")
```

### 5.3 Configure and Submit the Pipeline

Define the pipeline using the `chat_completion_pipeline` component from the system registry.

```python
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import Input

# Get the latest version of the pipeline component
pipeline_component_func = registry_ml_client.components.get(
    name="chat_completion_pipeline", label="latest"
)

@pipeline(name=pipeline_display_name)
def create_pipeline():
    chat_completion_pipeline = pipeline_component_func(
        mlflow_model_path=foundation_model.id,
        compute_model_import=compute_cluster,
        compute_preprocess=compute_cluster,
        compute_finetune=compute_cluster,
        compute_model_evaluation=compute_cluster,
        # Map the dataset splits
        train_file_path=Input(
            type="uri_file", path="./ultrachat_200k_dataset/train_sft.jsonl"
        ),
        test_file_path=Input(
            type="uri_file", path="./ultrachat_200k_dataset/test_sft.jsonl"
        ),
        # Training settings
        number_of_gpu_to_use_finetuning=gpus_per_node,
        **finetune_parameters
    )
    return {
        # Output for model registration
        "trained_model": chat_completion_pipeline.outputs.mlflow_model_folder
    }

# Create the pipeline object
pipeline_object = create_pipeline()
pipeline_object.settings.force_rerun = True       # Disable caching
pipeline_object.settings.continue_on_step_failure = False # Stop on failure

# Submit the job
pipeline_job = workspace_ml_client.jobs.create_or_update(
    pipeline_object, experiment_name=experiment_name
)

# Stream the job logs
workspace_ml_client.jobs.stream(pipeline_job.name)
```

Wait for the job to complete. You can monitor its progress in the Azure ML studio.

## 6. Register the Fine-Tuned Model

Once the job finishes successfully, register the output model to your workspace to track lineage and enable deployment.

```python
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

# Get the path to the trained model from the job outputs
model_path_from_job = "azureml://jobs/{0}/outputs/{1}".format(
    pipeline_job.name, "trained_model"
)

# Create a name for the fine-tuned model
finetuned_model_name = model_name + "-ultrachat-200k"
finetuned_model_name = finetuned_model_name.replace("/", "-")

print(f"Registering model from path: {model_path_from_job}")

# Prepare the model object for registration
prepare_to_register_model = Model(
    path=model_path_from_job,
    type=AssetTypes.MLFLOW_MODEL,
    name=finetuned_model_name,
    version=timestamp,  # Use timestamp for unique versioning
    description=model_name + " fine-tuned model for ultrachat 200k chat-completion",
)

# Register the model
registered_model = workspace_ml_client.models.create_or_update(
    prepare_to_register_model
)

print("Registered model:\n", registered_model)
```

## 7. Deploy the Model to a Managed Online Endpoint

Now, deploy the registered model to a REST endpoint for real-time inference.

### 7.1 Create the Online Endpoint

```python
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    ProbeSettings,
    OnlineRequestSettings,
)

# Create a unique endpoint name
online_endpoint_name = "ultrachat-completion-" + timestamp

endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="Online endpoint for "
    + registered_model.name
    + ", fine-tuned model for ultrachat-200k-chat-completion",
    auth_mode="key",
)

workspace_ml_client.begin_create_or_update(endpoint).wait()
```

### 7.2 Create the Deployment

Check the model's tags for supported inference SKUs and create a deployment.

```python
# Set your desired instance type for inference
instance_type = "Standard_NC6s_v3"

# Check for inference compute allow list
if "inference_compute_allow_list" in foundation_model.tags:
    inference_computes_allow_list = ast.literal_eval(
        foundation_model.tags["inference_compute_allow_list"]
    )
    print(f"Supported inference SKUs: {inference_computes_allow_list}")
else:
    inference_computes_allow_list = None
    print("`inference_compute_allow_list` tag not found.")

# Validate the chosen instance type
if (
    inference_computes_allow_list is not None
    and instance_type not in inference_computes_allow_list
):
    print(
        f"`instance_type` is not in the allow-listed compute. Please select from {inference_computes_allow_list}"
    )

# Configure the deployment
demo_deployment = ManagedOnlineDeployment(
    name="demo",
    endpoint_name=online_endpoint_name,
    model=registered_model.id,
    instance_type=instance_type,
    instance_count=1,
    liveness_probe=ProbeSettings(initial_delay=600),
    request_settings=OnlineRequestSettings(request_timeout_ms=90000),
)

workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()

# Route 100% of traffic to the new deployment
endpoint.traffic = {"demo": 100}
workspace_ml_client.begin_create_or_update(endpoint).result()
```

## 8. Test the Deployed Endpoint

### 8.1 Prepare a Test Sample

Load a sample from the test dataset to use as input.

```python
import pandas as pd

test_df = pd.read_json("./ultrachat_200k_dataset/test_gen.jsonl", lines=True)
test_df = test_df.sample(n=1)
test_df.reset_index(drop=True, inplace=True)
test_df.head(2)
```

### 8.2 Create the Inference Request Payload

Format the sample into the required JSON structure for the endpoint.

```python
import json

parameters = {
    "temperature": 0.6,
    "top_p": 0.9,
    "do_sample": True,
    "max_new_tokens": 200,
}

test_json = {
    "input_data": {
        "input_string": [test_df["messages"][0]],
        "parameters": parameters,
    },
    "params": {},
}

with open("./ultrachat_200k_dataset/sample_score.json", "w") as f:
    json.dump(test_json, f)
```

### 8.3 Invoke the Endpoint

Send the request to your deployed model and print the response.

```python
response = workspace_ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name="demo",
    request_file="./ultrachat_200k_dataset/sample_score.json",
)

print("Raw response:\n", response, "\n")
```

## 9. Clean Up Resources

To avoid incurring unnecessary costs, delete the online endpoint when you are finished testing.

```python
workspace_ml_client.online_endpoints.begin_delete(name=online_endpoint_name).wait()
```

**Note:** This only deletes the endpoint and its deployments. The compute cluster, registered model, and job history remain in your workspace and can be managed via the Azure ML studio.

## Summary

You have successfully:
1.  Set up your Azure ML environment and selected a foundation model.
2.  Prepared a subset of the `ultrachat_200k` dataset for fine-tuning.
3.  Configured and submitted a fine-tuning job using LoRA and DeepSpeed for efficiency.
4.  Registered the resulting fine-tuned model to your workspace.
5.  Deployed the model to a managed online endpoint for real-time inference.
6.  Tested the endpoint with a sample conversation.
7.  Cleaned up the endpoint to stop billing for the deployment compute.

This fine-tuned model is now ready to be integrated into applications requiring conversational AI capabilities.