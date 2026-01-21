# Fine-tune and Integrate Custom Phi-3 Models with Prompt Flow: A Step-by-Step Guide

This guide walks you through the complete process of fine-tuning a Phi-3 model, deploying it on Azure Machine Learning, and integrating it into an interactive chat application using Prompt Flow.

## Overview

You will learn how to establish a workflow for customizing, deploying, and utilizing AI models. This tutorial is divided into three main scenarios:

1.  **Set up Azure resources and prepare for fine-tuning.**
2.  **Fine-tune the Phi-3 model and deploy it in Azure Machine Learning Studio.**
3.  **Integrate the model with Prompt Flow and interact with it.**

## Prerequisites

Before you begin, ensure you have:
*   An active Azure subscription.
*   [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) installed and authenticated (`az login`).
*   Python 3.10+ installed on your local machine.

---

## Scenario 1: Set Up Azure Resources and Prepare for Fine-Tuning

### Step 1: Create an Azure Machine Learning Workspace

1.  In the Azure portal, search for and select **Azure Machine Learning**.
2.  Click **+ Create** and select **New workspace**.
3.  Fill in the required details:
    *   **Subscription:** Your Azure subscription.
    *   **Resource Group:** Create a new one or select an existing group.
    *   **Workspace Name:** A unique name for your workspace.
    *   **Region:** Your preferred Azure region.
    *   **Storage Account, Key Vault, Application Insights, Container Registry:** Create new resources or select existing ones.
4.  Click **Review + create**, then **Create**.

### Step 2: Request GPU Quotas (Optional but Recommended)

For optimal fine-tuning performance, you'll need a GPU quota. This tutorial uses the `Standard_NC24ads_A100_v4` SKU.

> **Note:** Pay-As-You-Go subscriptions are required for GPU allocation. If you are using a benefit subscription (like Visual Studio Enterprise) or wish to test quickly, you can follow the CPU-based fine-tuning guidance provided later.

1.  Go to [Azure ML Studio](https://ml.azure.com).
2.  In the left menu, select **Quota**.
3.  Select **+ Request quota**.
4.  For **Virtual machine family**, choose **Standard NCADSA100v4 Family Cluster Dedicated vCPUs**.
5.  Enter your desired **New cores limit** (e.g., 24) and submit the request.

### Step 3: Create and Configure a User-Assigned Managed Identity (UAI)

A Managed Identity is required for secure authentication during deployment.

1.  In the Azure portal, search for and select **Managed Identities**.
2.  Click **+ Create**.
3.  Provide a **Name**, select your **Subscription**, **Resource Group**, and **Region**.
4.  Click **Review + create**, then **Create**.

#### Assign Required Roles to the Managed Identity

The identity needs permissions to manage resources and access storage.

1.  **Add Contributor Role:**
    *   Navigate to your newly created Managed Identity.
    *   Go to **Azure role assignments** > **+ Add role assignment**.
    *   Set **Scope** to **Resource group**, select your subscription and resource group.
    *   Choose the **Contributor** role and save.

2.  **Add Storage Blob Data Reader Role:**
    *   Navigate to the **Storage Account** associated with your Azure ML workspace.
    *   Go to **Access Control (IAM)** > **+ Add** > **Add role assignment**.
    *   Select the **Storage Blob Data Reader** role.
    *   On the **Members** tab, choose **Managed identity**, select your UAI, and complete the assignment.

3.  **Add AcrPull Role:**
    *   Navigate to the **Container Registry** associated with your Azure ML workspace.
    *   Go to **Access Control (IAM)** > **+ Add** > **Add role assignment**.
    *   Select the **AcrPull** role.
    *   Assign it to your Managed Identity as done in the previous step.

### Step 4: Set Up the Local Project Environment

Now, let's prepare your local development environment.

1.  **Create a project folder and virtual environment:**
    ```bash
    mkdir finetune-phi
    cd finetune-phi
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate.bat
    # On macOS/Linux:
    # source .venv/bin/activate
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install datasets==2.19.1 transformers==4.41.1 azure-ai-ml==1.16.0 torch==2.3.1 trl==0.9.4 promptflow==1.12.0
    ```

### Step 5: Create and Configure Project Files

Create the following files in your `finetune-phi` directory. The final structure should look like this:
```
finetune-phi/
├── finetuning_dir/
│   └── fine_tune.py
├── conda.yml
├── config.py
├── deploy_model.py
├── download_dataset.py
├── flow.dag.yml
├── integrate_with_promptflow.py
└── setup_ml.py
```

#### File 1: `conda.yml`
This defines the environment for the fine-tuning job in Azure ML.

```yaml
name: phi-3-training-env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - numpy<2.0
  - pip:
      - torch==2.4.0
      - torchvision==0.19.0
      - trl==0.8.6
      - transformers==4.41
      - datasets==2.21.0
      - azureml-core==1.57.0
      - azure-storage-blob==12.19.0
      - azure-ai-ml==1.16
      - azure-identity==1.17.1
      - accelerate==0.33.0
      - mlflow==2.15.1
      - azureml-mlflow==1.57.0
```

#### File 2: `config.py`
This file holds your Azure configuration. You will fill in the values as you progress.

```python
# Azure settings
AZURE_SUBSCRIPTION_ID = "your_subscription_id"
AZURE_RESOURCE_GROUP_NAME = "your_resource_group_name"

# Azure Machine Learning settings
AZURE_ML_WORKSPACE_NAME = "your_workspace_name"

# Azure Managed Identity settings
AZURE_MANAGED_IDENTITY_CLIENT_ID = "your_azure_managed_identity_client_id"
AZURE_MANAGED_IDENTITY_NAME = "your_azure_managed_identity_name"
AZURE_MANAGED_IDENTITY_RESOURCE_ID = f"/subscriptions/{AZURE_SUBSCRIPTION_ID}/resourceGroups/{AZURE_RESOURCE_GROUP_NAME}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/{AZURE_MANAGED_IDENTITY_NAME}"

# Dataset file paths
TRAIN_DATA_PATH = "data/train_data.jsonl"
TEST_DATA_PATH = "data/test_data.jsonl"

# Fine-tuned model settings
AZURE_MODEL_NAME = "your_fine_tuned_model_name"
AZURE_ENDPOINT_NAME = "your_fine_tuned_model_endpoint_name"
AZURE_DEPLOYMENT_NAME = "your_fine_tuned_model_deployment_name"

AZURE_ML_API_KEY = "your_fine_tuned_model_api_key"
AZURE_ML_ENDPOINT = "your_fine_tuned_model_endpoint_uri"
```

**Populate `config.py`:**
*   **Subscription ID, Resource Group, Workspace Name:** Find these in the "Overview" section of your Azure ML resource in the portal.
*   **Managed Identity Name & Client ID:** Find these in the "Overview" section of your Managed Identity resource.
*   Leave `AZURE_ML_API_KEY` and `AZURE_ML_ENDPOINT` blank for now; you will get them after deployment.

### Step 6: Prepare the Dataset for Fine-Tuning

You will download and prepare a subset of the ULTRACHAT_200k dataset.

#### File 3: `download_dataset.py`

```python
import json
import os
from datasets import load_dataset
from config import TRAIN_DATA_PATH, TEST_DATA_PATH

def load_and_split_dataset(dataset_name, config_name, split_ratio):
    dataset = load_dataset(dataset_name, config_name, split=split_ratio)
    print(f"Original dataset size: {len(dataset)}")
    split_dataset = dataset.train_test_split(test_size=0.2)
    print(f"Train dataset size: {len(split_dataset['train'])}")
    print(f"Test dataset size: {len(split_dataset['test'])}")
    return split_dataset

def save_dataset_to_jsonl(dataset, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in dataset:
            json.dump(record, f)
            f.write('\n')
    print(f"Dataset saved to {filepath}")

def main():
    # Load 1% of the dataset for faster fine-tuning in this tutorial
    dataset = load_and_split_dataset("HuggingFaceH4/ultrachat_200k", 'default', 'train_sft[:1%]')
    save_dataset_to_jsonl(dataset['train'], TRAIN_DATA_PATH)
    save_dataset_to_jsonl(dataset['test'], TEST_DATA_PATH)

if __name__ == "__main__":
    main()
```

Run the script to download the data:
```bash
python download_dataset.py
```

> **Tip for CPU Fine-Tuning:** If you are using a CPU, reduce the dataset size further for a quicker test by changing the split to `'train_sft[:10]'` (10 samples) in the `load_and_split_dataset` call.

---

## Scenario 2: Fine-tune the Phi-3 Model and Deploy It

### Step 1: Define the Fine-Tuning Script

#### File 4: `finetuning_dir/fine_tune.py`

This script contains the core logic for fine-tuning the model using the TRL library.

```python
import argparse
import sys
import logging
import os
from datasets import load_dataset
import torch
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING
)
logger = logging.getLogger(__name__)

def initialize_model_and_tokenizer(model_name, model_kwargs):
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
    return model, tokenizer

def apply_chat_template(example, tokenizer):
    messages = example["messages"]
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return example

def load_and_preprocess_data(train_filepath, test_filepath, tokenizer):
    train_dataset = load_dataset('json', data_files=train_filepath, split='train')
    test_dataset = load_dataset('json', data_files=test_filepath, split='train')
    column_names = list(train_dataset.features)

    train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to train dataset",
    )
    test_dataset = test_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to test dataset",
    )
    return train_dataset, test_dataset

def train_and_evaluate_model(train_dataset, test_dataset, model, tokenizer, output_dir):
    training_args = TrainingArguments(
        bf16=True,
        do_eval=True,
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=5.0e-06,
        logging_steps=20,
        lr_scheduler_type="cosine",
        num_train_epochs=3,
        overwrite_output_dir=True,
        per_device_eval_batch_size=4,
        per_device_train_batch_size=4,
        remove_unused_columns=True,
        save_steps=500,
        seed=0,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        warmup_ratio=0.2,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        max_seq_length=2048,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=True
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    mlflow.transformers.log_model(
        transformers_model={"model": trainer.model, "tokenizer": tokenizer},
        artifact_path=output_dir,
    )
    tokenizer.padding_side = 'left'
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(test_dataset)
    trainer.log_metrics("eval", eval_metrics)

def main(train_file, eval_file, model_output_dir):
    model_kwargs = {
        "use_cache": False,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": None,
        "attn_implementation": "eager"
    }
    # You can change this to "microsoft/Phi-3.5-mini-instruct" or other models
    pretrained_model_name = "microsoft/Phi-3-mini-4k-instruct"

    with mlflow.start_run():
        model, tokenizer = initialize_model_and_tokenizer(pretrained_model_name, model_kwargs)
        train_dataset, test_dataset = load_and_preprocess_data(train_file, eval_file, tokenizer)
        train_and_evaluate_model(train_dataset, test_dataset, model, tokenizer, model_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, required=True, help="Path to the training data")
    parser.add_argument("--eval-file", type=str, required=True, help="Path to the evaluation data")
    parser.add_argument("--model_output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    args = parser.parse_args()
    main(args.train_file, args.eval_file, args.model_output_dir)
```

### Step 2: Configure and Submit the Azure ML Fine-Tuning Job

#### File 5: `setup_ml.py`

This script configures the Azure ML environment, compute cluster, and submits the fine-tuning job.

```python
import logging
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment, AmlCompute
from azure.identity import AzureCliCredential
from config import (
    AZURE_SUBSCRIPTION_ID,
    AZURE_RESOURCE_GROUP_NAME,
    AZURE_ML_WORKSPACE_NAME,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH
)

# --- CONFIGURATION ---
# For GPU training (Recommended):
COMPUTE_INSTANCE_TYPE = "Standard_NC24ads_A100_v4"
COMPUTE_NAME = "gpu-nc24s-a100-v4"
DOCKER_IMAGE_NAME = "mcr.microsoft.com/azureml/curated/acft-hf-nlp-gpu:59"

# For CPU training (Testing):
# COMPUTE_INSTANCE_TYPE = "Standard_E16s_v3"
# COMPUTE_NAME = "cpu-e16s-v3"
# DOCKER_IMAGE_NAME = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"

CONDA_FILE = "conda.yml"
LOCATION = "eastus2"  # Change to your region
FINETUNING_DIR = "./finetuning_dir"
TRAINING_ENV_NAME = "phi-3-training-environment"
MODEL_OUTPUT_DIR = "./model_output"
# --- END CONFIGURATION ---

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.WARNING
)

def get_ml_client():
    credential = AzureCliCredential()
    return MLClient(credential, AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP_NAME, AZURE_ML_WORKSPACE_NAME)

def create_or_get_environment(ml_client):
    env = Environment(
        image=DOCKER_IMAGE_NAME,
        conda_file=CONDA_FILE,
        name=TRAINING_ENV_NAME,
    )
    return ml_client.environments.create_or_update(env)

def create_or_get_compute_cluster(ml_client, compute_name, COMPUTE_INSTANCE_TYPE, location):
    try:
        compute_cluster = ml_client.compute.get(compute_name)
        logger.info(f"Compute cluster '{compute_name}' already exists.")
    except Exception:
        logger.info(f"Creating new compute cluster '{compute_name}'.")
        compute_cluster = AmlCompute(
            name=compute_name,
            size=COMPUTE_INSTANCE_TYPE,
            location=location,
            tier="Dedicated",
            min_instances=0,
            max_instances=1
        )
        ml_client.compute.begin_create