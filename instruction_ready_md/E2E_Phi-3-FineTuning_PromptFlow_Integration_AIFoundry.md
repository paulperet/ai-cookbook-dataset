# Fine-tune and Integrate Custom Phi-3 Models with Prompt Flow in Azure AI Foundry

This end-to-end guide demonstrates how to fine-tune, deploy, and integrate a custom Phi-3 model using Azure AI Foundry and Azure Machine Learning Studio. You will establish a complete workflow for creating and utilizing a custom AI model, from setting up resources to interacting with it via a chat interface.

## Overview

This tutorial is divided into three main scenarios:

1.  **Set up Azure resources and prepare for fine-tuning.**
2.  **Fine-tune the Phi-3 model and deploy it in Azure Machine Learning Studio.**
3.  **Integrate the model with Prompt Flow and chat with it in Azure AI Foundry.**

By the end, you will have a deployed, fine-tuned Phi-3 model accessible through a custom Prompt Flow application.

## Prerequisites

*   An active Azure subscription (Pay-As-You-Go type is required for GPU quota requests).
*   Basic familiarity with the Azure portal and Python.

## Scenario 1: Set Up Azure Resources and Prepare for Fine-Tuning

### Step 1: Create an Azure Machine Learning Workspace

1.  In the Azure portal, search for **Azure Machine Learning** and select it from the results.
2.  Click **+ Create** and then select **New workspace**.
3.  Fill in the required details:
    *   **Subscription:** Your Azure subscription.
    *   **Resource group:** Create a new one or select an existing group.
    *   **Workspace name:** Choose a unique name.
    *   **Region:** Select your preferred region.
    *   **Storage account, Key vault, Application insights, Container registry:** You can create new instances for each or select existing ones associated with your workspace.
4.  Click **Review + create**, then **Create** to provision the workspace.

### Step 2: Request GPU Quotas

Fine-tuning and deployment require specific GPU SKUs. You must request quota increases for them.

> **Note:** Only Pay-As-You-Go subscriptions are eligible for GPU quota requests.

1.  Navigate to [Azure ML Studio](https://ml.azure.com/).
2.  In the left sidebar, select **Quota**.
3.  To request quota for the fine-tuning GPU (`Standard_NC24ads_A100_v4`):
    *   Under **Virtual machine family**, select **Standard NCADSA100v4 Family Cluster Dedicated vCPUs**.
    *   Click **Request quota**.
    *   Enter your desired **New cores limit** (e.g., `24`) and click **Submit**.
4.  To request quota for the deployment GPU (`Standard_NC6s_v3`):
    *   Under **Virtual machine family**, select **Standard NCSv3 Family Cluster Dedicated vCPUs**.
    *   Click **Request quota**.
    *   Enter your desired **New cores limit** (e.g., `24`) and click **Submit**.

### Step 3: Create and Configure a User-Assigned Managed Identity (UAI)

A Managed Identity is required for secure authentication during model deployment.

#### Create the Managed Identity
1.  In the Azure portal, search for **Managed Identities** and select it.
2.  Click **+ Create**.
3.  Provide the details: your **Subscription**, **Resource group**, **Region**, and a unique **Name**.
4.  Click **Review + create**, then **Create**.

#### Assign Required Roles
The identity needs specific permissions to access resources.

1.  **Add Contributor Role:**
    *   Navigate to your newly created Managed Identity resource.
    *   Select **Azure role assignments** > **+ Add role assignment**.
    *   Set **Scope** to `Resource group`, select your subscription and resource group.
    *   Choose the **Contributor** role and click **Save**.

2.  **Add Storage Blob Data Reader Role:**
    *   In the portal, search for **Storage accounts** and select the storage account linked to your Azure ML workspace.
    *   Go to **Access Control (IAM)** > **+ Add** > **Add role assignment**.
    *   Search for and select the **Storage Blob Data Reader** role. Click **Next**.
    *   For **Members**, select **Managed identity**. Click **+ Select members**.
    *   Choose your subscription, select **Managed identity**, pick the UAI you created, and click **Select**.
    *   Click **Review + assign**.

3.  **Add AcrPull Role:**
    *   Search for **Container registries** and select the registry linked to your Azure ML workspace.
    *   Go to **Access Control (IAM)** > **+ Add** > **Add role assignment**.
    *   Search for and select the **AcrPull** role. Click **Next**.
    *   Assign this role to your Managed Identity using the same member selection process as above.
    *   Click **Review + assign**.

### Step 4: Set Up a Local Project Environment

You'll prepare a dataset locally before uploading it for fine-tuning.

1.  **Create a Project Folder:**
    Open a terminal and run the following commands to create and navigate into a working directory.
    ```bash
    mkdir finetune-phi
    cd finetune-phi
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate.bat
    # On macOS/Linux:
    # source .venv/bin/activate
    ```

3.  **Install Required Package:**
    ```bash
    pip install datasets==2.19.1
    ```

### Step 5: Prepare the Fine-Tuning Dataset

You will download and prepare a subset of the `ultrachat_200k` dataset.

1.  **Create the Download Script:**
    In your `finetune-phi` folder, create a file named `download_dataset.py` and add the following code:

    ```python
    import json
    import os
    from datasets import load_dataset

    def load_and_split_dataset(dataset_name, config_name, split_ratio):
        """
        Load and split a dataset.
        """
        dataset = load_dataset(dataset_name, config_name, split=split_ratio)
        print(f"Original dataset size: {len(dataset)}")
        
        split_dataset = dataset.train_test_split(test_size=0.2)
        print(f"Train dataset size: {len(split_dataset['train'])}")
        print(f"Test dataset size: {len(split_dataset['test'])}")
        
        return split_dataset

    def save_dataset_to_jsonl(dataset, filepath):
        """
        Save a dataset to a JSONL file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for record in dataset:
                json.dump(record, f)
                f.write('\n')
        
        print(f"Dataset saved to {filepath}")

    def main():
        """
        Main function to load, split, and save the dataset.
        """
        # Using only 1% of the dataset for tutorial speed
        dataset = load_and_split_dataset("HuggingFaceH4/ultrachat_200k", 'default', 'train_sft[:1%]')
        
        train_dataset = dataset['train']
        test_dataset = dataset['test']

        save_dataset_to_jsonl(train_dataset, "data/train_data.jsonl")
        save_dataset_to_jsonl(test_dataset, "data/test_data.jsonl")

    if __name__ == "__main__":
        main()
    ```

2.  **Run the Script:**
    Execute the script to download and split the dataset. The files will be saved in a `data/` subfolder.
    ```bash
    python download_dataset.py
    ```

> **Note:** Using `'train_sft[:1%]'` significantly reduces the dataset size, making the fine-tuning process faster for this tutorial. You can adjust this percentage based on your needs.

## Scenario 2: Fine-tune and Deploy the Phi-3 Model

### Step 1: Create a Compute Cluster for Fine-Tuning

1.  In [Azure ML Studio](https://ml.azure.com/), select **Compute** from the left sidebar.
2.  Go to **Compute clusters** and click **+ New**.
3.  Configure the cluster:
    *   **Virtual machine tier:** Dedicated
    *   **Virtual machine type:** GPU
    *   **Virtual machine size:** `Standard_NC24ads_A100_v4`
4.  Click **Next**.
5.  Provide a unique **Compute name**, set **Minimum nodes** to `0`, **Maximum nodes** to `1`, and configure the idle time before scale-down.
6.  Click **Create**.

### Step 2: Fine-Tune the Phi-3 Model

1.  In your Azure ML workspace, go to **Model catalog**.
2.  Search for `phi-3-mini-4k` and select the **Phi-3-mini-4k-instruct** model.
3.  Click the **Fine-tune** button.
4.  Configure the fine-tuning job:
    *   **Task type:** Chat completion
    *   **Training data:** Upload your local `train_data.jsonl` file.
    *   **Validation data:** Select "Provide different validation data" and upload your `test_data.jsonl` file.
5.  (Optional) Click **Advanced settings** to customize hyperparameters like `learning_rate`.
6.  Click **Finish** to submit the job.

The fine-tuning process will take some time. You can monitor its progress under the **Jobs** section of your workspace.

### Step 3: Register the Fine-Tuned Model

Once the job completes successfully, register the output as a model.

1.  In your workspace, go to **Models** and click **+ Register**.
2.  Select **From a job output**.
3.  Choose the fine-tuning job you just ran.
4.  Click **Next**.
5.  Set **Model type** to `MLflow`. Ensure the correct job output is selected.
6.  Click **Next**, then **Register**.

Your model will now appear in the **Models** list.

### Step 4: Deploy the Model to an Online Endpoint

1.  Navigate to **Endpoints** > **Real-time endpoints** and click **Create**.
2.  In the model selection step, choose the Phi-3 model you just registered.
3.  Configure the deployment:
    *   **Virtual machine:** `Standard_NC6s_v3`
    *   **Instance count:** `1`
    *   **Endpoint:** Create a **New** endpoint and give it a unique name.
    *   **Deployment name:** Provide a unique name for this deployment.
4.  Click **Deploy**.

> **Warning:** Remember to delete this endpoint when you're finished to avoid incurring unnecessary costs.

### Step 5: Verify Deployment and Configure Traffic

1.  Go to **Endpoints** and select your newly created endpoint.
2.  Wait for the deployment status to change to **Healthy**.
3.  Once healthy, ensure **Live traffic** is set to **100%** for the deployment. If it's `0%`, click **Update traffic** to adjust it. You cannot test the model if traffic is set to `0%`.

## Scenario 3: Integrate with Prompt Flow and Chat

### Step 1: Create an Azure AI Foundry Hub and Project

A Hub organizes your AI projects within Azure AI Foundry.

1.  Go to [Azure AI Foundry](https://ai.azure.com/).
2.  Select **All hubs** > **+ New hub**.
3.  Provide a unique **Hub name**, select your **Subscription**, **Resource group**, and **Location**. You can skip connecting Azure AI Search for this tutorial.
4.  Click **Next** to create the hub.
5.  Inside your new hub, go to **All projects** > **+ New project**.
6.  Give your project a unique name and click **Create a project**.

### Step 2: Create a Custom Connection for Your Model

You need to store your model endpoint's URL and key securely.

1.  **Get Endpoint Details:**
    *   In Azure ML Studio, go to **Endpoints** and select your deployed endpoint.
    *   Go to the **Consume** tab.
    *   Copy the **REST endpoint** URL and the **Primary key**.

2.  **Create the Connection in AI Foundry:**
    *   In your AI Foundry project, go to **Settings**.
    *   Click **+ New connection**.
    *   Select **Custom keys**.
    *   Add two key-value pairs:
        *   Key: `endpoint`, Value: `[Your REST endpoint URL]`
        *   Key: `key`, Value: `[Your Primary key]`
    *   For the `key` entry, check the **is secret** box.
    *   Click **Add connection**.

### Step 3: Create and Configure a Prompt Flow

1.  In your AI Foundry project, select **Prompt flow** > **+ Create**.
2.  Choose **Chat flow**, provide a **Folder name**, and click **Create**.
3.  You will rebuild the default flow. Enable **Raw file mode**.
4.  Replace the entire contents of the `flow.dag.yml` file with the following configuration:

    ```yaml
    inputs:
      input_data:
        type: string
        default: "Who founded Microsoft?"

    outputs:
      answer:
        type: string
        reference: ${integrate_with_promptflow.output}

    nodes:
    - name: integrate_with_promptflow
      type: python
      source:
        type: code
        path: integrate_with_promptflow.py
      inputs:
        input_data: ${inputs.input_data}
    ```
    Click **Save**.

5.  Now, open or create the `integrate_with_promptflow.py` file and paste the following code:

    ```python
    import logging
    import requests
    from promptflow import tool
    from promptflow.connections import CustomConnection

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG
    )
    logger = logging.getLogger(__name__)

    def query_phi3_model(input_data: str, connection: CustomConnection) -> str:
        """
        Send a request to the Phi-3 model endpoint.
        """
        endpoint_url = connection.endpoint
        api_key = connection.key

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "input_data": {
                "input_string": [
                    {"role": "user", "content": input_data}
                ],
                "parameters": {
                    "temperature": 0.7,
                    "max_new_tokens": 128
                }
            }
        }
        try:
            response = requests.post(endpoint_url, json=data, headers=headers)
            response.raise_for_status()
            
            logger.debug(f"Full JSON response: {response.json()}")
            result = response.json()["output"]
            logger.info("Successfully received response from Azure ML Endpoint.")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying Azure ML Endpoint: {e}")
            raise

    @tool
    def my_python_tool(input_data: str, connection: CustomConnection) -> str:
        """
        Tool function to process input data and query the Phi-3 model.
        """
        return query_phi3_model(input_data, connection)
    ```

6.  In the flow designer, select the **Chat input** and **Chat output** nodes to enable the interactive chat interface.

### Step 4: Run the Flow and Chat with Your Model

1.  Click **Start compute sessions** to provision the runtime for your flow.
2.  Click **Validate and parse input** to refresh the node parameters.
3.  In the properties panel for the `integrate_with_promptflow` node, set the **connection** value to the name of the custom connection you created earlier.
4.  Click the **Chat** button to open the interactive chat pane.
5.  You can now ask questions and receive responses from your fine-tuned custom Phi-3 model. For best results, try questions related to the chat data used during fine-tuning.

## Conclusion

You have successfully completed an end-to-end workflow for customizing a Phi-3 model. You set up the necessary Azure resources, fine-tuned the model on a custom dataset, deployed it for real-time inference, and integrated it into an interactive chat application using Prompt Flow in Azure AI Foundry.