# Guide: Evaluating a Fine-tuned Phi-3 Model in Azure AI Foundry

This guide walks you through evaluating a fine-tuned Phi-3 or Phi-3.5 model in Azure AI Foundry, focusing on Microsoft's Responsible AI principles. You will learn to assess both the safety and performance of your model using built-in evaluation tools.

## Prerequisites

Before starting, ensure you have:
* An Azure subscription.
* A fine-tuned and deployed Phi-3 or Phi-3.5 model in Azure Machine Learning.
* A Prompt flow integrated with your model in Azure AI Foundry.
* A test dataset (e.g., `test_data.jsonl` from the ULTRACHAT_200k dataset).

## Part 1: Understanding Evaluation in Azure AI Foundry

Azure AI Foundry provides a structured evaluation process for generative AI applications. Evaluation is split into two key areas: **Safety** and **Performance**.

### Safety Evaluation & Responsible AI Principles

Safety evaluation assesses your model's alignment with Microsoft's Responsible AI Principles:
* **Fairness and Inclusiveness**: AI should treat everyone fairly.
* **Reliability and Safety**: Systems must operate reliably and resist harmful manipulation.
* **Transparency**: Users should understand how decisions are made.
* **Privacy and Security**: Data must be protected.
* **Accountability**: Designers are accountable for their system's operations.

**Safety Metrics** measure the model's tendency to generate:
* Self-harm-related content.
* Hateful and unfair content.
* Violent content.
* Sexual content.

### Performance Evaluation

Performance evaluation measures the model's effectiveness using metrics like:
* **Groundedness**: Alignment of answers with source information.
* **Relevance**: Pertinence of responses to questions.
* **Coherence**: Logical flow and readability.
* **Fluency**: Language proficiency.
* **GPT Similarity**: Comparison to a ground truth answer.
* **F1 Score**: Ratio of shared words between response and source.

## Part 2: Prepare Your Environment

This tutorial assumes you have already fine-tuned a Phi-3 model and integrated it with a Prompt flow using a code-first approach. If you used a low-code approach, you can skip to [Part 3](#part-3-deploy-an-azure-openai-evaluator).

### Step 1: Create an Azure AI Foundry Hub and Project

A Hub organizes your projects. A Project is your working environment.

1.  Sign in to [Azure AI Foundry](https://ai.azure.com/).
2.  Select **All hubs** from the left sidebar and click **+ New hub**.
3.  Fill in the details:
    *   **Hub name**: A unique name.
    *   **Subscription**: Your Azure subscription.
    *   **Resource group**: Select or create a new one.
    *   **Location**: Choose a region.
    *   **Connect Azure AI Services**: Create a new one if needed.
    *   **Connect Azure AI Search**: Select **Skip connecting**.
4.  Click **Next** and create the hub.
5.  Inside your new hub, select **All projects** and click **+ New project**.
6.  Enter a unique **Project name** and click **Create a project**.

### Step 2: Add a Custom Connection for Your Model

To allow Prompt flow to call your fine-tuned model, you must store its endpoint and key in a custom connection.

1.  Go to your Azure Machine Learning workspace and navigate to **Endpoints**.
2.  Select the endpoint for your fine-tuned Phi-3 model.
3.  Go to the **Consume** tab and copy the **REST endpoint** and **Primary key**.
4.  Return to your Azure AI Foundry project.
5.  Go to **Settings** > **Connections** and click **+ New connection**.
6.  Select **Custom keys**.
7.  Add two key-value pairs:
    *   Key: `endpoint`, Value: *(paste your REST endpoint)*. **Do not** mark as secret.
    *   Key: `key`, Value: *(paste your Primary key)*. **Check** the `is secret` box.
8.  Click **Add connection**.

### Step 3: Create and Configure a Prompt Flow

Now, create a Prompt flow that uses your custom model via the connection.

1.  In your project, go to **Prompt flow** and click **+ Create**.
2.  Select **Chat flow**.
3.  Enter a **Folder name** and click **Create**.
4.  In the new flow editor, enable **Raw file mode**.
5.  Replace the entire contents of the `flow.dag.yml` file with the following configuration:

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

6.  Create a new file called `integrate_with_promptflow.py` in the same directory and paste the following code:

```python
import logging
import requests
from promptflow import tool
from promptflow.connections import CustomConnection

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

def query_phi3_model(input_data: str, connection: CustomConnection) -> str:
    """
    Send a request to the Phi-3 / Phi-3.5 model endpoint with the given input data using Custom Connection.
    """
    # "connection" is the name of the Custom Connection, "endpoint", "key" are the keys in the Custom Connection
    endpoint_url = connection.endpoint
    api_key = connection.key

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "input_data": [input_data],
        "params": {
            "temperature": 0.7,
            "max_new_tokens": 128,
            "do_sample": True,
            "return_full_text": True
        }
    }
    try:
        response = requests.post(endpoint_url, json=data, headers=headers)
        response.raise_for_status()
        
        # Log the full JSON response
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
    Tool function to process input data and query the Phi-3 / Phi-3.5 model.
    """
    return query_phi3_model(input_data, connection)
```

7.  Save all files.
8.  In the flow editor, enable **Chat input** and **Chat output** from the top menu.
9.  Click **Start compute sessions** to initialize the runtime.
10. Click **Validate and parse input** to refresh the node parameters.
11. For the `connection` input of the `integrate_with_promptflow` node, select the custom connection you created earlier.
12. You can now test the flow by clicking the **Chat** button and sending a message.

## Part 3: Deploy an Azure OpenAI Evaluator

To evaluate your model, you need another AI model as the judge. Azure OpenAI's GPT-4 is commonly used for this.

1.  In your Azure AI Foundry project, go to **Deployments**.
2.  Click **+ Deploy model** and select **Deploy base model**.
3.  Choose an Azure OpenAI model like **gpt-4o** and click **Confirm**. Wait for the deployment to complete.

## Part 4: Run the Evaluation

With your Prompt flow and evaluator ready, you can now run a comprehensive evaluation.

1.  In your project, go to **Evaluation** and click **+ New evaluation**.
2.  Select **Prompt flow** evaluation.
3.  **Configure the evaluation**:
    *   **Evaluation name**: Enter a unique name.
    *   **Task type**: Select **Question and answer without context** (suitable for the ULTRACHAT dataset).
    *   **Target**: Select the Prompt flow you created.
    *   Click **Next**.
4.  **Select your dataset**:
    *   Click **Add your dataset** and upload your `test_data.jsonl` file.
    *   For the **Dataset column**, map it to `${data.prompt}`.
    *   Click **Next**.
5.  **Configure metrics**:
    *   **Performance & Quality Tab**: Select metrics like *Groundedness*, *Relevance*, *Coherence*, and *Fluency*. For the **Evaluator model**, select the Azure OpenAI model you deployed (e.g., `gpt-4o`).
    *   **Risk & Safety Tab**: Select safety metrics like *Hateful and unfair content*. Set the **Threshold** to `Medium`. Map the data sources:
        *   **question**: `${data.prompt}`
        *   **answer**: `${run.outputs.answer}`
        *   **ground_truth**: `${data.message}`
    *   Click **Next**.
6.  Review the summary and click **Submit** to start the evaluation. This process may take several minutes.

## Part 5: Review the Results

Once the evaluation completes, you can analyze the results.

1.  The main dashboard shows aggregate scores for **Performance and quality metrics** and **Risk and safety metrics**.
2.  Scroll down to view **Detailed metrics result**, which provides scores for each individual metric across your test dataset.
3.  Use these results to:
    *   Verify your model's effectiveness (high performance scores).
    *   Confirm its safety and alignment with Responsible AI principles (low risk scores).
    *   Identify specific areas for improvement.

## Cleanup

To avoid incurring unnecessary costs, remember to delete the Azure resources you created:
* The Azure Machine Learning workspace and model endpoint.
* The Azure AI Foundry Project and Hub.
* The deployed Azure OpenAI model.

## Next Steps and Resources

*   **Documentation**:
    *   [Assess AI systems by using the Responsible AI dashboard](https://learn.microsoft.com/azure/machine-learning/concept-responsible-ai-dashboard)
    *   [Evaluation metrics for generative AI](https://learn.microsoft.com/azure/ai-studio/concepts/evaluation-metrics-built-in)
    *   [Azure AI Foundry documentation](https://learn.microsoft.com/azure/ai-studio/)
*   **Training**:
    *   [Introduction to Microsoft's Responsible AI Approach](https://learn.microsoft.com/training/modules/introduction-to-microsofts-responsible-ai-approach/)
    *   [Introduction to Azure AI Foundry](https://learn.microsoft.com/training/modules/introduction-to-azure-ai-studio/)

By completing this guide, you have successfully evaluated your fine-tuned Phi-3 model, ensuring it is both performant and responsible, ready for real-world application.