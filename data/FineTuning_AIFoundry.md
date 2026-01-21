# Guide: Fine-tuning Phi-3 with Azure AI Foundry

This guide walks you through the process of fine-tuning Microsoft's Phi-3 Mini language model using Azure AI Foundry. Fine-tuning adapts the base model to your specific tasks, improving its performance and contextual awareness.

## Prerequisites & Setup

Before you begin, ensure you have the following:

*   An active Azure subscription. [Create a paid account](https://azure.microsoft.com/pricing/purchase-options/pay-as-you-go) if needed.
*   Access to [Azure AI Foundry](https://ai.azure.com).
*   Your user account must have the **Azure AI Developer role** assigned on the resource group you plan to use.

> **Important Regional Note:** For the Phi-3 model family, the pay-as-you-go fine-tuning service is currently only available in projects created within the **East US 2** region.

### Step 1: Register the Required Resource Provider

Your Azure subscription must be registered with the `Microsoft.Network` resource provider.

1.  Sign in to the [Azure portal](https://portal.azure.com).
2.  Navigate to **Subscriptions** and select your target subscription.
3.  Go to **AI project settings** > **Resource providers**.
4.  Verify that **Microsoft.Network** is listed. If it's not, select **+ Add** to register it.

## Part 1: Prepare Your Training Data

Effective fine-tuning requires a properly formatted dataset. For the `Phi-3-mini-128k-instruct` model, your data must be in the conversational format used by the Chat Completions API and saved as a JSON Lines (JSONL) file.

### Data Format Specification

Each line in your `.jsonl` file must be a valid JSON object containing a `"messages"` array. This array should include the conversation roles: `system`, `user`, and `assistant`.

**Example `training_data.jsonl`:**
```json
{"messages": [{"role": "system", "content": "You are an Xbox customer support agent whose primary goal is to help users with issues they are experiencing with their Xbox devices. You are friendly and concise. You only provide factual answers to queries, and do not provide answers that are not related to Xbox."}, {"role": "user", "content": "Is Xbox better than PlayStation?"}, {"role": "assistant", "content": "I apologize, but I cannot provide personal opinions. My primary job is to assist you with any issues related to your Xbox device. Do you have any Xbox-related issues that need addressing?"}]}
{"messages": [{"role": "system", "content": "You are an Xbox customer support agent whose primary goal is to help users with issues they are experiencing with their Xbox devices. You are friendly and concise. You only provide factual answers to queries, and do not provide answers that are not related to Xbox."}, {"role": "user", "content": "Can I play PC games on my Xbox?"}, {"role": "assistant", "content": "While Xbox consoles don't natively support PC games, they can play an array of Xbox games, many of which are also available on PC. Can I assist you with finding games compatible with your Xbox device?"}]}
```

**Best Practices:**
*   Ensure your dataset is balanced and diverse, covering various scenarios relevant to your task.
*   Clean your data by removing noise and handling any missing values.
*   Periodically refine the dataset to keep it aligned with real-world expectations.

## Part 2: Fine-Tune the Model in Azure AI Foundry

### Step 2: Create an AI Foundry Project and Hub

1.  Sign in to [Azure AI Foundry](https://ai.azure.com).
2.  Select **+ New project**.
3.  In the creation wizard:
    *   Provide a unique **Project name**.
    *   Select or create a new **Hub**. To create a hub:
        *   Enter a unique **Hub name**.
        *   Select your Azure **Subscription** and **Resource Group**.
        *   Set the **Location** to **East US 2**.
        *   For **Connect Azure AI Search**, select **Skip connecting**.
4.  Select **Next** and then **Create a project**.

### Step 3: Initiate the Fine-Tuning Job

1.  Within your project, select **Model catalog** from the left navigation pane.
2.  In the search bar, type `phi-3` and select the specific Phi-3 model you wish to fine-tune (e.g., `Phi-3-mini-128k-instruct`).
3.  On the model details page, select the **Fine-tune** button.
4.  **Basic Configuration:**
    *   Enter a descriptive **Fine-tuned model name**.
    *   Select **Next**.
5.  **Data Selection:**
    *   Set the **Task type** to **Chat completion**.
    *   Under **Training data**, upload your prepared `.jsonl` file from your local machine or select it from your project's data assets.
    *   Select **Next**.
6.  **Validation Data:**
    *   You can either upload a separate validation dataset or select **Automatic split of training data** to let the system create a validation set for you.
    *   Select **Next**.
7.  **Hyperparameter Tuning (Optional):**
    *   Adjust the training parameters if needed:
        *   **Batch size multiplier**
        *   **Learning rate**
        *   **Number of epochs**
    *   Select **Next**.
8.  **Review and Submit:**
    *   Review your configuration.
    *   Select **Submit** to start the fine-tuning job.

### Step 4: Monitor and Deploy the Fine-Tuned Model

1.  The fine-tuning job will now run. You can monitor its progress from the **Fine-tuning** section of your project.
2.  Once the status changes to **Completed**, your model is ready.
3.  You can now deploy this custom model for inference. You can deploy it to a real-time endpoint for use in your applications, test it in the playground, or integrate it into a Prompt Flow.
    > **Next Step:** For detailed deployment instructions, refer to the guide: [How to deploy Phi-3 family of small language models with Azure AI Foundry](https://learn.microsoft.com/azure/ai-studio/how-to/deploy-models-phi-3?tabs=phi-3-5&pivots=programming-language-python).

## Part 3: Post-Training Management

### Cleaning Up Resources

You can delete a fine-tuned model to manage costs.
1.  Navigate to the **Fine-tuning** page in your AI Foundry project.
2.  Select the model you wish to delete.
3.  Select the **Delete** button.
    > **Note:** You must delete any existing deployments of the model before you can delete the model itself.

### Understanding Costs and Safety

*   **Pricing:** Fine-tuning and deploying Phi-3 models incurs costs. You can view the specific pricing during the deployment or fine-tuning workflow under the **Pricing and terms** tab.
*   **Content Safety:** Models deployed as a pay-as-you-go service are protected by **Azure AI Content Safety** by default. This filters both input prompts and output completions for harmful content. You can opt out of this feature when deploying to a real-time endpoint. Learn more about [Azure AI Content Safety](https://learn.microsoft.com/azure/ai-studio/concepts/content-filtering).

## Summary and Iteration

Fine-tuning is an iterative process. To improve your model:
1.  **Evaluate** its performance using validation metrics.
2.  **Refine** your training dataset by adding more examples or correcting existing ones.
3.  **Adjust** hyperparameters like learning rate or number of epochs.
4.  **Re-run** the fine-tuning job and compare results.

By following this guide, you can successfully customize the Phi-3 model for your specific application within Azure AI Foundry.

> **For more detailed information, visit the official documentation:** [Fine-tune Phi-3 models in Azure AI Foundry](https://learn.microsoft.com/azure/ai-studio/how-to/fine-tune-phi-3?tabs=phi-3-mini).