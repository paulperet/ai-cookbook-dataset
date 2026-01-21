# Using Phi-3 in Azure AI Foundry: A Step-by-Step Guide

Azure AI Foundry is an enterprise-grade platform for managing generative AI applications. It provides a unified environment for working with large and small language models (LLMs/SLMs), integrating enterprise data, performing fine-tuning and RAG operations, and evaluating model performance. This guide walks you through deploying and using the Phi-3 model within Azure AI Foundry.

## Prerequisites

Before you begin, ensure you have:
*   An active Azure subscription.
*   Access to [Azure AI Foundry](https://ai.azure.com).
*   (Optional) The [Azure Developer CLI](https://learn.microsoft.com/azure/developer/azure-developer-cli/overview) installed for streamlined project creation.

## Step 1: Create an Azure AI Foundry Hub and Project

Your work in AI Foundry is organized within a **Hub** (a top-level container) and **Projects** (workspaces for specific tasks).

### 1.1 Create a Hub
1.  Navigate to the [Azure AI Foundry portal](https://ai.azure.com) and sign in.
2.  From the left menu, select **Management center**.
3.  Click **All resources**, then click the dropdown arrow next to **+ New project** and select **+ New hub**.
4.  In the dialog:
    *   Enter a name for your hub (e.g., `contoso-hub`).
    *   Configure the remaining fields (Subscription, Resource Group, Region) as needed.
5.  Click **Next**, review the summary, and click **Create**.

### 1.2 Create a Project
1.  From the AI Foundry Home page, click **+ Create project**.
2.  Provide a name for your project.
3.  Select the hub you created in the previous step from the dropdown. If you need a new hub, select **Create new hub**.
4.  Click **Create**.

You can now access your studio workspace via [ai.azure.com](https://ai.azure.com/). For more detailed instructions, refer to the official [Microsoft documentation on creating projects](https://learn.microsoft.com/azure/ai-studio/how-to/create-projects).

## Step 2: Deploy the Phi-3 Model

With your project ready, you can deploy a model from the catalog.

1.  Inside your project, navigate to the **Explore** section to open the **Model Catalog**.
2.  Search for and select the **Phi-3** model family.
3.  Choose the **Phi-3-mini-4k-instruct** variant.
4.  Click the **Deploy** button.
5.  In the deployment wizard, configure the deployment:
    *   **Deployment Type:** Select **Serverless API with Azure AI Content Safety**.
    *   **Project:** Ensure your current project is selected.
    *   **Deployment Name:** Assign a custom name for your deployment.
    *   **Compute:** Select the appropriate compute power for your needs.
    *   **Region:** Your workspace must be in **East US 2** or **Sweden Central** to use the Serverless API offering.
6.  Review the pricing and terms, then click **Deploy**.
7.  Wait for the deployment to complete. You will be redirected to the **Deployments** page upon success.

> **Note:** Your Azure account requires **Azure AI Developer** role permissions on the Resource Group to perform deployment.

## Step 3: Interact with Phi-3 in the Playground

The Playground provides a quick, no-code interface to test your deployed model.

1.  From your project's **Deployments** page, locate your Phi-3 deployment.
2.  Click **Open in playground**.
3.  A chat interface will open. You can now start a conversation with the Phi-3 model to test its capabilities.

## Step 4: Access Deployment Details for API Use

To integrate the model into your applications, you need the endpoint and credentials.

1.  Return to your project's **Deployments** page.
2.  Select your Phi-3 deployment.
3.  Note the following critical information:
    *   **Target URL:** The HTTP endpoint for your model.
    *   **Secret Key:** The authentication key required for API calls.

You can always find this information later by navigating to the **Build** tab and selecting **Deployments** from the **Components** section.

## Step 5: Explore and Use the Model API

Azure AI Foundry provides a Swagger specification for its inference endpoints, making it easy to understand the API.

1.  Construct your model's Swagger URL: `https://{Your-Deployment-Name}.{region}.inference.ml.azure.com/swagger.json`
    *   Replace placeholders with your actual deployment name and region (e.g., `eastus2`).
2.  You can access this URL directly in a browser or import it into a tool like **Postman**.
3.  Use the **Secret Key** (from Step 4) as the `Authorization` header (typically as a `Bearer` token) in your API requests.

The Swagger documentation will detail all available endpoints (e.g., `/chat/completions`), required request parameters, and the structure of the response, allowing you to integrate Phi-3 into your applications seamlessly.