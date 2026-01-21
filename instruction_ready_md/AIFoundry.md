# Evaluating Generative AI Applications with Azure AI Foundry

This guide walks you through the process of evaluating your generative AI application using Azure AI Foundry. You'll learn how to assess both single-turn and multi-turn conversations, as well as Retrieval Augmented Generation (RAG) scenarios, using built-in metrics and custom evaluation flows.

## Prerequisites

Before you begin, ensure you have the following:

1.  **A Test Dataset:** Your evaluation data in CSV or JSON format.
2.  **A Deployed Model:** A generative AI model (e.g., Phi-3, GPT-3.5, GPT-4, or Davinci models) deployed within your Azure AI Foundry environment.
3.  **Runtime Compute:** A compute instance configured to run the evaluation.

## Step 1: Understand Built-in Evaluation Metrics

Azure AI Foundry provides several built-in metrics to assess your model's performance. The metrics you can use depend on your scenario:

*   **For RAG Scenarios:** Evaluate how well the model generates answers grounded in your provided data.
*   **For General QA (Non-RAG):** Assess the quality of single-turn question-answering.

## Step 2: Create an Evaluation Run

You can initiate an evaluation from two primary locations in the Azure AI Foundry UI.

1.  Navigate to either the **Evaluate** page or the **Prompt Flow** page.
2.  Launch the evaluation creation wizard.
3.  Provide an optional name for your evaluation run to help identify it later.
4.  **Select your scenario** (e.g., "RAG Evaluation" or "Single-turn QA").
5.  **Choose one or more evaluation metrics** from the available list that align with your assessment goals.

## Step 3: Configure a Custom Evaluation Flow (Optional)

For more specific or complex evaluation needs, you can create a custom evaluation flow.

1.  Within the evaluation setup, select the option to create or use a custom flow.
2.  Customize the evaluation steps, logic, and scoring based on your unique requirements. This allows for tailored assessment beyond the standard metrics.

## Step 4: Run the Evaluation and View Results

Once your evaluation is configured, execute the run.

1.  Start the evaluation. The system will process your test dataset against the chosen model and metrics.
2.  After completion, navigate to the results section.
3.  **Analyze the detailed metrics** provided. Azure AI Foundry presents logs and visualizations to help you understand your application's performance, strengths, and areas for improvement.

## Important Note

Azure AI Foundry is currently in **public preview**. It is excellent for experimentation, development, and proof-of-concept work. For production-grade workloads, please evaluate and consider other supported Azure AI services.

For comprehensive details and step-by-step instructions, refer to the official [Azure AI Foundry documentation](https://learn.microsoft.com/azure/ai-studio/?WT.mc_id=aiml-138114-kinfeylo).